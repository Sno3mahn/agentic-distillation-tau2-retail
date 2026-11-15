"""S7b fallback on Modal: run named test tasks x 3 trials with the frozen epoch3 adapter (one T4 container per task).

Used for the "even" test-half chunks the Colab run did not finish (Colab quota ran out). Same code path and adapter
hash check as notebooks/s7b_test_eval_colab.ipynb; the environment differs (see the versions each container prints).
Prereq: modal token + secret `agentic-distill` (HF_TOKEN, API_DEEPSEEK); scripts/make_kaggle_bundle.sh after committing.
Run:  .venv/bin/modal run scripts/modal_s7b_missing.py --tasks 17,18,19,20
Output: data/modal_out_s7b/task<ID>/... (raw eval records, one run dir per task).
"""
import hashlib
import io
import pathlib
import re
import tarfile

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAU2_COMMIT = "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4"
ADAPTER = ROOT / "data/colab_upload/agentic_distill_colab/epoch3"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "unzip")
    .run_commands(
        "mkdir -p /root/repo/third_party",
        "git clone -q https://github.com/sierra-research/tau2-bench /root/repo/third_party/tau2-bench",
        f"git -C /root/repo/third_party/tau2-bench checkout -q {TAU2_COMMIT}",
        "pip install -q -e /root/repo/third_party/tau2-bench audioop-lts pyyaml",
        "pip install -q torch transformers peft accelerate safetensors",
        "pip uninstall -y -q torchao || true",
    )
    .add_local_file("/tmp/agentic_distill_src.zip", "/root/src.zip")
    .add_local_dir(str(ADAPTER), "/root/epoch3")
)
app = modal.App("agentic-distill-s7b-missing", image=image)


@app.function(gpu="T4", timeout=4 * 3600, secrets=[modal.Secret.from_name("agentic-distill")])
def run(task: str) -> bytes:
    import os, subprocess, sys
    subprocess.run(["unzip", "-q", "-o", "/root/src.zip", "-d", "/root/repo"], check=True)
    frozen = re.search(r"sha256 `([0-9a-f]{64})`", open("/root/repo/results/threshold.md").read()).group(1)
    h = hashlib.sha256(open("/root/epoch3/adapter_model.safetensors", "rb").read()).hexdigest()
    assert h == frozen, "adapter differs from the one frozen in threshold.md"
    import torch, transformers, peft
    print(f"[{task}] adapter hash ok; torch {torch.__version__} transformers {transformers.__version__} "
          f"peft {peft.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    env = {**os.environ, "PYTHONPATH": "src", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [sys.executable, "-m", "agentic_distill.student_run", "--tasks", task, "--split", "test", "--trials", "3",
           "--slice", "eval_student", "--label", "s2-epoch3", "--adapter", "/root/epoch3", "--out-root", "/root/out"]
    p = subprocess.Popen(cmd, cwd="/root/repo", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith(("task=", "slice", "STOP", "Traceback")) or "Error" in line[:40]:
            print(f"[{task}] {line}", end="", flush=True)
    print(f"[{task}] exit code {p.wait()}", flush=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        if os.path.exists("/root/out"):
            t.add("/root/out", arcname=f"task{task}")
    return buf.getvalue()


@app.local_entrypoint()
def main(tasks: str):
    ids = [t.strip() for t in tasks.split(",") if t.strip()]
    dst = ROOT / "data/modal_out_s7b"
    dst.mkdir(parents=True, exist_ok=True)
    split = __import__("json").load(open(ROOT / "data/splits/retail_split.json"))["test_ids"]
    assert all(t in map(str, split) for t in ids), "every task must be a test-split id"
    for t, blob in zip(ids, run.map(ids)):
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            tf.extractall(dst, filter="data")
        print("saved", dst / f"task{t}")
