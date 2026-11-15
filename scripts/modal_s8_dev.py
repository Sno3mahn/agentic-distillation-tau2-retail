"""S8 dev evaluation on Modal: the continued-training checkpoints on the first 8 dev tasks x 1 trial.

Same code path as the Kaggle S6/S8 notebooks (agentic_distill.student_run), one checkpoint per T4 container, in parallel.
Prereqs (run by the user, values never printed):
  .venv/bin/modal token new
  .venv/bin/modal secret create agentic-distill HF_TOKEN=... API_DEEPSEEK=...
Run:  scripts/make_kaggle_bundle.sh && .venv/bin/modal run scripts/modal_s8_dev.py
Output: data/modal_out_s8dev/<epoch>/... (raw eval records), plus the printed metrics table.
"""
import io
import pathlib
import tarfile

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAU2_COMMIT = "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4"
ADAPTERS = ROOT / "data/kaggle_out_s8/out/lora_ext"
EPOCHS = ["epoch1", "epoch2"]
N_DEV = 8  # first 8 dev tasks, same as notebooks/s8_dev_eval_kaggle.ipynb

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "unzip")
    .run_commands(
        "mkdir -p /root/repo/third_party",
        "git clone -q https://github.com/sierra-research/tau2-bench /root/repo/third_party/tau2-bench",
        f"git -C /root/repo/third_party/tau2-bench checkout -q {TAU2_COMMIT}",
        "pip install -q -e /root/repo/third_party/tau2-bench audioop-lts pyyaml",
        "pip install -q torch==2.15.0 transformers==5.23.0 peft==0.21.3 accelerate==1.15.0 safetensors==0.8.0",  # = requirements-gpu.txt
        "pip uninstall -y -q torchao || true",
    )
    .add_local_file("/tmp/agentic_distill_src.zip", "/root/src.zip")
    .add_local_dir(str(ADAPTERS), "/root/adapters", ignore=["trainer_state.pt", "train_log.jsonl"])
)
app = modal.App("agentic-distill-s8-dev", image=image)


@app.function(gpu="T4", timeout=3 * 3600, secrets=[modal.Secret.from_name("agentic-distill")])
def run(epoch: str) -> bytes:
    import json, os, subprocess, sys
    subprocess.run(["unzip", "-q", "-o", "/root/src.zip", "-d", "/root/repo"], check=True)
    import torch, transformers, peft
    print(epoch, "torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__,
          torch.cuda.get_device_name(0), flush=True)
    dev = json.load(open("/root/repo/data/splits/retail_split.json"))["dev_ids"][:N_DEV]
    env = {**os.environ, "PYTHONPATH": "src", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [sys.executable, "-m", "agentic_distill.student_run", "--tasks", *dev, "--trials", "1",
           "--slice", "eval_student", "--label", f"s8-{epoch}", "--adapter", f"/root/adapters/{epoch}",
           "--out-root", "/root/out"]
    p = subprocess.Popen(cmd, cwd="/root/repo", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith(("task=", "slice", "STOP", "Traceback")) or "Error" in line[:40]:
            print(f"[{epoch}] {line}", end="", flush=True)
    print(f"[{epoch}] exit code {p.wait()}", flush=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        if os.path.exists("/root/out"):
            t.add("/root/out", arcname=epoch)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    dst = ROOT / "data/modal_out_s8dev"
    dst.mkdir(parents=True, exist_ok=True)
    for epoch, blob in zip(EPOCHS, run.map(EPOCHS)):
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
            t.extractall(dst, filter="data")
        print("saved", dst / epoch)
