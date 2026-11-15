"""Run agentic_distill.student_run on Modal T4s: tasks split into groups, one container per group, in parallel.

Used for DAgger rollouts (train split), and DAgger dev/test evaluation. Same code path as every other student run.
Prereq: modal token + secret `agentic-distill` (HF_TOKEN, API_DEEPSEEK); commit, then scripts/make_kaggle_bundle.sh.
Example:
  .venv/bin/modal run scripts/modal_student_eval.py --split train --tasks all --trials 2 --per-container 3 \
      --adapter data/colab_upload/agentic_distill_colab/epoch3 --label s2-epoch3 --slice dagger --out data/modal_out_dagger
Output: <out>/g<k>/data/raw/<run>/ (records + traces) per container group.
"""
import io
import json
import pathlib
import tarfile

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAU2_COMMIT = "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4"

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
)
app = modal.App("agentic-distill-student-eval", image=image)
vol = modal.Volume.from_name("agentic-distill-adapters", create_if_missing=True)


@app.function(gpu="T4", timeout=6 * 3600, secrets=[modal.Secret.from_name("agentic-distill")], volumes={"/vol": vol},
              retries=modal.Retries(max_retries=2, initial_delay=5.0))   # preempted containers ("Runner terminated") are retried
def run(job: dict) -> bytes:
    import hashlib, os, subprocess, sys
    subprocess.run(["unzip", "-q", "-o", "/root/src.zip", "-d", "/root/repo"], check=True)
    ad = f"/vol/{job['adapter']}" if job["adapter"] else None
    if ad:
        h = hashlib.sha256(open(f"{ad}/adapter_model.safetensors", "rb").read()).hexdigest()
        assert h == job["adapter_sha256"], "adapter on the volume differs from the local one"
    import torch, transformers, peft
    tag = f"g{job['group']}"
    print(f"[{tag}] tasks={job['tasks']} adapter_sha256={h[:12] if ad else None} torch {torch.__version__} "
          f"transformers {transformers.__version__} peft {peft.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    import shutil
    shutil.rmtree("/root/out", ignore_errors=True)   # a retried input may land on a used container
    env = {**os.environ, "PYTHONPATH": "src", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [sys.executable, "-m", "agentic_distill.student_run", "--tasks", *job["tasks"], "--split", job["split"],
           "--trials", str(job["trials"]), "--slice", job["slice"], "--label", job["label"], "--out-root", "/root/out"]
    if ad:
        cmd += ["--adapter", ad]
    p = subprocess.Popen(cmd, cwd="/root/repo", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith(("task=", "slice", "STOP", "Traceback")) or "Error" in line[:40]:
            print(f"[{tag}] {line}", end="", flush=True)
    print(f"[{tag}] exit code {p.wait()}", flush=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        if os.path.exists("/root/out"):
            t.add("/root/out", arcname=tag)
    return buf.getvalue()


@app.local_entrypoint()
def main(split: str, tasks: str, trials: int, label: str, slice: str, out: str, adapter: str = "",
         per_container: int = 1, adapter_name: str = ""):
    ids = json.load(open(ROOT / "data/splits/retail_split.json"))[f"{split}_ids"]
    ids = [str(t) for t in ids]
    sel = ids if tasks == "all" else [t.strip() for t in tasks.split(",") if t.strip()]
    assert all(t in ids for t in sel), f"every task must be a {split}-split id"
    import hashlib, subprocess
    name, sha = "", ""
    if adapter:   # upload once to the volume as <adapter name>/ (e.g. "epoch3"); hash-checked in each container
        a = ROOT / adapter
        name = adapter_name or a.name
        sha = hashlib.sha256((a / "adapter_model.safetensors").read_bytes()).hexdigest()
        for n in ("adapter_config.json", "adapter_model.safetensors"):
            subprocess.run([str(ROOT / ".venv/bin/modal"), "volume", "put", "-f", "agentic-distill-adapters",
                            str(a / n), f"/{name}/{n}"], check=True)
        print(f"adapter {name} sha256 {sha[:12]} on volume")
    groups = [sel[i:i + per_container] for i in range(0, len(sel), per_container)]
    jobs = [{"group": k, "tasks": g, "split": split, "trials": trials, "label": label, "slice": slice,
             "adapter": name, "adapter_sha256": sha} for k, g in enumerate(groups)]
    dst = ROOT / out
    dst.mkdir(parents=True, exist_ok=True)
    print(f"{len(sel)} tasks in {len(groups)} containers -> {dst}")
    for job, blob in zip(jobs, run.map(jobs, return_exceptions=True)):
        if isinstance(blob, Exception):
            print(f"g{job['group']} FAILED: {blob!r}")
            continue
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            tf.extractall(dst, filter="data")
        print("saved", dst / f"g{job['group']}")
