"""DAgger round 1 training on Modal (L4): continue the epoch3 adapter on the relabelled states + a replay sample.

Mix (D13): every example in --dagger (loss on the teacher's single turn) + a seeded 25% replay sample of the original
SFT set (full episodes, loss on all assistant turns). Training is token-weighted (train_lora), so without the
upweighting the ~100-token DAgger labels would be a few % of the gradient.
Run:  .venv/bin/modal run scripts/modal_dagger_train.py --dagger data/sft/dagger_r1.jsonl   [--tag dagger_r2 for round 2]
Output: data/modal_out_dagger_train/{epoch1,epoch2,train_log.jsonl}, data/sft/dagger_r1_mix.jsonl (the exact mix).
"""
import io
import json
import pathlib
import random
import tarfile

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("unzip")
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt")).pip_install("pyyaml")
    .run_commands("pip uninstall -y -q torchao || true")
    .add_local_file("/tmp/agentic_distill_src.zip", "/root/src.zip")
)
app = modal.App("agentic-distill-dagger-train", image=image)
vol = modal.Volume.from_name("agentic-distill-adapters", create_if_missing=True)


@app.function(gpu="L4", timeout=4 * 3600, secrets=[modal.Secret.from_name("agentic-distill")], volumes={"/vol": vol},
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def train(epochs: int, lr: float, seed: int, tag: str) -> bytes:
    """Checkpoints live on the volume and are committed every minute, so a preempted retry resumes (--resume)."""
    import threading, time
    import os, subprocess, sys
    subprocess.run(["unzip", "-q", "-o", "/root/src.zip", "-d", "/root/repo"], check=True)
    import torch, transformers, peft
    print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__,
          torch.cuda.get_device_name(0), flush=True)
    env = {**os.environ, "PYTHONPATH": "src", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [sys.executable, "-m", "agentic_distill.train_lora", "--data", f"/vol/data/{tag}_mix.jsonl",
           "--out", f"/vol/train/{tag}_s{seed}", "--init-adapter", "/vol/epoch3", "--epochs", str(epochs), "--lr", str(lr),
           "--seed", str(seed), "--max-tokens", "13000", "--resume"]
    done = threading.Event()
    def committer():
        while not done.wait(60):
            vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    p = subprocess.Popen(cmd, cwd="/root/repo", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait(); done.set(); vol.commit()
    assert rc == 0, f"train_lora exited {rc}"
    out = f"/vol/train/{tag}_s{seed}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in os.listdir(out):
            if name != "trainer_state.pt":
                t.add(f"{out}/{name}", arcname=name)
    return buf.getvalue()


@app.local_entrypoint()
def main(dagger: str = "data/sft/dagger_r1.jsonl", base: str = "data/sft/train_v2.jsonl", replay_frac: float = 0.25,
         epochs: int = 2, lr: float = 5e-5, seed: int = 20261117, tag: str = "dagger_r1"):
    import subprocess
    new = [l for l in open(ROOT / dagger)]
    old = [l for l in open(ROOT / base)]
    rng = random.Random(seed)
    k = max(1, round(replay_frac * len(new)))
    mix = new + rng.sample(old, k)
    rng.shuffle(mix)
    mp = ROOT / f"data/sft/{tag}_mix.jsonl"
    mp.write_text("".join(mix))
    print(f"mix: {len(new)} dagger + {k} replay = {len(mix)} examples -> {mp}")
    subprocess.run([str(ROOT / ".venv/bin/modal"), "volume", "put", "-f", "agentic-distill-adapters", str(mp),
                    f"/data/{tag}_mix.jsonl"], check=True)
    blob = train.remote(epochs, lr, seed, tag)
    dst = ROOT / f"data/modal_out_{tag}_train" if tag != "dagger_r1" else ROOT / "data/modal_out_dagger_train"
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(dst, filter="data")
    print("saved", dst)
