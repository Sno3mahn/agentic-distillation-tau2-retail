"""C5: LoRA SFT of the student on the pre-tokenized, pre-masked dataset (data/sft/*.jsonl).

Why not TRL SFTTrainer: with a 262K vocabulary, full logits for a 13K-token example are ~7 GB in fp16
and ~13 GB once upcast for the loss, on a 15 GB T4. Only ~13% of positions carry loss, so
`selective_loss` runs the transformer body once and applies lm_head only at supervised positions.
tests/test_loss_positions.py checks it equals the standard HF loss.

Modes:
    --memtest            one forward+backward on the longest example: peak memory, tokens/s
    --limit N            train on the first N examples only (overfit smoke test)
    (default)            full run; adapter + optimizer state saved after every epoch; --resume continues

    python -m agentic_distill.train_lora --data data/sft/train_v2.jsonl --out out/lora --memtest
"""

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

IGNORE = -100
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def causal_lm(model):
    """The underlying *ForCausalLM, whether or not it is wrapped by PEFT."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def selective_loss(model, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Summed next-token cross-entropy over supervised positions, and their count.

    Same maths as HF's loss (shift by one, ignore -100, Cohere's logit_scale) but lm_head is applied
    only where labels[t+1] != -100. Returns a sum so gradient accumulation can divide by the
    total number of supervised tokens in the accumulation window.
    """
    lm = causal_lm(model)
    hidden = lm.model(input_ids=input_ids).last_hidden_state          # (1, T, H)
    target = labels[:, 1:]
    keep = target != IGNORE
    h = hidden[:, :-1][keep]                                          # (n, H)
    logits = lm.lm_head(h).float() * getattr(lm, "logit_scale", 1.0)
    return F.cross_entropy(logits, target[keep], reduction="sum"), int(keep.sum())


def load_data(path: str, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def build_model(model_id: str, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
        token=os.environ.get("HF_TOKEN"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    for p in model.parameters():          # fp16 base frozen; trainable LoRA weights kept in fp32
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


def sdpa_context(name: str):
    """'efficient' forces PyTorch's memory-efficient attention kernel (no full T x T score matrix).
    With the default dispatcher, the longest example fell back to the math kernel and asked for ~10 GiB.
    If the kernel cannot handle our masks, PyTorch raises and warns with the reason instead of silently falling back."""
    if name == "default":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)


def disable_sdpa_gqa() -> None:
    """Make transformers expand K/V to all query heads (repeat_kv) instead of passing enable_gqa=True.

    PyTorch's memory-efficient/flash kernels reject enable_gqa and fall back to the math kernel, which builds
    the full T x T score matrix (~10 GiB at 12.9K tokens). Depends on transformers 5.23's sdpa_attention module;
    fails loudly if that function is gone.
    """
    import transformers.integrations.sdpa_attention as sa
    assert hasattr(sa, "use_gqa_in_sdpa"), "transformers changed: re-check the SDPA GQA path"
    sa.use_gqa_in_sdpa = lambda attention_mask, key, value: False


def memtest(model, rows: list[dict], out: Path, backend: str) -> None:
    warnings.simplefilter("always")           # kernel-selection warnings carry the reason for a fallback
    out.mkdir(parents=True, exist_ok=True)
    model.train()   # HF only applies gradient checkpointing in training mode; eval mode would overstate memory
    ex = max(rows, key=lambda r: r["n_tokens"])
    ids = torch.tensor([ex["input_ids"]], device="cuda")
    lab = torch.tensor([ex["labels"]], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        with sdpa_context(backend), torch.autocast("cuda", dtype=torch.float16):
            loss, n = selective_loss(model, ids, lab)
        (loss / n).backward()
        torch.cuda.synchronize()
    except (torch.OutOfMemoryError, RuntimeError) as e:
        res = {"ok": False, "backend": backend, "memtest_tokens": ex["n_tokens"], "error": str(e)[:800],
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        print(json.dumps(res)); (out / "result.json").write_text(json.dumps(res))
        sys.exit(1)
    dt = time.time() - t0
    res = {"ok": True, "backend": backend, "memtest_tokens": ex["n_tokens"], "supervised": n,
           "loss": (loss / n).item(), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
           "seconds": round(dt, 1), "tokens_per_s": round(ex["n_tokens"] / dt)}
    print(json.dumps(res)); (out / "result.json").write_text(json.dumps(res))
    model.zero_grad(set_to_none=True)


def train(model, rows, args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(rows) / args.grad_accum)
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0
    state_path = out / "trainer_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, weights_only=False)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(out / f"epoch{st['epoch']}" / "adapter_model.safetensors"))
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_epoch = st["epoch"]
        print(f"resumed after epoch {start_epoch}")
    log = open(out / "train_log.jsonl", "a")
    step, bad = start_epoch * steps_per_epoch, 0
    model.train()
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for i in range(0, len(order), args.grad_accum):
            batch = [rows[j] for j in order[i:i + args.grad_accum]]
            n_sup = sum(sum(l != IGNORE for l in r["labels"][1:]) for r in batch)
            t0, loss_sum, toks = time.time(), 0.0, 0
            for r in batch:
                ids = torch.tensor([r["input_ids"]], device="cuda")
                lab = torch.tensor([r["labels"]], device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, _ = selective_loss(model, ids, lab)
                scaler.scale(loss / n_sup).backward()   # token-weighted mean over the window
                loss_sum += loss.item()
                toks += r["n_tokens"]
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < scale_before      # inf/NaN gradient: step skipped, scale lowered
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            loss_mean = loss_sum / n_sup
            bad = bad + 1 if (skipped or not math.isfinite(loss_mean)) else 0
            rec = {"epoch": epoch + 1, "step": step, "of": total, "loss": round(loss_mean, 4),
                   "lr": sched.get_last_lr()[0], "grad_norm": round(gnorm, 3), "skipped": skipped,
                   "loss_scale": scaler.get_scale(), "tokens_per_s": round(toks / (time.time() - t0)),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if bad >= 5:
                raise RuntimeError("5 consecutive non-finite/skipped steps: stopping (fp16 instability)")
            if args.max_steps and step >= args.max_steps:
                break
        ck = out / f"epoch{epoch + 1}"
        model.save_pretrained(ck)
        torch.save({"epoch": epoch + 1, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict()}, state_path)
        print(f"saved {ck}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-en-thinker")
    ap.add_argument("--memtest", action="store_true")
    ap.add_argument("--sdpa", choices=["efficient", "default"], default="efficient")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.sdpa == "efficient":
        disable_sdpa_gqa()
    rows = load_data(args.data, args.limit)
    print(f"{len(rows)} examples; {sum(r['n_tokens'] for r in rows)} tokens; args={vars(args)}")
    model = build_model(args.model, args.r, args.alpha, args.dropout)
    if args.memtest:
        memtest(model, rows, Path(args.out), args.sdpa)
    else:
        with sdpa_context(args.sdpa):
            train(model, rows, args)


if __name__ == "__main__":
    main()
