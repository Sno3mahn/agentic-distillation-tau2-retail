"""C6/C7: run the student (B0 or fine-tuned) as the tau2 retail agent against the DeepSeek user simulator.

Same reward as the teacher (D4). Writes one eval record per episode (design interface 5).
Dev tasks only for now (test is reserved until threshold.md exists, C9).

    python -m agentic_distill.student_run --tasks 1 --trials 1 --slice s3_pilot --label b0
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import litellm
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner import get_tasks, run_single_task
from transformers import AutoTokenizer

from agentic_distill import REPO_ROOT, load_config
from agentic_distill.formatter import Formatter
from agentic_distill.ledger import Ledger, episode_records, is_peak
from agentic_distill.splits import load_split
from agentic_distill.student_agent import ABORT, GenResult, register
from agentic_distill.teacher_run import score


def make_hf_generate(model_id: str, tok, gen_cfg: dict, adapter: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    from agentic_distill.train_lora import disable_sdpa_gqa
    disable_sdpa_gqa()   # same attention path as training; avoids the T x T math-kernel fallback on long prompts
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, gen_cfg["dtype"]), device_map="cuda",
        attn_implementation="sdpa", token=os.environ.get("HF_TOKEN"))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    stop_ids = [tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>"), tok.eos_token_id]

    def generate(prompt: str) -> GenResult:
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=gen_cfg["max_new_tokens"], do_sample=gen_cfg["do_sample"],
                                 eos_token_id=stop_ids, pad_token_id=tok.pad_token_id,
                                 stop_strings=["<tool_result"], tokenizer=tok)   # D7: saves tokens; agent also truncates
        new = out[0, ids.shape[1]:]
        return GenResult(tok.decode(new, skip_special_tokens=False), int(ids.shape[1]), int(len(new)))
    return generate


def fake_invalid_generate(prompt: str) -> GenResult:
    """Test backend: always unparseable, so the first agent turn aborts (exercises plumbing cheaply)."""
    return GenResult("I am not following the format.", 1, 1)


def eval_record(sim, sc: dict, label: str, ctx: dict, cost_usd: float) -> dict:
    msgs = sim.messages or []
    agent_msgs = [m for m in msgs if m.role == "assistant" and m.raw_data]
    stats = [m.raw_data for m in agent_msgs]
    calls = [(c.name, json.dumps(c.arguments, sort_keys=True)) for m in msgs if m.role == "assistant"
             for c in (m.tool_calls or [])]
    run, loop = 1, False
    for a, b in zip(calls, calls[1:]):
        run = run + 1 if a == b else 1
        loop = loop or run >= 3
    env_errors = sum(1 for m in msgs if m.role == "tool" and m.error)
    parse_invalid = sum(len(s["invalid_attempts"]) for s in stats)
    aborted = any(s["aborted"] for s in stats)
    reason = sc["termination_reason"]
    terminated_by = "abort" if aborted else "done" if reason in ("user_stop", "agent_stop") else \
        "max_steps" if reason == "max_steps" else "error"
    reward = 0.0 if aborted else sc["reward"]          # an abort is a failure (proposed; flagged in report)
    return {**ctx, "model": label, "reward": reward, "reward_if_abort_ignored": sc["reward"],
            "db_reward": sc["db_reward"], "communicate_reward": sc["communicate_reward"],
            "n_steps": sum(1 for m in msgs if m.role == "assistant"), "n_tool_calls": len(calls),
            "n_invalid_calls": parse_invalid + env_errors, "n_parse_invalid": parse_invalid,
            "n_env_tool_errors": env_errors, "terminated_by": terminated_by, "loop_detected": loop,
            "tokens": {"in": sum(s["n_in"] for s in stats), "out": sum(s["n_out"] for s in stats)},
            "gen_wall_s": sum(s["wall_s"] for s in stats), "wall_s": sim.duration, "cost_usd": cost_usd}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True, help="task IDs of the chosen split, or 'all'")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev",
                    help="test is locked until results/threshold.md exists (C9)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--label", default="b0", help="model label in records, e.g. b0 or s2-ckpt1")
    ap.add_argument("--backend", choices=["hf", "fake_invalid"], default="hf")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path/id (fine-tuned student)")
    ap.add_argument("--allow-peak", action="store_true")
    ap.add_argument("--out-root", default=None, help="override output root (e.g. /kaggle/working)")
    args = ap.parse_args()

    cfg = load_config()
    if not os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("API_DEEPSEEK"):
        os.environ["DEEPSEEK_API_KEY"] = os.environ["API_DEEPSEEK"]
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("API_DEEPSEEK (or DEEPSEEK_API_KEY) is not set")
    if not args.allow_peak and is_peak(datetime.now(timezone.utc), cfg["prices"]["peak_windows_utc"]):
        sys.exit("DeepSeek peak hours (2x price): refusing to run. Wait or pass --allow-peak.")

    split = load_split(REPO_ROOT / cfg["split"]["path"])
    if args.split == "test" and not (REPO_ROOT / "results" / "threshold.md").exists():
        sys.exit("test split is locked: results/threshold.md must exist first (C9)")
    ids = split[f"{args.split}_ids"]
    if args.tasks == ["all"]:
        args.tasks = list(ids)
    bad = [t for t in args.tasks if t not in set(ids)]
    if bad:
        sys.exit(f"not {args.split} task IDs: {bad}")

    out_root = REPO_ROOT if args.out_root is None else __import__("pathlib").Path(args.out_root)
    ledger = Ledger(out_root / cfg["paths"]["ledger"], cfg["budget"]["slices_usd"][args.slice])
    # NOTE: on Kaggle the ledger starts empty, so the slice cap there only bounds this notebook's spend.
    if ledger.remaining(slice=args.slice) <= 0:
        sys.exit(f"budget slice {args.slice} exhausted")
    litellm.max_budget = ledger.remaining(slice=args.slice)

    st = cfg["student"]
    tok_src = REPO_ROOT / st["tokenizer_dir"]
    tok = AutoTokenizer.from_pretrained(tok_src if (tok_src / "tokenizer.json").exists() else st["model"],
                                        token=os.environ.get("HF_TOKEN"))
    gen = (fake_invalid_generate if args.backend == "fake_invalid"
           else make_hf_generate(st["model"], tok, st["generation"], args.adapter))
    agent_name = register(Formatter(tok), gen)

    scoring = CommunicateScoring(scorer=cfg["reward"]["communicate_scorer"])
    ep = cfg["episodes"]
    config = TextRunConfig.model_validate({
        "domain": cfg["domain"], "agent": agent_name, "llm_agent": args.label, "llm_args_agent": {},
        "user": "user_simulator", "llm_user": cfg["user_sim"]["model"], "llm_args_user": cfg["user_sim"]["llm_args"],
        "max_steps": ep["max_steps"], "max_errors": ep["max_errors"], "seed": ep["base_seed"],
        "communicate_scoring": scoring})

    # pid keeps names unique when two processes (one per GPU) start in the same second with the same label
    run_name = f"{args.slice}_{args.label}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_p{os.getpid()}"
    out_dir = out_root / cfg["paths"]["raw"] / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    model_by_role = {"user_sim": cfg["user_sim"]["model"]}
    for task in get_tasks(cfg["domain"], task_ids=args.tasks):
        for trial in range(args.trials):
            if ledger.remaining(slice=args.slice) <= 0:
                print(f"STOP: budget slice {args.slice} exhausted")
                return
            seed = ep["base_seed"] + trial
            ctx = {"slice": args.slice, "run": run_name, "task_id": task.id, "trial": trial, "seed": seed,
                   "user_sim_model": cfg["user_sim"]["model"], "split": args.split}
            try:
                sim = run_single_task(config, task, seed=seed, evaluation_type=EvaluationType.ENV)
            except Exception as e:            # e.g. CUDA OOM on a very long episode: a failure, but the run goes on
                import gc
                gc.collect()
                if args.backend == "hf":
                    import torch
                    torch.cuda.empty_cache()
                rec = {**ctx, "model": args.label, "reward": 0.0, "reward_if_abort_ignored": 0.0, "db_reward": 0.0,
                       "communicate_reward": 0.0, "n_steps": 0, "n_tool_calls": 0, "n_invalid_calls": 0,
                       "n_parse_invalid": 0, "n_env_tool_errors": 0, "terminated_by": "crash", "loop_detected": False,
                       "tokens": {"in": 0, "out": 0}, "gen_wall_s": 0, "wall_s": 0, "cost_usd": 0.0,
                       "error": f"{type(e).__name__}: {str(e)[:300]}"}
                with open(out_dir / "episodes.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"task={task.id} trial={trial} reward=0 term=crash {rec['error']}", flush=True)
                continue
            sim.trial = trial
            recs = episode_records(sim, model_by_role, cfg["prices"], ctx, skip_roles=("agent",))
            ledger.append(recs)
            rec = eval_record(sim, score(sim, task, scoring), args.label, ctx, sum(r["cost_usd"] for r in recs))
            (out_dir / f"task{task.id}_trial{trial}.json").write_text(sim.model_dump_json(indent=1))
            with open(out_dir / "episodes.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"task={task.id} trial={trial} reward={rec['reward']:.0f} term={rec['terminated_by']} "
                  f"steps={rec['n_steps']} calls={rec['n_tool_calls']} invalid={rec['n_invalid_calls']} "
                  f"tok in/out={rec['tokens']['in']}/{rec['tokens']['out']} gen_wall={rec['gen_wall_s']:.0f}s "
                  f"cost=${rec['cost_usd']:.4f}", flush=True)
    print(f"slice {args.slice} total=${ledger.total(slice=args.slice):.4f}; records in {out_dir}")


if __name__ == "__main__":
    main()
