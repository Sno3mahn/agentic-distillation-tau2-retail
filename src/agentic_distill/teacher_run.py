"""C2/C3: run the DeepSeek teacher on tau2 retail train tasks and store raw traces.

Reward (user-decided 2026-11-13, option B):
    reward = DB check (tau2 EvaluationType.ENV) x substring COMMUNICATE on communicate_info
The task's NL_ASSERTION basis (LLM judge) is deliberately not evaluated.

Example:
    python -m agentic_distill.teacher_run --tasks 0 1 --trials 1 --slice s1_smoke
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import litellm
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import TerminationReason, TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.runner import get_tasks, run_single_task

from agentic_distill import REPO_ROOT, load_config
from agentic_distill.ledger import Ledger, episode_records, is_peak
from agentic_distill.splits import load_split


def score(sim, task, scoring: CommunicateScoring) -> dict:
    ri = sim.reward_info
    db = ri.db_check.db_reward if (ri and ri.db_check) else 0.0
    env_reward = ri.reward if ri else 0.0  # DB x env_assertions; 0 if terminated prematurely
    finished = sim.termination_reason in {TerminationReason.AGENT_STOP, TerminationReason.USER_STOP}
    comm = CommunicateEvaluator.calculate_reward(task=task, full_trajectory=sim.messages, scoring=scoring)
    comm_reward = comm.reward if finished else 0.0
    return {"reward": env_reward * comm_reward, "db_reward": db, "env_reward": env_reward,
            "communicate_reward": comm_reward,
            "communicate_checks": [c.model_dump() for c in (comm.communicate_checks or [])],
            "termination_reason": sim.termination_reason.value}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True, help="train task IDs")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--slice", required=True, help="budget slice name from configs/run.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--allow-peak", action="store_true",
                    help="override the default off-peak-only guard (peak = 2x price)")
    args = ap.parse_args()

    cfg = load_config()
    # The user's shell variable is API_DEEPSEEK; LiteLLM reads DEEPSEEK_API_KEY.
    if not os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("API_DEEPSEEK"):
        os.environ["DEEPSEEK_API_KEY"] = os.environ["API_DEEPSEEK"]
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("API_DEEPSEEK (or DEEPSEEK_API_KEY) is not set")
    if not args.allow_peak and is_peak(datetime.now(timezone.utc), cfg["prices"]["peak_windows_utc"]):
        sys.exit("DeepSeek peak hours (Mon-Fri 01-04 and 06-10 UTC, 2x price): refusing to run. "
                 "Wait for off-peak or pass --allow-peak.")

    split = load_split(REPO_ROOT / cfg["split"]["path"])
    bad = [t for t in args.tasks if t not in set(split["train_ids"])]
    if bad:
        sys.exit(f"refusing non-train task IDs (contamination guard): {bad}")

    slice_budget = cfg["budget"]["slices_usd"][args.slice]
    ledger = Ledger(REPO_ROOT / cfg["paths"]["ledger"], slice_budget)
    if ledger.remaining(slice=args.slice) <= 0:
        sys.exit(f"budget slice {args.slice} exhausted")
    # In-flight hard stop (LiteLLM's own cost table; raises BudgetExceededError before a call).
    litellm.max_budget = ledger.remaining(slice=args.slice)

    scoring = CommunicateScoring(scorer=cfg["reward"]["communicate_scorer"])
    ep = cfg["episodes"]
    # Built from a dict: tau2's pydantic fields are declared as Annotated[..., Field(default=...)],
    # which static checkers misread as required keyword arguments.
    config = TextRunConfig.model_validate({
        "domain": cfg["domain"],
        "agent": "llm_agent", "llm_agent": cfg["teacher"]["model"], "llm_args_agent": cfg["teacher"]["llm_args"],
        "user": "user_simulator", "llm_user": cfg["user_sim"]["model"], "llm_args_user": cfg["user_sim"]["llm_args"],
        "max_steps": ep["max_steps"], "max_errors": ep["max_errors"], "seed": ep["base_seed"],
        "communicate_scoring": scoring,
    })
    model_by_role = {"agent": cfg["teacher"]["model"], "user_sim": cfg["user_sim"]["model"]}

    run_name = args.run_name or f"{args.slice}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_dir = REPO_ROOT / cfg["paths"]["raw"] / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    tasks = get_tasks(cfg["domain"], task_ids=args.tasks)
    for task in tasks:
        for trial in range(args.trials):
            if ledger.remaining(slice=args.slice) <= 0:
                print(f"STOP: budget slice {args.slice} exhausted")
                return
            seed = ep["base_seed"] + trial
            sim = run_single_task(config, task, seed=seed, evaluation_type=EvaluationType.ENV)
            sim.trial = trial
            ctx = {"slice": args.slice, "run": run_name, "task_id": task.id, "trial": trial}
            recs = episode_records(sim, model_by_role, cfg["prices"], ctx)
            ledger.append(recs)
            sc = score(sim, task, scoring)
            usage = {k: sum(r[k] for r in recs) for k in ("hit", "miss", "out", "reasoning")}
            meta = {**ctx, "seed": seed, "model": cfg["teacher"]["model"], "mode": "non-thinking",
                    "user_sim_model": cfg["user_sim"]["model"], "sim_id": sim.id,
                    "n_llm_calls": len(recs), "usage": usage,
                    "cost_usd": sum(r["cost_usd"] for r in recs),
                    "cost_agent_usd": sum(r["cost_usd"] for r in recs if r["role"] == "agent"),
                    "cost_user_sim_usd": sum(r["cost_usd"] for r in recs if r["role"] == "user_sim"),
                    "thinking_detected": any(r["has_reasoning_content"] or r["reasoning"] for r in recs),
                    "n_messages": len(sim.messages or []), **sc}
            (out_dir / f"task{task.id}_trial{trial}.json").write_text(sim.model_dump_json(indent=1))
            with open(out_dir / "episodes.jsonl", "a") as f:
                f.write(json.dumps(meta) + "\n")
            print(f"task={task.id} trial={trial} reward={sc['reward']:.0f} "
                  f"(db={sc['db_reward']:.0f} env={sc['env_reward']:.0f} comm={sc['communicate_reward']:.0f}) "
                  f"term={sc['termination_reason']} msgs={meta['n_messages']} calls={len(recs)} "
                  f"tok hit/miss/out={usage['hit']}/{usage['miss']}/{usage['out']} "
                  f"cost=${meta['cost_usd']:.4f}")
            if meta["thinking_detected"]:
                print("STOP: reasoning content detected; non-thinking mode is not taking effect")
                return
    print(f"slice {args.slice} total=${ledger.total(slice=args.slice):.4f} / ${slice_budget:.2f}; "
          f"traces in {out_dir}")


if __name__ == "__main__":
    main()
