"""DAgger round 2: success-filtered teacher labels at states the STUDENT visited.

Round 1 (dagger_relabel.py) asked the teacher for one turn at each state and could not tell whether that turn leads
to success. Here the teacher is rolled forward from each state to the end of the episode (tau2 resumes from
`task.initial_state.message_history`, replaying the student's tool calls on the database), K times, and scored with
the same reward as everywhere else (DB x substring COMMUNICATE). A label is kept only if a rollout that SUCCEEDED
begins with it; states where no rollout succeeds are dropped.

States are the same ones round 1 used (`select_states`, the student's first error turn + earlier write calls).
Cost accounting counts only the LLM calls made after the prefix (the prefix carries the old run's raw_data).

    python -m agentic_distill.dagger_rollout "data/modal_out_dagger/*/data/raw/*" --out data/sft/dagger_r2.jsonl
"""

import argparse
import glob
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

EXCLUDE_TASKS = {"59"}      # gold answer and simulator disagree (E2); same exclusion as round 1


def collect_states(runs: list[str], train: set[str], per_episode: int):
    from agentic_distill import REPO_ROOT
    from agentic_distill.dagger_relabel import select_states
    states, seen = [], set()
    for pattern in runs:
        for d in sorted(glob.glob(str(REPO_ROOT / pattern)) or glob.glob(pattern)):
            for line in open(Path(d) / "episodes.jsonl"):
                meta = json.loads(line)
                tid = str(meta["task_id"])
                assert tid in train, f"non-train task {tid} in {d}"
                key = (tid, meta["trial"])
                if key in seen:
                    continue
                seen.add(key)
                if meta["reward"] >= 0.5 or meta["terminated_by"] == "crash" or tid in EXCLUDE_TASKS:
                    continue
                sim = json.load(open(Path(d) / f"task{tid}_trial{meta['trial']}.json"))
                for i in select_states(sim["messages"], per_episode):
                    states.append({"run": Path(d).name, "task_id": tid, "trial": meta["trial"], "turn": i,
                                   "messages": sim["messages"]})
    return states


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="data/sft/dagger_r2.jsonl")
    ap.add_argument("--rollouts", default="data/raw/dagger_r2_rollouts.jsonl", help="every rollout, appended (resumable)")
    ap.add_argument("--k", type=int, default=4, help="teacher rollouts per state")
    ap.add_argument("--per-episode", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=13000)
    ap.add_argument("--slice", default="dagger2")
    ap.add_argument("--allow-peak", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-states", type=int, default=None, help="smoke test: only the first N states")
    args = ap.parse_args()

    import litellm
    from tau2.data_model.semantic_communication import CommunicateScoring
    from tau2.data_model.simulation import TextRunConfig
    from tau2.data_model.tasks import InitialState
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.runner import build_environment, get_tasks, run_single_task
    from transformers import AutoTokenizer

    from agentic_distill import REPO_ROOT, agents, load_config  # noqa: F401  (agents registers variants)
    from agentic_distill.build_sft import IGNORE, build_example
    from agentic_distill.dagger_relabel import to_tau2
    from agentic_distill.formatter import Formatter, from_tau2
    from agentic_distill.ledger import Ledger, episode_records, is_peak
    from agentic_distill.teacher_run import score

    cfg = load_config()
    if not os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("API_DEEPSEEK"):
        os.environ["DEEPSEEK_API_KEY"] = os.environ["API_DEEPSEEK"]
    train = set(json.load(open(REPO_ROOT / cfg["split"]["path"]))["train_ids"])
    states = collect_states(args.runs, train, args.per_episode)[: args.limit_states]
    print(f"states={len(states)} tasks={len({s['task_id'] for s in states})} rollouts planned={len(states) * args.k}", flush=True)
    if args.dry_run:
        return
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("API_DEEPSEEK (or DEEPSEEK_API_KEY) is not set")
    if not args.allow_peak and is_peak(datetime.now(timezone.utc), cfg["prices"]["peak_windows_utc"]):
        sys.exit("DeepSeek peak hours (2x price): refusing to run. Wait or pass --allow-peak.")

    var = cfg["teacher"]["variants"]["rationale"]
    ep = cfg["episodes"]
    ledger = Ledger(REPO_ROOT / cfg["paths"]["ledger"], cfg["budget"]["slices_usd"][args.slice])
    if ledger.remaining(slice=args.slice) <= 0:
        sys.exit(f"budget slice {args.slice} exhausted")
    litellm.max_budget = ledger.remaining(slice=args.slice)
    scoring = CommunicateScoring(scorer=cfg["reward"]["communicate_scorer"])
    config = TextRunConfig.model_validate({
        "domain": cfg["domain"], "agent": var["agent"], "llm_agent": cfg["teacher"]["model"],
        "llm_args_agent": var["llm_args"], "user": "user_simulator", "llm_user": cfg["user_sim"]["model"],
        "llm_args_user": cfg["user_sim"]["llm_args"], "max_steps": ep["max_steps"], "max_errors": ep["max_errors"],
        "seed": ep["base_seed"], "communicate_scoring": scoring})
    model_by_role = {"agent": cfg["teacher"]["model"], "user_sim": cfg["user_sim"]["model"]}

    rpath = REPO_ROOT / args.rollouts
    rpath.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if rpath.exists():
        for line in open(rpath):
            r = json.loads(line)
            done[(r["run"], r["task_id"], r["trial"], r["turn"], r["k"])] = r
    lock = threading.Lock()
    tasks_by_id = {t.id: t for t in get_tasks(cfg["domain"], task_ids=sorted({s["task_id"] for s in states}))}

    def rollout(job):
        s, k = job
        key = (s["run"], s["task_id"], s["trial"], s["turn"], k)
        if key in done:
            return done[key]
        with lock:
            if ledger.remaining(slice=args.slice) <= 0:
                return None
        prefix = to_tau2(s["messages"][: s["turn"]])
        task = tasks_by_id[s["task_id"]].model_copy(deep=True)
        init = task.initial_state or InitialState()
        task.initial_state = init.model_copy(update={"message_history": prefix})
        rec = {"run": s["run"], "task_id": s["task_id"], "trial": s["trial"], "turn": s["turn"], "k": k}
        try:
            sim = run_single_task(config, task, seed=ep["base_seed"] + 100 + k, evaluation_type=EvaluationType.ENV)
        except Exception as e:
            return {**rec, "reward": 0.0, "error": f"{type(e).__name__}: {str(e)[:200]}", "cost_usd": 0.0}
        sim.messages = sim.messages or []
        new = sim.messages[len(prefix):]
        view = type("V", (), {"messages": new})()
        recs = episode_records(view, model_by_role, cfg["prices"],
                               {"slice": args.slice, "run": "dagger_rollout", "task_id": s["task_id"],
                                "trial": s["trial"], "turn": s["turn"], "k": k})
        # tau2 scored this run against a gold DB that already contains the prefix's writes, so a wrong write in the
        # student's prefix would also be in the gold state. Re-score the FULL trajectory against the original task.
        clean = tasks_by_id[s["task_id"]]
        sim.reward_info = evaluate_simulation(simulation=sim, task=clean, evaluation_type=EvaluationType.ENV,
                                              solo_mode=False, domain=cfg["domain"], communicate_scoring=scoring)
        sc = score(sim, clean, scoring)
        first = new[0] if new else None
        label = None
        if first is not None and first.role == "assistant":
            label = {"role": "assistant", "content": first.content,
                     "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in (first.tool_calls or [])]}
        out = {**rec, "reward": sc["reward"], "termination": sc["termination_reason"], "label": label,
               "n_new_msgs": len(new), "cost_usd": sum(r["cost_usd"] for r in recs)}
        with lock:
            ledger.append(recs)
        return out

    jobs = [(s, k) for s in states for k in range(args.k)]
    results = []
    with ThreadPoolExecutor(args.workers) as ex, open(rpath, "a") as f:
        for r in ex.map(rollout, jobs):
            if r is None:
                continue
            results.append(r)
            if (r["run"], r["task_id"], r["trial"], r["turn"], r["k"]) not in done:
                f.write(json.dumps(r) + "\n"); f.flush()
            if len(results) % 20 == 0:
                print(f"{len(results)}/{len(jobs)} rollouts, wins={sum(x['reward'] > 0.5 for x in results)}, "
                      f"slice total=${ledger.total(slice=args.slice):.4f}", flush=True)
    print(f"rollouts={len(results)} wins={sum(x['reward'] > 0.5 for x in results)} "
          f"errors={sum('error' in x for x in results)} slice total=${ledger.total(slice=args.slice):.4f}", flush=True)

    env = build_environment(cfg["domain"])
    policy, schemas, tool_names = env.get_policy(), [t.openai_schema for t in env.get_tools()], {t.name for t in env.get_tools()}
    tok = AutoTokenizer.from_pretrained(REPO_ROOT / cfg["student"]["tokenizer_dir"])
    fmt = Formatter(tok)
    by_state = {}
    for r in results:
        by_state.setdefault((r["run"], r["task_id"], r["trial"], r["turn"]), []).append(r)
    kept = agree = dropped_len = no_win = 0
    out = REPO_ROOT / args.out
    with open(out, "w") as f:
        for s in states:
            rs = by_state.get((s["run"], s["task_id"], s["trial"], s["turn"]), [])
            wins = [r for r in rs if r["reward"] > 0.5 and r.get("label")
                    and (r["label"]["tool_calls"] or (r["label"]["content"] or "").strip())
                    and all(c["name"] in tool_names for c in r["label"]["tool_calls"])]
            if not wins:
                no_win += 1
                continue
            label = wins[0]["label"]
            prefix = s["messages"][: s["turn"]]
            ex = build_example(fmt, tok, policy, schemas, from_tau2(to_tau2(prefix)) + [label], loss_on="last")
            n = len(ex["input_ids"])
            if n > args.max_tokens:
                dropped_len += 1
                continue
            student = from_tau2([s["messages"][s["turn"]]])[0]
            same = [(c["name"], c["arguments"]) for c in student.get("tool_calls", [])] == \
                   [(c["name"], c["arguments"]) for c in label["tool_calls"]] and bool(label["tool_calls"])
            agree += same
            kept += 1
            f.write(json.dumps({**ex, "variant": "dagger_r2", "task_id": s["task_id"], "state_turn": s["turn"],
                                "source_episode_id": f"{s['run']}/task{s['task_id']}_trial{s['trial']}",
                                "teacher_agrees": same, "rollout_wins": f"{len(wins)}/{len(rs)}", "n_tokens": n,
                                "n_loss_tokens": sum(l != IGNORE for l in ex["labels"])}) + "\n")
    print(f"states={len(states)} kept={kept} (tasks {len({s['task_id'] for s in states})}) no_successful_rollout={no_win} "
          f"dropped_over_length={dropped_len} teacher_agrees={agree}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
