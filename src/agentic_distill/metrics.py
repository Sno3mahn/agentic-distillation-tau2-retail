"""C7 metrics: pass^k, valid-call rate, steps, loop/abort rate, cost per success, bootstrap CIs over tasks.

Input is a list of episode records (student eval records, or teacher episodes.jsonl enriched from the saved
simulation). All CIs resample TASKS (not episodes), so repeated trials of a task are never treated as independent.

Definitions
- pass^k (per task, n trials, c successes) = C(c,k)/C(n,k): the chance that k trials drawn without replacement
  all succeed (tau-bench's reliability estimator). Averaged over tasks with n >= k.
- valid-call rate = 1 - env_tool_errors / (executed calls + parse-failed outputs). Parse failures are counted as
  attempted-but-invalid calls (approximation: a failed output may not have been a call).
- loop/abort rate = share of episodes that ended by abort or max_steps, or contained 3+ identical consecutive calls.
- cost per success = total USD cost / number of successful episodes (nan if none).
"""

import argparse
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def pass_hat_k(c: int, n: int, k: int) -> float:
    return math.comb(c, k) / math.comb(n, k) if n >= k else float("nan")


def group_by_task(records: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for r in records:
        g[r["task_id"]].append(r)
    return dict(g)


def metrics_of(groups: list[list[dict]]) -> dict:
    eps = [r for g in groups for r in g]
    out = {"n_tasks": len(groups), "n_episodes": len(eps)}
    for k in (1, 2, 3):
        vals = [pass_hat_k(sum(r["reward"] > 0.5 for r in g), len(g), k) for g in groups if len(g) >= k]
        out[f"pass^{k}"] = sum(vals) / len(vals) if vals else float("nan")
    calls = sum(r["n_tool_calls"] for r in eps) + sum(r.get("n_parse_invalid", 0) for r in eps)
    out["valid_call_rate"] = 1 - sum(r.get("n_env_tool_errors", 0) for r in eps) / calls if calls else float("nan")
    out["steps"] = sum(r["n_steps"] for r in eps) / len(eps)
    out["loop_abort_rate"] = sum(r["terminated_by"] in ("abort", "max_steps") or r.get("loop_detected", False)
                                 for r in eps) / len(eps)
    wins = sum(r["reward"] > 0.5 for r in eps)
    out["cost_per_success"] = sum(r["cost_usd"] for r in eps) / wins if wins else float("nan")
    return out


def bootstrap(by_task: dict[str, list[dict]], n_boot: int = 2000, seed: int = 0, other: dict | None = None):
    """95% percentile CIs over tasks. With `other`, also returns the paired difference (this - other) on the
    tasks both have, resampled jointly."""
    rng = random.Random(seed)
    if other is not None:
        ids = sorted(set(by_task) & set(other))
        point = {k: v - metrics_of([other[t] for t in ids])[k] for k, v in metrics_of([by_task[t] for t in ids]).items()
                 if k not in ("n_tasks", "n_episodes")}
        draw = lambda s: {k: metrics_of([by_task[t] for t in s])[k] - metrics_of([other[t] for t in s])[k] for k in point}
    else:
        ids = sorted(by_task)
        point = {k: v for k, v in metrics_of([by_task[t] for t in ids]).items()}
        draw = lambda s: metrics_of([by_task[t] for t in s])
    samples = defaultdict(list)
    for _ in range(n_boot):
        for k, v in draw([rng.choice(ids) for _ in ids]).items():
            if not math.isnan(v) and k not in ("n_tasks", "n_episodes"):
                samples[k].append(v)
    cis = {}
    for k, xs in samples.items():
        xs.sort()
        cis[k] = (xs[int(0.025 * len(xs))], xs[min(len(xs) - 1, int(0.975 * len(xs)))])
    return point, cis


def _trajectory_stats(messages: list[dict]) -> dict:
    calls = [(c["name"], json.dumps(c["arguments"], sort_keys=True)) for m in messages if m["role"] == "assistant"
             for c in (m.get("tool_calls") or [])]
    run, loop = 1, False
    for a, b in zip(calls, calls[1:]):
        run = run + 1 if a == b else 1
        loop = loop or run >= 3
    return {"n_steps": sum(m["role"] == "assistant" for m in messages), "n_tool_calls": len(calls),
            "n_env_tool_errors": sum(m["role"] == "tool" and bool(m.get("error")) for m in messages),
            "n_parse_invalid": 0, "loop_detected": loop}


def load_records(run_dir: str | Path) -> list[dict]:
    """Read a run directory's episodes.jsonl; teacher records are completed from the saved simulation."""
    run_dir = Path(run_dir)
    out = []
    for line in open(run_dir / "episodes.jsonl"):
        r = json.loads(line)
        if "terminated_by" not in r:                                   # teacher record
            sim = json.load(open(run_dir / f"task{r['task_id']}_trial{r['trial']}.json"))
            reason = r["termination_reason"]
            r = {**r, **_trajectory_stats(sim["messages"]), "terminated_by":
                 "done" if reason in ("user_stop", "agent_stop") else "max_steps" if reason == "max_steps" else "error"}
        out.append(r)
    return out


def _fmt(v, ci=None, pct=False):
    s = f"{100 * v:.0f}%" if pct else f"{v:.3g}"
    return s if ci is None else f"{s} [{(100 * ci[0]) if pct else ci[0]:.3g}, {(100 * ci[1]) if pct else ci[1]:.3g}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=glob of run directories (first is the paired-comparison reference)")
    args = ap.parse_args()
    sets = {}
    for spec in args.runs:
        name, pattern = spec.split("=", 1)
        sets[name] = group_by_task([r for d in sorted(glob.glob(pattern)) for r in load_records(d)])
    names = list(sets)
    keys = ["pass^1", "pass^2", "pass^3", "valid_call_rate", "steps", "loop_abort_rate", "cost_per_success"]
    print("| metric | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    res = {n: bootstrap(sets[n]) for n in names}
    print("| tasks / episodes | " + " | ".join(f"{res[n][0]['n_tasks']} / {res[n][0]['n_episodes']}" for n in names) + " |")
    for k in keys:
        print(f"| {k} | " + " | ".join(_fmt(res[n][0][k], res[n][1].get(k), pct=k.startswith(("pass", "valid", "loop")))
                                      for n in names) + " |")
    for n in names[1:]:
        point, cis = bootstrap(sets[n], other=sets[names[0]])
        print(f"\npaired {n} - {names[0]} (tasks in common): " + "; ".join(
            f"{k} {_fmt(point[k], cis.get(k), pct=k.startswith(('pass', 'valid', 'loop')))}" for k in ("pass^1", "pass^2", "steps", "cost_per_success")))


if __name__ == "__main__":
    main()
