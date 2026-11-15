"""Pool the test-split records into one file per model and print the pre-registered comparison.

The epoch3 student's 40 test tasks were run in three places (D11): the "odd" half on Kaggle, the "even" half
on Colab (chunks 0-2) and Modal (chunks 3-4, after Colab's quota ran out). Run directories are listed
explicitly so a stray directory can never be pooled by accident; integrity checks fail loudly.

    PYTHONPATH=src python scripts/pool_test.py            # writes results/test_pooled_*.jsonl, prints tables
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from agentic_distill.metrics import bootstrap, group_by_task, load_records

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {   # model -> list of (environment, run-directory glob)
    "student_epoch3": [("kaggle_t4x2", "data/kaggle_out_s7/out/data/raw/eval_student_s2-epoch3_*"),
                       ("colab_t4", "data/colab_out/results/chunk*/eval_student_s2-epoch3_*"),
                       ("modal_t4", "data/modal_out_s7b/task*/data/raw/eval_student_s2-epoch3_*")],
    "b0": [("kaggle_t4x2", "data/kaggle_out_b0/out/data/raw/eval_student_b0_*")],
    "teacher_base": [("mac_api", "data/raw/eval_teacher_base_t*")],
}
TRIALS = 3


def pool(model: str, test_ids: list[str]) -> list[dict]:
    recs = []
    for env, pattern in SOURCES[model]:
        dirs = sorted(glob.glob(str(ROOT / pattern)))
        assert dirs, f"{model}: no run directory matches {pattern}"
        for d in dirs:
            recs += [{**r, "task_id": str(r["task_id"]), "env": env, "run_dir": str(Path(d).relative_to(ROOT))}
                     for r in load_records(d)]
    keys = Counter((r["task_id"], r["trial"]) for r in recs)
    dup = [k for k, n in keys.items() if n > 1]
    assert not dup, f"{model}: duplicate (task, trial) records, e.g. {dup[:3]}"
    assert {t for t, _ in keys} == set(test_ids), f"{model}: tasks differ from the test split"
    assert all(n == TRIALS for n in Counter(t for t, _ in keys).values()), f"{model}: not {TRIALS} trials per task"
    return recs


def _row(name, point, cis, keys):
    cell = lambda k: (f"{100 * point[k]:.1f}% [{100 * cis[k][0]:.1f}, {100 * cis[k][1]:.1f}]" if k in cis
                      else f"{100 * point[k]:.1f}%")
    return f"| {name} | " + " | ".join(cell(k) for k in keys) + " |"


def main() -> None:
    test_ids = [str(t) for t in json.load(open(ROOT / "data/splits/retail_split.json"))["test_ids"]]
    pooled = {m: pool(m, test_ids) for m in SOURCES}
    for m, recs in pooled.items():
        out = ROOT / f"results/test_pooled_{m}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in sorted(recs, key=lambda r: (r["task_id"], r["trial"]))))
        print(f"{m}: {len(recs)} episodes, {len(set(r['task_id'] for r in recs))} tasks, "
              f"envs {dict(Counter(r['env'] for r in recs))}, terminations {dict(Counter(r['terminated_by'] for r in recs))}"
              f" -> {out.relative_to(ROOT)}")
    keys = ["pass^1", "pass^2", "pass^3", "valid_call_rate", "loop_abort_rate", "crash_rate"]
    print("\n| model | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    g = {m: group_by_task(r) for m, r in pooled.items()}
    for m in pooled:
        print(_row(m, *bootstrap(g[m]), keys))
    print("\nPaired over the 40 test tasks (95% task-bootstrap CI):")
    for a, b in (("student_epoch3", "b0"), ("teacher_base", "student_epoch3")):
        point, cis = bootstrap(g[a], other=g[b])
        print(f"  {a} - {b}: " + "; ".join(f"{k} {100 * point[k]:+.1f} pts [{100 * cis[k][0]:+.1f}, {100 * cis[k][1]:+.1f}]"
                                          for k in ("pass^1", "pass^3")))
    point, cis = bootstrap(g["student_epoch3"], other=g["b0"])
    lo = cis["pass^1"][0]
    print(f"\nPre-registered bar (results/threshold.md): student - B0 pass^1 CI lower bound > 0 -> "
          f"{'MET' if lo > 0 else 'NOT MET'} (lower bound {100 * lo:+.1f} pts)")
    # Sensitivity (D10): crashed episodes removed instead of scored 0
    nc = {m: group_by_task([r for r in pooled[m] if r["terminated_by"] != "crash"]) for m in ("student_epoch3", "b0")}
    point, cis = bootstrap(nc["student_epoch3"], other=nc["b0"])
    print(f"Sensitivity, crashes excluded: student - B0 pass^1 {100 * point['pass^1']:+.1f} pts "
          f"[{100 * cis['pass^1'][0]:+.1f}, {100 * cis['pass^1'][1]:+.1f}]")


if __name__ == "__main__":
    sys.exit(main())
