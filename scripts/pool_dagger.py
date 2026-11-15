"""Pool the DAgger round-1 records and print the comparisons fixed in results/threshold_dagger.md.

    PYTHONPATH=src python scripts/pool_dagger.py dev     # dev gate: DAgger vs epoch3, 14 dev tasks x 1 trial (Modal)
    PYTHONPATH=src python scripts/pool_dagger.py test    # DAgger vs epoch3 and vs B0, 40 test tasks x 3 trials
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from agentic_distill.metrics import bootstrap, group_by_task, load_records

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["pass^1", "pass^3", "valid_call_rate", "loop_abort_rate", "crash_rate"]


def pool(pattern: str, ids: list[str], trials: int) -> list[dict]:
    dirs = sorted(glob.glob(str(ROOT / pattern)))
    assert dirs, f"no run directory matches {pattern}"
    recs = [{**r, "task_id": str(r["task_id"]), "env": "modal_t4", "run_dir": str(Path(d).relative_to(ROOT))}
            for d in dirs for r in load_records(d)]
    keys = Counter((r["task_id"], r["trial"]) for r in recs)
    assert all(n == 1 for n in keys.values()), "duplicate (task, trial) records"
    assert {t for t, _ in keys} == set(ids), f"tasks differ from the split: missing {sorted(set(ids) - {t for t, _ in keys})}"
    assert all(n == trials for n in Counter(t for t, _ in keys).values()), f"not {trials} trials per task"
    return recs


def table(sets: dict[str, list[dict]]) -> None:
    print("| model | episodes | wins | " + " | ".join(KEYS) + " | terminations |\n|---|" + "---|" * (len(KEYS) + 3))
    for m, recs in sets.items():
        point, cis = bootstrap(group_by_task(recs))
        cells = [f"{100 * point[k]:.1f}%" + (f" [{100 * cis[k][0]:.1f}, {100 * cis[k][1]:.1f}]" if k.startswith("pass") and k in cis else "")
                 for k in KEYS if not (k == "pass^3" and point[k] != point[k])]
        if len(cells) < len(KEYS):
            cells.insert(1, "n/a")
        print(f"| {m} | {len(recs)} | {sum(r['reward'] > 0.5 for r in recs)} | " + " | ".join(cells)
              + f" | {dict(Counter(r['terminated_by'] for r in recs))} |")


def paired(a: str, b: str, sets: dict, ks=("pass^1", "pass^3")) -> float:
    point, cis = bootstrap(group_by_task(sets[a]), other=group_by_task(sets[b]))
    print(f"  {a} - {b}: " + "; ".join(f"{k} {100 * point[k]:+.1f} pts [{100 * cis[k][0]:+.1f}, {100 * cis[k][1]:+.1f}]"
                                      for k in ks if k in cis))
    return cis["pass^1"][0]


def main() -> None:
    mode = sys.argv[1]
    split = json.load(open(ROOT / "data/splits/retail_split.json"))
    if mode == "dev":
        ids = [str(t) for t in split["dev_ids"]]
        sets = {"epoch3": pool("data/modal_out_daggerdev_epoch3/g*/data/raw/*", ids, 1),
                "dagger_r1": pool("data/modal_out_daggerdev_dagger/g*/data/raw/*", ids, 1)}
        table(sets)
        paired("dagger_r1", "epoch3", sets, ks=("pass^1",))
        p = {m: sum(r["reward"] > 0.5 for r in recs) / len(recs) for m, recs in sets.items()}
        print(f"\nDev gate (results/threshold_dagger.md): DAgger pass^1 >= epoch3 pass^1 -> "
              f"{'PASS' if p['dagger_r1'] >= p['epoch3'] else 'FAIL'} ({p['dagger_r1']:.3f} vs {p['epoch3']:.3f})")
        for m, recs in sets.items():
            (ROOT / f"results/dagger_dev_{m}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in sorted(recs, key=lambda r: (r["task_id"], r["trial"]))))
    else:
        ids = [str(t) for t in split["test_ids"]]
        sets = {"dagger_r1": pool("data/modal_out_daggertest/g*/data/raw/*", ids, 3)}
        (ROOT / "results/test_pooled_dagger_r1.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in sorted(sets["dagger_r1"], key=lambda r: (r["task_id"], r["trial"]))))
        for m in ("student_epoch3", "b0"):
            sets[m] = [json.loads(l) for l in open(ROOT / f"results/test_pooled_{m}.jsonl")]
        table(sets)
        print("\nPaired over the 40 test tasks (95% task-bootstrap CI):")
        lo = paired("dagger_r1", "student_epoch3", sets)
        paired("dagger_r1", "b0", sets)
        print(f"\nPre-registered rule: DAgger is called better than epoch3 only if the CI lower bound of the pass^1 "
              f"difference is above 0 -> {'BETTER' if lo > 0 else 'NO IMPROVEMENT SHOWN'} (lower bound {100 * lo:+.1f} pts)")
        nc = {m: [r for r in sets[m] if r["terminated_by"] != "crash"] for m in ("dagger_r1", "student_epoch3")}
        print("Sensitivity, crashes excluded:")
        paired("dagger_r1", "student_epoch3", nc, ks=("pass^1",))


if __name__ == "__main__":
    main()
