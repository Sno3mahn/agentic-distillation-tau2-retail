"""C1: fixed retail split (train / dev / test) derived from tau2's official split.

Dev is a seeded sample of the official train tasks; test is the official test set,
untouched. The sha256 covers only the three ID lists, so any change to membership
changes the hash.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from agentic_distill import REPO_ROOT, TAU2_ROOT, load_config

OFFICIAL_SPLIT = TAU2_ROOT / "data" / "tau2" / "domains" / "retail" / "split_tasks.json"


def _hash(ids: dict) -> str:
    canon = json.dumps(ids, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def build_split(official: dict, dev_size: int, seed: int, tau2_commit: str) -> dict:
    official_train = sorted(official["train"], key=int)
    test = sorted(official["test"], key=int)
    dev = sorted(random.Random(seed).sample(official_train, dev_size), key=int)
    dev_set = set(dev)
    train = [t for t in official_train if t not in dev_set]
    ids = {"train_ids": train, "dev_ids": dev, "test_ids": test}
    return {**ids, "seed": seed, "dev_size": dev_size,
            "tau2_commit": tau2_commit, "sha256": _hash(ids)}


def load_split(path: Path | str) -> dict:
    """Load a split file and verify its hash."""
    split = json.loads(Path(path).read_text())
    ids = {k: split[k] for k in ("train_ids", "dev_ids", "test_ids")}
    if _hash(ids) != split["sha256"]:
        raise ValueError(f"split hash mismatch in {path}")
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=REPO_ROOT / "configs" / "run.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    official = json.loads(OFFICIAL_SPLIT.read_text())
    split = build_split(official, cfg["split"]["dev_size"], cfg["split"]["seed"], cfg["tau2_commit"])
    out = REPO_ROOT / cfg["split"]["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2) + "\n")
    print(f"wrote {out}: train={len(split['train_ids'])} dev={len(split['dev_ids'])} "
          f"test={len(split['test_ids'])} sha256={split['sha256'][:12]}")


if __name__ == "__main__":
    main()
