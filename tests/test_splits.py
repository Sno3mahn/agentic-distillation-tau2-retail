import json

from agentic_distill.splits import OFFICIAL_SPLIT, build_split

OFFICIAL = json.loads(OFFICIAL_SPLIT.read_text())


def _split(seed=20261113):
    return build_split(OFFICIAL, dev_size=14, seed=seed, tau2_commit="x")


def test_counts():
    s = _split()
    assert (len(s["train_ids"]), len(s["dev_ids"]), len(s["test_ids"])) == (60, 14, 40)


def test_disjoint_and_complete():
    s = _split()
    tr, dv, te = map(set, (s["train_ids"], s["dev_ids"], s["test_ids"]))
    assert not (tr & dv) and not (tr & te) and not (dv & te)
    assert tr | dv == set(OFFICIAL["train"])
    assert te == set(OFFICIAL["test"])


def test_hash_stable_and_seed_sensitive():
    assert _split()["sha256"] == _split()["sha256"]
    assert _split()["sha256"] != _split(seed=1)["sha256"]
