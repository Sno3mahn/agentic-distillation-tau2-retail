"""Agentic distillation on tau2-bench retail."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TAU2_ROOT = REPO_ROOT / "third_party" / "tau2-bench"


def load_config(path: Path | str = REPO_ROOT / "configs" / "run.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
