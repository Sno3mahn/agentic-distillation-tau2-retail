"""S2a length report: rendered token counts of saved teacher episodes vs the student's 32K window."""

import glob
import json
import statistics as st
from collections import defaultdict

from transformers import AutoTokenizer

from agentic_distill import REPO_ROOT, load_config
from agentic_distill.formatter import Formatter, from_tau2


def main() -> None:
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(REPO_ROOT / cfg["student"]["tokenizer_dir"])
    from tau2.runner import build_environment
    env = build_environment(cfg["domain"])
    pol, tools = env.get_policy(), [t.openai_schema for t in env.get_tools()]
    f = Formatter(tok)
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    print(f"fixed system prefix: {n(f.head(pol, tools))} tokens (policy {len(pol)} chars, {len(tools)} tools)")
    by_variant = defaultdict(list)
    for path in sorted(glob.glob(str(REPO_ROOT / cfg["paths"]["raw"] / "*/task*_trial*.json"))):
        run = path.split("/")[-2]                      # variant is one underscore-separated part of the run name
        variant = next((p for p in run.split("_") if p in cfg["teacher"]["variants"]), "other")
        variant += "-pilot10" if "pilot10" in run else ""
        chat = from_tau2(json.load(open(path))["messages"])
        by_variant[variant].append(n(f.render(pol, tools, chat)))
    limit = cfg["student"]["max_len"]
    for v, xs in sorted(by_variant.items()):
        xs.sort()
        p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
        print(f"{v:10} n={len(xs):2} median={int(st.median(xs))} p90={p90} max={xs[-1]} "
              f"over_{limit}={sum(x > limit for x in xs)}")


if __name__ == "__main__":
    main()
