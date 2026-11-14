"""C4: success-filtered SFT dataset with assistant-only loss masks.

Per episode: keep reward == 1, render with the formatter, tokenize piece by piece, and set labels to -100
everywhere except what the model itself generates: each assistant turn from <|START_RESPONSE|> through
<|END_OF_TURN_TOKEN|>. Masked: system turn, user/tool turns, the generation prompt (SOT, CHATBOT,
empty thinking block), and the scripted greeting (tau2 injects it; the student never generates it).
Over-length episodes are rejected and counted, never truncated.

    python -m agentic_distill.build_sft "data/raw/s4_rationale_p*" --out data/sft/train.jsonl
"""

import argparse
import glob
import json
import statistics as st
from pathlib import Path

from transformers import AutoTokenizer

from agentic_distill import REPO_ROOT, load_config
from agentic_distill.formatter import GEN_PROMPT, Formatter, from_tau2

IGNORE = -100


def build_example(fmt: Formatter, tok, policy: str, tools: list[dict], chat: list[dict]) -> dict:
    """-> {input_ids, labels}. Tokenizes piece by piece (pieces are delimited by special tokens)."""
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    ids, labels = enc(fmt.head(policy, tools)), None
    labels = [IGNORE] * len(ids)
    greeting_done = False
    for role, text in fmt.render_turns(chat):
        if role == "assistant" and greeting_done:
            assert text.startswith(GEN_PROMPT), "assistant turn must start with the generation prompt"
            prompt_ids, gen_ids = enc(GEN_PROMPT), enc(text[len(GEN_PROMPT):])
            ids += prompt_ids + gen_ids
            labels += [IGNORE] * len(prompt_ids) + gen_ids
        else:
            piece = enc(text)
            ids += piece
            labels += [IGNORE] * len(piece)
            if role == "assistant":
                greeting_done = True          # the first assistant turn is tau2's scripted greeting
    return {"input_ids": ids, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help="glob of run directories")
    ap.add_argument("--out", default="data/sft/train.jsonl")
    args = ap.parse_args()
    cfg = load_config()
    split = json.load(open(REPO_ROOT / cfg["split"]["path"]))
    train = set(split["train_ids"])
    tok = AutoTokenizer.from_pretrained(REPO_ROOT / cfg["student"]["tokenizer_dir"])
    from tau2.runner import build_environment
    env = build_environment(cfg["domain"])
    policy, tools = env.get_policy(), [t.openai_schema for t in env.get_tools()]
    fmt, max_len = Formatter(tok), cfg["student"]["max_len"]

    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = failed = over = 0
    tasks_all, tasks_kept, lengths, trained = set(), set(), [], []
    with open(out, "w") as f:
        for d in sorted(glob.glob(str(REPO_ROOT / args.runs))):
            for line in open(Path(d) / "episodes.jsonl"):
                meta = json.loads(line)
                assert meta["task_id"] in train, f"non-train task {meta['task_id']} in {d}"
                seen += 1
                tasks_all.add(meta["task_id"])
                if meta["reward"] < 0.5:
                    failed += 1
                    continue
                sim = json.load(open(Path(d) / f"task{meta['task_id']}_trial{meta['trial']}.json"))
                ex = build_example(fmt, tok, policy, tools, from_tau2(sim["messages"]))
                n = len(ex["input_ids"])
                if n > max_len:
                    over += 1
                    continue
                kept += 1
                tasks_kept.add(meta["task_id"])
                lengths.append(n)
                trained.append(sum(l != IGNORE for l in ex["labels"]))
                f.write(json.dumps({**ex, "source_episode_id": f"{Path(d).name}/{sim['id']}",
                                    "task_id": meta["task_id"], "n_tokens": n}) + "\n")
    lengths.sort()
    print(f"episodes seen={seen} failed(reward<1)={failed} over_length(>{max_len})={over} kept={kept}")
    print(f"tasks covered: {len(tasks_kept)}/{len(tasks_all)} seen; train tasks total {len(train)}")
    if kept:
        print(f"tokens/episode median={int(st.median(lengths))} p90={lengths[int(0.9 * len(lengths))]} max={lengths[-1]} "
              f"total={sum(lengths)}; trained tokens (loss) total={sum(trained)} "
              f"({100 * sum(trained) / sum(lengths):.1f}% of tokens)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
