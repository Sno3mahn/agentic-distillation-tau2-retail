"""C4: success-filtered SFT dataset with assistant-only loss masks.

Per episode: keep reward == 1, render with the formatter, tokenize piece by piece, and set labels to -100
everywhere except what the model itself generates: each assistant turn from <|START_RESPONSE|> through
<|END_OF_TURN_TOKEN|>. Masked: system turn, user/tool turns, the generation prompt (SOT, CHATBOT,
empty thinking block), and the scripted greeting (tau2 injects it; the student never generates it).
Over-length episodes are rejected and counted, never truncated.

    python -m agentic_distill.build_sft "data/raw/s4_rationale_p*" "data/raw/s4_thinkrat_h*" --cap-per-task 8

With --cap-per-task N, a task's reward==1 episodes beyond N are dropped by a seeded shuffle (seed from
configs/run.yaml split.seed), so the choice is reproducible and not biased towards either source.
"""

import argparse
import glob
import json
import random
import statistics as st
from pathlib import Path

from transformers import AutoTokenizer

from agentic_distill import REPO_ROOT, load_config
from agentic_distill.formatter import GEN_PROMPT, Formatter, from_tau2

IGNORE = -100


def build_example(fmt: Formatter, tok, policy: str, tools: list[dict], chat: list[dict],
                  loss_on: str = "all") -> dict:
    """-> {input_ids, labels}. Tokenizes piece by piece (pieces are delimited by special tokens).

    loss_on="all": loss on every assistant turn after the greeting (S4 data).
    loss_on="last": loss only on the final assistant turn (DAgger labels; the prefix is the student's own)."""
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    ids, labels = enc(fmt.head(policy, tools)), None
    labels = [IGNORE] * len(ids)
    greeting_done = False
    turns = fmt.render_turns(chat)
    last = max(i for i, (r, _) in enumerate(turns) if r == "assistant")
    for k, (role, text) in enumerate(turns):
        if role == "assistant" and greeting_done and (loss_on == "all" or k == last):
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
    ap.add_argument("runs", nargs="+", help="globs of run directories")
    ap.add_argument("--out", default="data/sft/train.jsonl")
    ap.add_argument("--cap-per-task", type=int, default=None, help="max kept episodes per task")
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
    seen = kept = failed = over = capped = 0
    tasks_all, tasks_kept, lengths, trained = set(), set(), [], []
    cands = []                                   # reward==1 episodes: (run dir, meta)
    for pattern in args.runs:
        for d in sorted(glob.glob(str(REPO_ROOT / pattern))):
            for line in open(Path(d) / "episodes.jsonl"):
                meta = json.loads(line)
                assert meta["task_id"] in train, f"non-train task {meta['task_id']} in {d}"
                seen += 1
                tasks_all.add(meta["task_id"])
                if meta["reward"] < 0.5:
                    failed += 1
                else:
                    cands.append((d, meta))
    if args.cap_per_task:
        rng = random.Random(cfg["split"]["seed"])
        by_task = {}
        for c in cands:
            by_task.setdefault(c[1]["task_id"], []).append(c)
        cands = []
        for t in sorted(by_task):
            group = by_task[t]
            rng.shuffle(group)
            capped += max(0, len(group) - args.cap_per_task)
            cands += group[:args.cap_per_task]
    with open(out, "w") as f:
        for d, meta in sorted(cands, key=lambda c: (c[0], c[1]["task_id"], c[1]["trial"])):
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
                                "variant": meta["variant"], "task_id": meta["task_id"], "n_tokens": n}) + "\n")
    lengths.sort()
    print(f"episodes seen={seen} failed(reward<1)={failed} dropped_by_cap={capped} over_length(>{max_len})={over} kept={kept}")
    print(f"tasks covered: {len(tasks_kept)}/{len(tasks_all)} seen; train tasks total {len(train)}")
    if kept:
        print(f"tokens/episode median={int(st.median(lengths))} p90={lengths[int(0.9 * len(lengths))]} max={lengths[-1]} "
              f"total={sum(lengths)}; trained tokens (loss) total={sum(trained)} "
              f"({100 * sum(trained) / sum(lengths):.1f}% of tokens)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
