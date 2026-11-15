"""DAgger-lite relabelling (one round): teacher labels at states the STUDENT visited.

For each FAILED student episode on a train task, pick up to `per_episode` states = assistant turns
(after tau2's scripted greeting) that either (a) are the student's first error turn (a parse error, an
abort, or a tool call returning an error) or (b) make a write call (cancel_/exchange_/modify_/return_),
stopping at the first error. The conversation BEFORE that turn is the student's own history with real
tool results; the teacher (the `rationale` variant, same prompt/settings as the S4 data) writes the next
assistant turn. Training example = that prefix with loss masked + the teacher turn with loss.

No environment replay is needed: the teacher only writes one turn at a state already reached.
Labels are NOT success-filtered (a single turn has no reward); this is the main quality risk.

    python -m agentic_distill.dagger_relabel "data/modal_out_dagger/*/data/raw/*" --out data/sft/dagger_r1.jsonl
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WRITE_PREFIXES = ("cancel_", "exchange_", "modify_", "return_")


def _role(m: dict) -> str:
    return m["role"]


def first_error_turn(messages: list[dict]) -> int | None:
    """Index of the first assistant turn that is an error: invalid attempts, abort, or an erroring tool result."""
    for i, m in enumerate(messages):
        if _role(m) != "assistant" or i == 0:
            continue
        raw = m.get("raw_data") or {}
        if raw.get("invalid_attempts") or raw.get("aborted"):
            return i
        if m.get("tool_calls"):
            j = i + 1
            while j < len(messages) and _role(messages[j]) == "tool":
                if messages[j].get("error") or (messages[j].get("content") or "").startswith("Error"):
                    return i
                j += 1
    return None


def select_states(messages: list[dict], per_episode: int = 3) -> list[int]:
    """Assistant-turn indices to relabel, earliest first (see module docstring)."""
    err = first_error_turn(messages)
    stop = err if err is not None else len(messages) - 1
    cands = []
    for i, m in enumerate(messages[: stop + 1]):
        if _role(m) != "assistant" or i == 0:
            continue
        writes = any(c["name"].startswith(WRITE_PREFIXES) for c in (m.get("tool_calls") or []))
        if writes or i == err:
            cands.append(i)
    if not cands:                          # no write and no error: the student ended without acting
        last = [i for i, m in enumerate(messages) if _role(m) == "assistant" and i > 0]
        cands = last[-1:]
    if err is not None and err in cands:   # always keep the error turn, then the earliest others
        others = [i for i in cands if i != err][: per_episode - 1]
        return sorted(others + [err])
    return cands[:per_episode]


def to_tau2(messages: list[dict]):
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    cls = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if k != "raw_data"}
        out.append(cls[_role(m)].model_validate(m))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="globs of student rollout run directories (train split)")
    ap.add_argument("--out", default="data/sft/dagger_r1.jsonl")
    ap.add_argument("--per-episode", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=13000, help="drop (never truncate) longer examples")
    ap.add_argument("--slice", default="dagger")
    ap.add_argument("--allow-peak", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="select states and print counts; no API calls")
    args = ap.parse_args()

    import litellm
    from transformers import AutoTokenizer
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
    from tau2.data_model.message import SystemMessage
    from tau2.runner import build_environment
    from tau2.utils.llm_utils import generate

    from agentic_distill import REPO_ROOT, load_config
    from agentic_distill.agents import RATIONALE_INSTRUCTION
    from agentic_distill.build_sft import IGNORE, build_example
    from agentic_distill.formatter import Formatter, from_tau2
    from agentic_distill.ledger import Ledger, call_cost, is_peak, usage_from_raw

    cfg = load_config()
    if not os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("API_DEEPSEEK"):
        os.environ["DEEPSEEK_API_KEY"] = os.environ["API_DEEPSEEK"]
    if not args.dry_run:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            sys.exit("API_DEEPSEEK (or DEEPSEEK_API_KEY) is not set")
        if not args.allow_peak and is_peak(datetime.now(timezone.utc), cfg["prices"]["peak_windows_utc"]):
            sys.exit("DeepSeek peak hours (2x price): refusing to run. Wait or pass --allow-peak.")
    train = set(json.load(open(REPO_ROOT / cfg["split"]["path"]))["train_ids"])
    var = cfg["teacher"]["variants"]["rationale"]           # same teacher behaviour as the S4 data (D8)
    model = cfg["teacher"]["model"]
    env = build_environment(cfg["domain"])
    tools, policy = env.get_tools(), env.get_policy()
    schemas = [t.openai_schema for t in tools]
    tool_names = {t.name for t in tools}
    system = SystemMessage(role="system", content=SYSTEM_PROMPT.format(
        domain_policy=policy, agent_instruction=f"{AGENT_INSTRUCTION}\n{RATIONALE_INSTRUCTION}"))
    tok = AutoTokenizer.from_pretrained(REPO_ROOT / cfg["student"]["tokenizer_dir"])
    fmt = Formatter(tok)
    ledger = Ledger(REPO_ROOT / cfg["paths"]["ledger"], cfg["budget"]["slices_usd"][args.slice])

    states = []                                              # (run dir, meta, sim messages, turn index)
    n_eps = n_failed = 0
    for pattern in args.runs:
        for d in sorted(glob.glob(str(REPO_ROOT / pattern)) or glob.glob(pattern)):
            for line in open(Path(d) / "episodes.jsonl"):
                meta = json.loads(line)
                assert str(meta["task_id"]) in train, f"non-train task {meta['task_id']} in {d}"
                n_eps += 1
                if meta["reward"] >= 0.5 or meta["terminated_by"] == "crash":
                    continue
                n_failed += 1
                sim = json.load(open(Path(d) / f"task{meta['task_id']}_trial{meta['trial']}.json"))
                for i in select_states(sim["messages"], args.per_episode):
                    states.append((d, meta, sim, i))
    print(f"episodes={n_eps} failed={n_failed} states selected={len(states)}", flush=True)
    if args.dry_run:
        return

    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = agree = dropped_len = dropped_bad = 0
    with open(out, "w") as f:
        for d, meta, sim, i in states:
            if ledger.remaining(slice=args.slice) <= 0:
                print(f"STOP: budget slice {args.slice} exhausted")
                break
            prefix = sim["messages"][:i]
            msg = generate(model=model, tools=tools, messages=[system] + to_tau2(prefix),
                           call_name="agent_response", **var["llm_args"])
            raw = msg.raw_data or {}
            usage = usage_from_raw(raw)
            created = datetime.fromtimestamp(raw["created"], tz=timezone.utc)
            peak = is_peak(created, cfg["prices"]["peak_windows_utc"])
            cost = call_cost(usage, cfg["prices"][model]["peak" if peak else "off_peak"])
            ledger.append([{"slice": args.slice, "run": "dagger_relabel", "task_id": meta["task_id"],
                            "trial": meta["trial"], "msg_idx": i, "role": "agent", "model": model,
                            "created_utc": created.isoformat(), "peak": peak, **usage, "cost_usd": cost}])
            calls = [{"name": c.name, "arguments": c.arguments} for c in (msg.tool_calls or [])]
            if (not calls and not (msg.content or "").strip()) or any(c["name"] not in tool_names for c in calls):
                dropped_bad += 1
                continue
            label = {"role": "assistant", "content": msg.content, "tool_calls": calls}
            chat = from_tau2(prefix) + [label]
            ex = build_example(fmt, tok, policy, schemas, chat, loss_on="last")
            n = len(ex["input_ids"])
            if n > args.max_tokens:
                dropped_len += 1
                continue
            student = from_tau2([sim["messages"][i]])[0] if from_tau2([sim["messages"][i]]) else {}
            same = [(c["name"], c["arguments"]) for c in student.get("tool_calls", [])] == \
                   [(c["name"], c["arguments"]) for c in calls] and bool(calls)
            agree += same
            kept += 1
            f.write(json.dumps({**ex, "source_episode_id": f"{Path(d).name}/{sim['id']}", "variant": "dagger_r1",
                                "task_id": meta["task_id"], "state_turn": i, "teacher_agrees": same,
                                "n_tokens": n, "n_loss_tokens": sum(l != IGNORE for l in ex["labels"])}) + "\n")
            print(f"task={meta['task_id']} trial={meta['trial']} turn={i} agree={same} tokens={n} cost=${cost:.5f}",
                  flush=True)
    print(f"kept={kept} teacher_agrees={agree} dropped_invalid={dropped_bad} dropped_over_length={dropped_len}; "
          f"slice {args.slice} total=${ledger.total(slice=args.slice):.4f}; wrote {out}")


if __name__ == "__main__":
    main()
