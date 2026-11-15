"""DAgger relabelling: state selection and last-turn-only loss masks."""

import glob
import json

import pytest

from agentic_distill import REPO_ROOT
from agentic_distill.dagger_relabel import first_error_turn, select_states


def _a(calls=(), invalid=False):
    return {"role": "assistant", "content": "x", "raw_data": {"invalid_attempts": [1] if invalid else []},
            "tool_calls": [{"id": f"c{n}", "name": n, "arguments": {}} for n in calls] or None}


def _t(err=False):
    return {"role": "tool", "content": "Error: no" if err else "ok", "error": err}


U = {"role": "user", "content": "hi"}


def test_first_error_is_tool_error_or_invalid_attempt():
    msgs = [_a(), U, _a(["get_order_details"]), _t(), _a(["modify_pending_order_items"]), _t(err=True), U, _a(invalid=True)]
    assert first_error_turn(msgs) == 4
    assert first_error_turn([_a(), U, _a(invalid=True)]) == 2
    assert first_error_turn([_a(), U, _a(["x"]), _t()]) is None


def test_select_keeps_error_and_earlier_writes_never_the_greeting():
    msgs = [_a(["cancel_pending_order"]), U, _a(["cancel_pending_order"]), _t(), U,
            _a(["return_delivered_order_items"]), _t(), U, _a(["get_user_details"]), _t(err=True), U,
            _a(["exchange_delivered_order_items"]), _t()]
    assert select_states(msgs, per_episode=3) == [2, 5, 8]      # writes before the error + the error turn
    assert select_states(msgs, per_episode=2) == [2, 8]         # the error turn is always kept
    assert 11 not in select_states(msgs, per_episode=5)         # nothing after the first error


def test_select_falls_back_to_last_assistant_turn():
    msgs = [_a(), U, _a(["get_user_details"]), _t(), U, _a()]
    assert select_states(msgs) == [5]


TOK_DIR = REPO_ROOT / "data" / "student_tok"


@pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="tokenizer files not downloaded")
def test_loss_on_last_masks_all_but_final_assistant_turn():
    from tau2.runner import build_environment
    from transformers import AutoTokenizer
    from agentic_distill.build_sft import IGNORE, build_example
    from agentic_distill.formatter import Formatter, from_tau2
    env = build_environment("retail")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    fmt, pol, tools = Formatter(tok), env.get_policy(), [t.openai_schema for t in env.get_tools()]
    p = sorted(glob.glob(str(REPO_ROOT / "data/raw/s3_pilot10_rationale/task*_trial*.json")))[0]
    chat = from_tau2(json.load(open(p))["messages"])
    full = build_example(fmt, tok, pol, tools, chat)
    last = build_example(fmt, tok, pol, tools, chat, loss_on="last")
    assert last["input_ids"] == full["input_ids"]                  # same text, only the mask differs
    on = [i for i, l in enumerate(last["labels"]) if l != IGNORE]
    assert on and all(full["labels"][i] == last["labels"][i] for i in on)
    tail = tok.decode([last["input_ids"][i] for i in on])
    body = [t for r, t in fmt.render_turns(chat) if r == "assistant"][-1]
    assert tail.strip() and tail.strip() in body                    # the unmasked span is the final assistant turn
    assert sum(l != IGNORE for l in last["labels"]) < sum(l != IGNORE for l in full["labels"])
