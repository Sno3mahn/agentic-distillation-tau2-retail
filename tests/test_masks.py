"""Loss-mask correctness (P0): only generated assistant tokens carry labels."""

import glob
import json

import pytest
from transformers import AutoTokenizer

from agentic_distill import REPO_ROOT
from agentic_distill.build_sft import IGNORE, build_example
from agentic_distill.formatter import ER, GEN_PROMPT, SR, Formatter, from_tau2, parse

TOK_DIR = REPO_ROOT / "data" / "student_tok"
pytestmark = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="tokenizer files not downloaded")


@pytest.fixture(scope="module")
def ctx():
    from tau2.runner import build_environment
    env = build_environment("retail")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    return tok, Formatter(tok), env.get_policy(), [t.openai_schema for t in env.get_tools()]


def _chats(n=4):
    for p in sorted(glob.glob(str(REPO_ROOT / "data/raw/s3_pilot10_rationale/task*_trial*.json")))[:n]:
        yield from_tau2(json.load(open(p))["messages"])


def test_piecewise_tokenization_equals_whole(ctx):
    tok, fmt, pol, tools = ctx
    for chat in _chats():
        ex = build_example(fmt, tok, pol, tools, chat)
        whole = tok(fmt.render(pol, tools, chat), add_special_tokens=False)["input_ids"]
        assert ex["input_ids"] == whole and len(ex["labels"]) == len(whole)


def test_unmasked_tokens_are_exactly_the_generated_assistant_turns(ctx):
    tok, fmt, pol, tools = ctx
    for chat in _chats():
        ex = build_example(fmt, tok, pol, tools, chat)
        # split labelled tokens into contiguous runs; each run is one generated assistant turn
        runs, cur = [], []
        for t, l in zip(ex["input_ids"], ex["labels"]):
            if l != IGNORE:
                assert l == t, "label must equal its input token"
                cur.append(t)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        generated = [m for m in chat[1:] if m["role"] == "assistant"]       # chat[0] = scripted greeting
        assert len(runs) == len(generated)
        for run, m in zip(runs, generated):
            text = tok.decode(run, skip_special_tokens=False)
            assert text.startswith(SR) and text.endswith("<|END_OF_TURN_TOKEN|>")
            got = parse(text)
            assert got.text == (m["content"] or "").strip() and got.tool_calls == m["tool_calls"]
            assert "<tool_result" not in text and "<|USER_TOKEN|>" not in text and GEN_PROMPT not in text


def test_greeting_and_prompts_are_masked(ctx):
    tok, fmt, pol, tools = ctx
    chat = next(_chats(1))
    assert chat[0]["role"] == "assistant"
    ex = build_example(fmt, tok, pol, tools, chat)
    labelled = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE], skip_special_tokens=False)
    assert chat[0]["content"] not in labelled and "# System Preamble" not in labelled
    assert "<|START_THINKING|>" not in labelled      # generation-prompt tokens are context, not targets
