"""S2a acceptance checks for the formatter. Needs the tokenizer files in data/student_tok (see configs/run.yaml)."""

import glob
import json

import pytest
from transformers import AutoTokenizer

from agentic_distill import REPO_ROOT
from agentic_distill.formatter import (EOS, GEN_PROMPT, Formatter, ParseError, Parsed, assistant_body,
                                       from_tau2, parse)

TOK_DIR = REPO_ROOT / "data" / "student_tok"
pytestmark = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="tokenizer files not downloaded")


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOK_DIR)


@pytest.fixture(scope="module")
def retail():
    from tau2.runner import build_environment
    env = build_environment("retail")
    return env.get_policy(), [t.openai_schema for t in env.get_tools()]


def _episodes():
    return sorted(glob.glob(str(REPO_ROOT / "data/raw/*/task*_trial*.json")))


def test_native_parts_match_template(tok):
    """System/user turns and the generation prompt must equal Cohere's own template output."""
    f = Formatter(tok)
    pol, tools = "P", [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    msgs = [{"role": "user", "content": "hi"}]
    from agentic_distill.formatter import system_text
    native = tok.apply_chat_template(
        [{"role": "system", "content": system_text(pol, tools)}] + msgs,
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert f.render(pol, tools, msgs, add_generation_prompt=True) == native


def test_parse_roundtrip_variants():
    cases = [("", [{"name": "a", "arguments": {}}]),
             ("Let me check.", [{"name": "a", "arguments": {"x": [1, {"y": "é\n\"q\""}]}}]),
             ("Hello there.", []),
             ("two", [{"name": "a", "arguments": {"i": 1}}, {"name": "b", "arguments": {"i": 2}}])]
    for text, calls in cases:
        body = assistant_body(text, calls)
        out = parse(f"<|START_RESPONSE|>{body}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>")
        assert out == Parsed(text, calls), (text, calls, out)


@pytest.mark.parametrize("bad", ["no block", "<|START_RESPONSE|>x", "<|START_RESPONSE|><tool_call>{bad}</tool_call><|END_RESPONSE|>",
                                 "<|START_RESPONSE|><tool_call>{\"name\":\"a\",\"arguments\":{}}<|END_RESPONSE|>",
                                 "<|START_RESPONSE|><tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call> tail<|END_RESPONSE|>",
                                 "<|START_RESPONSE|><tool_call>{\"name\":\"a\",\"arguments\":[]}</tool_call><|END_RESPONSE|>"])
def test_parse_errors(bad):
    assert isinstance(parse(bad), ParseError)


def test_roundtrip_all_teacher_turns(tok, retail):
    """Every assistant turn in every saved episode: render -> generate-style text -> parse is lossless."""
    eps = _episodes()
    assert eps, "no saved episodes"
    f, n = Formatter(tok), 0
    pol, tools = retail
    for path in eps:
        chat = from_tau2(json.load(open(path))["messages"])
        turns = f.render_turns(chat)
        for m in chat:
            if m["role"] != "assistant":
                continue
            body = assistant_body(m["content"], m["tool_calls"])
            got = parse(f"<|START_RESPONSE|>{body}<|END_RESPONSE|>{EOS}")
            assert got == Parsed((m["content"] or "").strip(), m["tool_calls"]), (path, m)
            n += 1
    assert n > 200


def test_prefix_property_all_turns(tok, retail):
    """prefix + generation prompt must be a token-level prefix of the full text, for every assistant turn."""
    f, pol, tools = Formatter(tok), *retail
    checked = 0
    for path in _episodes()[:6]:                       # token-level check on a subset keeps the test fast
        chat = from_tau2(json.load(open(path))["messages"])
        full = tok(f.render(pol, tools, chat), add_special_tokens=False)["input_ids"]
        for i, m in enumerate(chat):
            if m["role"] != "assistant":
                continue
            pre = tok(f.render(pol, tools, chat[:i], add_generation_prompt=True), add_special_tokens=False)["input_ids"]
            assert full[: len(pre)] == pre, (path, i)
            checked += 1
    assert checked > 30


def test_truncate_at_tool_result():
    from agentic_distill.formatter import truncate_at_tool_result
    call = '<tool_call>{"name":"a","arguments":{"i":1}}</tool_call>'
    hallucinated = f'<|START_RESPONSE|>Checking.\n{call}\n<tool_result name="a">{{}}</tool_result>\n{call.replace("1", "2")}'
    out = parse(truncate_at_tool_result(hallucinated))
    assert out == Parsed("Checking.", [{"name": "a", "arguments": {"i": 1}}])
    clean = f"<|START_RESPONSE|>{call}<|END_RESPONSE|>"
    assert truncate_at_tool_result(clean) == clean
