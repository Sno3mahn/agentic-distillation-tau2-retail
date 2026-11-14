"""StudentAgent behaviour with a scripted generator (no GPU, no API)."""

import pytest
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from transformers import AutoTokenizer

from agentic_distill import REPO_ROOT
from agentic_distill.formatter import Formatter
from agentic_distill.student_agent import ABORT, GenResult, StudentAgent

TOK_DIR = REPO_ROOT / "data" / "student_tok"
pytestmark = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="tokenizer files not downloaded")


class Script:
    def __init__(self, *outs):
        self.outs, self.prompts = list(outs), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return GenResult(self.outs.pop(0), 10, 5)


@pytest.fixture(scope="module")
def parts():
    from tau2.runner import build_environment
    env = build_environment("retail")
    return Formatter(AutoTokenizer.from_pretrained(TOK_DIR)), env.get_tools(), env.get_policy()


def _agent(parts, *outs):
    fmt, tools, pol = parts
    gen = Script(*outs)
    return StudentAgent(tools, pol, fmt, gen), gen


def _hello():
    return UserMessage(role="user", content="hi, order #W1 please")


def test_text_reply(parts):
    a, gen = _agent(parts, "<|START_RESPONSE|>Sure, what is your email?<|END_RESPONSE|><|END_OF_TURN_TOKEN|>")
    s = a.get_init_state([AssistantMessage(role="assistant", content="Hi! How can I help you today?")])
    m, s = a.generate_next_message(_hello(), s)
    assert m.content == "Sure, what is your email?" and not m.tool_calls and not a.is_stop(m)
    assert gen.prompts[0].endswith("<|START_THINKING|><|END_THINKING|>") and m.raw_data["invalid_attempts"] == []


def test_multi_call_gets_unique_ids(parts):
    out = ('<|START_RESPONSE|>Checking.\n<tool_call>{"name":"get_order_details","arguments":{"order_id":"#W1"}}</tool_call>\n'
           '<tool_call>{"name":"get_order_details","arguments":{"order_id":"#W2"}}</tool_call><|END_RESPONSE|>')
    a, _ = _agent(parts, out)
    m, _ = a.generate_next_message(_hello(), a.get_init_state([]))
    assert [c.name for c in m.tool_calls] == ["get_order_details"] * 2
    assert len({c.id for c in m.tool_calls}) == 2 and m.content == "Checking."


def test_one_retry_then_success(parts):
    a, gen = _agent(parts, "garbage", "<|START_RESPONSE|>ok<|END_RESPONSE|>")
    m, _ = a.generate_next_message(_hello(), a.get_init_state([]))
    assert m.content == "ok" and len(m.raw_data["invalid_attempts"]) == 1 and not m.raw_data["aborted"]
    assert "[format error" in gen.prompts[1] and gen.prompts[1] != gen.prompts[0]
    assert m.raw_data["n_in"] == 20          # tokens of both attempts are counted


def test_second_failure_aborts(parts):
    a, _ = _agent(parts, "garbage", "<|START_RESPONSE|>x")
    m, _ = a.generate_next_message(_hello(), a.get_init_state([]))
    assert a.is_stop(m) and m.content == ABORT and m.raw_data["aborted"] and len(m.raw_data["invalid_attempts"]) == 2


def test_tool_result_round_trip(parts):
    call = ('<|START_RESPONSE|><tool_call>{"name":"get_order_details","arguments":{"order_id":"#W1"}}'
            '</tool_call><|END_RESPONSE|>')
    a, gen = _agent(parts, call, "<|START_RESPONSE|>Done.<|END_RESPONSE|>")
    s = a.get_init_state([])
    m, s = a.generate_next_message(_hello(), s)
    tm = ToolMessage(id=m.tool_calls[0].id, role="tool", content='{"status":"pending"}', requestor="assistant")
    m2, s = a.generate_next_message(tm, s)
    assert '<tool_result name="get_order_details">{"status":"pending"}</tool_result>' in gen.prompts[1]
    assert m2.content == "Done."


def test_hallucinated_tool_result_is_cut_not_aborted(parts):
    out = ('<|START_RESPONSE|><tool_call>{"name":"get_user_details","arguments":{"user_id":"x"}}</tool_call>\n'
           '<tool_result name="get_user_details">{"fake":1}</tool_result>\n'
           '<tool_call>{"name":"cancel_pending_order","arguments":{"order_id":"y"}}</tool_call>')
    a, _ = _agent(parts, out)
    m, _ = a.generate_next_message(_hello(), a.get_init_state([]))
    assert [c.name for c in m.tool_calls] == ["get_user_details"] and not m.raw_data["aborted"]
