"""C6: the student as a tau2 HalfDuplexAgent (identical for B0 and the fine-tuned model).

render (formatter) -> generate -> parse. D3.6: an unparseable output gets ONE error message fed back
(internally, not shown in the tau2 transcript); a second failure ends the episode as an abort.
Per-turn stats (tokens, wall time, invalid attempts) are stored in AssistantMessage.raw_data so eval
records can be rebuilt from saved traces.
"""

import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, ToolCall
from tau2.registry import registry

from agentic_distill.formatter import Formatter, ParseError, from_tau2, parse

ABORT = "###ABORT###"
_SPECIAL = re.compile(r"<\|[A-Z_]+\|>|<[A-Z_]+_TOKEN>")


@dataclass
class GenResult:
    text: str
    n_in: int
    n_out: int


GenerateFn = Callable[[str], GenResult]


class StudentAgent(HalfDuplexAgent[LLMAgentState]):
    def __init__(self, tools, domain_policy: str, formatter: Formatter, generate_fn: GenerateFn):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.fmt, self.gen = formatter, generate_fn
        self.tool_schemas = [t.openai_schema for t in tools]
        self._n_calls = 0

    def is_stop(self, message: AssistantMessage) -> bool:
        return message.content == ABORT and not message.tool_calls

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> LLMAgentState:
        return LLMAgentState(system_messages=[], messages=list(message_history or []))

    def _attempt(self, chat: list[dict], stats: dict):
        prompt = self.fmt.render(self.domain_policy, self.tool_schemas, chat, add_generation_prompt=True)
        t0 = time.perf_counter()
        res = self.gen(prompt)
        stats["wall_s"] += time.perf_counter() - t0
        stats["n_in"] += res.n_in
        stats["n_out"] += res.n_out
        out = parse(res.text)
        if isinstance(out, ParseError):
            stats["invalid_attempts"].append({"reason": out.reason, "raw": res.text[:2000]})
            return out
        if not out.text and not out.tool_calls:
            err = ParseError("empty response", res.text)
            stats["invalid_attempts"].append({"reason": err.reason, "raw": res.text[:2000]})
            return err
        return out

    def generate_next_message(self, message, state: LLMAgentState):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        chat = from_tau2(state.messages)
        stats = {"n_in": 0, "n_out": 0, "wall_s": 0.0, "invalid_attempts": [], "aborted": False}

        out = self._attempt(chat, stats)
        if isinstance(out, ParseError):
            bad = _SPECIAL.sub("", out.raw).strip()[:1500]
            feedback = [{"role": "assistant", "content": bad or "(empty)"},
                        {"role": "user", "content": f"[format error: {out.reason}] Reply again using the required format."}]
            out = self._attempt(chat + feedback, stats)

        if isinstance(out, ParseError):
            stats["aborted"] = True
            msg = AssistantMessage(role="assistant", content=ABORT, raw_data=stats)
        else:
            calls = []
            for c in out.tool_calls:
                self._n_calls += 1
                calls.append(ToolCall(id=f"call_{self._n_calls}", name=c["name"], arguments=c["arguments"]))
            msg = AssistantMessage(role="assistant", content=out.text or None,
                                   tool_calls=calls or None, raw_data=stats)
        state.messages.append(msg)
        return msg, state


def register(formatter: Formatter, generate_fn: GenerateFn, name: str = "student_llm_agent") -> str:
    def factory(tools, domain_policy, **kwargs):
        return StudentAgent(tools, domain_policy, formatter, generate_fn)
    if name in registry.get_agents():
        raise RuntimeError(f"agent {name} already registered")
    registry.register_agent_factory(factory, name)
    return name
