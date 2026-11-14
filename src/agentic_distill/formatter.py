"""C4 formatter: the single source of truth for how conversations become student text.

Used identically for SFT data and for inference, so train/inference formats cannot drift.

Chat dialect (decisions D3.5 + D5 + D6):
- system turn  = Cohere's native preamble + "developer preamble" (policy, tool schemas, response format).
- assistant turn = optional short text, then zero or more <tool_call>{"name":..,"arguments":{..}}</tool_call>
  blocks, wrapped in <|START_RESPONSE|>...<|END_RESPONSE|>. Every assistant turn begins with an
  EMPTY thinking block (the /no_think form), so history rendered at inference equals the training text.
- tool results come back in one user turn as <tool_result name="..."> blocks, in call order.
- user turns carry the template's " /no_think" suffix (applied to every user turn, as the template does).

Message dicts:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str | None, "tool_calls": [{"name": str, "arguments": dict}, ...]}
  {"role": "tool", "name": str, "content": str}         (consecutive tool messages -> one user turn)
"""

import json
import re
from dataclasses import dataclass

SOT, EOT = "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>"
USER, CHATBOT = "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>"
STH, ETH = "<|START_THINKING|>", "<|END_THINKING|>"
SR, ER = "<|START_RESPONSE|>", "<|END_RESPONSE|>"
MODE_SUFFIX = " /no_think"
GEN_PROMPT = f"{SOT}{CHATBOT}{STH}{ETH}"
EOS = "<EOS_TOKEN>"

_RESPONSE_FORMAT = """# Role
You are a customer service agent. Help the user according to the policy above and always follow it.

# Response format
Each turn, either reply to the user or call tools. To call a tool, write
<tool_call>{"name": "<tool name>", "arguments": {<arguments as JSON>}}</tool_call>
You may write one short sentence before the calls. You may make several calls in one turn; put each in its own <tool_call> block. Do not write anything after the last call.
Tool results arrive in the next user turn as <tool_result name="<tool name>">...</tool_result> blocks, in the same order as your calls."""

_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class Parsed:
    text: str
    tool_calls: list[dict]


@dataclass
class ParseError:
    reason: str
    raw: str


def system_text(policy: str, tools: list[dict]) -> str:
    """tools: openai-style schemas ({"type":"function","function":{...}}) or bare function dicts."""
    lines = [json.dumps(t.get("function", t), ensure_ascii=False, separators=(",", ":")) for t in tools]
    return (f"<policy>\n{policy}\n</policy>\n\n# Tools\nFunction schemas, one per line:\n"
            + "\n".join(lines) + "\n\n" + _RESPONSE_FORMAT)


def assistant_body(text: str | None, tool_calls: list[dict] | None) -> str:
    parts = []
    if text and text.strip():
        parts.append(text.strip())
    for c in tool_calls or []:
        parts.append("<tool_call>" + json.dumps({"name": c["name"], "arguments": c["arguments"]},
                                               ensure_ascii=False) + "</tool_call>")
    return "\n".join(parts)


def _tool_results_text(results: list[dict]) -> str:
    return "\n".join(f'<tool_result name="{r["name"]}">{r["content"]}</tool_result>' for r in results)


class Formatter:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def head(self, policy: str, tools: list[dict]) -> str:
        """BOS + native system turn. Uses the real template so the Cohere preamble is never retyped."""
        return self.tok.apply_chat_template(
            [{"role": "system", "content": system_text(policy, tools)}], tokenize=False, enable_thinking=False)

    def render_turns(self, messages: list[dict]) -> list[tuple[str, str]]:
        """-> [(role, text)] with role in {'user','assistant'}; tool messages are merged into user turns."""
        out, pending = [], []
        def flush():
            if pending:
                out.append(("user", f"{SOT}{USER}{_tool_results_text(pending)}{MODE_SUFFIX}{EOT}"))
                pending.clear()
        for m in messages:
            if m["role"] == "tool":
                pending.append(m)
                continue
            flush()
            if m["role"] == "user":
                out.append(("user", f"{SOT}{USER}{m['content']}{MODE_SUFFIX}{EOT}"))
            elif m["role"] == "assistant":
                body = assistant_body(m.get("content"), m.get("tool_calls"))
                out.append(("assistant", f"{SOT}{CHATBOT}{STH}{ETH}{SR}{body}{ER}{EOT}"))
            else:
                raise ValueError(f"unsupported role {m['role']}")
        flush()
        return out

    def render(self, policy: str, tools: list[dict], messages: list[dict],
               add_generation_prompt: bool = False) -> str:
        text = self.head(policy, tools) + "".join(t for _, t in self.render_turns(messages))
        return text + GEN_PROMPT if add_generation_prompt else text


def truncate_at_tool_result(text: str) -> str:
    """D7: the model must stop after its calls. If it starts writing a <tool_result>, cut there and close
    the response block, so a hallucinated continuation is dropped instead of failing the whole turn."""
    i = text.find("<tool_result")
    if i < 0:
        return text
    return text[:i].rstrip() + ER


def parse(completion: str) -> Parsed | ParseError:
    """Parse the model's raw completion (the text generated after GEN_PROMPT)."""
    s = completion.strip()
    for tail in (EOS, EOT):
        if s.endswith(tail):
            s = s[: -len(tail)].rstrip()
    if not s.startswith(SR):
        return ParseError("missing <|START_RESPONSE|>", completion)
    if not s.endswith(ER):
        return ParseError("missing <|END_RESPONSE|>", completion)
    body = s[len(SR): -len(ER)]
    calls, pos, text = [], None, None
    for mt in _CALL_RE.finditer(body):
        if pos is None:
            text = body[: mt.start()].strip()
        elif body[pos: mt.start()].strip():
            return ParseError("text between tool calls", completion)
        pos = mt.end()
        try:
            obj = json.loads(mt.group(1))
        except json.JSONDecodeError as e:
            return ParseError(f"invalid JSON in tool_call: {e.msg}", completion)
        if not (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                and isinstance(obj.get("arguments"), dict)):
            return ParseError("tool_call needs a string name and dict arguments", completion)
        calls.append({"name": obj["name"], "arguments": obj["arguments"]})
    if "<tool_call>" in _CALL_RE.sub("", body):
        return ParseError("unclosed <tool_call>", completion)
    if pos is None:
        return Parsed(body.strip(), [])
    if body[pos:].strip():
        return ParseError("text after tool call", completion)
    return Parsed(text, calls)


def from_tau2(messages) -> list[dict]:
    """Convert tau2 messages (pydantic or dicts) to formatter dicts; drops the user simulator's final stop token."""
    from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER
    get = lambda m, k: m[k] if isinstance(m, dict) else getattr(m, k)
    names, out = {}, []
    for m in messages:
        role = get(m, "role")
        role = getattr(role, "value", role)
        if role == "assistant":
            calls = get(m, "tool_calls") or []
            calls = [c if isinstance(c, dict) else c.model_dump() for c in calls]
            for c in calls:
                names[c["id"]] = c["name"]
            out.append({"role": "assistant", "content": get(m, "content"),
                        "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls]})
        elif role == "user":
            out.append({"role": "user", "content": get(m, "content")})
        elif role == "tool":
            out.append({"role": "tool", "name": names[get(m, "id")], "content": get(m, "content")})
    while out and out[-1]["role"] == "user" and (out[-1]["content"] or "").strip() in {STOP, TRANSFER, OUT_OF_SCOPE}:
        out.pop()
    return out
