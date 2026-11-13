"""C8: cost ledger and budget guard.

Per-call usage is read from the raw provider response that tau2 stores on every
LLM-generated message (`raw_data`), so the ledger is reproducible from saved
traces. DeepSeek reports `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`;
if those fields are absent we fail loudly rather than guess.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def is_peak(ts: datetime, windows_utc: list[list[int]]) -> bool:
    """Peak = weekday (Mon-Fri) and UTC hour in any [start, end) window."""
    ts = ts.astimezone(timezone.utc)
    return ts.weekday() < 5 and any(a <= ts.hour < b for a, b in windows_utc)


def usage_from_raw(raw: dict) -> dict:
    u = raw.get("usage") or {}
    hit, miss = u.get("prompt_cache_hit_tokens"), u.get("prompt_cache_miss_tokens")
    if hit is None or miss is None:
        raise KeyError(f"response lacks DeepSeek cache fields; usage keys={sorted(u)}")
    if hit + miss != u["prompt_tokens"]:
        raise ValueError(f"cache hit+miss {hit}+{miss} != prompt_tokens {u['prompt_tokens']}")
    details = u.get("completion_tokens_details") or {}
    return {"hit": hit, "miss": miss, "out": u["completion_tokens"],
            "reasoning": details.get("reasoning_tokens") or 0}


def call_cost(usage: dict, rates: dict) -> float:
    """rates are USD per 1M tokens: {cache_hit, cache_miss, output}."""
    return (usage["hit"] * rates["cache_hit"] + usage["miss"] * rates["cache_miss"]
            + usage["out"] * rates["output"]) / 1e6


def episode_records(sim, model_by_role: dict, prices: dict, context: dict) -> list[dict]:
    """One ledger record per LLM call in a tau2 SimulationRun.

    role 'assistant' -> teacher agent, role 'user' -> user simulator.
    """
    out = []
    for i, msg in enumerate(sim.messages or []):
        raw = getattr(msg, "raw_data", None)
        if not raw:
            continue  # tool results, initial greeting, etc. — no LLM call
        role = "agent" if msg.role == "assistant" else "user_sim"
        model = model_by_role[role]
        usage = usage_from_raw(raw)
        created = datetime.fromtimestamp(raw["created"], tz=timezone.utc)
        peak = is_peak(created, prices["peak_windows_utc"])
        rates = prices[model]["peak" if peak else "off_peak"]
        reasoning_text = ((raw.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content")
        out.append({**context, "msg_idx": i, "role": role, "model": model,
                    "response_model": raw.get("model"), "created_utc": created.isoformat(),
                    "peak": peak, **usage, "has_reasoning_content": bool(reasoning_text),
                    "cost_usd": call_cost(usage, rates)})
    return out


class Ledger:
    """Append-only JSONL ledger with a hard budget."""

    def __init__(self, path: Path | str, budget_usd: float):
        self.path = Path(path)
        self.budget_usd = budget_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, records: list[dict]) -> None:
        with open(self.path, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def total(self, **match) -> float:
        if not self.path.exists():
            return 0.0
        tot = 0.0
        for line in self.path.read_text().splitlines():
            r = json.loads(line)
            if all(r.get(k) == v for k, v in match.items()):
                tot += r["cost_usd"]
        return tot

    def remaining(self, **match) -> float:
        return self.budget_usd - self.total(**match)
