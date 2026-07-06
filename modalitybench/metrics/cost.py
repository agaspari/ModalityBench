"""Cost accounting from token usage.

Prices are USD per 1M tokens. Cache reads are ~0.1x input; cache writes (5-minute TTL)
~1.25x input. Update the table as pricing changes; unknown models fall back to Opus rates
with a flag in the returned breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass

from modalitybench.agents.model_client import Usage


@dataclass(frozen=True)
class Price:
    input: float  # $ / 1M input tokens
    output: float  # $ / 1M output tokens


# Cached 2026-07; refresh from platform.claude.com/pricing (Anthropic) and each provider's
# pricing page. The cache multipliers below are Anthropic-calibrated; OpenAI-compatible
# providers price cache reads even cheaper (e.g. DeepSeek ~0.02x), so cache-read cost is a
# slight over-estimate for them — negligible next to the input/output terms.
MODEL_PRICES: dict[str, Price] = {
    "claude-fable-5": Price(10.00, 50.00),
    "claude-opus-4-8": Price(5.00, 25.00),
    "claude-opus-4-7": Price(5.00, 25.00),
    "claude-opus-4-6": Price(5.00, 25.00),
    "claude-sonnet-5": Price(3.00, 15.00),
    "claude-sonnet-4-6": Price(3.00, 15.00),
    "claude-haiku-4-5": Price(1.00, 5.00),
    # DeepSeek (api-docs.deepseek.com, cache-miss input) — deepseek-chat == v4-flash non-think.
    # deepseek-reasoner intentionally omitted (thinking-mode price unverified) → fallback-flagged.
    "deepseek-chat": Price(0.14, 0.28),
    # Zhipu GLM (docs.z.ai/guides/overview/pricing).
    "glm-4.6": Price(0.60, 2.20),
    "glm-4.5": Price(0.60, 2.20),
    "glm-4.5-air": Price(0.20, 1.10),
    "mock": Price(0.0, 0.0),
}

_CACHE_READ_MULT = 0.10
_CACHE_WRITE_MULT = 1.25
_FALLBACK = MODEL_PRICES["claude-opus-4-8"]


def cost_for_usage(model: str, usage: Usage) -> dict[str, float]:
    """Return a cost breakdown (USD) for a usage record on ``model``."""
    price = MODEL_PRICES.get(model, _FALLBACK)
    in_cost = usage.input_tokens / 1e6 * price.input
    cache_read_cost = usage.cache_read_input_tokens / 1e6 * price.input * _CACHE_READ_MULT
    cache_write_cost = (
        usage.cache_creation_input_tokens / 1e6 * price.input * _CACHE_WRITE_MULT
    )
    out_cost = usage.output_tokens / 1e6 * price.output
    total = in_cost + cache_read_cost + cache_write_cost + out_cost
    return {
        "input_cost": round(in_cost, 6),
        "cache_read_cost": round(cache_read_cost, 6),
        "cache_write_cost": round(cache_write_cost, 6),
        "output_cost": round(out_cost, 6),
        "total_cost": round(total, 6),
        "priced_as_fallback": model not in MODEL_PRICES,
    }
