"""$/token pricing, in one place.

Before Week 11 this was duplicated independently in eval/week7/race.py and
eval/week10/race.py (each a local module-level constant), and nowhere in
app/ at all - there was no per-request cost to price, since no caller of
structured() kept the token counts (see app/llm/base.py). One table now, so
a price change or a new model is one edit, not a grep across eval scripts.
"""

from __future__ import annotations

# USD per 1M tokens. Source: provider pricing pages, captured at the price in
# effect when each model was adopted in this project - update on a real
# pricing change, not automatically.
_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # model -> (input, output)
    "gpt-4o-mini": (0.15, 0.60),
}

# Local models (Ollama) have no per-token $ cost - self-hosted compute instead.
_ZERO_COST_PREFIXES = ("qwen", "llama", "mistral", "phi", "gemma")


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """0.0 for an unpriced or local model - a missing price must never raise
    partway through a cost report, and a silent 0 is visible in a total in a
    way an exception during logging is not."""
    if any(model.lower().startswith(p) for p in _ZERO_COST_PREFIXES):
        return 0.0
    rates = _PRICING_PER_1M.get(model)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    return input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate
