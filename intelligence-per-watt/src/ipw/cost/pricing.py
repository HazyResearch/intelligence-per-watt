"""Pricing tables and cost calculation for LLM API providers.

All prices are per 1M tokens in USD unless otherwise noted.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# OpenAI pricing (per 1M tokens)
# Source: https://openai.com/api/pricing/
# ---------------------------------------------------------------------------
OPENAI_PRICING: dict[str, dict[str, float]] = {
    # GPT-4 series
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # o1 reasoning models
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    # GPT-5 series
    "gpt-5.2-2025-12-11": {"input": 30.00, "output": 120.00},
    "gpt-5-mini-2025-08-07": {"input": 5.00, "output": 20.00},
    "gpt-5-nano-2025-08-07": {"input": 1.00, "output": 4.00},
}

# ---------------------------------------------------------------------------
# Anthropic pricing (per 1M tokens)
# Source: https://www.anthropic.com/pricing
# ---------------------------------------------------------------------------
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    # Claude 4.5 models
    "claude-opus-4-5-20251101": {"input": 20.00, "output": 100.00},
    "claude-sonnet-4-5-20250929": {"input": 4.00, "output": 20.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    # Claude 4 models
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    # Claude 3.5 models
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # Claude 3 models
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

# ---------------------------------------------------------------------------
# Google Gemini pricing (per 1M tokens)
# Source: https://ai.google.dev/pricing
# ---------------------------------------------------------------------------
GEMINI_PRICING: dict[str, dict[str, float]] = {
    # Gemini 3.0 Flash (preview)
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    # Gemini 2.0 models
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    # Gemini 1.5 models
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
}

# ---------------------------------------------------------------------------
# Tool pricing (flat per-call)
# ---------------------------------------------------------------------------
TAVILY_COST_PER_SEARCH: float = 0.01

# ---------------------------------------------------------------------------
# Unified lookup
# ---------------------------------------------------------------------------
PROVIDER_PRICING: dict[str, dict[str, dict[str, float]]] = {
    "openai": OPENAI_PRICING,
    "anthropic": ANTHROPIC_PRICING,
    "gemini": GEMINI_PRICING,
    "google": GEMINI_PRICING,
}


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate API cost in USD for a given provider, model, and token usage.

    Args:
        provider: Provider name (openai, anthropic, gemini, google).
        model: Model identifier as used in the provider's API.
        input_tokens: Number of input / prompt tokens.
        output_tokens: Number of output / completion tokens.

    Returns:
        Estimated cost in USD.  Returns 0.0 if the provider or model is
        not found in the pricing tables.
    """
    pricing_table = PROVIDER_PRICING.get(provider.lower())
    if pricing_table is None:
        return 0.0

    model_pricing = pricing_table.get(model)
    if model_pricing is None:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    return input_cost + output_cost


__all__ = [
    "ANTHROPIC_PRICING",
    "GEMINI_PRICING",
    "OPENAI_PRICING",
    "PROVIDER_PRICING",
    "TAVILY_COST_PER_SEARCH",
    "calculate_cost",
]
