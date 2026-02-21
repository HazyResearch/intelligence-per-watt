"""Tests for cost/pricing.py — cost calculation and pricing tables."""

from __future__ import annotations

import pytest

from ipw.cost.pricing import (
    ANTHROPIC_PRICING,
    GEMINI_PRICING,
    OPENAI_PRICING,
    PROVIDER_PRICING,
    TAVILY_COST_PER_SEARCH,
    calculate_cost,
)
from ipw.execution.types import CostMetrics


class TestCalculateCost:
    """Test calculate_cost() for various providers."""

    def test_openai_gpt4o_cost(self) -> None:
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = calculate_cost("openai", "gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(12.50)

    def test_openai_gpt4o_small_usage(self) -> None:
        # 1000 input + 500 output tokens
        cost = calculate_cost("openai", "gpt-4o", 1000, 500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected)

    def test_anthropic_claude_sonnet(self) -> None:
        cost = calculate_cost(
            "anthropic", "claude-sonnet-4-20250514", 1_000_000, 1_000_000
        )
        assert cost == pytest.approx(18.00)

    def test_gemini_flash(self) -> None:
        cost = calculate_cost("gemini", "gemini-2.0-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.50)

    def test_google_alias_uses_gemini_pricing(self) -> None:
        cost_gemini = calculate_cost("gemini", "gemini-1.5-pro", 1000, 1000)
        cost_google = calculate_cost("google", "gemini-1.5-pro", 1000, 1000)
        assert cost_gemini == cost_google

    def test_unknown_provider_returns_zero(self) -> None:
        cost = calculate_cost("unknown_provider", "some-model", 1000, 1000)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self) -> None:
        cost = calculate_cost("openai", "nonexistent-model", 1000, 1000)
        assert cost == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        cost = calculate_cost("openai", "gpt-4o", 0, 0)
        assert cost == 0.0

    def test_case_insensitive_provider(self) -> None:
        cost_lower = calculate_cost("openai", "gpt-4o", 1000, 1000)
        cost_upper = calculate_cost("OpenAI", "gpt-4o", 1000, 1000)
        assert cost_lower == cost_upper

    def test_tavily_cost_per_search(self) -> None:
        assert TAVILY_COST_PER_SEARCH == 0.01


class TestPricingTables:
    """Test that pricing tables contain expected models."""

    def test_openai_contains_gpt4o(self) -> None:
        assert "gpt-4o" in OPENAI_PRICING
        assert "input" in OPENAI_PRICING["gpt-4o"]
        assert "output" in OPENAI_PRICING["gpt-4o"]

    def test_anthropic_contains_claude_models(self) -> None:
        assert "claude-sonnet-4-20250514" in ANTHROPIC_PRICING
        assert "claude-3-5-sonnet-20241022" in ANTHROPIC_PRICING

    def test_gemini_contains_flash(self) -> None:
        assert "gemini-2.0-flash" in GEMINI_PRICING

    def test_provider_pricing_maps_providers(self) -> None:
        assert "openai" in PROVIDER_PRICING
        assert "anthropic" in PROVIDER_PRICING
        assert "gemini" in PROVIDER_PRICING
        assert "google" in PROVIDER_PRICING


class TestCostMetrics:
    """Test CostMetrics dataclass from execution/types.py."""

    def test_default_values(self) -> None:
        cm = CostMetrics()
        assert cm.input_cost_usd is None
        assert cm.output_cost_usd is None
        assert cm.tool_cost_usd is None
        assert cm.total_cost_usd is None

    def test_set_values(self) -> None:
        cm = CostMetrics(
            input_cost_usd=0.01,
            output_cost_usd=0.02,
            tool_cost_usd=0.005,
            total_cost_usd=0.035,
        )
        assert cm.total_cost_usd == 0.035
