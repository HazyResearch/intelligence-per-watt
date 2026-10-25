"""Tests for cost/pricing.py — cost calculation and pricing tables."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ipw.cost.aggregation import sum_costs_if_complete
from ipw.cost.pricing import (
    ANTHROPIC_PRICING,
    GEMINI_PRICING,
    OPENAI_PRICING,
    PROVIDER_PRICING,
    TAVILY_COST_PER_SEARCH,
    CostQuote,
    PriceCatalog,
    PriceStatus,
    PricingUnavailableWarning,
    TokenPrice,
    calculate_cost,
    calculate_cost_or_none,
    calculate_cost_quote,
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

    def test_unknown_provider_compatibility_return_is_not_silent(self) -> None:
        with pytest.warns(PricingUnavailableWarning, match="unknown_provider"):
            cost = calculate_cost("unknown_provider", "some-model", 1000, 1000)
        assert cost == 0.0

    def test_unknown_model_compatibility_return_is_not_silent(self) -> None:
        with pytest.warns(PricingUnavailableWarning, match="unknown_model"):
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

    def test_runtime_legacy_table_addition_remains_compatible(self, monkeypatch) -> None:
        monkeypatch.setitem(
            OPENAI_PRICING,
            "runtime-model",
            {"input": 1.0, "output": 2.0},
        )

        with pytest.warns(PricingUnavailableWarning, match="unversioned runtime"):
            cost = calculate_cost("openai", "runtime-model", 1_000_000, 1_000_000)

        assert cost == pytest.approx(3.0)

    def test_runtime_legacy_table_mutation_remains_compatible(self, monkeypatch) -> None:
        monkeypatch.setitem(
            OPENAI_PRICING,
            "gpt-4o",
            {"input": 1.0, "output": 2.0},
        )

        assert calculate_cost("openai", "gpt-4o", 1_000_000, 1_000_000) == pytest.approx(3.0)


class TestExactCostQuotes:
    def test_known_quote_has_exact_components(self) -> None:
        catalog = PriceCatalog(
            "pinned-fixture-v1",
            [
                TokenPrice(
                    provider="openai",
                    model="gpt-4o",
                    input_per_million_usd=Decimal("2.50"),
                    output_per_million_usd=Decimal("10.00"),
                    source_url="https://openai.com/api/pricing/",
                    source_retrieved_at=date(2026, 8, 11),
                    effective_from=date(2026, 8, 11),
                    effective_until=date(2026, 8, 12),
                )
            ],
        )
        quote = calculate_cost_quote(
            "openai",
            "gpt-4o",
            1_000,
            500,
            as_of=date(2026, 8, 11),
            catalog=catalog,
        )

        assert isinstance(quote, CostQuote)
        assert quote.status is PriceStatus.AVAILABLE
        assert quote.input_cost_usd == Decimal("0.0025")
        assert quote.output_cost_usd == Decimal("0.005")
        assert quote.total_cost_usd == Decimal("0.0075")
        assert quote.catalog_id == "pinned-fixture-v1"

    def test_undated_builtin_price_is_not_strictly_available(self) -> None:
        quote = calculate_cost_quote(
            "openai",
            "gpt-4o",
            1_000,
            500,
            as_of=date(2100, 1, 1),
        )

        assert quote.status is PriceStatus.UNVERIFIED
        assert quote.total_cost_usd is None
        assert calculate_cost_or_none(
            "openai",
            "gpt-4o",
            1_000,
            500,
            as_of=date(2100, 1, 1),
        ) is None

    def test_optional_api_keeps_unknown_cost_unknown(self) -> None:
        assert calculate_cost_or_none("openai", "nonexistent-model", 1000, 1000) is None

    def test_unknown_provider_and_model_are_distinct(self) -> None:
        provider_quote = calculate_cost_quote("missing", "model", 1, 1)
        model_quote = calculate_cost_quote("openai", "missing", 1, 1)

        assert provider_quote.status is PriceStatus.UNKNOWN_PROVIDER
        assert provider_quote.total_cost_usd is None
        assert model_quote.status is PriceStatus.UNKNOWN_MODEL
        assert model_quote.total_cost_usd is None

    def test_expired_price_fails_closed_as_stale(self) -> None:
        catalog = PriceCatalog(
            "stale-fixture-v1",
            [
                TokenPrice(
                    provider="fixture",
                    model="dated-model",
                    input_per_million_usd=Decimal("1.25"),
                    output_per_million_usd=Decimal("10"),
                    effective_from=date(2026, 1, 1),
                    effective_until=date(2026, 7, 1),
                )
            ],
        )

        quote = calculate_cost_quote(
            "fixture",
            "dated-model",
            1000,
            1000,
            as_of=date(2026, 8, 11),
            catalog=catalog,
        )

        assert quote.status is PriceStatus.STALE
        assert quote.total_cost_usd is None

    def test_expired_price_is_checked_when_as_of_is_omitted(self) -> None:
        catalog = PriceCatalog(
            "stale-fixture-v1",
            [
                TokenPrice(
                    provider="fixture",
                    model="dated-model",
                    input_per_million_usd=Decimal(1),
                    output_per_million_usd=Decimal(1),
                    effective_until=date(2020, 1, 1),
                )
            ],
        )

        quote = calculate_cost_quote("fixture", "dated-model", 1, 1, catalog=catalog)

        assert quote.status is PriceStatus.STALE
        assert quote.total_cost_usd is None

    def test_future_price_is_not_used_early(self) -> None:
        catalog = PriceCatalog(
            "future-fixture-v1",
            [
                TokenPrice(
                    provider="fixture",
                    model="future-model",
                    input_per_million_usd=Decimal(1),
                    output_per_million_usd=Decimal(1),
                    effective_from=date(2027, 1, 1),
                )
            ],
        )

        quote = calculate_cost_quote(
            "fixture", "future-model", 1, 1, as_of=date(2026, 8, 11), catalog=catalog
        )

        assert quote.status is PriceStatus.NOT_YET_EFFECTIVE
        assert quote.total_cost_usd is None

    @pytest.mark.parametrize("tokens", [-1, True, 1.5])
    def test_invalid_token_counts_are_rejected(self, tokens: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            calculate_cost_quote("openai", "gpt-4o", tokens, 0)  # type: ignore[arg-type]


class TestCompleteCostAggregation:
    def test_known_values_sum_and_preserve_zero(self) -> None:
        assert sum_costs_if_complete([0.0, 0.0]) == 0.0
        assert sum_costs_if_complete([0.0, 0.25]) == pytest.approx(0.25)

    def test_all_unknown_and_mixed_values_fail_closed(self) -> None:
        assert sum_costs_if_complete([None, None]) is None
        assert sum_costs_if_complete([0.25, None]) is None

    @pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan"), True, "0.25"])
    def test_invalid_values_fail_closed(self, invalid: object) -> None:
        assert sum_costs_if_complete([0.25, invalid]) is None

    def test_empty_semantics_are_explicit(self) -> None:
        assert sum_costs_if_complete([]) is None
        assert sum_costs_if_complete([], empty=0.0) == 0.0


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
