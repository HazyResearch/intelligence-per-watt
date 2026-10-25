"""Pricing tables and cost calculation for LLM API providers.

All prices are per 1M tokens in USD unless otherwise noted.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

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
    "gpt-5.4": {"input": 30.00, "output": 120.00},
    "gpt-5-mini": {"input": 5.00, "output": 20.00},
    "gpt-5-mini-2025-08-07": {"input": 5.00, "output": 20.00},
    "gpt-5-nano-2025-08-07": {"input": 1.00, "output": 4.00},
}

# ---------------------------------------------------------------------------
# Anthropic pricing (per 1M tokens)
# Source: https://www.anthropic.com/pricing
# ---------------------------------------------------------------------------
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    # Claude 4.6 models
    "claude-opus-4-6": {"input": 20.00, "output": 100.00},
    "claude-sonnet-4-6": {"input": 4.00, "output": 20.00},
    # Claude 4.5 models
    "claude-opus-4-5-20251101": {"input": 20.00, "output": 100.00},
    "claude-sonnet-4-5-20250929": {"input": 4.00, "output": 20.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
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
    # Gemini 3.1 models
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 5.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.075, "output": 0.30},
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


class PriceStatus(str, Enum):
    """Availability of a price for a particular date."""

    AVAILABLE = "available"
    UNKNOWN_PROVIDER = "unknown_provider"
    UNKNOWN_MODEL = "unknown_model"
    UNVERIFIED = "unverified"
    NOT_YET_EFFECTIVE = "not_yet_effective"
    STALE = "stale"


class PricingUnavailableWarning(RuntimeWarning):
    """A compatibility call could not resolve a model price."""


def _decimal_price(value: Decimal | int | str | float) -> Decimal:
    """Convert legacy table values without inheriting binary-float error."""
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("token prices must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class TokenPrice:
    """An immutable token-price record suitable for a versioned catalog.

    ``effective_until`` is an exclusive bound. Strict quotes require source
    provenance, a retrieval date, and explicit effective bounds. This keeps an
    undated compatibility table from becoming a perpetually current price.
    """

    provider: str
    model: str
    input_per_million_usd: Decimal
    output_per_million_usd: Decimal
    source_url: str | None = None
    source_retrieved_at: date | None = None
    verified: bool = True
    effective_from: date | None = None
    effective_until: date | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        model = self.model.strip()
        if not provider or not model:
            raise ValueError("provider and model must be non-empty")
        input_price = _decimal_price(self.input_per_million_usd)
        output_price = _decimal_price(self.output_per_million_usd)
        if self.effective_from and self.effective_until and self.effective_from >= self.effective_until:
            raise ValueError("effective_from must be before effective_until")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "input_per_million_usd", input_price)
        object.__setattr__(self, "output_per_million_usd", output_price)

    def status_at(self, as_of: date | None) -> PriceStatus:
        if as_of is not None:
            if self.effective_from is not None and as_of < self.effective_from:
                return PriceStatus.NOT_YET_EFFECTIVE
            if self.effective_until is not None and as_of >= self.effective_until:
                return PriceStatus.STALE
        has_strict_provenance = (
            self.verified
            and bool(self.source_url and self.source_url.strip())
            and self.source_retrieved_at is not None
            and self.effective_from is not None
            and self.effective_until is not None
        )
        return (
            PriceStatus.AVAILABLE
            if has_strict_provenance
            else PriceStatus.UNVERIFIED
        )


class PriceCatalog:
    """Read-only, snapshot-identified collection of provider token prices."""

    __slots__ = ("catalog_id", "_records")

    def __init__(self, catalog_id: str, records: Iterable[TokenPrice]) -> None:
        if not catalog_id.strip():
            raise ValueError("catalog_id must be non-empty")
        by_provider: dict[str, dict[str, TokenPrice]] = {}
        for record in records:
            models = by_provider.setdefault(record.provider, {})
            if record.model in models:
                raise ValueError(f"duplicate price record: {record.provider}/{record.model}")
            models[record.model] = record
        self.catalog_id = catalog_id.strip()
        self._records: Mapping[str, Mapping[str, TokenPrice]] = MappingProxyType(
            {provider: MappingProxyType(models) for provider, models in by_provider.items()}
        )

    def has_provider(self, provider: str) -> bool:
        return provider.strip().lower() in self._records

    def get(self, provider: str, model: str) -> TokenPrice | None:
        models = self._records.get(provider.strip().lower())
        return None if models is None else models.get(model.strip())


@dataclass(frozen=True, slots=True)
class CostQuote:
    """Exact cost components, or an explicit reason that cost is unavailable."""

    status: PriceStatus
    catalog_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    as_of: date
    input_cost_usd: Decimal | None = None
    output_cost_usd: Decimal | None = None
    total_cost_usd: Decimal | None = None
    price: TokenPrice | None = None

    @property
    def available(self) -> bool:
        return self.status is PriceStatus.AVAILABLE


_PROVIDER_SOURCE_URLS: Mapping[str, str] = MappingProxyType(
    {
        "openai": "https://openai.com/api/pricing/",
        "anthropic": "https://www.anthropic.com/pricing",
        "gemini": "https://ai.google.dev/pricing",
        "google": "https://ai.google.dev/pricing",
    }
)

def _catalog_from_legacy_tables() -> PriceCatalog:
    records = (
        TokenPrice(
            provider=provider,
            model=model,
            input_per_million_usd=_decimal_price(prices["input"]),
            output_per_million_usd=_decimal_price(prices["output"]),
            source_url=_PROVIDER_SOURCE_URLS.get(provider),
        )
        for provider, models in PROVIDER_PRICING.items()
        for model, prices in models.items()
    )
    # This ID versions the in-repository snapshot. It deliberately does not
    # claim a retrieval date for legacy entries that did not record one.
    return PriceCatalog("ipw-builtin-v1", records)


DEFAULT_PRICE_CATALOG = _catalog_from_legacy_tables()


def _validate_tokens(input_tokens: int, output_tokens: int) -> None:
    for name, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def calculate_cost_quote(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    as_of: date | None = None,
    catalog: PriceCatalog = DEFAULT_PRICE_CATALOG,
) -> CostQuote:
    """Return an exact token-cost quote without using zero as missing data.

    ``as_of`` defaults to the current UTC calendar date for freshness
    enforcement. Reproducible experiments should pass and persist it.
    """
    _validate_tokens(input_tokens, output_tokens)
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()
    quote_date = as_of or datetime.now(timezone.utc).date()
    if not catalog.has_provider(normalized_provider):
        return CostQuote(
            status=PriceStatus.UNKNOWN_PROVIDER,
            catalog_id=catalog.catalog_id,
            provider=normalized_provider,
            model=normalized_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            as_of=quote_date,
        )

    price = catalog.get(normalized_provider, normalized_model)
    if price is None:
        return CostQuote(
            status=PriceStatus.UNKNOWN_MODEL,
            catalog_id=catalog.catalog_id,
            provider=normalized_provider,
            model=normalized_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            as_of=quote_date,
        )

    status = price.status_at(quote_date)
    if status is not PriceStatus.AVAILABLE:
        return CostQuote(
            status=status,
            catalog_id=catalog.catalog_id,
            provider=normalized_provider,
            model=normalized_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            as_of=quote_date,
            price=price,
        )

    million = Decimal(1_000_000)
    input_cost = Decimal(input_tokens) * price.input_per_million_usd / million
    output_cost = Decimal(output_tokens) * price.output_per_million_usd / million
    return CostQuote(
        status=status,
        catalog_id=catalog.catalog_id,
        provider=normalized_provider,
        model=normalized_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        as_of=quote_date,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
        price=price,
    )


def calculate_cost_or_none(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    as_of: date | None = None,
    catalog: PriceCatalog = DEFAULT_PRICE_CATALOG,
) -> float | None:
    """Return a float for legacy trace fields, or ``None`` when unavailable."""
    quote = calculate_cost_quote(
        provider,
        model,
        input_tokens,
        output_tokens,
        as_of=as_of,
        catalog=catalog,
    )
    return None if quote.total_cost_usd is None else float(quote.total_cost_usd)


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
        Estimated cost in USD. For compatibility with IPW 0.1, an unknown
        lookup returns ``0.0`` and an unverified/out-of-window table entry
        returns its legacy estimate; both emit ``PricingUnavailableWarning``.
        New code must use ``calculate_cost_quote`` or ``calculate_cost_or_none``
        so unknown, unverified, and stale prices stay unknown.
    """
    _validate_tokens(input_tokens, output_tokens)
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()
    live_provider_table = PROVIDER_PRICING.get(normalized_provider)
    live_price = (
        live_provider_table.get(normalized_model)
        if live_provider_table is not None
        else None
    )
    quote = calculate_cost_quote(provider, model, input_tokens, output_tokens)

    if live_price is not None:
        million = Decimal(1_000_000)
        legacy_cost = (
            Decimal(input_tokens) * _decimal_price(live_price["input"])
            + Decimal(output_tokens) * _decimal_price(live_price["output"])
        ) / million
        if quote.status is not PriceStatus.AVAILABLE:
            warning_reason = (
                "unversioned runtime table entry"
                if quote.price is None
                else quote.status.value
            )
            warnings.warn(
                f"price is {warning_reason} for {provider}/{model}; returning the mutable legacy "
                "table estimate only for calculate_cost() compatibility",
                PricingUnavailableWarning,
                stacklevel=2,
            )
        return float(legacy_cost)

    if quote.price is not None:
        million = Decimal(1_000_000)
        legacy_cost = (
            Decimal(input_tokens) * quote.price.input_per_million_usd
            + Decimal(output_tokens) * quote.price.output_per_million_usd
        ) / million
        warnings.warn(
            f"price is {quote.status.value} for {provider}/{model}; returning the legacy table "
            "estimate only for calculate_cost() compatibility",
            PricingUnavailableWarning,
            stacklevel=2,
        )
        return float(legacy_cost)
    warnings.warn(
        f"price unavailable ({quote.status.value}) for {provider}/{model}; "
        "returning 0.0 only for calculate_cost() compatibility",
        PricingUnavailableWarning,
        stacklevel=2,
    )
    return 0.0


__all__ = [
    "ANTHROPIC_PRICING",
    "CostQuote",
    "DEFAULT_PRICE_CATALOG",
    "GEMINI_PRICING",
    "OPENAI_PRICING",
    "PriceCatalog",
    "PriceStatus",
    "PricingUnavailableWarning",
    "PROVIDER_PRICING",
    "TAVILY_COST_PER_SEARCH",
    "TokenPrice",
    "calculate_cost",
    "calculate_cost_or_none",
    "calculate_cost_quote",
]
