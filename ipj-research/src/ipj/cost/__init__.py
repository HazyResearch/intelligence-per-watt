"""IPJ-side cost helpers + pricing overlay.

Local dollars = measured energy at an electricity rate; cloud dollars = API token price.

DECIDED 2026-08-12 (ADR-0004, `ipj/planning/decisions/0004-cost-basis-and-anchors.md`): local
dollars are the **electricity to run the job**, priced from the *measured* Joules -- not amortized
GPU-hours and not a flat per-task constant. This makes the dollar axis a direct function of the
Joules axis, which is the axis the paper argues about.

Read the caveat before quoting any local dollar figure: this basis **excludes hardware capital,
cooling/PUE, staffing and margin**, so a local $/task under it is a *marginal energy* cost and a
genuine lower bound. Cloud $/task, by contrast, is a list price that has all of those baked in.
Any local-vs-cloud multiplier built from the two is therefore comparing a lower bound against a
retail price, and must say so wherever it appears. ADR-0004 requires the amortized-GPU-hour
sensitivity row (`dollars_local`) to be published alongside the headline, not as a follow-up.

Historical note: this used to be a single module `ipj/src/ipj/cost.py`. When the pricing
overlay landed (PR #132), a same-named package was created next to it, and Python's
import machinery preferred the package — silently shadowing the module and breaking
`from ipj.cost import H100_HOURLY_USD` at collection time. The historical helpers
now live here so both `import ipj.cost` and `from ipj.cost import H100_HOURLY_USD` work
identically to the pre-#132 world.
"""

from __future__ import annotations

from . import pricing_overlay as pricing_overlay
from .accounting import (
    ApiChargeKind,
    CostAccountingView,
    CostComponents,
    IpdResult,
    MoneyValue,
    TaskCostObservation,
    dollars_tokens_decimal,
    workload_ipd,
)

# --- historical helpers (were ipj/src/ipj/cost.py before PR #132) --------------

# Illustrative June-2026 prices, ($/M input, $/M output). VERIFY before publishing.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-fable-5": (10.0, 50.0),
}

# Illustrative datacenter GPU rate ($/GPU-hour). VERIFY for the actual cluster.
# NOTE: no longer the local cost basis -- kept for the amortized-hours sensitivity row only.
H100_HOURLY_USD: float = 2.49

# The decided local cost basis: US average retail price of electricity to the INDUSTRIAL sector.
# Source: U.S. EIA, Electric Power Monthly, Table 5.3 -- 8.71 cents/kWh, May 2026.
#   https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_3
# Industrial (not commercial or residential) is the right series: it is the tariff class a
# datacenter buys under. The series is revised monthly, so pin the month you cite.
# A large operator on a negotiated contract pays LESS than this, which keeps the figure
# conservative in the direction that matters -- it does not flatter the local tier.
US_INDUSTRIAL_USD_PER_KWH: float = 0.0871
EIA_RATE_PERIOD: str = "2026-05"

JOULES_PER_KWH: float = 3.6e6


def dollars_energy(joules: float, usd_per_kwh: float = US_INDUSTRIAL_USD_PER_KWH) -> float:
    """USD of electricity for a measured energy figure.

    `joules` must be a MEASURED integral over the run's own devices. Feeding this a token-derived
    estimate silently launders an estimate into the dollar axis, which ADR-0003 forbids -- cloud
    tiers have no measured Joules and must be priced with `dollars_tokens` instead.
    """
    return (joules / JOULES_PER_KWH) * usd_per_kwh


def dollars_tokens(tokens_in: int, tokens_out: int, price_in_per_m: float, price_out_per_m: float) -> float:
    """USD for a completion given per-million-token prices."""
    return (tokens_in / 1_000_000.0) * price_in_per_m + (tokens_out / 1_000_000.0) * price_out_per_m


def dollars_local(wall_clock_s: float, gpu_hourly_rate: float, n_gpus: int = 1) -> float:
    """Amortized USD for a local run: wall-clock GPU-hours * rate * #GPUs."""
    return (wall_clock_s / 3600.0) * gpu_hourly_rate * n_gpus


__all__ = [
    "ApiChargeKind",
    "CostAccountingView",
    "CostComponents",
    "EIA_RATE_PERIOD",
    "H100_HOURLY_USD",
    "IpdResult",
    "JOULES_PER_KWH",
    "MoneyValue",
    "PRICES",
    "TaskCostObservation",
    "US_INDUSTRIAL_USD_PER_KWH",
    "dollars_energy",
    "dollars_local",
    "dollars_tokens",
    "dollars_tokens_decimal",
    "pricing_overlay",
    "workload_ipd",
]
