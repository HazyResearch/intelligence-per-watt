"""Per-task cost helpers for the efficiency framework.

DECIDED cost methodology (2026-07-08, see plans/2026-07-08-frontier-push-execution-plan.md):

* **Open-source / local:** ``$ = latency · n_GPUs · H100_HOURLY_USD`` — a legacy amortized
  GPU-hours view using the repository's one shared reference rate. Per-task cost amortizes the
  run window over the tasks run in it (``local_dollars_per_task``).
* **Closed-source / cloud:** ``$ = tok_in·price_in + tok_out·price_out`` at each provider's public
  reported per-million-token list price, from ``PRICING`` below.

Both the ≥20× headline and its sensitivity band (``multiplier`` / ``sensitivity_band``) are built
from these two rules.
"""

from __future__ import annotations

from ..cost import H100_HOURLY_USD, PRICES, dollars_tokens_decimal

# ---------------------------------------------------------------------------
# Cloud pricing table — lowercased model id → (price_in_per_M_usd, price_out_per_M_usd).
# ONLY verified public list prices belong here. A model absent from this table makes ``cost_for``
# return None; the caller then falls back to the captured Harbor ``agent_result.cost_usd`` (which
# the provider billed at list price). Do NOT add unverified prices — cite the source per entry.
# ---------------------------------------------------------------------------
# Compatibility alias over the single IPJ table. Entries in that table are
# explicitly illustrative and must not back headline IPD; strict accounting
# uses ``TaskCostObservation`` plus a pinned snapshot or a provider bill.
PRICING: dict[str, tuple[float, float]] = PRICES


def local_dollars(latency_s: float, n_gpus: int, *, hourly_rate: float = H100_HOURLY_USD) -> float:
    """USD to rent ``n_gpus`` for ``latency_s`` seconds: ``latency · n_GPUs · hourly_rate``.

    This is the raw open-source cost rule. For a whole run, ``latency_s`` is the serve window; for
    per-task cost use ``local_dollars_per_task`` (which amortizes the window over the batch).
    """
    return (latency_s / 3600.0) * n_gpus * hourly_rate


def local_dollars_per_task(
    window_s: float, n_gpus: int, n_tasks: int, *, hourly_rate: float = H100_HOURLY_USD
) -> float:
    """Amortized open-source $/task = ``local_dollars(window_s) / n_tasks``.

    Continuous batching shares the GPUs across concurrent tasks, so per-task cost is the run
    window's rental cost spread over the tasks completed in it — not each task's isolated latency.
    """
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    return local_dollars(window_s, n_gpus, hourly_rate=hourly_rate) / n_tasks


def throughput_dollars_per_task(
    tokens_in: int,
    tokens_out: int,
    *,
    decode_tok_s: float,
    prefill_tok_s: float,
    n_gpus: int,
    cache_frac: float = 0.0,
    hourly_rate: float = H100_HOURLY_USD,
) -> float:
    """Deployment-realistic local $/task from a SATURATED serve's measured throughput.

    The honest reading of "latency" in the local cost rule is the GPU-time a task's tokens actually
    occupy at saturation — NOT single-stream wall-clock, which bills the GPU while it sits idle
    during agentic tool execution (that idle time inflates the wall-clock number ~50-200×). At
    saturation many concurrent tasks keep the GPU busy, so per-task GPU-time is:

        gpu_s = prefill_tokens · (1 - cache_frac) / prefill_tok_s  +  output_tokens / decode_tok_s

    ``cache_frac`` discounts the prompt tokens served from the prefix cache (agent turns re-send a
    growing context; typical reuse is high). ``decode_tok_s``/``prefill_tok_s`` are the serve's
    measured saturated aggregate throughputs on ``n_gpus``. This is the honest local cost basis; the
    wall-clock ``local_dollars_per_task`` is a single-stream upper bound.
    """
    gpu_s = (tokens_in * (1.0 - cache_frac)) / prefill_tok_s + tokens_out / decode_tok_s
    return (gpu_s / 3600.0) * n_gpus * hourly_rate


def multiplier(our_dollars: float, reference_dollars: float) -> float:
    """Iso-accuracy cost multiplier: how many times cheaper we are than the reference (e.g. Fable)."""
    if our_dollars <= 0:
        raise ValueError("our_dollars must be positive")
    return reference_dollars / our_dollars


def sensitivity_band(
    our_dollars: float, reference_dollars: float, *, band: float = 0.5
) -> tuple[float, float]:
    """Range of the iso-accuracy multiplier as OUR cost scales over ``[1-band, 1+band]``.

    The reference (cloud Fable) is fixed; only our cost — dominated by the local GPU-hr rate on the
    open side — is perturbed. Returns ``(low_multiplier, high_multiplier)`` so the ≥20× claim can be
    reported as robust to a ±band swing in the $3.00 rate rather than a point estimate.
    """
    hi_cost, lo_cost = our_dollars * (1 + band), our_dollars * (1 - band)
    return multiplier(hi_cost, reference_dollars), multiplier(lo_cost, reference_dollars)


def cost_from_tokens(
    input_tokens: int,
    output_tokens: int,
    price_in_per_m: float,
    price_out_per_m: float,
) -> float:
    """USD cost for one completion given per-million-token prices.

    Args:
        input_tokens: Number of prompt/input tokens consumed.
        output_tokens: Number of completion/output tokens generated.
        price_in_per_m: Price per million input tokens in USD.
        price_out_per_m: Price per million output tokens in USD.

    Returns:
        Total USD cost for this completion.
    """
    return float(
        dollars_tokens_decimal(
            input_tokens,
            output_tokens,
            str(price_in_per_m),
            str(price_out_per_m),
        )
    )


def cost_for(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD cost for a completion using the PRICING table.

    Args:
        model_id: Model identifier (case-insensitive lookup in PRICING).
        input_tokens: Number of prompt/input tokens consumed.
        output_tokens: Number of completion/output tokens generated.

    Returns:
        USD cost if the model is in PRICING, else None.
    """
    entry = PRICING.get(model_id.lower())
    if entry is None:
        return None
    price_in, price_out = entry
    return cost_from_tokens(input_tokens, output_tokens, price_in, price_out)
