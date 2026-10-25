"""Central-percentile amortization of a batched run's cost metrics (the MANDATORY IPJ metric
methodology — see ipj/experiments/INFERENCE_CONFIG.md).

One saturated-batch run produces a per-query event log + a board-power time series; from those we
amortize energy / $ / latency / power / tokens / throughput over several **central-percentile
windows** (total, P90, P80, P50). P50 (central 50% by completion time) is the headline — it removes
the startup ramp and the lagging tail, leaving steady state.

Window definition (confirmed 2026-06-20): order queries by COMPLETION time; for P_X drop the first
and last (100-X)/2 %. Window energy = integral of board power over [earliest start, latest
completion] of the central queries, amortized over the X%*N central queries. Energy is a
time-window integral, NOT per-query attribution (continuous batching makes per-query energy
unmeasurable). Accuracy is always reported over the full N elsewhere — these windows are for the
efficiency/cost metrics only.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipj.fast_measure import trapezoid_joules

WINDOWS = (100, 90, 80, 50)


@dataclass(slots=True)
class QueryEvent:
    start: float          # epoch seconds the query began
    end: float            # epoch seconds the query completed
    tokens_in: int = 0
    tokens_out: int = 0
    resolved: bool = False


@dataclass(slots=True)
class WindowMetrics:
    percentile: int
    n_queries: int
    window_s: float
    energy_j: float
    energy_per_query_j: float
    avg_power_w: float
    tokens_in_per_query: float
    tokens_out_per_query: float
    decode_throughput_tok_s: float
    latency_mean_s: float
    latency_p50_s: float
    latency_p90_s: float
    dollars: float | None = None
    dollars_per_query: float | None = None


def _pct(sorted_vals: list[float], q: float) -> float:
    """Simple nearest-rank percentile (q in [0,1]) on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def window_amortize(events, power_samples, percentile: int, *, gpu_hourly_rate=None,
                    n_gpus: int = 1, price_in: float = 0.0, price_out: float = 0.0) -> WindowMetrics:
    """Amortize the run's metrics over the central `percentile`% of queries (by completion time)."""
    evs = sorted(events, key=lambda e: e.end)
    n = len(evs)
    trim = (100 - percentile) / 200.0          # fraction dropped from EACH end
    lo = int(n * trim)
    central = evs[lo:n - lo] if percentile < 100 and lo > 0 else evs
    nc = len(central)
    if nc == 0:
        return WindowMetrics(percentile, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    win_start = min(e.start for e in central)
    win_end = max(e.end for e in central)
    dur = win_end - win_start
    energy = trapezoid_joules(power_samples, win_start, win_end)
    lats = sorted(e.end - e.start for e in central)
    tin = sum(e.tokens_in for e in central)
    tout = sum(e.tokens_out for e in central)

    dollars = dpq = None
    if gpu_hourly_rate is not None:                       # local: amortized GPU-hours over the window
        dollars = (dur / 3600.0) * gpu_hourly_rate * n_gpus
        dpq = dollars / nc
    elif price_in or price_out:                           # cloud: token-priced
        dollars = (tin / 1e6) * price_in + (tout / 1e6) * price_out
        dpq = dollars / nc

    return WindowMetrics(
        percentile=percentile, n_queries=nc, window_s=dur, energy_j=energy,
        energy_per_query_j=energy / nc, avg_power_w=(energy / dur if dur > 0 else 0.0),
        tokens_in_per_query=tin / nc, tokens_out_per_query=tout / nc,
        decode_throughput_tok_s=(tout / dur if dur > 0 else 0.0),
        latency_mean_s=sum(lats) / nc, latency_p50_s=_pct(lats, 0.5), latency_p90_s=_pct(lats, 0.9),
        dollars=dollars, dollars_per_query=dpq,
    )


def amortize_all(events, power_samples, **kw) -> dict[int, WindowMetrics]:
    """Return {percentile: WindowMetrics} for total/P90/P80/P50."""
    return {p: window_amortize(events, power_samples, p, **kw) for p in WINDOWS}
