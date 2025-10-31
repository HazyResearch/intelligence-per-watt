"""Regression utilities for profiling analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RegressionDict = Dict[str, List["RegressionSample"]]
ZeroCountDict = Dict[str, int]
ZERO_EPSILON = 1e-12


@dataclass(slots=True)
class RegressionSample:
    x: float
    y: float


def create_regression_containers() -> Tuple[RegressionDict, ZeroCountDict]:
    regressions: RegressionDict = {
        "input_tokens_vs_ttft": [],
        "total_tokens_vs_energy": [],
        "total_tokens_vs_latency": [],
        "total_tokens_vs_power": [],
    }
    zero_counts: ZeroCountDict = {
        "energy": 0,
        "power": 0,
        "ttft": 0,
        "latency": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }
    return regressions, zero_counts


def register_regression_sample(
    regressions: RegressionDict,
    zero_counts: ZeroCountDict,
    *,
    prompt_tokens: Optional[float],
    completion_tokens: Optional[float],
    total_tokens: Optional[float],
    ttft_seconds: Optional[float],
    total_latency_seconds: Optional[float],
    per_query_joules: Optional[float],
    per_query_watts: Optional[float],
) -> None:
    if per_query_joules is not None and abs(per_query_joules) < ZERO_EPSILON:
        zero_counts["energy"] += 1
    if per_query_watts is not None and abs(per_query_watts) < ZERO_EPSILON:
        zero_counts["power"] += 1
    if ttft_seconds is not None and abs(ttft_seconds) < ZERO_EPSILON:
        zero_counts["ttft"] += 1
    if total_latency_seconds is not None and abs(total_latency_seconds) < ZERO_EPSILON:
        zero_counts["latency"] += 1
    if completion_tokens is not None and abs(completion_tokens) < ZERO_EPSILON:
        zero_counts["output_tokens"] += 1
    if prompt_tokens is not None and abs(prompt_tokens) < ZERO_EPSILON:
        zero_counts["prompt_tokens"] += 1
    if total_tokens is not None and abs(total_tokens) < ZERO_EPSILON:
        zero_counts["total_tokens"] += 1

    if prompt_tokens is not None and ttft_seconds is not None:
        regressions["input_tokens_vs_ttft"].append(RegressionSample(prompt_tokens, ttft_seconds))

    if total_tokens is not None and per_query_joules is not None:
        regressions["total_tokens_vs_energy"].append(RegressionSample(total_tokens, per_query_joules))

    if total_tokens is not None and total_latency_seconds is not None:
        regressions["total_tokens_vs_latency"].append(RegressionSample(total_tokens, total_latency_seconds))

    if total_tokens is not None and per_query_watts is not None and abs(per_query_watts) >= ZERO_EPSILON:
        regressions["total_tokens_vs_power"].append(RegressionSample(total_tokens, per_query_watts))


def finalize_regressions(
    regressions: RegressionDict,
    *,
    include_power_log: bool = True,
) -> Dict[str, Dict[str, Optional[float]]]:
    results: Dict[str, Dict[str, Optional[float]]] = {
        "input_tokens_vs_ttft": _regression_with_average(regressions["input_tokens_vs_ttft"]),
        "total_tokens_vs_energy": _regression_with_average(regressions["total_tokens_vs_energy"]),
        "total_tokens_vs_latency": _regression_with_average(regressions["total_tokens_vs_latency"]),
        "total_tokens_vs_power": _regression_with_average(regressions["total_tokens_vs_power"]),
    }
    if include_power_log:
        results["total_tokens_vs_power_log"] = _regression_with_average(
            regressions["total_tokens_vs_power"],
            log_x=True,
        )
    return results


def build_zero_warnings(zero_counts: ZeroCountDict, *, context: str = "") -> List[str]:
    warnings: List[str] = []
    if zero_counts["energy"]:
        warnings.append(
            f"encountered {zero_counts['energy']} per-query energy samples equal to 0.0{context}"
        )
    if zero_counts["power"]:
        warnings.append(
            f"encountered {zero_counts['power']} per-query power samples equal to 0.0{context}"
        )
    if zero_counts["ttft"]:
        warnings.append(
            f"encountered {zero_counts['ttft']} TTFT samples equal to 0.0{context}"
        )
    if zero_counts["latency"]:
        warnings.append(
            f"encountered {zero_counts['latency']} latency samples equal to 0.0{context}"
        )
    if zero_counts["output_tokens"]:
        warnings.append(
            f"encountered {zero_counts['output_tokens']} completion-token samples equal to 0.0{context}"
        )
    if zero_counts["prompt_tokens"]:
        warnings.append(
            f"encountered {zero_counts['prompt_tokens']} prompt-token samples equal to 0.0{context}"
        )
    if zero_counts["total_tokens"]:
        warnings.append(
            f"encountered {zero_counts['total_tokens']} total-token samples equal to 0.0{context}"
        )
    return warnings


def _regression_with_average(
    samples: Sequence[RegressionSample],
    *,
    log_y: bool = False,
    log_x: bool = False,
) -> Dict[str, Optional[float]]:
    stats = _compute_regression(samples, log_y=log_y, log_x=log_x)
    stats["avg_y"] = _compute_average(samples)
    return stats


def _compute_regression(
    samples: Sequence[RegressionSample],
    *,
    log_y: bool = False,
    log_x: bool = False,
) -> Dict[str, Optional[float]]:
    if len(samples) < 2:
        return {"count": len(samples), "slope": None, "intercept": None, "r2": None}

    x = np.array([s.x for s in samples], dtype=np.float64)
    y = np.array([s.y for s in samples], dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if log_x:
        mask &= x > 0
    if log_y:
        mask &= y > 0
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return {"count": len(x), "slope": None, "intercept": None, "r2": None}

    if log_y:
        y = np.log(y)
    if log_x:
        x = np.log(x)

    if np.unique(x).size < 2:
        return {"count": int(len(x)), "slope": None, "intercept": None, "r2": None}

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        return {"count": int(len(x)), "slope": None, "intercept": None, "r2": None}
    predictions = slope * x + intercept
    residuals = y - predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

    return {
        "count": int(len(x)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def _compute_average(samples: Sequence[RegressionSample]) -> Optional[float]:
    if not samples:
        return None
    y_values = [s.y for s in samples]
    return float(np.mean(y_values))


__all__ = [
    "RegressionDict",
    "RegressionSample",
    "ZeroCountDict",
    "ZERO_EPSILON",
    "build_zero_warnings",
    "create_regression_containers",
    "finalize_regressions",
    "register_regression_sample",
]

