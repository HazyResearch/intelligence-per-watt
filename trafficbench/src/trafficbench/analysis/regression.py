"""Regression utilities and default analysis implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.registry import AnalysisRegistry
from .base import AnalysisProvider, AnalysisContext, AnalysisResult
from .helpers import (
    iter_model_entries,
    load_metrics_dataset,
    resolve_model_name,
)

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
        regressions["input_tokens_vs_ttft"].append(
            RegressionSample(prompt_tokens, ttft_seconds)
        )

    if total_tokens is not None and per_query_joules is not None:
        regressions["total_tokens_vs_energy"].append(
            RegressionSample(total_tokens, per_query_joules)
        )

    if total_tokens is not None and total_latency_seconds is not None:
        regressions["total_tokens_vs_latency"].append(
            RegressionSample(total_tokens, total_latency_seconds)
        )

    if (
        total_tokens is not None
        and per_query_watts is not None
        and abs(per_query_watts) >= ZERO_EPSILON
    ):
        regressions["total_tokens_vs_power"].append(
            RegressionSample(total_tokens, per_query_watts)
        )


def finalize_regressions(
    regressions: RegressionDict,
    *,
    include_power_log: bool = True,
) -> Dict[str, Dict[str, Optional[float]]]:
    results: Dict[str, Dict[str, Optional[float]]] = {
        "input_tokens_vs_ttft": _regression_with_average(
            regressions["input_tokens_vs_ttft"]
        ),
        "total_tokens_vs_energy": _regression_with_average(
            regressions["total_tokens_vs_energy"]
        ),
        "total_tokens_vs_latency": _regression_with_average(
            regressions["total_tokens_vs_latency"]
        ),
        "total_tokens_vs_power": _regression_with_average(
            regressions["total_tokens_vs_power"]
        ),
    }
    if include_power_log:
        results["total_tokens_vs_power_log"] = _regression_with_average(
            regressions["total_tokens_vs_power"],
            log_x=True,
        )
    return results


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


@AnalysisRegistry.register("regression")
class RegressionAnalysis(AnalysisProvider):
    """Default analysis computing regression statistics for metrics runs."""

    analysis_id = "regression"

    def run(self, context: AnalysisContext) -> AnalysisResult:
        results_dir = context.results_dir
        options = dict(context.options)
        requested_model = options.get("model")
        skip_zeroes = bool(options.get("skip_zeroes", False))

        dataset = load_metrics_dataset(results_dir)
        active_model = resolve_model_name(dataset, requested_model, results_dir)

        entries = list(iter_model_entries(dataset, active_model))
        if not entries:
            raise RuntimeError(
                f"No usable metrics found for model '{active_model}' in dataset at '{results_dir}'."
            )

        regressions, zero_counts = create_regression_containers()

        samples_collected = 0
        for entry in entries:
            token_metrics = _get_mapping(entry.get("token_metrics"))
            latency_metrics = _get_mapping(entry.get("latency_metrics"))
            energy_metrics = _get_mapping(entry.get("energy_metrics"))
            power_metrics = _get_mapping(entry.get("power_metrics"))

            prompt_tokens = to_float(token_metrics.get("input"))
            completion_tokens = to_float(token_metrics.get("output"))
            total_tokens = to_float(token_metrics.get("total"))
            if total_tokens is None:
                total_tokens = derive_total_tokens(prompt_tokens, completion_tokens)

            ttft_value = to_float(latency_metrics.get("time_to_first_token_seconds"))
            total_latency_value = to_float(latency_metrics.get("total_query_seconds"))

            energy_value = to_float(energy_metrics.get("per_query_joules"))
            power_value = _extract_power_value(power_metrics)

            register_regression_sample(
                regressions,
                zero_counts,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ttft_seconds=ttft_value,
                total_latency_seconds=total_latency_value,
                per_query_joules=energy_value,
                per_query_watts=power_value,
            )
            samples_collected += 1

        if samples_collected == 0:
            raise RuntimeError(
                f"No usable metrics found for model '{active_model}' in dataset at '{results_dir}'."
            )

        regression_results = finalize_regressions(regressions)
        if skip_zeroes:
            regression_results = _filter_none_regressions(regression_results)

        warnings = build_zero_warnings(zero_counts, context=" in dataset")

        summary_payload = {
            "total_samples": samples_collected,
        }
        data_payload: Dict[str, Any] = {
            "regressions": dict(regression_results),
        }

        artifact_payload = {
            "analysis": self.analysis_id,
            "summary": summary_payload,
            "warnings": list(warnings),
            "data": data_payload,
        }

        artifact_dir = results_dir / "analysis"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{self.analysis_id}.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, default=str))

        return AnalysisResult(
            analysis=self.analysis_id,
            summary=summary_payload,
            data=data_payload,
            warnings=tuple(warnings),
            artifacts={"report": artifact_path},
        )


def _get_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _extract_power_value(power_metrics: Mapping[str, Any]) -> Optional[float]:
    gpu_metrics = power_metrics.get("gpu")
    if isinstance(gpu_metrics, Mapping):
        per_query = gpu_metrics.get("per_query_watts")
        if isinstance(per_query, Mapping):
            for key in ("avg", "median", "max", "min"):
                candidate = to_float(per_query.get(key))
                if candidate is not None:
                    return candidate
    return None


def _filter_none_regressions(
    regressions: Mapping[str, Mapping[str, Optional[float]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    filtered: Dict[str, Dict[str, Optional[float]]] = {}
    for name, stats in regressions.items():
        if any(
            stats.get(field) is None for field in ("slope", "intercept", "r2", "avg_y")
        ):
            continue
        filtered[name] = dict(stats)
    return filtered


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def derive_total_tokens(
    prompt_tokens: Optional[float],
    completion_tokens: Optional[float],
) -> Optional[float]:
    if prompt_tokens is None and completion_tokens is None:
        return None
    prompt_val = prompt_tokens or 0.0
    completion_val = completion_tokens or 0.0
    return prompt_val + completion_val


__all__ = [
    "RegressionAnalysis",
    "RegressionDict",
    "RegressionSample",
    "ZeroCountDict",
    "ZERO_EPSILON",
    "build_zero_warnings",
    "create_regression_containers",
    "finalize_regressions",
    "register_regression_sample",
    "to_float",
    "derive_total_tokens",
]
