"""Report builder for regression analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ..execution.hardware import derive_hardware_label
from .loader import collect_run_metadata, iter_model_entries, load_metrics_dataset, resolve_model_name
from .regression import (
    ZeroCountDict,
    build_zero_warnings,
    create_regression_containers,
    finalize_regressions,
    register_regression_sample,
)


@dataclass(slots=True)
class RegressionReport:
    model: str
    hardware_label: str
    regressions: Dict[str, Dict[str, Optional[float]]]
    zero_counts: ZeroCountDict
    warnings: list[str]
    total_samples: int
    summary_path: Path
    summary: Dict[str, Any]
    system_info: Optional[Dict[str, Any]] = None
    gpu_info: Optional[Dict[str, Any]] = None


def build_regression_report(
    metrics_dir: Path,
    *,
    model_name: str | None = None,
    skip_zeroes: bool = False,
) -> RegressionReport:
    """Compute regression statistics and persist them into the run summary."""

    dataset = load_metrics_dataset(metrics_dir)
    active_model = resolve_model_name(dataset, model_name, metrics_dir)

    entries = list(iter_model_entries(dataset, active_model))
    if not entries:
        raise RuntimeError(
            f"No usable metrics found for model '{active_model}' in dataset at '{metrics_dir}'."
        )

    system_info, gpu_info = collect_run_metadata(entries)
    regressions, zero_counts = create_regression_containers()

    samples_collected = 0
    for entry in entries:
        token_metrics = _get_mapping(entry.get("token_metrics"))
        latency_metrics = _get_mapping(entry.get("latency_metrics"))
        energy_metrics = _get_mapping(entry.get("energy_metrics"))
        power_metrics = _get_mapping(entry.get("power_metrics"))

        prompt_tokens = to_float(token_metrics.get("input"))
        completion_tokens = to_float(token_metrics.get("output"))
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
            f"No usable metrics found for model '{active_model}' in dataset at '{metrics_dir}'."
        )

    regression_results = finalize_regressions(regressions)

    if skip_zeroes:
        regression_results = _filter_none_regressions(regression_results)

    sys_arg = system_info or None
    gpu_arg = gpu_info or None
    hardware_label = derive_hardware_label(sys_arg, gpu_arg)

    warnings = build_zero_warnings(zero_counts, context=" in dataset")

    summary_data = _load_summary(metrics_dir)
    summary_path = metrics_dir / "summary.json"
    now = time.time()

    summary_data.setdefault("model", active_model)
    summary_data["regressions"] = regression_results
    summary_data["hardware_label"] = hardware_label
    if system_info and not summary_data.get("system_info"):
        summary_data["system_info"] = system_info
    if gpu_info and not summary_data.get("gpu_info"):
        summary_data["gpu_info"] = gpu_info
    summary_data["regression_analysis"] = {
        "model": active_model,
        "generated_at": now,
        "hardware_label": hardware_label,
        "total_samples": samples_collected,
        "zero_counts": dict(zero_counts),
        "warnings": warnings,
        "skip_zeroes": skip_zeroes,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_data, indent=2))

    return RegressionReport(
        model=active_model,
        hardware_label=hardware_label,
        regressions=dict(regression_results),
        zero_counts=dict(zero_counts),
        warnings=list(warnings),
        total_samples=samples_collected,
        summary_path=summary_path,
        summary=dict(summary_data),
        system_info=system_info or None,
        gpu_info=gpu_info or None,
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
    regressions: Dict[str, Dict[str, Optional[float]]]
) -> Dict[str, Dict[str, Optional[float]]]:
    filtered: Dict[str, Dict[str, Optional[float]]] = {}
    for name, stats in regressions.items():
        if any(stats.get(field) is None for field in ("slope", "intercept", "r2", "avg_y")):
            continue
        filtered[name] = stats
    return filtered


def _load_summary(metrics_dir: Path) -> Dict[str, Any]:
    summary_path = metrics_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        raw = summary_path.read_text()
    except OSError as exc:
        raise RuntimeError(f"Failed to read summary at '{summary_path}': {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Summary file at '{summary_path}' is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Summary file at '{summary_path}' does not contain an object."
        )
    return data


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


__all__ = ["RegressionReport", "build_regression_report"]
