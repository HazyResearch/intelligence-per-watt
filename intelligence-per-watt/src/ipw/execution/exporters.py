"""Export functions for agent run traces and profiling records."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from ..core.accounting import (
    RIGOROUS_ENERGY_PROVENANCE,
    rigorous_utility_population_is_complete,
)
from ..cost.aggregation import is_valid_cost, sum_costs_if_complete
from .public_artifact import (
    ArtifactProvenance,
    PublicArtifactPolicy,
    create_public_trace_artifact,
)
from .trace import QueryTrace

_LEGACY_COST_COMPATIBILITY_STATUS = "legacy_lm_only_partial"


def _finite_nonnegative(value: Any) -> Optional[float]:
    """Return a normalized physical measurement or ``None`` when invalid."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        return None
    return normalized


def _eligible_trace_energy(
    trace: QueryTrace,
    value: Optional[float],
) -> Optional[float]:
    """Gate per-query energy on attribution and isolation metadata."""
    if not trace.lm_energy_measurable or trace.shared_device_warning:
        return None
    return _finite_nonnegative(value)


def _rigorous_trace_gpu_energy(trace: QueryTrace) -> Optional[float]:
    """Return GPU energy only with explicit complete measured provenance."""
    if trace.gpu_energy_complete is not True:
        return None
    if trace.accelerator_device_coverage_complete is not True:
        return None
    if trace.gpu_energy_provenance not in RIGOROUS_ENERGY_PROVENANCE:
        return None
    return _eligible_trace_energy(trace, trace.complete_total_gpu_energy_joules)


def _eligible_benchmark_energy(
    bench_energy: dict[str, Any],
    resource: str,
) -> Optional[float]:
    """Validate a full-window benchmark measurement for rigorous aggregation."""
    if bench_energy.get(f"{resource}_energy_complete") is not True:
        return None
    coverage_key = (
        "accelerator_device_coverage_complete" if resource == "gpu" else "host_energy_domain_coverage_complete"
    )
    if bench_energy.get(coverage_key) is not True:
        return None
    if bench_energy.get(f"{resource}_energy_provenance") not in RIGOROUS_ENERGY_PROVENANCE:
        return None
    return _finite_nonnegative(bench_energy.get(f"{resource}_energy_joules"))


def _benchmark_energy_covers_workload(traces: Sequence[QueryTrace]) -> bool:
    """Return whether one benchmark meter can cover every executed LM trace."""

    return bool(traces) and all(trace.lm_energy_measurable and not trace.shared_device_warning for trace in traces)


def _agg_stats(values: Sequence[Optional[float]]) -> dict[str, Optional[float]]:
    """Return {avg, median, min, max, std} filtering None values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"avg": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "avg": statistics.mean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def _agg_stats_if_complete(values: Sequence[Optional[float]]) -> dict[str, Optional[float]]:
    """Compute statistics only when every required observation is known."""
    if any(value is None for value in values):
        return {"avg": None, "median": None, "min": None, "max": None, "std": None}
    return _agg_stats(values)


def _sum_nonnegative_if_complete(
    values: Sequence[Optional[float]],
) -> Optional[float]:
    """Return a finite total only when every required observation is valid."""
    if not values:
        return None
    if any(value is None or not math.isfinite(value) or value < 0 for value in values):
        return None
    return math.fsum(value for value in values if value is not None)


def _compute_stats(traces: list[QueryTrace]) -> dict[str, dict[str, Any]]:
    """Compute per-metric statistics for a list of traces."""
    stats: dict[str, dict[str, Any]] = {
        "wall_clock_s": _agg_stats([t.total_wall_clock_s for t in traces]),
        "gpu_energy_joules": _agg_stats([t.total_gpu_energy_joules for t in traces]),
        "complete_gpu_energy_joules": _agg_stats_if_complete([t.complete_total_gpu_energy_joules for t in traces]),
        "judge_gpu_energy_joules": _agg_stats([t.judge_gpu_energy_joules for t in traces]),
        "total_task_gpu_energy_joules": _agg_stats([t.total_task_gpu_energy_joules for t in traces]),
        "cpu_energy_joules": _agg_stats([t.total_cpu_energy_joules for t in traces]),
        "complete_cpu_energy_joules": _agg_stats_if_complete([t.complete_total_cpu_energy_joules for t in traces]),
        "judge_cpu_energy_joules": _agg_stats([t.judge_cpu_energy_joules for t in traces]),
        "total_task_cpu_energy_joules": _agg_stats([t.total_task_cpu_energy_joules for t in traces]),
        "gpu_power_watts": _agg_stats([t.avg_gpu_power_watts for t in traces]),
        "judge_gpu_power_watts": _agg_stats([t.judge_gpu_power_avg_watts for t in traces]),
        "cpu_power_watts": _agg_stats([t.avg_cpu_power_watts for t in traces]),
        "judge_cpu_power_watts": _agg_stats([t.judge_cpu_power_avg_watts for t in traces]),
        "judge_wall_clock_s": _agg_stats([t.judge_wall_clock_s for t in traces]),
        "input_tokens": _agg_stats(
            [float(t.total_input_tokens) if t.total_input_tokens is not None else None for t in traces]
        ),
        "output_tokens": _agg_stats(
            [float(t.total_output_tokens) if t.total_output_tokens is not None else None for t in traces]
        ),
        "total_tokens": _agg_stats([float(t.total_tokens) if t.total_tokens is not None else None for t in traces]),
        "throughput_tokens_per_sec": _agg_stats([t.throughput_tokens_per_sec for t in traces]),
        "energy_per_token_joules": _agg_stats([t.energy_per_token_joules for t in traces]),
        "complete_energy_per_token_joules": _agg_stats_if_complete(
            [t.complete_energy_per_token_joules for t in traces]
        ),
        # Two-release compatibility: this established key remains the partial
        # LM-only view. Scientific consumers must use complete_cost_usd.
        "cost_usd": {
            **_agg_stats([t.total_cost_usd for t in traces]),
            "compatibility_status": _LEGACY_COST_COMPATIBILITY_STATUS,
        },
        "complete_cost_usd": _agg_stats_if_complete([t.complete_total_cost_usd for t in traces]),
        "legacy_lm_only_cost_usd": _agg_stats([t.legacy_lm_only_cost_usd for t in traces]),
        "turns": _agg_stats([float(t.num_turns) for t in traces]),
        "tool_calls": _agg_stats([float(t.total_tool_calls) for t in traces]),
        "mbu_avg_pct": _agg_stats([t.query_mbu_avg_pct for t in traces]),
    }
    return stats


def _compute_efficiency(
    traces: list[QueryTrace],
    stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute legacy diagnostics and additive rigorous workload efficiency."""
    resolved_count = sum(1 for t in traces if t.is_resolved is True)
    unresolved_count = sum(1 for t in traces if t.is_resolved is False)
    scored_count = resolved_count + unresolved_count
    accuracy = resolved_count / scored_count if scored_count > 0 else None

    # Preserve the established IPW summary contract for two public releases:
    # known per-query energy is partially summed and IPJ means accuracy divided
    # by that sum. This is descriptive only; the rigorous sibling below is the
    # ratio of workload utility totals over complete measured accelerator energy.
    gpu_energy_values = [t.total_gpu_energy_joules for t in traces if t.total_gpu_energy_joules is not None]
    total_gpu_energy = sum(gpu_energy_values) if gpu_energy_values else None
    gpu_energy_known = len(gpu_energy_values)

    rigorous_gpu_energy_values = [_rigorous_trace_gpu_energy(t) for t in traces]
    rigorous_total_gpu_energy = _sum_nonnegative_if_complete(rigorous_gpu_energy_values)
    rigorous_gpu_energy_known = sum(value is not None for value in rigorous_gpu_energy_values)

    cpu_energy_values = [t.total_cpu_energy_joules for t in traces if t.total_cpu_energy_joules is not None]
    total_cpu_energy = sum(cpu_energy_values) if cpu_energy_values else None
    cpu_energy_known = len(cpu_energy_values)

    avg_gpu_power = stats["gpu_power_watts"]["avg"]
    avg_cpu_power = stats["cpu_power_watts"]["avg"]

    utility_complete = scored_count == len(traces) and scored_count > 0
    rigorous_observations = [(trace.rigorous_utility_basis, trace.utility_task_membership_id) for trace in traces]
    eligible_bases = {basis for basis, _ in rigorous_observations if basis is not None}
    if len(eligible_bases) == 1:
        only_basis = next(iter(eligible_bases))
        rigorous_utility_expected = only_basis[-1]
        rigorous_utility_known = len(
            {
                membership_id.strip()
                for basis, membership_id in rigorous_observations
                if basis == only_basis and isinstance(membership_id, str) and membership_id.strip()
            }
        )
        rigorous_utility_known = min(rigorous_utility_known, rigorous_utility_expected)
    else:
        rigorous_utility_expected = len(traces)
        rigorous_utility_known = 0
    rigorous_utility_complete = (
        utility_complete
        and len({trace.workload_type for trace in traces}) == 1
        and rigorous_utility_population_is_complete(rigorous_observations)
    )
    ipj_eligible = accuracy is not None and accuracy > 0 and total_gpu_energy is not None and total_gpu_energy > 0
    ipj = accuracy / total_gpu_energy if ipj_eligible else None
    rigorous_ipj_eligible = (
        rigorous_utility_complete and rigorous_total_gpu_energy is not None and rigorous_total_gpu_energy > 0
    )
    rigorous_ipj = resolved_count / rigorous_total_gpu_energy if rigorous_ipj_eligible else None
    ipw = (
        accuracy / avg_gpu_power if (accuracy is not None and avg_gpu_power is not None and avg_gpu_power > 0) else None
    )

    return {
        "accuracy": accuracy,
        "utility_total": resolved_count,
        "scored_queries": scored_count,
        "unscored_queries": len(traces) - scored_count,
        "rigorous_utility_known_queries": rigorous_utility_known,
        "rigorous_utility_unknown_queries": max(rigorous_utility_expected - rigorous_utility_known, 0),
        "rigorous_utility_expected_queries": rigorous_utility_expected,
        "rigorous_utility_coverage": (
            rigorous_utility_known / rigorous_utility_expected if rigorous_utility_expected > 0 else None
        ),
        "total_gpu_energy_joules": total_gpu_energy,
        "gpu_energy_known_queries": gpu_energy_known,
        "gpu_energy_unknown_queries": len(traces) - gpu_energy_known,
        "gpu_energy_coverage": (gpu_energy_known / len(traces) if traces else None),
        "total_cpu_energy_joules": total_cpu_energy,
        "cpu_energy_known_queries": cpu_energy_known,
        "cpu_energy_unknown_queries": len(traces) - cpu_energy_known,
        "cpu_energy_coverage": (cpu_energy_known / len(traces) if traces else None),
        "avg_gpu_power_watts": avg_gpu_power,
        "avg_cpu_power_watts": avg_cpu_power,
        "ipj_eligible": ipj_eligible,
        "ipj_energy_basis": "legacy_partial_observed_gpu_energy",
        "ipj_definition": "accuracy / sum(known_gpu_energy_joules)",
        "ipj": ipj,
        "legacy_ipj_eligible": ipj_eligible,
        "legacy_ipj": ipj,
        "ipj_compatibility_status": "legacy_nonrigorous",
        "rigorous_total_gpu_energy_joules": rigorous_total_gpu_energy,
        "rigorous_gpu_energy_known_queries": rigorous_gpu_energy_known,
        "rigorous_gpu_energy_unknown_queries": len(traces) - rigorous_gpu_energy_known,
        "rigorous_gpu_energy_coverage": (rigorous_gpu_energy_known / len(traces) if traces else None),
        "rigorous_ipj_eligible": rigorous_ipj_eligible,
        "rigorous_ipj": rigorous_ipj,
        "ipw": ipw,
    }


def export_jsonl(traces: list[QueryTrace], path: Path) -> Path:
    """Export traces as JSONL (one JSON object per line).

    Args:
        traces: List of QueryTrace objects to export.
        path: Output file path. Parent directories are created if needed.

    Returns:
        The path to the written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict()) + "\n")
    return path


def export_public_trace_artifact(
    traces: list[QueryTrace],
    path: Path,
    *,
    provenance: ArtifactProvenance,
    policy: PublicArtifactPolicy | None = None,
    id_key: bytes | None = None,
) -> Path:
    """Export a fail-closed public trace bundle.

    Unlike :func:`export_jsonl`, this API removes prompts, responses, tool
    names/results, errors, arbitrary metadata, raw trace IDs, and infrastructure
    details by default.  Content retention requires an explicit policy and is
    still subject to the credential/private-infrastructure scanner.

    The historical raw exporters remain available for authorized local use.
    Their output is not accepted by the public registry upload path.
    """

    return create_public_trace_artifact(
        traces,
        path,
        provenance=provenance,
        policy=policy,
        id_key=id_key,
    )


def export_hf_dataset(traces: list[QueryTrace], path: Path) -> Path:
    """Export traces as a HuggingFace Arrow dataset.

    Args:
        traces: List of QueryTrace objects to export.
        path: Output directory for the Arrow dataset.

    Returns:
        The path to the saved dataset directory.
    """
    ds = QueryTrace.to_hf_dataset(traces)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    return path


def export_summary_json(
    traces: list[QueryTrace],
    config: dict[str, Any],
    path: Path,
    *,
    bench_energy: dict[str, Any] | None = None,
) -> Path:
    """Export aggregate summary as JSON.

    Args:
        traces: List of QueryTrace objects.
        config: Run configuration dictionary.
        path: Output file path.
        bench_energy: Optional benchmark-level energy metrics.  When
            telemetry is collected at aggregate (benchmark) granularity
            rather than per-query, per-trace energy fields will be
            ``None``.  This dict supplies the aggregate values so the
            summary still includes energy/power/efficiency data.

    Returns:
        The path to the written file.
    """
    total_queries = len(traces)
    completed = sum(1 for t in traces if t.completed)
    total_turns = sum(t.num_turns for t in traces)
    total_tool_calls = sum(t.total_tool_calls for t in traces)

    # Aggregate tokens. If any query lacks measured token usage, the aggregate
    # is unknown; do not estimate or partially sum it.
    input_token_values = [t.total_input_tokens for t in traces]
    output_token_values = [t.total_output_tokens for t in traces]
    total_input_tokens = None if any(v is None for v in input_token_values) else sum(input_token_values)
    total_output_tokens = None if any(v is None for v in output_token_values) else sum(output_token_values)
    total_tokens = (
        total_input_tokens + total_output_tokens
        if total_input_tokens is not None and total_output_tokens is not None
        else None
    )

    # Aggregate wall clock
    total_wall_clock_s = sum(t.total_wall_clock_s for t in traces)

    # Aggregate legacy energy exactly as the established exporter did: partial
    # sums of known observations. Complete measured totals remain separate.
    gpu_energy_values = [t.total_gpu_energy_joules for t in traces if t.total_gpu_energy_joules is not None]
    total_gpu_energy = sum(gpu_energy_values) if gpu_energy_values else None

    judge_gpu_energy_values = [t.judge_gpu_energy_joules for t in traces if t.judge_gpu_energy_joules is not None]
    total_judge_gpu_energy = sum(judge_gpu_energy_values) if judge_gpu_energy_values else None

    total_task_gpu_energy_values = [
        t.total_task_gpu_energy_joules for t in traces if t.total_task_gpu_energy_joules is not None
    ]
    total_task_gpu_energy = sum(total_task_gpu_energy_values) if total_task_gpu_energy_values else None

    cpu_energy_values = [t.total_cpu_energy_joules for t in traces if t.total_cpu_energy_joules is not None]
    total_cpu_energy = sum(cpu_energy_values) if cpu_energy_values else None

    judge_cpu_energy_values = [t.judge_cpu_energy_joules for t in traces if t.judge_cpu_energy_joules is not None]
    total_judge_cpu_energy = sum(judge_cpu_energy_values) if judge_cpu_energy_values else None

    # Aggregate resolved/unresolved
    resolved = sum(1 for t in traces if t.is_resolved is True)
    unresolved = sum(1 for t in traces if t.is_resolved is False)
    unscorable = sum(
        1 for t in traces if t.is_resolved is None and (t.unscorable_reason or t.timed_out or not t.completed)
    )

    # Aggregate cost
    legacy_cost_values = [trace.total_cost_usd for trace in traces if trace.total_cost_usd is not None]
    legacy_total_cost = sum(legacy_cost_values) if legacy_cost_values else None
    complete_cost_values = [trace.complete_total_cost_usd for trace in traces]
    complete_total_cost = sum_costs_if_complete(complete_cost_values)
    cost_known_queries = sum(is_valid_cost(value) for value in complete_cost_values)
    cost_unknown_queries = total_queries - cost_known_queries
    cost_coverage = cost_known_queries / total_queries if total_queries > 0 else None

    # Per-query averages
    avg_turns = total_turns / total_queries if total_queries > 0 else 0
    avg_wall_clock = total_wall_clock_s / total_queries if total_queries > 0 else 0
    avg_gpu_energy = total_gpu_energy / total_queries if total_gpu_energy is not None and total_queries > 0 else None

    # Per-metric statistics (includes mbu_avg_pct)
    stats = _compute_stats(traces)

    # Accuracy and efficiency
    efficiency = _compute_efficiency(traces, stats)
    accuracy = efficiency["accuracy"]

    # Normalized statistics: trim top/bottom 5% by wall_clock_s
    normalized_statistics: Optional[dict[str, Any]] = None
    normalized_efficiency: Optional[dict[str, Any]] = None
    if len(traces) >= 4:
        sorted_traces = sorted(traces, key=lambda t: t.total_wall_clock_s)
        n = len(sorted_traces)
        trim_count = max(1, math.floor(n * 0.05))
        trimmed = sorted_traces[trim_count : n - trim_count]
        outlier_meta = {
            "top_pct": 5,
            "bottom_pct": 5,
            "queries_before": n,
            "queries_after": len(trimmed),
        }

        norm_stats = _compute_stats(trimmed)
        norm_stats["_description"] = (
            "Statistics recomputed after removing the top 5% and bottom 5% of queries by wall_clock_s"
        )
        norm_stats["_outliers_removed"] = outlier_meta
        normalized_statistics = norm_stats

        norm_eff = _compute_efficiency(trimmed, norm_stats)
        norm_eff["_description"] = (
            "Legacy compatibility efficiency recomputed on the post-hoc trimmed "
            "query set; not eligible for a rigorous workload IPJ claim"
        )
        norm_eff["_outliers_removed"] = outlier_meta
        # Latency-based post-hoc trimming changes the workload task set. Keep
        # the historical diagnostic, but never label its ratio headline-ready.
        norm_eff["rigorous_ipj_eligible"] = False
        norm_eff["rigorous_ipj"] = None
        normalized_efficiency = norm_eff

    # Preserve finite benchmark-level measurements in the legacy compatibility
    # totals. Rigorous fields additionally require explicit provenance,
    # component completeness, and device/domain coverage.
    be = bench_energy or {}
    legacy_benchmark_gpu_energy = _finite_nonnegative(be.get("gpu_energy_joules"))
    legacy_benchmark_cpu_energy = _finite_nonnegative(be.get("cpu_energy_joules"))
    rigorous_benchmark_gpu_energy = _eligible_benchmark_energy(be, "gpu")
    if not _benchmark_energy_covers_workload(traces):
        rigorous_benchmark_gpu_energy = None
    if total_gpu_energy is None and legacy_benchmark_gpu_energy is not None:
        total_gpu_energy = legacy_benchmark_gpu_energy
        avg_gpu_energy = total_gpu_energy / total_queries if total_queries > 0 else None
    if total_cpu_energy is None and legacy_benchmark_cpu_energy is not None:
        total_cpu_energy = legacy_benchmark_cpu_energy

    bench_avg_gpu_power = _finite_nonnegative(be.get("avg_gpu_power_watts"))
    bench_avg_cpu_power = _finite_nonnegative(be.get("avg_cpu_power_watts"))

    # Recompute efficiency with any newly available energy data.
    if legacy_benchmark_gpu_energy is not None and efficiency.get("total_gpu_energy_joules") is None:
        avg_gpu_power = (
            bench_avg_gpu_power if bench_avg_gpu_power is not None else efficiency.get("avg_gpu_power_watts")
        )
        utility_total = efficiency["utility_total"]
        utility_complete = efficiency["unscored_queries"] == 0 and total_queries > 0
        rigorous_utility_complete = (
            utility_complete
            and efficiency["rigorous_utility_unknown_queries"] == 0
            and len({trace.workload_type for trace in traces}) == 1
            and rigorous_utility_population_is_complete(
                (trace.rigorous_utility_basis, trace.utility_task_membership_id) for trace in traces
            )
        )
        ipj_eligible = accuracy is not None and accuracy > 0 and total_gpu_energy > 0
        ipj = accuracy / total_gpu_energy if ipj_eligible else None
        rigorous_ipj_eligible = (
            rigorous_utility_complete
            and rigorous_benchmark_gpu_energy is not None
            and rigorous_benchmark_gpu_energy > 0
        )
        rigorous_ipj = utility_total / rigorous_benchmark_gpu_energy if rigorous_ipj_eligible else None
        ipw = (
            accuracy / avg_gpu_power
            if (accuracy is not None and avg_gpu_power is not None and avg_gpu_power > 0)
            else None
        )
        efficiency = {
            **efficiency,
            "total_gpu_energy_joules": total_gpu_energy,
            "total_cpu_energy_joules": total_cpu_energy,
            "gpu_energy_known_queries": total_queries,
            "gpu_energy_unknown_queries": 0,
            "gpu_energy_coverage": 1.0 if total_queries > 0 else None,
            "avg_gpu_power_watts": avg_gpu_power,
            "avg_cpu_power_watts": (
                bench_avg_cpu_power if bench_avg_cpu_power is not None else efficiency.get("avg_cpu_power_watts")
            ),
            "ipj_eligible": ipj_eligible,
            "ipj_energy_basis": "legacy_benchmark_aggregate",
            "ipj": ipj,
            "legacy_ipj_eligible": ipj_eligible,
            "legacy_ipj": ipj,
            "ipj_compatibility_status": "legacy_nonrigorous",
            "rigorous_total_gpu_energy_joules": rigorous_benchmark_gpu_energy,
            "rigorous_gpu_energy_known_queries": (total_queries if rigorous_benchmark_gpu_energy is not None else 0),
            "rigorous_gpu_energy_unknown_queries": (0 if rigorous_benchmark_gpu_energy is not None else total_queries),
            "rigorous_gpu_energy_coverage": (
                1.0 if rigorous_benchmark_gpu_energy is not None and total_queries > 0 else 0.0
            ),
            "rigorous_ipj_eligible": rigorous_ipj_eligible,
            "rigorous_ipj": rigorous_ipj,
            "ipw": ipw,
        }

    summary = {
        "generated_at": time.time(),
        "config": config,
        "totals": {
            "queries": total_queries,
            "completed": completed,
            "resolved": resolved,
            "unresolved": unresolved,
            "unscorable": unscorable,
            "turns": total_turns,
            "tool_calls": total_tool_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "wall_clock_s": total_wall_clock_s,
            "gpu_energy_joules": total_gpu_energy,
            "judge_gpu_energy_joules": total_judge_gpu_energy,
            "total_task_gpu_energy_joules": total_task_gpu_energy,
            "cpu_energy_joules": total_cpu_energy,
            "judge_cpu_energy_joules": total_judge_cpu_energy,
            # Compatibility-only LM/API partial. Rigorous IPD must consume the
            # additive complete field and its causal coverage metadata below.
            "cost_usd": legacy_total_cost,
            "cost_usd_compatibility_status": _LEGACY_COST_COMPATIBILITY_STATUS,
            "complete_cost_usd": complete_total_cost,
            "legacy_lm_only_cost_usd": legacy_total_cost,
            "cost_known_queries": cost_known_queries,
            "cost_unknown_queries": cost_unknown_queries,
            "cost_coverage": cost_coverage,
            "accuracy": accuracy,
        },
        "efficiency": efficiency,
        "averages": {
            "turns_per_query": avg_turns,
            "wall_clock_per_query_s": avg_wall_clock,
            "gpu_energy_per_query_joules": avg_gpu_energy,
        },
        "statistics": stats,
        "normalized_statistics": normalized_statistics,
        "normalized_efficiency": normalized_efficiency,
    }

    # Include benchmark-level telemetry metrics that aren't per-query.
    if be:
        summary["bench_telemetry"] = {k: v for k, v in be.items() if v is not None}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def export_artifacts_manifest(run_dir: Path) -> Optional[Path]:
    """Scan ``{run_dir}/artifacts/`` and write ``artifacts_manifest.json``.

    The manifest lists every per-query artifact directory together with
    the files it contains, making it easy for downstream tools to discover
    what was produced without walking the directory tree themselves.

    Returns:
        The manifest path. If there is no artifacts directory, an empty
        manifest is still written so every run has the same output contract.
    """
    artifacts_root = run_dir / "artifacts"

    entries: list[dict[str, object]] = []
    if artifacts_root.is_dir():
        for query_dir in sorted(artifacts_root.iterdir()):
            if not query_dir.is_dir():
                continue
            files = sorted(str(p.relative_to(artifacts_root)) for p in query_dir.rglob("*") if p.is_file())
            entries.append({"query_dir": query_dir.name, "files": files})

    manifest_path = run_dir / "artifacts_manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return manifest_path


__all__ = [
    "export_jsonl",
    "export_hf_dataset",
    "export_summary_json",
    "export_public_trace_artifact",
    "export_artifacts_manifest",
]
