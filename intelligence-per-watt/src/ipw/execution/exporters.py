"""Export functions for agent run traces and profiling records."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from ..execution.trace import QueryTrace


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


def _compute_stats(traces: list[QueryTrace]) -> dict[str, dict[str, Optional[float]]]:
    """Compute per-metric statistics for a list of traces."""
    stats: dict[str, dict[str, Optional[float]]] = {
        "wall_clock_s": _agg_stats([t.total_wall_clock_s for t in traces]),
        "gpu_energy_joules": _agg_stats([t.total_gpu_energy_joules for t in traces]),
        "cpu_energy_joules": _agg_stats([t.total_cpu_energy_joules for t in traces]),
        "gpu_power_watts": _agg_stats([t.avg_gpu_power_watts for t in traces]),
        "cpu_power_watts": _agg_stats([t.avg_cpu_power_watts for t in traces]),
        "input_tokens": _agg_stats([float(t.total_input_tokens) for t in traces]),
        "output_tokens": _agg_stats([float(t.total_output_tokens) for t in traces]),
        "total_tokens": _agg_stats([float(t.total_tokens) for t in traces]),
        "throughput_tokens_per_sec": _agg_stats([t.throughput_tokens_per_sec for t in traces]),
        "energy_per_token_joules": _agg_stats([t.energy_per_token_joules for t in traces]),
        "cost_usd": _agg_stats([t.total_cost_usd for t in traces]),
        "turns": _agg_stats([float(t.num_turns) for t in traces]),
        "tool_calls": _agg_stats([float(t.total_tool_calls) for t in traces]),
        "mbu_avg_pct": _agg_stats([t.query_mbu_avg_pct for t in traces]),
    }
    return stats


def _compute_efficiency(
    traces: list[QueryTrace],
    stats: dict[str, dict[str, Optional[float]]],
) -> dict[str, Optional[float]]:
    """Compute accuracy, IPJ, and IPW from traces and their statistics."""
    resolved_count = sum(1 for t in traces if t.is_resolved is True)
    unresolved_count = sum(1 for t in traces if t.is_resolved is False)
    scored_count = resolved_count + unresolved_count
    accuracy = resolved_count / scored_count if scored_count > 0 else None

    gpu_energy_values = [
        t.total_gpu_energy_joules for t in traces
        if t.total_gpu_energy_joules is not None
    ]
    total_gpu_energy = sum(gpu_energy_values) if gpu_energy_values else None

    cpu_energy_values: list[float] = []
    for trace in traces:
        cpu_vals = [
            turn.cpu_energy_joules for turn in trace.turns
            if turn.cpu_energy_joules is not None
        ]
        if cpu_vals:
            cpu_energy_values.append(sum(cpu_vals))
    total_cpu_energy = sum(cpu_energy_values) if cpu_energy_values else None

    avg_gpu_power = stats["gpu_power_watts"]["avg"]
    avg_cpu_power = stats["cpu_power_watts"]["avg"]

    ipj = (
        accuracy / total_gpu_energy
        if (accuracy and accuracy > 0 and total_gpu_energy and total_gpu_energy > 0)
        else None
    )
    ipw = (
        accuracy / avg_gpu_power
        if (accuracy and accuracy > 0 and avg_gpu_power and avg_gpu_power > 0)
        else None
    )

    return {
        "accuracy": accuracy,
        "total_gpu_energy_joules": total_gpu_energy,
        "total_cpu_energy_joules": total_cpu_energy,
        "avg_gpu_power_watts": avg_gpu_power,
        "avg_cpu_power_watts": avg_cpu_power,
        "ipj": ipj,
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

    # Aggregate tokens
    total_input_tokens = sum(t.total_input_tokens for t in traces)
    total_output_tokens = sum(t.total_output_tokens for t in traces)

    # Aggregate wall clock
    total_wall_clock_s = sum(t.total_wall_clock_s for t in traces)

    # Aggregate energy
    gpu_energy_values = [
        t.total_gpu_energy_joules for t in traces
        if t.total_gpu_energy_joules is not None
    ]
    total_gpu_energy = sum(gpu_energy_values) if gpu_energy_values else None

    cpu_energy_values: list[float] = []
    for trace in traces:
        cpu_vals = [
            turn.cpu_energy_joules for turn in trace.turns
            if turn.cpu_energy_joules is not None
        ]
        if cpu_vals:
            cpu_energy_values.append(sum(cpu_vals))
    total_cpu_energy = sum(cpu_energy_values) if cpu_energy_values else None

    # Aggregate resolved/unresolved
    resolved = sum(1 for t in traces if t.is_resolved is True)
    unresolved = sum(1 for t in traces if t.is_resolved is False)

    # Aggregate cost
    cost_values = [
        t.total_cost_usd for t in traces
        if t.total_cost_usd is not None
    ]
    total_cost = sum(cost_values) if cost_values else None

    # Per-query averages
    avg_turns = total_turns / total_queries if total_queries > 0 else 0
    avg_wall_clock = total_wall_clock_s / total_queries if total_queries > 0 else 0
    avg_gpu_energy = (
        total_gpu_energy / total_queries
        if total_gpu_energy is not None and total_queries > 0
        else None
    )

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
        trimmed = sorted_traces[trim_count: n - trim_count]
        outlier_meta = {
            "top_pct": 5,
            "bottom_pct": 5,
            "queries_before": n,
            "queries_after": len(trimmed),
        }

        norm_stats = _compute_stats(trimmed)
        norm_stats["_description"] = (
            "Statistics recomputed after removing the top 5% and bottom 5% "
            "of queries by wall_clock_s"
        )
        norm_stats["_outliers_removed"] = outlier_meta
        normalized_statistics = norm_stats

        norm_eff = _compute_efficiency(trimmed, norm_stats)
        norm_eff["_description"] = (
            "Efficiency recomputed on the trimmed query set"
        )
        norm_eff["_outliers_removed"] = outlier_meta
        normalized_efficiency = norm_eff

    # When per-query energy is unavailable but benchmark-level energy
    # was collected, use the aggregate values for totals and efficiency.
    be = bench_energy or {}
    if total_gpu_energy is None and be.get("gpu_energy_joules") is not None:
        total_gpu_energy = be["gpu_energy_joules"]
        avg_gpu_energy = total_gpu_energy / total_queries if total_queries > 0 else None
    if total_cpu_energy is None and be.get("cpu_energy_joules") is not None:
        total_cpu_energy = be["cpu_energy_joules"]

    bench_avg_gpu_power = be.get("avg_gpu_power_watts")
    bench_avg_cpu_power = be.get("avg_cpu_power_watts")

    # Recompute efficiency with any newly available energy data.
    if be and efficiency.get("total_gpu_energy_joules") is None and total_gpu_energy is not None:
        avg_gpu_power = bench_avg_gpu_power or efficiency.get("avg_gpu_power_watts")
        ipj = (
            accuracy / total_gpu_energy
            if (accuracy and accuracy > 0 and total_gpu_energy > 0)
            else None
        )
        ipw = (
            accuracy / avg_gpu_power
            if (accuracy and accuracy > 0 and avg_gpu_power and avg_gpu_power > 0)
            else None
        )
        efficiency = {
            "accuracy": accuracy,
            "total_gpu_energy_joules": total_gpu_energy,
            "total_cpu_energy_joules": total_cpu_energy,
            "avg_gpu_power_watts": avg_gpu_power,
            "avg_cpu_power_watts": bench_avg_cpu_power or efficiency.get("avg_cpu_power_watts"),
            "ipj": ipj,
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
            "turns": total_turns,
            "tool_calls": total_tool_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "wall_clock_s": total_wall_clock_s,
            "gpu_energy_joules": total_gpu_energy,
            "cpu_energy_joules": total_cpu_energy,
            "cost_usd": total_cost,
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
        summary["bench_telemetry"] = {
            k: v for k, v in be.items() if v is not None
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def export_artifacts_manifest(run_dir: Path) -> Optional[Path]:
    """Scan ``{run_dir}/artifacts/`` and write ``artifacts_manifest.json``.

    The manifest lists every per-query artifact directory together with
    the files it contains, making it easy for downstream tools to discover
    what was produced without walking the directory tree themselves.

    Returns:
        The manifest path, or ``None`` if there is no artifacts directory.
    """
    artifacts_root = run_dir / "artifacts"
    if not artifacts_root.is_dir():
        return None

    entries: list[dict[str, object]] = []
    for query_dir in sorted(artifacts_root.iterdir()):
        if not query_dir.is_dir():
            continue
        files = sorted(
            str(p.relative_to(artifacts_root))
            for p in query_dir.rglob("*")
            if p.is_file()
        )
        entries.append({"query_dir": query_dir.name, "files": files})

    manifest_path = run_dir / "artifacts_manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return manifest_path


__all__ = [
    "export_jsonl",
    "export_hf_dataset",
    "export_summary_json",
    "export_artifacts_manifest",
]
