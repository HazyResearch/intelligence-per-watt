"""Export functions for agent run traces and profiling records."""

from __future__ import annotations

import json
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
) -> Path:
    """Export aggregate summary as JSON.

    Args:
        traces: List of QueryTrace objects.
        config: Run configuration dictionary.
        path: Output file path.

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

    judge_gpu_energy_values = [
        t.judge_gpu_energy_joules for t in traces
        if t.judge_gpu_energy_joules is not None
    ]
    total_judge_gpu_energy = (
        sum(judge_gpu_energy_values) if judge_gpu_energy_values else None
    )

    total_task_gpu_energy_values = [
        t.total_task_gpu_energy_joules for t in traces
        if t.total_task_gpu_energy_joules is not None
    ]
    total_task_gpu_energy = (
        sum(total_task_gpu_energy_values) if total_task_gpu_energy_values else None
    )

    cpu_energy_values = []
    for trace in traces:
        cpu_vals = [
            turn.cpu_energy_joules for turn in trace.turns
            if turn.cpu_energy_joules is not None
        ]
        if cpu_vals:
            cpu_energy_values.append(sum(cpu_vals))
    total_cpu_energy = sum(cpu_energy_values) if cpu_energy_values else None

    judge_cpu_energy_values = [
        t.judge_cpu_energy_joules for t in traces
        if t.judge_cpu_energy_joules is not None
    ]
    total_judge_cpu_energy = (
        sum(judge_cpu_energy_values) if judge_cpu_energy_values else None
    )

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

    # Per-metric statistics
    stats = {
        "wall_clock_s": _agg_stats([t.total_wall_clock_s for t in traces]),
        "gpu_energy_joules": _agg_stats([t.total_gpu_energy_joules for t in traces]),
        "judge_gpu_energy_joules": _agg_stats([t.judge_gpu_energy_joules for t in traces]),
        "total_task_gpu_energy_joules": _agg_stats([t.total_task_gpu_energy_joules for t in traces]),
        "cpu_energy_joules": _agg_stats([t.total_cpu_energy_joules for t in traces]),
        "judge_cpu_energy_joules": _agg_stats([t.judge_cpu_energy_joules for t in traces]),
        "gpu_power_watts": _agg_stats([t.avg_gpu_power_watts for t in traces]),
        "judge_gpu_power_watts": _agg_stats([t.judge_gpu_power_avg_watts for t in traces]),
        "cpu_power_watts": _agg_stats([t.avg_cpu_power_watts for t in traces]),
        "judge_wall_clock_s": _agg_stats([t.judge_wall_clock_s for t in traces]),
        "input_tokens": _agg_stats([float(t.total_input_tokens) for t in traces]),
        "output_tokens": _agg_stats([float(t.total_output_tokens) for t in traces]),
        "total_tokens": _agg_stats([float(t.total_tokens) for t in traces]),
        "throughput_tokens_per_sec": _agg_stats([t.throughput_tokens_per_sec for t in traces]),
        "energy_per_token_joules": _agg_stats([t.energy_per_token_joules for t in traces]),
        "cost_usd": _agg_stats([t.total_cost_usd for t in traces]),
        "turns": _agg_stats([float(t.num_turns) for t in traces]),
        "tool_calls": _agg_stats([float(t.total_tool_calls) for t in traces]),
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
            "judge_gpu_energy_joules": total_judge_gpu_energy,
            "total_task_gpu_energy_joules": total_task_gpu_energy,
            "cpu_energy_joules": total_cpu_energy,
            "judge_cpu_energy_joules": total_judge_cpu_energy,
            "cost_usd": total_cost,
        },
        "averages": {
            "turns_per_query": avg_turns,
            "wall_clock_per_query_s": avg_wall_clock,
            "gpu_energy_per_query_joules": avg_gpu_energy,
        },
        "statistics": stats,
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
