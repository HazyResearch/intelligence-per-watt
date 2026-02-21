"""Export functions for agent run traces and profiling records."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from ..execution.trace import QueryTrace


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

    cpu_energy_values = []
    for trace in traces:
        cpu_vals = [
            turn.cpu_energy_joules for turn in trace.turns
            if turn.cpu_energy_joules is not None
        ]
        if cpu_vals:
            cpu_energy_values.append(sum(cpu_vals))
    total_cpu_energy = sum(cpu_energy_values) if cpu_energy_values else None

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

    summary = {
        "generated_at": time.time(),
        "config": config,
        "totals": {
            "queries": total_queries,
            "completed": completed,
            "turns": total_turns,
            "tool_calls": total_tool_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "wall_clock_s": total_wall_clock_s,
            "gpu_energy_joules": total_gpu_energy,
            "cpu_energy_joules": total_cpu_energy,
            "cost_usd": total_cost,
        },
        "averages": {
            "turns_per_query": avg_turns,
            "wall_clock_per_query_s": avg_wall_clock,
            "gpu_energy_per_query_joules": avg_gpu_energy,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


__all__ = ["export_jsonl", "export_hf_dataset", "export_summary_json"]
