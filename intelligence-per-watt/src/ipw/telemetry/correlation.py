"""Correlate energy samples with agent events for per-action energy attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .events import AgentEvent


@dataclass
class ActionEnergyBreakdown:
    """Energy breakdown for a single agent action."""

    action_type: str  # 'lm_inference', 'tool_call', etc.
    step_number: int
    gpu_energy_joules: float
    cpu_energy_joules: float
    total_energy_joules: float
    duration_ms: float
    max_power_watts: Optional[float] = None
    avg_power_watts: Optional[float] = None
    memory_bandwidth_gbps: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # GPU/CPU power split
    gpu_max_power_watts: Optional[float] = None
    gpu_avg_power_watts: Optional[float] = None
    cpu_max_power_watts: Optional[float] = None
    cpu_avg_power_watts: Optional[float] = None

    # GPU compute utilization
    gpu_compute_utilization_pct_avg: Optional[float] = None
    gpu_compute_utilization_pct_max: Optional[float] = None
    gpu_memory_bw_utilization_pct_avg: Optional[float] = None
    gpu_memory_bw_utilization_pct_max: Optional[float] = None
    gpu_tensor_core_utilization_pct_avg: Optional[float] = None
    gpu_tensor_core_utilization_pct_max: Optional[float] = None

    # Dollar cost (cloud API calls only; None for local models)
    cost_usd: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"ActionEnergyBreakdown({self.action_type!r}, step={self.step_number}, "
            f"total={self.total_energy_joules:.2f}J, duration={self.duration_ms:.1f}ms)"
        )


def correlate_energy_to_events(
    energy_samples: List[Any],
    agent_events: List[AgentEvent],
    include_idle: bool = False,
) -> List[ActionEnergyBreakdown]:
    """Correlate energy samples with agent events to compute per-action energy.

    Algorithm:
    1. Pair start/end events into time windows
    2. For each window, find nearest energy samples at start and end
    3. Compute energy delta (cumulative: end_energy - start_energy)
    4. Handle edge cases: missing samples, counter resets, None values

    Args:
        energy_samples: List of telemetry samples with timestamp and reading attributes.
        agent_events: List of AgentEvent objects from EventRecorder.
        include_idle: If True, add breakdown entries for idle periods between actions.

    Returns:
        List of ActionEnergyBreakdown objects, one per paired start/end event.
    """
    if not energy_samples or not agent_events:
        return []

    event_pairs = _pair_events(agent_events)

    breakdowns = []
    for step_num, (start_event, end_event) in enumerate(event_pairs):
        start_sample = _find_nearest_sample(energy_samples, start_event.timestamp)
        end_sample = _find_nearest_sample(energy_samples, end_event.timestamp)

        if start_sample is None or end_sample is None:
            continue

        gpu_delta = _safe_delta(
            getattr(end_sample.reading, "energy_joules", None),
            getattr(start_sample.reading, "energy_joules", None),
        )
        cpu_delta = _safe_delta(
            getattr(end_sample.reading, "cpu_energy_joules", None),
            getattr(start_sample.reading, "cpu_energy_joules", None),
        )

        window_samples = _get_samples_in_window(
            energy_samples, start_event.timestamp, end_event.timestamp
        )
        gpu_max_p, gpu_avg_p, cpu_max_p, cpu_avg_p, combined_max, combined_avg = (
            _compute_power_metrics(window_samples)
        )
        (
            compute_avg, compute_max,
            membw_avg, membw_max,
            tensor_avg, tensor_max,
        ) = _compute_utilization_metrics(window_samples)
        memory_bw = _compute_memory_bandwidth(window_samples)

        action_type = start_event.event_type.replace("_start", "").replace("_end", "")

        merged_metadata = {
            **start_event.metadata,
            **end_event.metadata,
            "start_timestamp": start_event.timestamp,
            "end_timestamp": end_event.timestamp,
        }
        cost_usd = end_event.metadata.get("cost_usd")

        duration_ms = (end_event.timestamp - start_event.timestamp) * 1000
        breakdowns.append(
            ActionEnergyBreakdown(
                action_type=action_type,
                step_number=step_num,
                gpu_energy_joules=gpu_delta,
                cpu_energy_joules=cpu_delta,
                total_energy_joules=gpu_delta + cpu_delta,
                duration_ms=duration_ms,
                max_power_watts=combined_max,
                avg_power_watts=combined_avg,
                memory_bandwidth_gbps=memory_bw,
                metadata=merged_metadata,
                gpu_max_power_watts=gpu_max_p,
                gpu_avg_power_watts=gpu_avg_p,
                cpu_max_power_watts=cpu_max_p,
                cpu_avg_power_watts=cpu_avg_p,
                gpu_compute_utilization_pct_avg=compute_avg,
                gpu_compute_utilization_pct_max=compute_max,
                gpu_memory_bw_utilization_pct_avg=membw_avg,
                gpu_memory_bw_utilization_pct_max=membw_max,
                gpu_tensor_core_utilization_pct_avg=tensor_avg,
                gpu_tensor_core_utilization_pct_max=tensor_max,
                cost_usd=cost_usd,
            )
        )

    if include_idle:
        idle_breakdowns = _compute_idle_periods(breakdowns, energy_samples)
        breakdowns.extend(idle_breakdowns)
        breakdowns.sort(key=lambda b: b.metadata.get("start_timestamp", 0))

    return breakdowns


def compute_analysis(breakdowns: List[ActionEnergyBreakdown]) -> Dict[str, Any]:
    """Compute aggregate analysis from energy breakdowns."""
    if not breakdowns:
        return {
            "total_energy_joules": 0.0,
            "total_gpu_energy_joules": 0.0,
            "total_cpu_energy_joules": 0.0,
            "total_duration_ms": 0.0,
            "max_power_watts": None,
            "avg_power_watts": None,
            "avg_memory_bandwidth_gbps": None,
            "action_counts": {},
            "energy_by_action": {},
            "model_counts": {},
            "energy_by_model": {},
            "latency_by_model": {},
            "tokens_by_model": {},
            "gpu_max_power_watts": None,
            "gpu_avg_power_watts": None,
            "cpu_max_power_watts": None,
            "cpu_avg_power_watts": None,
            "gpu_compute_utilization_pct_avg": None,
            "gpu_compute_utilization_pct_max": None,
            "gpu_memory_bw_utilization_pct_avg": None,
            "gpu_memory_bw_utilization_pct_max": None,
            "gpu_tensor_core_utilization_pct_avg": None,
            "gpu_tensor_core_utilization_pct_max": None,
            "total_cost_usd": 0.0,
            "cost_by_model": {},
        }

    action_counts: Dict[str, int] = {}
    energy_by_action: Dict[str, float] = {}
    model_counts: Dict[str, int] = {}
    energy_by_model: Dict[str, float] = {}
    latency_by_model: Dict[str, float] = {}
    tokens_by_model: Dict[str, int] = {}

    max_power_values: List[float] = []
    avg_power_values: List[float] = []
    bandwidth_values: List[float] = []

    gpu_max_power_values: List[float] = []
    gpu_avg_power_values: List[float] = []
    cpu_max_power_values: List[float] = []
    cpu_avg_power_values: List[float] = []

    compute_avg_values: List[float] = []
    compute_max_values: List[float] = []
    membw_avg_values: List[float] = []
    membw_max_values: List[float] = []
    tensor_avg_values: List[float] = []
    tensor_max_values: List[float] = []

    cost_values: List[float] = []
    cost_by_model: Dict[str, float] = {}

    for b in breakdowns:
        action_counts[b.action_type] = action_counts.get(b.action_type, 0) + 1
        energy_by_action[b.action_type] = (
            energy_by_action.get(b.action_type, 0.0) + b.total_energy_joules
        )

        if b.max_power_watts is not None:
            max_power_values.append(b.max_power_watts)
        if b.avg_power_watts is not None:
            avg_power_values.append(b.avg_power_watts)
        if b.memory_bandwidth_gbps is not None:
            bandwidth_values.append(b.memory_bandwidth_gbps)

        if b.gpu_max_power_watts is not None:
            gpu_max_power_values.append(b.gpu_max_power_watts)
        if b.gpu_avg_power_watts is not None:
            gpu_avg_power_values.append(b.gpu_avg_power_watts)
        if b.cpu_max_power_watts is not None:
            cpu_max_power_values.append(b.cpu_max_power_watts)
        if b.cpu_avg_power_watts is not None:
            cpu_avg_power_values.append(b.cpu_avg_power_watts)

        if b.gpu_compute_utilization_pct_avg is not None:
            compute_avg_values.append(b.gpu_compute_utilization_pct_avg)
        if b.gpu_compute_utilization_pct_max is not None:
            compute_max_values.append(b.gpu_compute_utilization_pct_max)
        if b.gpu_memory_bw_utilization_pct_avg is not None:
            membw_avg_values.append(b.gpu_memory_bw_utilization_pct_avg)
        if b.gpu_memory_bw_utilization_pct_max is not None:
            membw_max_values.append(b.gpu_memory_bw_utilization_pct_max)
        if b.gpu_tensor_core_utilization_pct_avg is not None:
            tensor_avg_values.append(b.gpu_tensor_core_utilization_pct_avg)
        if b.gpu_tensor_core_utilization_pct_max is not None:
            tensor_max_values.append(b.gpu_tensor_core_utilization_pct_max)

        if b.cost_usd is not None:
            cost_values.append(b.cost_usd)
            model_id = b.metadata.get("model_id")
            if model_id:
                cost_by_model[model_id] = cost_by_model.get(model_id, 0.0) + b.cost_usd

        model_id = b.metadata.get("model_id")
        if model_id:
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
            energy_by_model[model_id] = (
                energy_by_model.get(model_id, 0.0) + b.total_energy_joules
            )
            latency_by_model[model_id] = (
                latency_by_model.get(model_id, 0.0) + b.duration_ms
            )
            total_tokens = b.metadata.get("total_tokens", 0)
            tokens_by_model[model_id] = (
                tokens_by_model.get(model_id, 0) + total_tokens
            )

    overall_max_power = max(max_power_values) if max_power_values else None
    overall_avg_power = _safe_mean(avg_power_values)
    overall_avg_bandwidth = _safe_mean(bandwidth_values)

    return {
        "total_energy_joules": sum(b.total_energy_joules for b in breakdowns),
        "total_gpu_energy_joules": sum(b.gpu_energy_joules for b in breakdowns),
        "total_cpu_energy_joules": sum(b.cpu_energy_joules for b in breakdowns),
        "total_duration_ms": sum(b.duration_ms for b in breakdowns),
        "max_power_watts": overall_max_power,
        "avg_power_watts": overall_avg_power,
        "avg_memory_bandwidth_gbps": overall_avg_bandwidth,
        "action_counts": action_counts,
        "energy_by_action": energy_by_action,
        "model_counts": model_counts,
        "energy_by_model": energy_by_model,
        "latency_by_model": latency_by_model,
        "tokens_by_model": tokens_by_model,
        "gpu_max_power_watts": max(gpu_max_power_values) if gpu_max_power_values else None,
        "gpu_avg_power_watts": _safe_mean(gpu_avg_power_values),
        "cpu_max_power_watts": max(cpu_max_power_values) if cpu_max_power_values else None,
        "cpu_avg_power_watts": _safe_mean(cpu_avg_power_values),
        "gpu_compute_utilization_pct_avg": _safe_mean(compute_avg_values),
        "gpu_compute_utilization_pct_max": max(compute_max_values) if compute_max_values else None,
        "gpu_memory_bw_utilization_pct_avg": _safe_mean(membw_avg_values),
        "gpu_memory_bw_utilization_pct_max": max(membw_max_values) if membw_max_values else None,
        "gpu_tensor_core_utilization_pct_avg": _safe_mean(tensor_avg_values),
        "gpu_tensor_core_utilization_pct_max": max(tensor_max_values) if tensor_max_values else None,
        "total_cost_usd": sum(cost_values) if cost_values else 0.0,
        "cost_by_model": cost_by_model,
    }


def _compute_idle_periods(
    action_breakdowns: List[ActionEnergyBreakdown],
    energy_samples: List[Any],
) -> List[ActionEnergyBreakdown]:
    """Compute energy during idle periods between actions."""
    if len(action_breakdowns) < 2:
        return []

    idle_breakdowns: List[ActionEnergyBreakdown] = []

    sorted_breakdowns = sorted(
        action_breakdowns,
        key=lambda b: b.metadata.get("end_timestamp", 0),
    )

    for i in range(len(sorted_breakdowns) - 1):
        current = sorted_breakdowns[i]
        next_action = sorted_breakdowns[i + 1]

        current_end = current.metadata.get("end_timestamp")
        next_start = next_action.metadata.get("start_timestamp")

        if current_end is None or next_start is None:
            continue

        if next_start > current_end:
            gap_start_sample = _find_nearest_sample(energy_samples, current_end)
            gap_end_sample = _find_nearest_sample(energy_samples, next_start)

            if gap_start_sample is None or gap_end_sample is None:
                continue

            gpu_delta = _safe_delta(
                getattr(gap_end_sample.reading, "energy_joules", None),
                getattr(gap_start_sample.reading, "energy_joules", None),
            )
            cpu_delta = _safe_delta(
                getattr(gap_end_sample.reading, "cpu_energy_joules", None),
                getattr(gap_start_sample.reading, "cpu_energy_joules", None),
            )

            idle_window = _get_samples_in_window(
                energy_samples, current_end, next_start
            )
            gpu_max_p, gpu_avg_p, cpu_max_p, cpu_avg_p, combined_max, combined_avg = (
                _compute_power_metrics(idle_window)
            )
            (
                compute_avg, compute_max,
                membw_avg, membw_max,
                tensor_avg, tensor_max,
            ) = _compute_utilization_metrics(idle_window)

            idle_step = current.step_number + 0.5

            idle_breakdowns.append(
                ActionEnergyBreakdown(
                    action_type="idle",
                    step_number=int(idle_step),
                    gpu_energy_joules=gpu_delta,
                    cpu_energy_joules=cpu_delta,
                    total_energy_joules=gpu_delta + cpu_delta,
                    duration_ms=(next_start - current_end) * 1000,
                    max_power_watts=combined_max,
                    avg_power_watts=combined_avg,
                    metadata={
                        "between_steps": [current.step_number, next_action.step_number],
                        "start_timestamp": current_end,
                        "end_timestamp": next_start,
                    },
                    gpu_max_power_watts=gpu_max_p,
                    gpu_avg_power_watts=gpu_avg_p,
                    cpu_max_power_watts=cpu_max_p,
                    cpu_avg_power_watts=cpu_avg_p,
                    gpu_compute_utilization_pct_avg=compute_avg,
                    gpu_compute_utilization_pct_max=compute_max,
                    gpu_memory_bw_utilization_pct_avg=membw_avg,
                    gpu_memory_bw_utilization_pct_max=membw_max,
                    gpu_tensor_core_utilization_pct_avg=tensor_avg,
                    gpu_tensor_core_utilization_pct_max=tensor_max,
                    cost_usd=None,
                )
            )

    return idle_breakdowns


def _pair_events(events: List[AgentEvent]) -> List[Tuple[AgentEvent, AgentEvent]]:
    """Pair start/end events by action type."""
    pairs: List[Tuple[AgentEvent, AgentEvent]] = []
    pending: Dict[str, AgentEvent] = {}

    for event in events:
        if event.event_type.endswith("_start"):
            base_type = event.event_type.replace("_start", "")
            pending[base_type] = event
        elif event.event_type.endswith("_end"):
            base_type = event.event_type.replace("_end", "")
            if base_type in pending:
                pairs.append((pending.pop(base_type), event))

    return pairs


def _find_nearest_sample(samples: List[Any], timestamp: float) -> Optional[Any]:
    """Find sample with timestamp nearest to given timestamp."""
    if not samples:
        return None
    return min(samples, key=lambda s: abs(s.timestamp - timestamp))


def _safe_delta(end_val: Optional[float], start_val: Optional[float]) -> float:
    """Compute delta handling None and counter resets."""
    if end_val is None:
        end_val = 0.0
    if start_val is None:
        start_val = 0.0

    delta = end_val - start_val
    if delta < 0:
        return end_val
    return delta


def _get_samples_in_window(
    samples: List[Any], start_time: float, end_time: float
) -> List[Any]:
    """Get all samples within a time window."""
    return [s for s in samples if start_time <= s.timestamp <= end_time]


def _safe_mean(values: List[float]) -> Optional[float]:
    """Compute mean of a list, returning None if empty."""
    return sum(values) / len(values) if values else None


def _compute_power_metrics(
    samples: List[Any],
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Compute max and average power from samples, split by GPU and CPU."""
    if not samples:
        return None, None, None, None, None, None

    gpu_values: List[float] = []
    cpu_values: List[float] = []
    combined_values: List[float] = []

    for s in samples:
        gpu_power = getattr(s.reading, "power_watts", None)
        cpu_power = getattr(s.reading, "cpu_power_watts", None)

        if gpu_power is not None and gpu_power >= 0:
            gpu_values.append(gpu_power)
        if cpu_power is not None and cpu_power >= 0:
            cpu_values.append(cpu_power)

        combined = 0.0
        has_any = False
        if gpu_power is not None and gpu_power >= 0:
            combined += gpu_power
            has_any = True
        if cpu_power is not None and cpu_power >= 0:
            combined += cpu_power
            has_any = True
        if has_any:
            combined_values.append(combined)

    gpu_max = max(gpu_values) if gpu_values else None
    gpu_avg = _safe_mean(gpu_values)
    cpu_max = max(cpu_values) if cpu_values else None
    cpu_avg = _safe_mean(cpu_values)
    combined_max = max(combined_values) if combined_values else None
    combined_avg = _safe_mean(combined_values)

    return gpu_max, gpu_avg, cpu_max, cpu_avg, combined_max, combined_avg


def _compute_utilization_metrics(
    samples: List[Any],
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Compute GPU utilization metrics from samples."""
    if not samples:
        return None, None, None, None, None, None

    compute_vals: List[float] = []
    membw_vals: List[float] = []
    tensor_vals: List[float] = []

    for s in samples:
        compute = getattr(s.reading, "gpu_compute_utilization_pct", None)
        membw = getattr(s.reading, "gpu_memory_bandwidth_utilization_pct", None)
        tensor = getattr(s.reading, "gpu_tensor_core_utilization_pct", None)

        if compute is not None and compute >= 0:
            compute_vals.append(compute)
        if membw is not None and membw >= 0:
            membw_vals.append(membw)
        if tensor is not None and tensor >= 0:
            tensor_vals.append(tensor)

    return (
        _safe_mean(compute_vals),
        max(compute_vals) if compute_vals else None,
        _safe_mean(membw_vals),
        max(membw_vals) if membw_vals else None,
        _safe_mean(tensor_vals),
        max(tensor_vals) if tensor_vals else None,
    )


def _compute_memory_bandwidth(samples: List[Any]) -> Optional[float]:
    """Compute GPU memory bandwidth from samples."""
    if len(samples) < 2:
        return None

    sorted_samples = sorted(samples, key=lambda s: s.timestamp)

    total_bytes_transferred = 0.0
    total_time_seconds = 0.0

    for i in range(1, len(sorted_samples)):
        prev_mem = getattr(sorted_samples[i - 1].reading, "gpu_memory_usage_mb", None)
        curr_mem = getattr(sorted_samples[i].reading, "gpu_memory_usage_mb", None)

        if prev_mem is None or curr_mem is None:
            continue

        mem_delta_mb = abs(curr_mem - prev_mem)
        time_delta = sorted_samples[i].timestamp - sorted_samples[i - 1].timestamp

        if time_delta > 0:
            total_bytes_transferred += mem_delta_mb * 1024 * 1024
            total_time_seconds += time_delta

    if total_time_seconds == 0:
        return None

    bandwidth_gbps = (total_bytes_transferred / total_time_seconds) / (1024**3)
    return bandwidth_gbps


__all__ = [
    "ActionEnergyBreakdown",
    "compute_analysis",
    "correlate_energy_to_events",
]
