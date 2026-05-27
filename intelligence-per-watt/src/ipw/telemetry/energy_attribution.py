"""Bus-driven per-window energy attribution.

Pairs *_START and *_END events by correlation_id, queries the TelemetrySession
for samples in the window, computes energy/power, publishes ENERGY_ATTRIBUTED.
Cloud LM inferences explicitly emit gpu_energy_j=None since no local GPU is involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Protocol

from .eventbus import Event, EventBus
from .events import EventType

_log = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Computed energy attribution for a single window."""

    window: str               # "tool" or "inference"
    subject_name: str         # tool name or model name
    gpu_energy_j: Optional[float]
    cpu_energy_j: Optional[float]
    dram_energy_j: Optional[float]
    avg_watts: Optional[float]
    peak_watts: Optional[float]
    duration_s: float
    gpu_util_pct_avg: Optional[float]
    shared_device_warning: bool


class _SessionProtocol(Protocol):
    def window(self, start_time: float, end_time: float) -> Iterable: ...


class EnergyAttribution:
    """Subscribes to tool/inference lifecycle events and emits per-window attributions."""

    def __init__(
        self,
        bus: EventBus,
        session: _SessionProtocol,
        *,
        is_cloud_fn: Callable[[Event], bool] = lambda evt: False,
        shared_device_warning_fn: Callable[[], bool] = lambda: False,
    ) -> None:
        self._bus = bus
        self._session = session
        self._is_cloud_fn = is_cloud_fn
        self._shared_device_warning_fn = shared_device_warning_fn
        self._open: Dict[str, Event] = {}

        bus.subscribe(EventType.TOOL_CALL_START, self._on_start)
        bus.subscribe(EventType.TOOL_CALL_END, self._on_end)
        bus.subscribe(EventType.LM_INFERENCE_START, self._on_start)
        bus.subscribe(EventType.LM_INFERENCE_END, self._on_end)

    def _on_start(self, evt: Event) -> None:
        cid = evt.correlation_id
        if cid is None:
            return
        self._open[cid] = evt

    def _on_end(self, evt: Event) -> None:
        cid = evt.correlation_id
        if cid is None or cid not in self._open:
            return
        start = self._open.pop(cid)
        result = self._compute(start, evt)
        if result is None:
            return
        self._bus.publish(Event(
            event_type=EventType.ENERGY_ATTRIBUTED,
            timestamp_ns=evt.timestamp_ns,
            correlation_id=cid,
            turn_id=evt.turn_id,
            payload={
                "window": result.window,
                "subject_name": result.subject_name,
                "gpu_energy_j": result.gpu_energy_j,
                "cpu_energy_j": result.cpu_energy_j,
                "dram_energy_j": result.dram_energy_j,
                "avg_watts": result.avg_watts,
                "peak_watts": result.peak_watts,
                "duration_s": result.duration_s,
                "gpu_util_pct_avg": result.gpu_util_pct_avg,
                "shared_device_warning": result.shared_device_warning,
            },
        ))

    def _compute(self, start: Event, end: Event) -> Optional[AttributionResult]:
        start_s = start.timestamp_ns / 1_000_000_000.0
        end_s = end.timestamp_ns / 1_000_000_000.0
        if end_s < start_s:
            return None

        samples = list(self._session.window(start_s, end_s))

        is_tool = start.event_type == EventType.TOOL_CALL_START
        window = "tool" if is_tool else "inference"
        subject = start.payload.get("tool", "unknown") if is_tool else start.payload.get("model", "unknown")

        is_cloud = (not is_tool) and self._is_cloud_fn(start)

        gpu_j = self._delta(samples, "energy_joules") if not is_cloud else None
        cpu_j = self._delta(samples, "cpu_energy_joules")
        dram_j = self._delta(samples, "dram_energy_joules")
        avg_w, peak_w = self._power_stats(samples)
        util_avg = self._util_avg(samples)

        return AttributionResult(
            window=window,
            subject_name=str(subject),
            gpu_energy_j=gpu_j,
            cpu_energy_j=cpu_j,
            dram_energy_j=dram_j,
            avg_watts=avg_w,
            peak_watts=peak_w,
            duration_s=end_s - start_s,
            gpu_util_pct_avg=util_avg,
            shared_device_warning=self._shared_device_warning_fn(),
        )

    @staticmethod
    def _delta(samples, attr: str) -> Optional[float]:
        vals = [getattr(s.reading, attr, None) for s in samples]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return None
        delta = vals[-1] - vals[0]
        return delta if delta >= 0 else None

    @staticmethod
    def _power_stats(samples):
        vals = [getattr(s.reading, "power_watts", None) for s in samples]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return sum(vals) / len(vals), max(vals)

    @staticmethod
    def _util_avg(samples) -> Optional[float]:
        vals = [getattr(s.reading, "gpu_utilization_pct", None) for s in samples]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)


__all__ = ["EnergyAttribution", "AttributionResult"]
