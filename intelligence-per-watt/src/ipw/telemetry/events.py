"""Event recording system for agent execution telemetry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .eventbus import EventBus


class EventType(str, Enum):
    """Supported event types for agent telemetry."""

    LM_INFERENCE_START = "lm_inference_start"
    LM_INFERENCE_END = "lm_inference_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    PREFILL_START = "prefill_start"
    PREFILL_END = "prefill_end"
    DECODE_START = "decode_start"
    DECODE_END = "decode_end"
    SUBMODEL_CALL_START = "submodel_call_start"
    SUBMODEL_CALL_END = "submodel_call_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    RETRY_ATTEMPT = "retry_attempt"
    ERROR_CLASSIFIED = "error_classified"
    ENERGY_ATTRIBUTED = "energy_attributed"
    TRAJECTORY_END = "trajectory_end"  # point event after AGENT_END + accuracy scoring; trajectory start is implicit at AGENT_START


@dataclass
class AgentEvent:
    """Single event recorded during agent execution."""

    event_type: str
    timestamp: float  # Unix timestamp from time.time()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"AgentEvent({self.event_type!r}, ts={self.timestamp:.3f}, meta={self.metadata})"


class EventRecorder:
    """Thread-safe event recorder. Publishes to an internal EventBus for subscriber fan-out.

    Bus publishing is a side effect of record(); it does not block or mutate list semantics.
    Event-type strings not in EventType are recorded to the list but not published; this
    preserves agent code that uses freeform event-type strings. Metadata keys ``turn_id``
    and ``correlation_id`` are promoted to first-class fields on the published Event.
    """

    def __init__(self, bus: Optional["EventBus"] = None) -> None:
        self._events: List[AgentEvent] = []
        self._lock = threading.Lock()
        if bus is None:
            from .eventbus import EventBus
            bus = EventBus()
        self._bus = bus

    @property
    def bus(self) -> "EventBus":
        return self._bus

    def record(self, event_type: str, **metadata: Any) -> None:
        ts = time.time()
        agent_event = AgentEvent(event_type=event_type, timestamp=ts, metadata=metadata)
        with self._lock:
            self._events.append(agent_event)
        try:
            bus_type = EventType(event_type)
        except ValueError:
            return
        from .eventbus import Event
        self._bus.publish(Event(
            event_type=bus_type,
            timestamp_ns=int(ts * 1_000_000_000),
            payload=dict(metadata),
            turn_id=metadata.get("turn_id"),
            correlation_id=metadata.get("correlation_id"),
        ))

    def get_events(self) -> List[AgentEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


__all__ = ["AgentEvent", "EventRecorder", "EventType"]
