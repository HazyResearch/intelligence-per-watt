"""In-process pub/sub event bus for agent telemetry.

Synchronous delivery with per-subscriber isolation — a failing subscriber does
not block or break other subscribers. Thread-safe for concurrent publish.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .events import EventType

_log = logging.getLogger(__name__)


@dataclass
class Event:
    """One event on the bus."""

    event_type: EventType
    timestamp_ns: int
    payload: Dict[str, Any] = field(default_factory=dict)
    turn_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class _Subscription:
    handle: str
    event_type: Optional[EventType]  # None = wildcard (all types)
    callback: Callable[[Event], None]


class EventBus:
    """Synchronous in-process pub/sub bus.

    Subscribers register a callback for a specific EventType (or None for all).
    publish() invokes matching callbacks inline; any callback raising an
    exception is logged and skipped — other subscribers still receive the event.
    """

    def __init__(self) -> None:
        self._subs: List[_Subscription] = []
        self._lock = threading.Lock()

    def subscribe(
        self,
        event_type: Optional[EventType],
        callback: Callable[[Event], None],
    ) -> str:
        """Register a callback. Returns a handle for later unsubscribe."""
        handle = str(uuid.uuid4())
        with self._lock:
            self._subs.append(
                _Subscription(handle=handle, event_type=event_type, callback=callback)
            )
        return handle

    def unsubscribe(self, handle: str) -> None:
        """Remove the subscription with the given handle. No-op if unknown."""
        with self._lock:
            self._subs = [s for s in self._subs if s.handle != handle]

    def publish(self, event: Event) -> None:
        """Deliver event to all matching subscribers. Callback errors are logged and skipped."""
        with self._lock:
            targets = [
                s for s in self._subs
                if s.event_type is None or s.event_type == event.event_type
            ]
        for sub in targets:
            try:
                sub.callback(event)
            except Exception:
                _log.exception("EventBus subscriber %s raised on %s", sub.handle, event.event_type)


__all__ = ["Event", "EventBus"]
