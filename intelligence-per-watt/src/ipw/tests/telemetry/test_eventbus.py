"""Tests for telemetry/eventbus.py — EventBus pub/sub."""

from __future__ import annotations

import threading
import time
from typing import List

from ipw.telemetry.eventbus import Event, EventBus
from ipw.telemetry.events import EventType


class TestEventDataclass:
    def test_creation(self) -> None:
        evt = Event(
            event_type=EventType.TURN_START,
            timestamp_ns=1_000_000_000,
            turn_id="t0",
            correlation_id="c0",
            payload={"k": "v"},
        )
        assert evt.event_type == EventType.TURN_START
        assert evt.timestamp_ns == 1_000_000_000
        assert evt.turn_id == "t0"
        assert evt.correlation_id == "c0"
        assert evt.payload == {"k": "v"}

    def test_optional_fields_default_none(self) -> None:
        evt = Event(event_type=EventType.AGENT_START, timestamp_ns=0, payload={})
        assert evt.turn_id is None
        assert evt.correlation_id is None


class TestEventBus:
    def test_subscribe_and_publish_delivers_event(self) -> None:
        bus = EventBus()
        received: List[Event] = []
        bus.subscribe(EventType.TURN_START, received.append)
        bus.publish(Event(event_type=EventType.TURN_START, timestamp_ns=1, payload={}))
        assert len(received) == 1
        assert received[0].event_type == EventType.TURN_START

    def test_subscriber_receives_only_subscribed_type(self) -> None:
        bus = EventBus()
        got: List[Event] = []
        bus.subscribe(EventType.TURN_START, got.append)
        bus.publish(Event(event_type=EventType.TURN_END, timestamp_ns=1, payload={}))
        bus.publish(Event(event_type=EventType.TURN_START, timestamp_ns=2, payload={}))
        assert len(got) == 1
        assert got[0].event_type == EventType.TURN_START

    def test_multiple_subscribers_all_receive(self) -> None:
        bus = EventBus()
        a: List[Event] = []
        b: List[Event] = []
        bus.subscribe(EventType.TOOL_CALL_START, a.append)
        bus.subscribe(EventType.TOOL_CALL_START, b.append)
        bus.publish(Event(event_type=EventType.TOOL_CALL_START, timestamp_ns=1, payload={}))
        assert len(a) == 1
        assert len(b) == 1

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = EventBus()
        got: List[Event] = []
        handle = bus.subscribe(EventType.TURN_START, got.append)
        bus.publish(Event(event_type=EventType.TURN_START, timestamp_ns=1, payload={}))
        bus.unsubscribe(handle)
        bus.publish(Event(event_type=EventType.TURN_START, timestamp_ns=2, payload={}))
        assert len(got) == 1

    def test_failing_subscriber_does_not_break_others(self) -> None:
        bus = EventBus()
        got: List[Event] = []

        def boom(_evt: Event) -> None:
            raise RuntimeError("intentional")

        bus.subscribe(EventType.TURN_START, boom)
        bus.subscribe(EventType.TURN_START, got.append)
        bus.publish(Event(event_type=EventType.TURN_START, timestamp_ns=1, payload={}))
        assert len(got) == 1

    def test_thread_safe_concurrent_publish(self) -> None:
        bus = EventBus()
        got: List[Event] = []
        lock = threading.Lock()

        def append_safe(evt: Event) -> None:
            with lock:
                got.append(evt)

        bus.subscribe(EventType.TURN_START, append_safe)

        def worker() -> None:
            for _ in range(100):
                bus.publish(Event(event_type=EventType.TURN_START, timestamp_ns=int(time.time_ns()), payload={}))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(got) == 400

    def test_subscribe_to_all_via_wildcard(self) -> None:
        bus = EventBus()
        got: List[Event] = []
        bus.subscribe(None, got.append)  # None means "all event types"
        bus.publish(Event(event_type=EventType.TURN_START, timestamp_ns=1, payload={}))
        bus.publish(Event(event_type=EventType.TURN_END, timestamp_ns=2, payload={}))
        assert len(got) == 2

    def test_unsubscribe_unknown_handle_is_noop(self) -> None:
        bus = EventBus()
        # Should not raise
        bus.unsubscribe("not-a-real-handle")
        # Bus is still usable
        got: List[Event] = []
        bus.subscribe(EventType.TURN_START, got.append)
        bus.publish(Event(event_type=EventType.TURN_START, timestamp_ns=1, payload={}))
        assert len(got) == 1
