"""Tests for telemetry/events.py — EventRecorder and AgentEvent."""

from __future__ import annotations

import threading
import time

from ipw.telemetry.events import AgentEvent, EventRecorder, EventType


class TestEventType:
    """Test EventType enum values."""

    def test_lm_inference_events_exist(self) -> None:
        assert EventType.LM_INFERENCE_START.value == "lm_inference_start"
        assert EventType.LM_INFERENCE_END.value == "lm_inference_end"

    def test_tool_call_events_exist(self) -> None:
        assert EventType.TOOL_CALL_START.value == "tool_call_start"
        assert EventType.TOOL_CALL_END.value == "tool_call_end"

    def test_prefill_decode_events_exist(self) -> None:
        assert EventType.PREFILL_START.value == "prefill_start"
        assert EventType.PREFILL_END.value == "prefill_end"
        assert EventType.DECODE_START.value == "decode_start"
        assert EventType.DECODE_END.value == "decode_end"

    def test_submodel_events_exist(self) -> None:
        assert EventType.SUBMODEL_CALL_START.value == "submodel_call_start"
        assert EventType.SUBMODEL_CALL_END.value == "submodel_call_end"

    def test_is_string_enum(self) -> None:
        assert isinstance(EventType.LM_INFERENCE_START, str)


class TestAgentEvent:
    """Test AgentEvent dataclass."""

    def test_creation(self) -> None:
        event = AgentEvent(
            event_type="tool_call_start",
            timestamp=1234567890.0,
            metadata={"tool": "calculator"},
        )
        assert event.event_type == "tool_call_start"
        assert event.timestamp == 1234567890.0
        assert event.metadata == {"tool": "calculator"}

    def test_default_metadata(self) -> None:
        event = AgentEvent(event_type="test", timestamp=0.0)
        assert event.metadata == {}

    def test_repr(self) -> None:
        event = AgentEvent(event_type="test", timestamp=1.0, metadata={"k": "v"})
        r = repr(event)
        assert "test" in r
        assert "1.000" in r


class TestEventRecorder:
    """Test EventRecorder functionality."""

    def test_record_creates_event_with_timestamp(self) -> None:
        recorder = EventRecorder()
        before = time.time()
        recorder.record("test_event", key="value")
        after = time.time()

        events = recorder.get_events()
        assert len(events) == 1
        assert events[0].event_type == "test_event"
        assert events[0].metadata == {"key": "value"}
        assert before <= events[0].timestamp <= after

    def test_get_events_returns_copy(self) -> None:
        recorder = EventRecorder()
        recorder.record("event1")
        events = recorder.get_events()
        events.append(AgentEvent(event_type="fake", timestamp=0.0))
        assert len(recorder.get_events()) == 1

    def test_clear_empties_events(self) -> None:
        recorder = EventRecorder()
        recorder.record("event1")
        recorder.record("event2")
        assert len(recorder) == 2

        recorder.clear()
        assert len(recorder) == 0
        assert recorder.get_events() == []

    def test_len(self) -> None:
        recorder = EventRecorder()
        assert len(recorder) == 0
        recorder.record("a")
        assert len(recorder) == 1
        recorder.record("b")
        assert len(recorder) == 2

    def test_thread_safety_concurrent_recording(self) -> None:
        recorder = EventRecorder()
        num_threads = 10
        events_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(events_per_thread):
                recorder.record(f"thread_{thread_id}", index=i)

        threads = [
            threading.Thread(target=worker, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = recorder.get_events()
        assert len(events) == num_threads * events_per_thread

    def test_events_maintain_chronological_order(self) -> None:
        recorder = EventRecorder()
        for i in range(10):
            recorder.record(f"event_{i}")

        events = recorder.get_events()
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_metadata_storage(self) -> None:
        recorder = EventRecorder()
        recorder.record(
            "tool_call_end",
            tool="calculator",
            result="42",
            latency_ms=5.0,
        )
        events = recorder.get_events()
        assert events[0].metadata["tool"] == "calculator"
        assert events[0].metadata["result"] == "42"
        assert events[0].metadata["latency_ms"] == 5.0


class TestEventRecorderFixture:
    """Test the sample_event_recorder fixture from conftest."""

    def test_has_pre_populated_events(self, sample_event_recorder: EventRecorder) -> None:
        events = sample_event_recorder.get_events()
        assert len(events) == 4
        assert events[0].event_type == "lm_inference_start"
        assert events[1].event_type == "tool_call_start"
        assert events[2].event_type == "tool_call_end"
        assert events[3].event_type == "lm_inference_end"
