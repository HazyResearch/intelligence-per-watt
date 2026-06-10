"""Tests for AgenticRunner EventBus + EnergyAttribution shadow wiring."""

from __future__ import annotations

from typing import List

import pytest

from ipw.telemetry.eventbus import Event, EventBus
from ipw.telemetry.events import EventRecorder, EventType


class TestAgenticRunnerBusWiring:
    def test_recorder_exposes_eventbus(self) -> None:
        recorder = EventRecorder()
        assert isinstance(recorder.bus, EventBus)

    def test_recorded_events_flow_to_bus(self) -> None:
        recorder = EventRecorder()
        got: List[Event] = []
        recorder.bus.subscribe(EventType.TOOL_CALL_START, got.append)
        recorder.record("tool_call_start", tool="calculator")
        assert len(got) == 1
        assert got[0].payload["tool"] == "calculator"

    def test_shadow_mode_does_not_alter_get_events(self) -> None:
        """Contract: recorder.get_events() returns the same AgentEvent
        list as before the bus was added."""
        recorder = EventRecorder()
        recorder.record("tool_call_start", tool="calculator")
        recorder.record("tool_call_end", tool="calculator")
        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "tool_call_start"
        assert events[1].event_type == "tool_call_end"

    def test_shadow_energy_attribution_does_not_consume_agent_events(self) -> None:
        """Attaching EnergyAttribution must not consume or alter AgentEvents seen
        by the existing correlate_energy_to_events path."""
        from ipw.telemetry.energy_attribution import EnergyAttribution

        class _Sess:
            def window(self, a, b): return iter([])

        recorder = EventRecorder()
        EnergyAttribution(bus=recorder.bus, session=_Sess(), is_cloud_fn=lambda e: False)
        recorder.record("tool_call_start", tool="x", correlation_id="c1")
        recorder.record("tool_call_end", tool="x", correlation_id="c1")

        events = recorder.get_events()
        # AgentEvent list unchanged by the subscriber
        assert len(events) == 2
        assert events[0].event_type == "tool_call_start"
        assert events[1].event_type == "tool_call_end"

    def test_runner_attaches_energy_attribution_when_telemetry_present(self) -> None:
        """The runner attaches an EnergyAttribution to the recorder's bus when a
        telemetry session is available."""
        from ipw.execution.agentic_runner import AgenticRunner

        # Build a fake session with a window() method
        class _Sess:
            def window(self, a, b): return iter([])

        runner = AgenticRunner.__new__(AgenticRunner)
        runner._event_recorder = EventRecorder()
        runner._telemetry = _Sess()
        runner._preflight_baseline_dirty = False
        runner._energy_attribution = None  # will be set by _attach_energy_attribution

        # Method should exist and create the subscriber
        runner._attach_energy_attribution()
        assert runner._energy_attribution is not None

    def test_runner_skips_energy_attribution_when_telemetry_missing(self) -> None:
        """If telemetry_session is None, runner should NOT attach an attribution
        subscriber (would have no samples to query)."""
        from ipw.execution.agentic_runner import AgenticRunner

        runner = AgenticRunner.__new__(AgenticRunner)
        runner._event_recorder = EventRecorder()
        runner._telemetry = None
        runner._preflight_baseline_dirty = False
        runner._energy_attribution = None

        runner._attach_energy_attribution()
        assert runner._energy_attribution is None

    def test_runner_attribution_emits_energy_attributed_events(self) -> None:
        """End-to-end: runner attaches subscriber, recorded START/END pair triggers
        ENERGY_ATTRIBUTED event publication."""
        from dataclasses import dataclass

        from ipw.execution.agentic_runner import AgenticRunner

        @dataclass
        class _Reading:
            energy_joules: float = 0.0
            cpu_energy_joules: float = 0.0

        @dataclass
        class _Sample:
            timestamp: float
            reading: _Reading

        class _Sess:
            def __init__(self, samples): self._samples = samples
            def window(self, a, b):
                return iter([s for s in self._samples if a <= s.timestamp <= b])

        samples = [
            _Sample(timestamp=10.0, reading=_Reading(energy_joules=0.0, cpu_energy_joules=0.0)),
            _Sample(timestamp=11.0, reading=_Reading(energy_joules=5.0, cpu_energy_joules=1.0)),
        ]

        runner = AgenticRunner.__new__(AgenticRunner)
        runner._event_recorder = EventRecorder()
        runner._telemetry = _Sess(samples)
        runner._preflight_baseline_dirty = False
        runner._energy_attribution = None
        runner._attach_energy_attribution()

        attributed: List[Event] = []
        runner._event_recorder.bus.subscribe(EventType.ENERGY_ATTRIBUTED, attributed.append)

        # Simulate a tool call window
        # Use deterministic timestamps that fall within the sample range
        # by recording with explicit metadata; recorder uses time.time() so we can't
        # control it directly. Instead, publish events directly to the bus to test
        # the attribution path with controlled timestamps:
        runner._event_recorder.bus.publish(Event(
            event_type=EventType.TOOL_CALL_START,
            timestamp_ns=10 * 1_000_000_000,
            correlation_id="c1",
            payload={"tool": "calc"},
        ))
        runner._event_recorder.bus.publish(Event(
            event_type=EventType.TOOL_CALL_END,
            timestamp_ns=11 * 1_000_000_000,
            correlation_id="c1",
            payload={"tool": "calc"},
        ))

        assert len(attributed) == 1
        assert attributed[0].payload["window"] == "tool"
        assert attributed[0].payload["gpu_energy_j"] == pytest.approx(5.0)
        assert attributed[0].payload["shared_device_warning"] is False

    def test_runner_attribution_propagates_preflight_dirty_flag(self) -> None:
        """Attribution events must reflect the runner's _preflight_baseline_dirty."""
        from dataclasses import dataclass

        from ipw.execution.agentic_runner import AgenticRunner

        @dataclass
        class _Reading:
            energy_joules: float = 0.0
            cpu_energy_joules: float = 0.0

        @dataclass
        class _Sample:
            timestamp: float
            reading: _Reading

        class _Sess:
            def __init__(self, samples): self._samples = samples
            def window(self, a, b):
                return iter([s for s in self._samples if a <= s.timestamp <= b])

        samples = [
            _Sample(timestamp=10.0, reading=_Reading()),
            _Sample(timestamp=11.0, reading=_Reading(energy_joules=1.0)),
        ]

        runner = AgenticRunner.__new__(AgenticRunner)
        runner._event_recorder = EventRecorder()
        runner._telemetry = _Sess(samples)
        runner._preflight_baseline_dirty = True  # contaminated baseline
        runner._energy_attribution = None
        runner._attach_energy_attribution()

        attributed: List[Event] = []
        runner._event_recorder.bus.subscribe(EventType.ENERGY_ATTRIBUTED, attributed.append)

        runner._event_recorder.bus.publish(Event(
            event_type=EventType.TOOL_CALL_START, timestamp_ns=10 * 1_000_000_000,
            correlation_id="x", payload={"tool": "t"},
        ))
        runner._event_recorder.bus.publish(Event(
            event_type=EventType.TOOL_CALL_END, timestamp_ns=11 * 1_000_000_000,
            correlation_id="x", payload={"tool": "t"},
        ))

        assert attributed[0].payload["shared_device_warning"] is True
