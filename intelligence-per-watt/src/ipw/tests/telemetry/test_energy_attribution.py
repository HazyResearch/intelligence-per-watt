"""Tests for telemetry/energy_attribution.py — bus-driven energy attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest

from ipw.telemetry.energy_attribution import AttributionResult, EnergyAttribution
from ipw.telemetry.eventbus import Event, EventBus
from ipw.telemetry.events import EventType


@dataclass
class _FakeReading:
    energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None
    dram_energy_joules: Optional[float] = None
    power_watts: Optional[float] = None
    cpu_power_watts: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    timestamp_nanos: int = 0


@dataclass
class _FakeSample:
    timestamp: float
    reading: _FakeReading


class _FakeSession:
    """Stub that yields samples within a time window."""

    def __init__(self, samples: List[_FakeSample]) -> None:
        self._samples = samples

    def window(self, start_time: float, end_time: float):
        return iter([s for s in self._samples if start_time <= s.timestamp <= end_time])


class TestEnergyAttribution:
    def test_emits_attributed_event_on_matching_tool_end(self) -> None:
        bus = EventBus()
        samples = [
            _FakeSample(timestamp=10.0, reading=_FakeReading(
                energy_joules=0.0, cpu_energy_joules=0.0, power_watts=100.0)),
            _FakeSample(timestamp=11.0, reading=_FakeReading(
                energy_joules=10.0, cpu_energy_joules=2.0, power_watts=110.0)),
        ]
        EnergyAttribution(bus=bus, session=_FakeSession(samples), is_cloud_fn=lambda evt: False)

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        bus.publish(Event(event_type=EventType.TOOL_CALL_START, timestamp_ns=10 * 1_000_000_000,
                          correlation_id="c1", payload={"tool": "calculator"}))
        bus.publish(Event(event_type=EventType.TOOL_CALL_END, timestamp_ns=11 * 1_000_000_000,
                          correlation_id="c1", payload={"tool": "calculator"}))

        assert len(got) == 1
        result = got[0].payload
        assert result["window"] == "tool"
        assert result["subject_name"] == "calculator"
        assert result["gpu_energy_j"] == pytest.approx(10.0)
        assert result["cpu_energy_j"] == pytest.approx(2.0)
        assert result["duration_s"] == pytest.approx(1.0)

    def test_emits_attributed_event_on_lm_inference_end(self) -> None:
        bus = EventBus()
        samples = [
            _FakeSample(timestamp=20.0, reading=_FakeReading(
                energy_joules=0.0, cpu_energy_joules=0.0, power_watts=200.0)),
            _FakeSample(timestamp=22.0, reading=_FakeReading(
                energy_joules=50.0, cpu_energy_joules=4.0, power_watts=300.0)),
        ]
        EnergyAttribution(bus=bus, session=_FakeSession(samples), is_cloud_fn=lambda evt: False)

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        bus.publish(Event(event_type=EventType.LM_INFERENCE_START, timestamp_ns=20 * 1_000_000_000,
                          correlation_id="lm1", payload={"model": "gpt-4o-mini", "is_cloud": False}))
        bus.publish(Event(event_type=EventType.LM_INFERENCE_END, timestamp_ns=22 * 1_000_000_000,
                          correlation_id="lm1", payload={"model": "gpt-4o-mini", "is_cloud": False}))

        assert len(got) == 1
        p = got[0].payload
        assert p["window"] == "inference"
        assert p["subject_name"] == "gpt-4o-mini"
        assert p["gpu_energy_j"] == pytest.approx(50.0)
        assert p["cpu_energy_j"] == pytest.approx(4.0)
        assert p["duration_s"] == pytest.approx(2.0)

    def test_cloud_lm_sets_gpu_energy_none(self) -> None:
        bus = EventBus()
        samples = [
            _FakeSample(timestamp=10.0, reading=_FakeReading(
                energy_joules=0.0, cpu_energy_joules=0.0)),
            _FakeSample(timestamp=11.0, reading=_FakeReading(
                energy_joules=10.0, cpu_energy_joules=2.0)),
        ]
        EnergyAttribution(bus=bus, session=_FakeSession(samples), is_cloud_fn=lambda evt: True)

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        bus.publish(Event(event_type=EventType.LM_INFERENCE_START, timestamp_ns=10 * 1_000_000_000,
                          correlation_id="c1", payload={"model": "gpt-4o", "is_cloud": True}))
        bus.publish(Event(event_type=EventType.LM_INFERENCE_END, timestamp_ns=11 * 1_000_000_000,
                          correlation_id="c1", payload={"model": "gpt-4o", "is_cloud": True}))

        assert len(got) == 1
        p = got[0].payload
        assert p["gpu_energy_j"] is None
        assert p["cpu_energy_j"] == pytest.approx(2.0)

    def test_cloud_check_does_not_apply_to_tool_calls(self) -> None:
        bus = EventBus()
        samples = [
            _FakeSample(timestamp=10.0, reading=_FakeReading(
                energy_joules=0.0, cpu_energy_joules=0.0)),
            _FakeSample(timestamp=11.0, reading=_FakeReading(
                energy_joules=5.0, cpu_energy_joules=1.0)),
        ]
        EnergyAttribution(bus=bus, session=_FakeSession(samples), is_cloud_fn=lambda evt: True)

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        bus.publish(Event(event_type=EventType.TOOL_CALL_START, timestamp_ns=10 * 1_000_000_000,
                          correlation_id="t1", payload={"tool": "shell"}))
        bus.publish(Event(event_type=EventType.TOOL_CALL_END, timestamp_ns=11 * 1_000_000_000,
                          correlation_id="t1", payload={"tool": "shell"}))

        # Tool calls always run on local hardware — gpu_energy should be populated
        assert got[0].payload["gpu_energy_j"] == pytest.approx(5.0)

    def test_mismatched_end_without_start_is_dropped(self) -> None:
        bus = EventBus()
        EnergyAttribution(bus=bus, session=_FakeSession([]), is_cloud_fn=lambda evt: False)

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        bus.publish(Event(event_type=EventType.TOOL_CALL_END, timestamp_ns=1, correlation_id="unknown", payload={}))

        assert got == []

    def test_event_with_no_correlation_id_is_dropped(self) -> None:
        bus = EventBus()
        EnergyAttribution(bus=bus, session=_FakeSession([]), is_cloud_fn=lambda evt: False)

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        # No correlation_id on either event
        bus.publish(Event(event_type=EventType.TOOL_CALL_START, timestamp_ns=1, payload={"tool": "x"}))
        bus.publish(Event(event_type=EventType.TOOL_CALL_END, timestamp_ns=2, payload={"tool": "x"}))

        assert got == []

    def test_subject_name_unknown_when_payload_missing(self) -> None:
        bus = EventBus()
        samples = [
            _FakeSample(timestamp=10.0, reading=_FakeReading(energy_joules=0.0, cpu_energy_joules=0.0)),
            _FakeSample(timestamp=11.0, reading=_FakeReading(energy_joules=1.0, cpu_energy_joules=0.5)),
        ]
        EnergyAttribution(bus=bus, session=_FakeSession(samples), is_cloud_fn=lambda evt: False)

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        bus.publish(Event(event_type=EventType.TOOL_CALL_START, timestamp_ns=10 * 1_000_000_000,
                          correlation_id="x", payload={}))
        bus.publish(Event(event_type=EventType.TOOL_CALL_END, timestamp_ns=11 * 1_000_000_000,
                          correlation_id="x", payload={}))

        assert got[0].payload["subject_name"] == "unknown"

    def test_shared_device_warning_propagates_from_callable(self) -> None:
        bus = EventBus()
        samples = [
            _FakeSample(timestamp=10.0, reading=_FakeReading(energy_joules=0.0, cpu_energy_joules=0.0)),
            _FakeSample(timestamp=11.0, reading=_FakeReading(energy_joules=1.0, cpu_energy_joules=0.5)),
        ]
        EnergyAttribution(
            bus=bus, session=_FakeSession(samples),
            is_cloud_fn=lambda evt: False,
            shared_device_warning_fn=lambda: True,
        )

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        bus.publish(Event(event_type=EventType.TOOL_CALL_START, timestamp_ns=10 * 1_000_000_000,
                          correlation_id="x", payload={"tool": "calc"}))
        bus.publish(Event(event_type=EventType.TOOL_CALL_END, timestamp_ns=11 * 1_000_000_000,
                          correlation_id="x", payload={"tool": "calc"}))

        assert got[0].payload["shared_device_warning"] is True

    def test_attribution_result_dataclass_fields(self) -> None:
        # Smoke test the dataclass exists with expected fields
        result = AttributionResult(
            window="tool", subject_name="calc",
            gpu_energy_j=1.0, cpu_energy_j=0.5, dram_energy_j=None,
            avg_watts=100.0, peak_watts=120.0,
            duration_s=1.0, gpu_util_pct_avg=None,
            shared_device_warning=False,
        )
        assert result.window == "tool"
        assert result.subject_name == "calc"
        assert result.gpu_energy_j == 1.0

    def test_turn_id_propagates_from_end_event(self) -> None:
        bus = EventBus()
        samples = [
            _FakeSample(timestamp=10.0, reading=_FakeReading(energy_joules=0.0, cpu_energy_joules=0.0)),
            _FakeSample(timestamp=11.0, reading=_FakeReading(energy_joules=1.0, cpu_energy_joules=0.5)),
        ]
        EnergyAttribution(bus=bus, session=_FakeSession(samples), is_cloud_fn=lambda evt: False)

        got: List[Event] = []
        bus.subscribe(EventType.ENERGY_ATTRIBUTED, got.append)

        bus.publish(Event(event_type=EventType.TOOL_CALL_START, timestamp_ns=10 * 1_000_000_000,
                          correlation_id="c1", turn_id="turn-42", payload={"tool": "calc"}))
        bus.publish(Event(event_type=EventType.TOOL_CALL_END, timestamp_ns=11 * 1_000_000_000,
                          correlation_id="c1", turn_id="turn-42", payload={"tool": "calc"}))

        assert got[0].turn_id == "turn-42"
