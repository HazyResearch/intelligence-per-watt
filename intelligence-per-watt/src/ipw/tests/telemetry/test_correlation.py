"""Tests for energy-event correlation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from ipw.telemetry.correlation import (
    ActionEnergyBreakdown,
    compute_analysis,
    correlate_energy_to_events,
)
from ipw.telemetry.events import AgentEvent


@dataclass
class MockReading:
    """Mock telemetry reading for tests."""
    energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None
    power_watts: Optional[float] = None
    cpu_power_watts: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None
    gpu_compute_utilization_pct: Optional[float] = None
    gpu_memory_bandwidth_utilization_pct: Optional[float] = None
    gpu_tensor_core_utilization_pct: Optional[float] = None
    platform: Optional[str] = None
    system_info: Optional[Any] = None
    gpu_info: Optional[Any] = None


@dataclass
class MockSample:
    """Mock telemetry sample for tests."""
    timestamp: float
    reading: MockReading


def _make_events(
    start_time: float = 1000.0,
    inference_duration: float = 1.0,
    tool_duration: float = 0.5,
) -> list[AgentEvent]:
    """Create a sequence of agent events."""
    events = [
        AgentEvent(
            event_type="lm_inference_start",
            timestamp=start_time,
            metadata={"model_id": "test-model"},
        ),
        AgentEvent(
            event_type="lm_inference_end",
            timestamp=start_time + inference_duration,
            metadata={
                "model_id": "test-model",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        ),
        AgentEvent(
            event_type="tool_call_start",
            timestamp=start_time + inference_duration + 0.1,
            metadata={"tool": "search"},
        ),
        AgentEvent(
            event_type="tool_call_end",
            timestamp=start_time + inference_duration + 0.1 + tool_duration,
            metadata={"tool": "search"},
        ),
    ]
    return events


def _make_samples(
    start_time: float = 999.5,
    count: int = 10,
    interval: float = 0.2,
    gpu_energy_start: float = 100.0,
    gpu_energy_rate: float = 10.0,
    gpu_power: float = 150.0,
    cpu_energy_start: float = 50.0,
    cpu_energy_rate: float = 3.0,
    cpu_power: float = 45.0,
) -> list[MockSample]:
    """Create a sequence of telemetry samples."""
    samples = []
    for i in range(count):
        t = start_time + i * interval
        samples.append(
            MockSample(
                timestamp=t,
                reading=MockReading(
                    energy_joules=gpu_energy_start + i * gpu_energy_rate * interval,
                    cpu_energy_joules=cpu_energy_start + i * cpu_energy_rate * interval,
                    power_watts=gpu_power,
                    cpu_power_watts=cpu_power,
                    gpu_memory_usage_mb=8000.0,
                ),
            )
        )
    return samples


class TestCorrelateEnergyToEvents:
    """Tests for energy-event correlation."""

    def test_empty_inputs(self) -> None:
        assert correlate_energy_to_events([], []) == []
        assert correlate_energy_to_events([], [AgentEvent("test", 0.0)]) == []
        assert correlate_energy_to_events([MockSample(0.0, MockReading())], []) == []

    def test_basic_correlation(self) -> None:
        events = _make_events()
        samples = _make_samples()

        breakdowns = correlate_energy_to_events(samples, events)

        assert len(breakdowns) == 2  # lm_inference + tool_call

        lm = breakdowns[0]
        assert lm.action_type == "lm_inference"
        assert lm.step_number == 0
        assert lm.gpu_energy_joules >= 0
        assert lm.duration_ms > 0

        tool = breakdowns[1]
        assert tool.action_type == "tool_call"
        assert tool.step_number == 1

    def test_energy_deltas_are_positive(self) -> None:
        events = _make_events()
        samples = _make_samples()

        breakdowns = correlate_energy_to_events(samples, events)

        for b in breakdowns:
            assert b.gpu_energy_joules >= 0
            assert b.cpu_energy_joules >= 0
            assert b.total_energy_joules >= 0

    def test_power_metrics_populated(self) -> None:
        events = _make_events()
        samples = _make_samples()

        breakdowns = correlate_energy_to_events(samples, events)

        for b in breakdowns:
            if b.duration_ms > 0:
                assert b.max_power_watts is not None or b.avg_power_watts is not None

    def test_metadata_preserved(self) -> None:
        events = _make_events()
        samples = _make_samples()

        breakdowns = correlate_energy_to_events(samples, events)

        lm = breakdowns[0]
        assert "start_timestamp" in lm.metadata
        assert "end_timestamp" in lm.metadata
        assert lm.metadata.get("model_id") == "test-model"

    def test_include_idle(self) -> None:
        events = _make_events(start_time=1000.0, inference_duration=1.0, tool_duration=0.5)
        samples = _make_samples(start_time=999.5, count=20)

        breakdowns = correlate_energy_to_events(samples, events, include_idle=True)

        idle_breakdowns = [b for b in breakdowns if b.action_type == "idle"]
        assert len(idle_breakdowns) >= 1

    def test_counter_reset_handled(self) -> None:
        """Counter resets should not produce negative energy."""
        events = [
            AgentEvent("lm_inference_start", 1000.0, {}),
            AgentEvent("lm_inference_end", 1001.0, {}),
        ]
        samples = [
            MockSample(999.9, MockReading(energy_joules=100.0, cpu_energy_joules=50.0)),
            MockSample(1001.1, MockReading(energy_joules=10.0, cpu_energy_joules=5.0)),
        ]

        breakdowns = correlate_energy_to_events(samples, events)

        assert len(breakdowns) == 1
        assert breakdowns[0].gpu_energy_joules >= 0
        assert breakdowns[0].cpu_energy_joules >= 0

    def test_unpaired_events_ignored(self) -> None:
        events = [
            AgentEvent("lm_inference_start", 1000.0, {}),
            # Missing lm_inference_end
            AgentEvent("tool_call_start", 1001.0, {}),
            AgentEvent("tool_call_end", 1001.5, {}),
        ]
        samples = _make_samples()

        breakdowns = correlate_energy_to_events(samples, events)

        # Only tool_call should be paired
        assert len(breakdowns) == 1
        assert breakdowns[0].action_type == "tool_call"


class TestComputeAnalysis:
    """Tests for aggregate analysis computation."""

    def test_empty_breakdowns(self) -> None:
        result = compute_analysis([])
        assert result["total_energy_joules"] == 0.0
        assert result["action_counts"] == {}
        assert result["total_cost_usd"] == 0.0

    def test_basic_analysis(self) -> None:
        breakdowns = [
            ActionEnergyBreakdown(
                action_type="lm_inference",
                step_number=0,
                gpu_energy_joules=10.0,
                cpu_energy_joules=2.0,
                total_energy_joules=12.0,
                duration_ms=1000.0,
                max_power_watts=200.0,
                avg_power_watts=150.0,
                metadata={"model_id": "test-model", "total_tokens": 150},
            ),
            ActionEnergyBreakdown(
                action_type="tool_call",
                step_number=1,
                gpu_energy_joules=5.0,
                cpu_energy_joules=1.0,
                total_energy_joules=6.0,
                duration_ms=500.0,
                max_power_watts=100.0,
                avg_power_watts=80.0,
            ),
        ]

        result = compute_analysis(breakdowns)

        assert result["total_energy_joules"] == 18.0
        assert result["total_gpu_energy_joules"] == 15.0
        assert result["total_cpu_energy_joules"] == 3.0
        assert result["total_duration_ms"] == 1500.0
        assert result["max_power_watts"] == 200.0
        assert result["action_counts"] == {"lm_inference": 1, "tool_call": 1}
        assert result["energy_by_action"]["lm_inference"] == 12.0
        assert result["energy_by_action"]["tool_call"] == 6.0

    def test_per_model_aggregation(self) -> None:
        breakdowns = [
            ActionEnergyBreakdown(
                action_type="lm_inference",
                step_number=0,
                gpu_energy_joules=10.0,
                cpu_energy_joules=2.0,
                total_energy_joules=12.0,
                duration_ms=1000.0,
                metadata={"model_id": "model-a", "total_tokens": 100},
            ),
            ActionEnergyBreakdown(
                action_type="lm_inference",
                step_number=1,
                gpu_energy_joules=8.0,
                cpu_energy_joules=1.5,
                total_energy_joules=9.5,
                duration_ms=800.0,
                metadata={"model_id": "model-b", "total_tokens": 80},
            ),
        ]

        result = compute_analysis(breakdowns)

        assert result["model_counts"]["model-a"] == 1
        assert result["model_counts"]["model-b"] == 1
        assert result["energy_by_model"]["model-a"] == 12.0
        assert result["tokens_by_model"]["model-a"] == 100

    def test_cost_aggregation(self) -> None:
        breakdowns = [
            ActionEnergyBreakdown(
                action_type="lm_inference",
                step_number=0,
                gpu_energy_joules=10.0,
                cpu_energy_joules=2.0,
                total_energy_joules=12.0,
                duration_ms=1000.0,
                cost_usd=0.05,
                metadata={"model_id": "model-a"},
            ),
            ActionEnergyBreakdown(
                action_type="lm_inference",
                step_number=1,
                gpu_energy_joules=8.0,
                cpu_energy_joules=1.5,
                total_energy_joules=9.5,
                duration_ms=800.0,
                cost_usd=0.03,
                metadata={"model_id": "model-a"},
            ),
        ]

        result = compute_analysis(breakdowns)

        assert result["total_cost_usd"] == pytest.approx(0.08)
        assert result["cost_by_model"]["model-a"] == pytest.approx(0.08)


class TestActionEnergyBreakdown:
    """Tests for ActionEnergyBreakdown dataclass."""

    def test_repr(self) -> None:
        b = ActionEnergyBreakdown(
            action_type="lm_inference",
            step_number=0,
            gpu_energy_joules=10.0,
            cpu_energy_joules=2.0,
            total_energy_joules=12.0,
            duration_ms=1000.0,
        )
        repr_str = repr(b)
        assert "lm_inference" in repr_str
        assert "12.00J" in repr_str
        assert "1000.0ms" in repr_str

    def test_defaults(self) -> None:
        b = ActionEnergyBreakdown(
            action_type="test",
            step_number=0,
            gpu_energy_joules=0.0,
            cpu_energy_joules=0.0,
            total_energy_joules=0.0,
            duration_ms=0.0,
        )
        assert b.max_power_watts is None
        assert b.cost_usd is None
        assert b.metadata == {}
