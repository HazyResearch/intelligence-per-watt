"""Tests for core/types.py — AgentRunResult and related types."""

from __future__ import annotations

import pytest

from ipw.core.types import (
    AgentRunResult,
    DatasetRecord,
    ProfilerConfig,
    TelemetryReading,
)


class TestAgentRunResult:
    """Test AgentRunResult construction and defaults."""

    def test_construction_with_all_fields(self) -> None:
        result = AgentRunResult(
            content="answer",
            tool_calls_attempted=3,
            tool_calls_succeeded=2,
            tool_names_used=["calc", "search"],
            num_turns=5,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        assert result.content == "answer"
        assert result.tool_calls_attempted == 3
        assert result.tool_calls_succeeded == 2
        assert result.tool_names_used == ["calc", "search"]
        assert result.num_turns == 5
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cost_usd == 0.01

    def test_default_field_values(self) -> None:
        result = AgentRunResult(content="hello")
        assert result.tool_calls_attempted == 0
        assert result.tool_calls_succeeded == 0
        assert result.tool_names_used == []
        assert result.num_turns == 0
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cost_usd == 0.0
        assert result.trace is None

    def test_slots_behavior(self) -> None:
        result = AgentRunResult(content="test")
        assert hasattr(result, "__slots__")
        with pytest.raises(AttributeError):
            result.nonexistent_attr = "should fail"

    def test_tool_names_used_is_mutable_list(self) -> None:
        r1 = AgentRunResult(content="a")
        r2 = AgentRunResult(content="b")
        r1.tool_names_used.append("tool1")
        assert r1.tool_names_used == ["tool1"]
        assert r2.tool_names_used == []


class TestTelemetryReading:
    """Test TelemetryReading defaults and slots."""

    def test_all_fields_default_to_none(self) -> None:
        reading = TelemetryReading()
        assert reading.power_watts is None
        assert reading.energy_joules is None
        assert reading.temperature_celsius is None
        assert reading.gpu_memory_usage_mb is None
        assert reading.gpu_memory_total_mb is None
        assert reading.cpu_memory_usage_mb is None
        assert reading.cpu_power_watts is None
        assert reading.cpu_energy_joules is None
        assert reading.ane_power_watts is None
        assert reading.ane_energy_joules is None
        assert reading.platform is None
        assert reading.timestamp_nanos is None
        assert reading.system_info is None
        assert reading.gpu_info is None

    def test_slots_behavior(self) -> None:
        reading = TelemetryReading()
        assert hasattr(reading, "__slots__")
        with pytest.raises(AttributeError):
            reading.nonexistent_attr = "fail"


class TestDatasetRecord:
    """Test DatasetRecord construction."""

    def test_minimal_construction(self) -> None:
        record = DatasetRecord(problem="Q?", answer="A", subject="math")
        assert record.problem == "Q?"
        assert record.answer == "A"
        assert record.subject == "math"
        assert record.dataset_metadata == {}

    def test_metadata_default_factory(self) -> None:
        r1 = DatasetRecord(problem="a", answer="b", subject="c")
        r2 = DatasetRecord(problem="x", answer="y", subject="z")
        r1.dataset_metadata["key"] = "val"
        assert "key" not in r2.dataset_metadata


class TestProfilerConfig:
    """Test ProfilerConfig defaults."""

    def test_defaults(self) -> None:
        config = ProfilerConfig(dataset_id="ipw", client_id="ollama")
        assert config.model == ""
        assert config.output_dir is None
        assert config.max_concurrency == 1
        assert config.max_queries is None
        assert config.phased_profiling is False
        assert config.run_hardware_benchmarks is True
