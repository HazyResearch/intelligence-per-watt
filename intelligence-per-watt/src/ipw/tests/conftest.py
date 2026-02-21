"""Shared test fixtures for the IPW test suite."""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from ipw.core.types import (
    AgentRunResult,
    DatasetRecord,
    GpuInfo,
    SystemInfo,
    TelemetryReading,
)
from ipw.telemetry.events import EventRecorder


@pytest.fixture()
def synthetic_telemetry_readings() -> List[TelemetryReading]:
    """List of TelemetryReading with all fields (GPU + CPU + ANE)."""
    return [
        TelemetryReading(
            power_watts=150.0,
            energy_joules=100.0,
            temperature_celsius=65.0,
            gpu_memory_usage_mb=4096.0,
            gpu_memory_total_mb=8192.0,
            cpu_memory_usage_mb=16384.0,
            cpu_power_watts=45.0,
            cpu_energy_joules=50.0,
            ane_power_watts=5.0,
            ane_energy_joules=10.0,
            gpu_compute_utilization_pct=85.0,
            gpu_memory_bandwidth_utilization_pct=60.0,
            gpu_tensor_core_utilization_pct=70.0,
            platform="linux",
            timestamp_nanos=1000000000,
            system_info=SystemInfo(
                os_name="Linux",
                os_version="6.8.0",
                kernel_version="6.8.0-60-generic",
                host_name="testhost",
                cpu_count=16,
                cpu_brand="AMD EPYC",
            ),
            gpu_info=GpuInfo(
                name="RTX 4090",
                vendor="NVIDIA",
                device_id=0,
                device_type="discrete",
                backend="cuda",
            ),
        ),
        TelemetryReading(
            power_watts=155.0,
            energy_joules=120.0,
            temperature_celsius=67.0,
            gpu_memory_usage_mb=4100.0,
            gpu_memory_total_mb=8192.0,
            cpu_memory_usage_mb=16400.0,
            cpu_power_watts=47.0,
            cpu_energy_joules=55.0,
            ane_power_watts=5.5,
            ane_energy_joules=12.0,
            gpu_compute_utilization_pct=88.0,
            gpu_memory_bandwidth_utilization_pct=62.0,
            gpu_tensor_core_utilization_pct=72.0,
            platform="linux",
            timestamp_nanos=2000000000,
            system_info=SystemInfo(
                os_name="Linux",
                os_version="6.8.0",
                kernel_version="6.8.0-60-generic",
                host_name="testhost",
                cpu_count=16,
                cpu_brand="AMD EPYC",
            ),
            gpu_info=GpuInfo(
                name="RTX 4090",
                vendor="NVIDIA",
                device_id=0,
                device_type="discrete",
                backend="cuda",
            ),
        ),
    ]


@pytest.fixture()
def mock_agent() -> MagicMock:
    """Returns a mock agent producing predictable AgentRunResult."""
    agent = MagicMock()
    agent.run.return_value = AgentRunResult(
        content="mock answer",
        tool_calls_attempted=2,
        tool_calls_succeeded=2,
        tool_names_used=["calculator", "web_search"],
        num_turns=3,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.005,
    )
    return agent


@pytest.fixture()
def mock_dataset() -> MagicMock:
    """DatasetProvider mock with N known records."""
    dataset = MagicMock()
    records = [
        DatasetRecord(
            problem=f"Question {i}",
            answer=f"Answer {i}",
            subject=f"subject_{i}",
            dataset_metadata={"index": i, "dataset_name": "test"},
        )
        for i in range(5)
    ]
    dataset.iter_records.return_value = iter(records)
    dataset.__iter__ = lambda self: iter(records)
    dataset.size.return_value = len(records)
    dataset.dataset_id = "test"
    dataset.dataset_name = "TestDataset"
    return dataset


@pytest.fixture()
def sample_event_recorder() -> EventRecorder:
    """EventRecorder with pre-populated events."""
    recorder = EventRecorder()
    recorder.record("lm_inference_start", model="test-model")
    recorder.record("tool_call_start", tool="calculator")
    recorder.record("tool_call_end", tool="calculator", result="42")
    recorder.record(
        "lm_inference_end",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=50,
    )
    return recorder


@pytest.fixture()
def mock_output_dir(tmp_path: Path) -> Path:
    """tmp_path-based output directory for test runs."""
    output = tmp_path / "test_output"
    output.mkdir(parents=True, exist_ok=True)
    return output
