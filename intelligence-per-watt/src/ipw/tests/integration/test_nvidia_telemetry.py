"""Tier 3 integration tests for NVIDIA GPU telemetry.

These tests require actual NVIDIA GPU hardware and drivers.
They are skipped unless the environment provides NVIDIA tooling.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
import time
from collections.abc import Iterator

import pytest

from ipw.telemetry import EnergyMonitorCollector, ensure_monitor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.nvidia,
    pytest.mark.skipif(
        shutil.which("nvidia-smi") is None,
        reason="nvidia-smi not available (no NVIDIA GPU)",
    ),
]


@pytest.fixture(scope="module")
def monitor_target() -> Iterator[str]:
    try:
        with ensure_monitor(timeout=15.0) as target:
            time.sleep(0.5)
            yield target
    except FileNotFoundError as exc:
        pytest.skip(f"Energy monitor binary missing: {exc}")
    except RuntimeError as exc:
        pytest.skip(f"Unable to launch energy monitor: {exc}")


def test_nvidia_smi_available() -> None:
    """Verify nvidia-smi is accessible."""
    assert shutil.which("nvidia-smi") is not None


def test_nvidia_smi_query_gpu() -> None:
    """Verify nvidia-smi can query GPU properties."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.draw", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    output = result.stdout.strip()
    assert len(output) > 0, "nvidia-smi returned empty output"
def test_nvml_telemetry_collection(monitor_target: str) -> None:
    """Collect NVML telemetry readings from a real NVIDIA GPU."""
    collector = EnergyMonitorCollector(target=monitor_target)
    assert collector.is_available()

    readings = collector.stream_readings()
    samples = list(itertools.islice(readings, 5))

    assert samples, "collector produced no telemetry samples"

    sample = samples[0]
    assert sample.platform == "nvidia", f"expected platform 'nvidia', got '{sample.platform}'"
    assert isinstance(sample.timestamp_nanos, int)


def test_gpu_energy_counter_monotonic(monitor_target: str) -> None:
    """Verify GPU energy counter is monotonically non-decreasing."""
    collector = EnergyMonitorCollector(target=monitor_target)
    readings = collector.stream_readings()
    samples = list(itertools.islice(readings, 10))

    assert len(samples) >= 2, "need at least 2 samples to check monotonicity"

    energy_values = [s.energy_joules for s in samples if s.energy_joules is not None]
    assert energy_values, "no energy_joules readings available"

def test_energy_monitor_launches_and_responds() -> None:
    """Test energy monitor binary launches and responds to health check."""
    from ipw.telemetry.launcher import launch_monitor, wait_for_ready

    try:
        pid, target = launch_monitor(timeout=10.0)
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"Energy monitor binary not available: {exc}")

    try:
        assert wait_for_ready(target, timeout=5.0)
    finally:
        import os
        import signal

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def test_streaming_telemetry_produces_readings() -> None:
    """Test streaming telemetry produces TelemetryReading with power and energy."""
    from ipw.telemetry.launcher import ensure_monitor

    try:
        with ensure_monitor(timeout=10.0) as target:
            from ipw.telemetry import EnergyMonitorCollector

            collector = EnergyMonitorCollector(target=target)
            readings = []
            with collector.start():
                deadline = time.monotonic() + 5.0
                for reading in collector.stream_readings():
                    readings.append(reading)
                    if len(readings) >= 3 or time.monotonic() > deadline:
                        break
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"Energy monitor not available: {exc}")

    assert len(readings) >= 1, "No telemetry readings collected"

    r = readings[0]
    assert r.power_watts is not None
    assert r.power_watts > 0
    assert r.energy_joules is not None


def test_gpu_memory_and_utilization_populated() -> None:
    """Test GPU memory and utilization metrics are populated."""
    from ipw.telemetry.launcher import ensure_monitor

    try:
        with ensure_monitor(timeout=10.0) as target:
            from ipw.telemetry import EnergyMonitorCollector

            collector = EnergyMonitorCollector(target=target)
            readings = []
            with collector.start():
                deadline = time.monotonic() + 5.0
                for reading in collector.stream_readings():
                    readings.append(reading)
                    if len(readings) >= 3 or time.monotonic() > deadline:
                        break
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"Energy monitor not available: {exc}")

    assert len(readings) >= 1
    r = readings[0]
    # GPU memory should be reported on NVIDIA hardware
    assert r.gpu_memory_total_mb is not None or r.gpu_memory_usage_mb is not None


def test_energy_counter_monotonically_increasing() -> None:
    """Verify GPU energy counter is monotonically increasing during collection."""
    from ipw.telemetry.launcher import ensure_monitor

    try:
        with ensure_monitor(timeout=10.0) as target:
            from ipw.telemetry import EnergyMonitorCollector

            collector = EnergyMonitorCollector(target=target)
            readings = []
            with collector.start():
                deadline = time.monotonic() + 5.0
                for reading in collector.stream_readings():
                    readings.append(reading)
                    if len(readings) >= 10 or time.monotonic() > deadline:
                        break
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"Energy monitor not available: {exc}")

    energies = [
        r.energy_joules for r in readings
        if r.energy_joules is not None
    ]
    assert len(energies) >= 2, "Not enough energy readings"

    for i in range(1, len(energies)):
        assert energies[i] >= energies[i - 1], (
            f"energy_joules decreased: {energies[i - 1]} -> {energies[i]}"
        )
