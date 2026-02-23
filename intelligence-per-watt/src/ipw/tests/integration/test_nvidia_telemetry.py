"""Tier 3 integration tests for NVIDIA GPU telemetry.

These tests require actual NVIDIA GPU hardware and drivers.
They are skipped unless the environment provides NVIDIA tooling.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.nvidia,
    pytest.mark.skipif(
        shutil.which("nvidia-smi") is None,
        reason="nvidia-smi not available (no NVIDIA GPU)",
    ),
]


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
            f"Energy counter decreased: {energies[i-1]} -> {energies[i]}"
        )
