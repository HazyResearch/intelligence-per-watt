"""Tier 3 integration tests for AMD GPU telemetry.

These tests require actual AMD GPU hardware with ROCm drivers.
They are skipped unless rocm-smi is available.
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
    pytest.mark.amd,
    pytest.mark.skipif(
        shutil.which("rocm-smi") is None,
        reason="rocm-smi not available (no AMD GPU with ROCm)",
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


def test_rocm_smi_available() -> None:
    """Verify rocm-smi is accessible."""
    assert shutil.which("rocm-smi") is not None


def test_rocm_smi_query_gpu() -> None:
    """Verify rocm-smi can query GPU properties."""
    result = subprocess.run(
        ["rocm-smi", "--showid"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0


def test_energy_monitor_launches_amd() -> None:
    """Test energy monitor binary launches on AMD hardware."""
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


def test_amd_telemetry_collection() -> None:
    """Collect AMD GPU telemetry readings and verify basic fields."""
    import time

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
    assert r.power_watts is not None


def test_amd_telemetry_collection_with_fixture(monitor_target: str) -> None:
    """Collect AMD GPU telemetry readings from a real AMD GPU."""
    collector = EnergyMonitorCollector(target=monitor_target)
    assert collector.is_available()

    readings = collector.stream_readings()
    samples = list(itertools.islice(readings, 5))

    assert samples, "collector produced no telemetry samples"

    sample = samples[0]
    assert sample.platform == "amd", f"expected platform 'amd', got '{sample.platform}'"
    assert isinstance(sample.timestamp_nanos, int)


def test_amd_gpu_energy_counter_monotonic(monitor_target: str) -> None:
    """Verify AMD GPU energy counter is monotonically non-decreasing."""
    collector = EnergyMonitorCollector(target=monitor_target)
    readings = collector.stream_readings()
    samples = list(itertools.islice(readings, 10))

    assert len(samples) >= 2, "need at least 2 samples to check monotonicity"

    energy_values = [s.energy_joules for s in samples if s.energy_joules is not None]
    assert energy_values, "no energy_joules readings available"

    for i in range(1, len(energy_values)):
        assert energy_values[i] >= energy_values[i - 1], (
            f"energy_joules decreased: {energy_values[i - 1]} -> {energy_values[i]}"
        )
