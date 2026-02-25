"""Tier 3 integration tests for Apple Silicon telemetry.

These tests require macOS with Apple Silicon and powermetrics access.
They are skipped on non-macOS platforms.
"""

from __future__ import annotations

import itertools
import platform
import shutil
import subprocess
import time
from collections.abc import Iterator

import pytest
from ipw.telemetry import EnergyMonitorCollector, ensure_monitor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.apple,
    pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="Apple Silicon telemetry only available on macOS",
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


def test_powermetrics_available() -> None:
    """Verify powermetrics is accessible on macOS."""
    assert shutil.which("powermetrics") is not None


def test_ioreg_gpu_info() -> None:
    """Verify ioreg can query GPU information on macOS."""
    result = subprocess.run(
        ["ioreg", "-l", "-w0"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0


def test_energy_monitor_launches_apple() -> None:
    """Test energy monitor binary launches on Apple Silicon."""
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


def test_apple_ane_telemetry() -> None:
    """Collect Apple Silicon telemetry readings including ANE power."""
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
    # On Apple Silicon we expect at least GPU or ANE power
    has_gpu = r.power_watts is not None
    has_ane = r.ane_power_watts is not None
    assert has_gpu or has_ane, "Neither GPU nor ANE power reported on Apple Silicon"
def test_apple_telemetry_collection(monitor_target: str) -> None:
    """Collect Apple Silicon telemetry readings."""
    collector = EnergyMonitorCollector(target=monitor_target)
    assert collector.is_available()

    readings = collector.stream_readings()
    samples = list(itertools.islice(readings, 5))

    assert samples, "collector produced no telemetry samples"

    sample = samples[0]
    assert sample.platform == "macos", f"expected platform 'macos', got '{sample.platform}'"
    assert isinstance(sample.timestamp_nanos, int)


def test_apple_cpu_power_reported(monitor_target: str) -> None:
    """Verify cpu_power_watts is reported on Apple Silicon."""
    collector = EnergyMonitorCollector(target=monitor_target)
    readings = collector.stream_readings()
    samples = list(itertools.islice(readings, 10))

    assert samples, "collector produced no telemetry samples"

    cpu_power_values = [s.cpu_power_watts for s in samples if s.cpu_power_watts is not None]
    assert cpu_power_values, "no cpu_power_watts readings reported on Apple Silicon"
    assert all(v >= 0 for v in cpu_power_values), "cpu_power_watts should be non-negative"
