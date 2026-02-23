"""Tier 3 integration tests for AMD GPU telemetry.

These tests require actual AMD GPU hardware with ROCm drivers.
They are skipped unless rocm-smi is available.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.amd,
    pytest.mark.skipif(
        shutil.which("rocm-smi") is None,
        reason="rocm-smi not available (no AMD GPU with ROCm)",
    ),
]


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
