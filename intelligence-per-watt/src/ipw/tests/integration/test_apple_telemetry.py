"""Tier 3 integration tests for Apple Silicon telemetry.

These tests require macOS with Apple Silicon and powermetrics access.
They are skipped on non-macOS platforms.
"""

from __future__ import annotations

import itertools
import platform
import shutil
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
