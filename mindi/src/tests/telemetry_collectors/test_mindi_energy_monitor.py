from __future__ import annotations

import itertools
import time

import pytest

from mindi.telemetry_collectors.mindi import (
    MindiEnergyMonitorCollector,
    ensure_monitor,
    wait_for_ready,
)


@pytest.fixture(scope="module")
def monitor_target() -> str:
    try:
        with ensure_monitor(timeout=15.0) as target:
            # Give the monitor a moment to begin streaming metrics
            time.sleep(0.5)
            yield target
    except FileNotFoundError as exc:
        pytest.skip(f"Energy monitor binary missing: {exc}")
    except RuntimeError as exc:
        pytest.skip(f"Unable to launch energy monitor: {exc}")


def test_wait_for_ready_returns_true(monitor_target: str) -> None:
    assert wait_for_ready(monitor_target, timeout=5.0)


def test_stream_readings_produces_samples(monitor_target: str) -> None:
    collector = MindiEnergyMonitorCollector(target=monitor_target)
    assert collector.is_available()

    readings = collector.stream_readings()
    samples = []
    for reading in itertools.islice(readings, 5):
        samples.append(reading)
        if reading.timestamp_nanos:
            break

    assert samples, "collector produced no telemetry samples"

    sample = samples[0]
    assert sample.timestamp_nanos is None or isinstance(sample.timestamp_nanos, int)
    assert sample.platform is None or isinstance(sample.platform, str)

