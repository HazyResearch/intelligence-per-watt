#!/usr/bin/env python3
"""Utility script to validate energy monitor telemetry streaming."""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING

from ipw.cli._console import error, info, success
from ipw.core.types import TelemetryReading

if TYPE_CHECKING:
    from ipw.telemetry import EnergyMonitorCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test that the energy monitor can stream telemetry readings."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Energy monitor gRPC target (host:port). Defaults to the local launcher.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between printed samples (default: 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ipw.telemetry import EnergyMonitorCollector

    collector = EnergyMonitorCollector(target=args.target)

    try:
        with collector.start():
            _run_monitor(collector, args.interval)
    except RuntimeError as exc:
        error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        info("\nStopping monitor")


def _run_monitor(collector: "EnergyMonitorCollector", interval: float) -> None:
    success(f"Streaming telemetry via collector '{collector.collector_name}'")

    info(
        "{:>12} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
            "Time",
            "Energy(J)",
            "Power(W)",
            "Temp(°C)",
            "GPU MB",
            "CPU MB",
        )
    )
    info("-" * 68)

    start = time.time()
    last_emit = start - interval

    for reading in collector.stream_readings():
        now = time.time()
        if now - last_emit < max(interval, 0.05):
            continue
        last_emit = now
        info(_format_line(now - start, reading))


def _format_line(elapsed: float, reading: TelemetryReading) -> str:
    return "{:>12} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        _format_elapsed(elapsed),
        _format_metric(reading.energy_joules, width=10, precision=3),
        _format_metric(reading.power_watts, width=10, precision=2),
        _format_metric(reading.temperature_celsius, width=10, precision=1),
        _format_metric(reading.gpu_memory_usage_mb, width=10, precision=1),
        _format_metric(reading.cpu_memory_usage_mb, width=8, precision=1),
    )


def _format_elapsed(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes:
        return f"{minutes}:{secs:02d}"
    return f"{secs}s"


def _format_metric(value: float | None, *, width: int, precision: int) -> str:
    if value is None or value < 0:
        return f"{'-':>{width}}"
    try:
        return f"{value:>{width}.{precision}f}"
    except (ValueError, TypeError):
        return f"{'-':>{width}}"


if __name__ == "__main__":
    main()
