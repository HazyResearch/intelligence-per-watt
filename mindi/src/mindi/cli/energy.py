"""Energy CLI backed by the bundled telemetry collector."""

from __future__ import annotations

import time

import click

from mindi.core.types import TelemetryReading
from mindi.telemetry import EnergyMonitorCollector

from ._console import info, success


@click.command(help="Ensure an energy monitor can run and stream telemetry.")
@click.option(
    "--target",
    type=str,
    help="Energy monitor gRPC target (host:port)",
)
@click.option(
    "-i",
    "--interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between printed samples",
)
def energy(
    target: str,
    interval: float,
) -> None:
    collector_instance = EnergyMonitorCollector(target=target or "")

    try:
        with collector_instance.start():
            _run_monitor(collector_instance, interval)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


def _run_monitor(collector: EnergyMonitorCollector, interval: float) -> None:
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

    try:
        for reading in collector.stream_readings():
            now = time.time()
            if now - last_emit < max(interval, 0.05):
                continue
            last_emit = now
            info(_format_line(now - start, reading))
    except KeyboardInterrupt:
        info("\nStopping monitor")


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


__all__ = ["energy"]
