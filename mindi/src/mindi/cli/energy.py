"""Energy CLI backed by abstract telemetry collectors."""

from __future__ import annotations

import time
from typing import Type

import click

from mindi.core.collector import HardwareCollector
from mindi.core.registry import CollectorRegistry
from mindi.core.types import TelemetryReading

from ._console import console, success


def _get_collector(name: str) -> Type[HardwareCollector]:
    try:
        return CollectorRegistry.get(name)
    except KeyError as exc:
        raise click.ClickException(f"Unknown collector '{name}'") from exc


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
@click.option(
    "--collector",
    type=str,
    default="mindi-energy-monitor",
    show_default=True,
    help="Hardware collector identifier",
)
@click.option(
    "--timeout",
    type=float,
    default=5.0,
    show_default=True,
    help="Seconds to wait for readiness checks",
)
def energy(
    target: str,
    interval: float,
    collector: str,
) -> None:
    collector_cls = _get_collector(collector)
    collector_instance = collector_cls(target=target or "")

    try:
        with collector_instance.start():
            _run_monitor(collector_instance, interval)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


def _run_monitor(collector: HardwareCollector, interval: float) -> None:
    success(f"Streaming telemetry via collector '{collector.collector_name}'")

    console.print(
        "{:>12} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
            "Time",
            "Energy(J)",
            "Power(W)",
            "Temp(°C)",
            "GPU MB",
            "CPU MB",
        )
    )
    console.print("-" * 68)

    start = time.time()
    last_emit = start - interval

    try:
        for reading in collector.stream_readings():
            now = time.time()
            if now - last_emit < max(interval, 0.05):
                continue
            last_emit = now
            console.print(_format_line(now - start, reading))
    except KeyboardInterrupt:
        console.print("\nStopping monitor")


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


