"""Tests for the telemetry timeline visualization."""

from __future__ import annotations

from pathlib import Path

from ipw.core.types import TelemetryReading
from ipw.execution.telemetry_timeline import QueryWindowWriter, TimelineWriter
from ipw.visualization.base import VisualizationContext
from ipw.visualization.timeline import TelemetryTimeline


def _write_run(results_dir: Path, *, with_power: bool = True) -> None:
    writer = TimelineWriter(results_dir / "telemetry_timeline.csv")
    for i in range(40):
        writer.write(
            1000.0 + i * 0.5,
            TelemetryReading(
                cpu_temperature_celsius=50.0 + i * 0.25,
                temperature_celsius=45.0 + i * 0.1,
                cpu_power_watts=5.0 if with_power else None,
                power_watts=8.0 if with_power else None,
                platform="macos",
            ),
        )
    writer.close()
    with QueryWindowWriter(results_dir / "query_windows.csv") as windows:
        windows.write(0, 1001.0, 1004.0, warmup=True)
        windows.write(1, 1005.0, 1015.0, warmup=False)


def _render(results_dir: Path, **options):
    context = VisualizationContext(
        results_dir=results_dir, output_dir=results_dir / "plots", options=options
    )
    return TelemetryTimeline().render(context)


def test_renders_plot_with_temperature_stats(tmp_path: Path) -> None:
    _write_run(tmp_path)

    result = _render(tmp_path)

    assert result.artifacts["timeline_plot"].exists()
    assert result.metadata["query_count"] == 1
    assert result.metadata["sample_count"] == 40
    assert result.metadata["cpu_temperature_celsius_min"] == 50.0
    assert result.metadata["cpu_temperature_celsius_max"] == 59.75
    assert result.warnings == ()


def test_skips_panel_without_data(tmp_path: Path) -> None:
    _write_run(tmp_path, with_power=False)

    result = _render(tmp_path)

    assert result.artifacts["timeline_plot"].exists()
    assert result.warnings == ("No power data in the timeline; panel skipped.",)


def test_all_zero_power_is_treated_as_missing(tmp_path: Path) -> None:
    # powermetrics without sudo reports 0 W rather than -1
    writer = TimelineWriter(tmp_path / "telemetry_timeline.csv")
    for i in range(5):
        writer.write(
            float(i),
            TelemetryReading(cpu_temperature_celsius=60.0, cpu_power_watts=0.0, power_watts=0.0),
        )
    writer.close()

    result = _render(tmp_path)

    assert result.artifacts["timeline_plot"].exists()
    assert result.warnings == ("No power data in the timeline; panel skipped.",)
    assert "cpu_power_watts_mean" not in result.metadata


def test_temperature_only_option(tmp_path: Path) -> None:
    _write_run(tmp_path)

    result = _render(tmp_path, panels="temperature")

    assert "power_watts_mean" not in result.metadata
    assert "cpu_temperature_celsius_mean" in result.metadata


def test_warns_when_timeline_missing(tmp_path: Path) -> None:
    result = _render(tmp_path)

    assert result.artifacts == {}
    assert "telemetry_timeline.csv" in result.warnings[0]


def test_rejects_unknown_panel(tmp_path: Path) -> None:
    _write_run(tmp_path)

    result = _render(tmp_path, panels="fans")

    assert result.artifacts == {}
    assert "Unknown panels option" in result.warnings[0]
