"""Telemetry timeline visualization - temperature and power against time."""

from __future__ import annotations

from pathlib import Path

# Force non-interactive backend for headless environments
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..core.registry import VisualizationRegistry
from ..execution.telemetry_timeline import QUERY_WINDOWS_FILENAME, TIMELINE_FILENAME
from .base import VisualizationContext, VisualizationProvider, VisualizationResult

# Colors follow the component, not the panel, so CPU is the same blue in both.
_CPU_COLOR = "#2a78d6"
_GPU_COLOR = "#eb6834"
_ANE_COLOR = "#1baf7a"

# (column, label, color) per panel. `temperature_celsius` is the GPU reading.
_PANELS = {
    "temperature": (
        "Temperature (°C)",
        (
            ("cpu_temperature_celsius", "CPU", _CPU_COLOR),
            ("temperature_celsius", "GPU", _GPU_COLOR),
        ),
    ),
    "power": (
        "Power (W)",
        (
            ("cpu_power_watts", "CPU", _CPU_COLOR),
            ("power_watts", "GPU", _GPU_COLOR),
            ("ane_power_watts", "ANE", _ANE_COLOR),
        ),
    ),
}
_DEFAULT_PANELS = ("temperature", "power")

# Past this, label the x axis in minutes rather than seconds.
_MINUTES_THRESHOLD_SECONDS = 600.0


def _has_data(values: pd.Series) -> bool:
    """Whether a series holds any real reading.

    All-zero counts as empty: powermetrics run without sudo reports 0 W rather
    than the -1 "unavailable" sentinel, and a flat zero line reads as a
    measurement.
    """
    present = values.dropna()
    return bool((present != 0).any())


def _load_query_windows(results_dir: Path, t0: float) -> pd.DataFrame | None:
    path = results_dir / QUERY_WINDOWS_FILENAME
    if not path.exists():
        return None
    windows = pd.read_csv(path)
    if windows.empty:
        return None
    windows["start_s"] = windows["start_timestamp"] - t0
    windows["end_s"] = windows["end_timestamp"] - t0
    return windows


def _create_timeline_plot(
    timeline: pd.DataFrame,
    windows: pd.DataFrame | None,
    panels: list[str],
    plotted: set[str],
    output_path: Path,
    title: str,
) -> dict[str, dict[str, float]]:
    """Draw one panel per metric group and return per-series summary stats."""
    duration = float(timeline["time_s"].iloc[-1])
    in_minutes = duration > _MINUTES_THRESHOLD_SECONDS
    scale = 60.0 if in_minutes else 1.0
    x = timeline["time_s"] / scale

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11, 3.2 * len(panels)), sharex=True, squeeze=False
    )
    stats: dict[str, dict[str, float]] = {}

    for ax, panel in zip(axes[:, 0], panels):
        ylabel, series = _PANELS[panel]
        if windows is not None:
            for position, row in enumerate(windows.itertuples()):
                if row.warmup:
                    # Lightest: warmup queries are excluded from results
                    alpha = 0.06
                else:
                    # Alternate so back-to-back queries stay distinguishable
                    alpha = 0.20 if position % 2 == 0 else 0.11
                ax.axvspan(
                    row.start_s / scale,
                    row.end_s / scale,
                    color="#9a9a9a",
                    alpha=alpha,
                    linewidth=0,
                )
        for column, label, color in series:
            if column not in plotted:
                continue
            values = timeline[column]
            ax.plot(x, values, color=color, linewidth=1.4, label=label)
            stats[column] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", frameon=False, ncol=3)
        if panel == "power":
            ax.set_ylim(bottom=0)

    axes[-1, 0].set_xlabel("Time (min)" if in_minutes else "Time (s)")
    axes[0, 0].set_title(title)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return stats


@VisualizationRegistry.register("timeline")
class TelemetryTimeline(VisualizationProvider):
    """Plot temperature and power over a run from ``telemetry_timeline.csv``.

    Shaded bands mark query windows from ``query_windows.csv`` when present
    (lighter for warmup queries).

    Options:
        panels: ``temperature``, ``power``, or ``both`` (default).
    """

    visualization_id = "timeline"

    def render(self, context: VisualizationContext) -> VisualizationResult:
        timeline_path = context.results_dir / TIMELINE_FILENAME
        if not timeline_path.exists():
            return VisualizationResult(
                visualization=self.visualization_id,
                artifacts={},
                warnings=(
                    f"No {TIMELINE_FILENAME} in {context.results_dir}. Runs made "
                    "before timeline recording was added don't have one.",
                ),
            )

        timeline = pd.read_csv(timeline_path)
        if timeline.empty:
            return VisualizationResult(
                visualization=self.visualization_id,
                artifacts={},
                warnings=(f"{TIMELINE_FILENAME} has no samples.",),
            )

        requested = str(context.options.get("panels", "both")).strip().lower()
        candidates = list(_DEFAULT_PANELS) if requested == "both" else [requested]
        unknown = [p for p in candidates if p not in _PANELS]
        if unknown:
            return VisualizationResult(
                visualization=self.visualization_id,
                artifacts={},
                warnings=(
                    f"Unknown panels option '{requested}'. "
                    "Use temperature, power, or both.",
                ),
            )

        warnings: list[str] = []
        panels = []
        plotted: set[str] = set()
        for panel in candidates:
            columns = [
                column for column, _, _ in _PANELS[panel][1] if _has_data(timeline[column])
            ]
            if columns:
                panels.append(panel)
                plotted.update(columns)
            else:
                warnings.append(f"No {panel} data in the timeline; panel skipped.")
        if not panels:
            return VisualizationResult(
                visualization=self.visualization_id,
                artifacts={},
                warnings=tuple(warnings),
            )

        t0 = float(timeline["timestamp"].iloc[0])
        windows = _load_query_windows(context.results_dir, t0)
        platform = timeline["platform"].dropna()
        platform_label = str(platform.iloc[0]) if not platform.empty else "unknown"

        output_path = context.output_dir / "telemetry_timeline.png"
        stats = _create_timeline_plot(
            timeline,
            windows,
            panels,
            plotted,
            output_path,
            title=f"Telemetry over time  |  {context.results_dir.name}  |  {platform_label}",
        )

        return VisualizationResult(
            visualization=self.visualization_id,
            artifacts={"timeline_plot": output_path},
            metadata={
                "duration_seconds": float(timeline["time_s"].iloc[-1]),
                "sample_count": len(timeline),
                "query_count": 0 if windows is None else int((windows["warmup"] == 0).sum()),
                **{f"{column}_{key}": round(value, 2)
                   for column, values in stats.items()
                   for key, value in values.items()},
            },
            warnings=tuple(warnings),
        )


__all__ = ["TelemetryTimeline"]
