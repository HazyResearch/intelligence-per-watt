"""Regression visualization provider - generates scatter plots from regression analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Force non-interactive backend for headless environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..core.registry import VisualizationRegistry
from .base import VisualizationContext, VisualizationProvider, VisualizationResult


def _load_regression_data(results_dir: Path) -> Mapping[str, Any]:
    """Load regression analysis results from disk."""
    regression_file = results_dir / "analysis" / "regression.json"
    if not regression_file.exists():
        raise FileNotFoundError(
            f"Regression analysis not found at {regression_file}. "
            "Run 'trafficbench analyze --analysis regression' first."
        )
    with open(regression_file) as f:
        return json.load(f)


def _load_dataset(results_dir: Path):
    """Load the HuggingFace dataset from disk."""
    from datasets import load_from_disk

    return load_from_disk(str(results_dir))


def _infer_model_name(dataset) -> str | None:
    """Infer the model name from the dataset."""
    for record in dataset:
        model_metrics = record.get("model_metrics", {})
        if model_metrics:
            return next(iter(model_metrics.keys()))
    return None


def _infer_hardware_label(dataset, model_name: str) -> str:
    """Infer hardware label from the first record with GPU info."""
    for record in dataset:
        model_metrics = record.get("model_metrics", {}).get(model_name)
        if not model_metrics:
            continue

        gpu_info = model_metrics.get("gpu_info", {})
        if gpu_info and isinstance(gpu_info, dict):
            gpu_name = gpu_info.get("name", "")
            if gpu_name:
                return gpu_name

        system_info = model_metrics.get("system_info", {})
        if system_info and isinstance(system_info, dict):
            cpu_brand = system_info.get("cpu_brand", "")
            if cpu_brand and cpu_brand != "Unknown CPU":
                return cpu_brand

    return "Unknown"


def _safe_get(obj: Any, key: str) -> Any:
    """Safely get a value from dict or dataclass.

    Args:
        obj: Dictionary or dataclass instance to extract value from
        key: Attribute or key name to retrieve

    Returns:
        The value if found, None otherwise
    """
    if isinstance(obj, dict):
        return obj.get(key)
    elif hasattr(obj, key):
        return getattr(obj, key)
    return None


def _extract_regression_samples(
    dataset, model_name: str, x_key_path: Sequence[str], y_key_path: Sequence[str]
) -> Tuple[List[float], List[float]]:
    """Extract x and y values from dataset for a given model and metric paths."""
    xs = []
    ys = []

    for record in dataset:
        model_metrics = record.get("model_metrics", {}).get(model_name)
        if not model_metrics:
            continue

        # Navigate nested dict/dataclass for x value
        x_val = model_metrics
        for key in x_key_path:
            if x_val is None:
                break
            x_val = _safe_get(x_val, key)

        # Backward compatibility: compute total tokens if not stored
        if x_val is None and x_key_path == ["token_metrics", "total"]:
            token_metrics = _safe_get(model_metrics, "token_metrics")
            if token_metrics is not None:
                input_tokens = _safe_get(token_metrics, "input")
                output_tokens = _safe_get(token_metrics, "output")
                if input_tokens is not None or output_tokens is not None:
                    x_val = (input_tokens or 0.0) + (output_tokens or 0.0)

        # Navigate nested dict/dataclass for y value
        y_val = model_metrics
        for key in y_key_path:
            if y_val is None:
                break
            y_val = _safe_get(y_val, key)

        if x_val is not None and y_val is not None:
            try:
                x_float = float(x_val)
                y_float = float(y_val)
                if math.isfinite(x_float) and math.isfinite(y_float):
                    xs.append(x_float)
                    ys.append(y_float)
            except (TypeError, ValueError):
                continue

    return xs, ys


def _generate_linear_fit(
    xs: Sequence[float], slope: float, intercept: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generate linear regression line."""
    if not xs or len(xs) < 2:
        return None

    x_arr = np.array(xs, dtype=np.float64)
    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    if math.isclose(x_min, x_max):
        return None

    x_line = np.linspace(x_min, x_max, 200)
    y_line = slope * x_line + intercept
    return x_line, y_line


def _generate_log_fit(
    xs: Sequence[float], slope: float, intercept: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generate log-x regression line: y = slope * log(x) + intercept."""
    if not xs or len(xs) < 2:
        return None

    x_arr = np.array(xs, dtype=np.float64)
    positive = x_arr[x_arr > 0]
    if len(positive) < 2:
        return None

    x_min, x_max = float(np.min(positive)), float(np.max(positive))
    if math.isclose(x_min, x_max):
        return None

    log_min, log_max = math.log(x_min), math.log(x_max)
    log_space = np.linspace(log_min, log_max, 200)
    x_line = np.exp(log_space)
    y_line = slope * log_space + intercept
    return x_line, y_line


def _create_scatter_plot(
    xs: Sequence[float],
    ys: Sequence[float],
    stats: Mapping[str, Any],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    model: str,
    hardware: str,
    log_fit_stats: Optional[Mapping[str, Any]] = None,
) -> None:
    """Create a scatter plot with regression lines."""
    if not xs or not ys:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, color="#1f77b4", alpha=0.35, s=12, label="samples")

    # Linear fit
    slope = stats.get("slope")
    intercept = stats.get("intercept")
    r2 = stats.get("r2")

    if slope is not None and intercept is not None:
        line = _generate_linear_fit(xs, slope, intercept)
        if line:
            x_line, y_line = line
            label = (
                f"linear fit (slope={slope:.3g}, r²={r2:.3f})" if r2 else "linear fit"
            )
            ax.plot(x_line, y_line, color="#d62728", linewidth=2.0, label=label)

    # Optional log fit
    if log_fit_stats:
        log_slope = log_fit_stats.get("slope")
        log_intercept = log_fit_stats.get("intercept")
        log_r2 = log_fit_stats.get("r2")

        if log_slope is not None and log_intercept is not None:
            log_line = _generate_log_fit(xs, log_slope, log_intercept)
            if log_line:
                x_line, y_line = log_line
                label = (
                    f"log-x fit (slope={log_slope:.3g}, r²={log_r2:.3f})"
                    if log_r2
                    else "log-x fit"
                )
                ax.plot(
                    x_line,
                    y_line,
                    color="#2ca02c",
                    linewidth=2.0,
                    linestyle=":",
                    label=label,
                )

    ax.set_title(f"{title}\nModel: {model}  |  Hardware: {hardware}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=280)
    plt.close(fig)


@VisualizationRegistry.register("regression")
class RegressionVisualization(VisualizationProvider):
    """Generate scatter plots with regression lines from regression analysis results."""

    visualization_id = "regression"

    def render(self, context: VisualizationContext) -> VisualizationResult:
        """Render regression scatter plots."""
        # Load regression analysis results
        regression_data = _load_regression_data(context.results_dir)

        data = regression_data.get("data", {})
        regressions = data.get("regressions", {})

        dataset = _load_dataset(context.results_dir)

        # Infer model name from options or dataset
        model_name = context.options.get("model")
        if not model_name:
            model_name = _infer_model_name(dataset)
            if not model_name:
                model_name = "unknown"

        hardware_label = _infer_hardware_label(dataset, model_name)

        artifacts = {}
        warnings = []

        # Define plots to generate
        plots = [
            {
                "key": "input_tokens_vs_ttft",
                "x_path": ["token_metrics", "input"],
                "y_path": ["latency_metrics", "time_to_first_token_seconds"],
                "title": "Prompt Tokens vs Time to First Token",
                "x_label": "Prompt tokens",
                "y_label": "TTFT (seconds)",
                "filename": "ttft.png",
            },
            {
                "key": "total_tokens_vs_latency",
                "x_path": ["token_metrics", "total"],
                "y_path": ["latency_metrics", "total_query_seconds"],
                "title": "Total Tokens vs Latency",
                "x_label": "Total tokens",
                "y_label": "Total latency (seconds)",
                "filename": "latency.png",
            },
            {
                "key": "total_tokens_vs_energy",
                "x_path": ["token_metrics", "total"],
                "y_path": ["energy_metrics", "per_query_joules"],
                "title": "Total Tokens vs Energy",
                "x_label": "Total tokens",
                "y_label": "Per-query energy (joules)",
                "filename": "energy.png",
            },
            {
                "key": "total_tokens_vs_power",
                "x_path": ["token_metrics", "total"],
                "y_path": ["power_metrics", "gpu", "per_query_watts", "avg"],
                "title": "Total Tokens vs Power",
                "x_label": "Total tokens",
                "y_label": "Per-query power (watts)",
                "filename": "power.png",
                "log_key": "total_tokens_vs_power_log",
            },
        ]

        for plot_spec in plots:
            key = plot_spec["key"]
            stats = regressions.get(key, {})

            if not stats or stats.get("count", 0) == 0:
                warnings.append(f"Skipping {plot_spec['filename']}: no regression data")
                continue

            # Extract samples from dataset
            xs, ys = _extract_regression_samples(
                dataset, model_name, plot_spec["x_path"], plot_spec["y_path"]
            )

            if not xs or not ys:
                warnings.append(f"Skipping {plot_spec['filename']}: no valid samples")
                continue

            output_path = context.output_dir / plot_spec["filename"]

            # Check for optional log fit
            log_stats = None
            if "log_key" in plot_spec:
                log_stats = regressions.get(plot_spec["log_key"])

            _create_scatter_plot(
                xs=xs,
                ys=ys,
                stats=stats,
                title=str(plot_spec["title"]),
                x_label=str(plot_spec["x_label"]),
                y_label=str(plot_spec["y_label"]),
                output_path=output_path,
                model=model_name,
                hardware=hardware_label,
                log_fit_stats=log_stats,
            )

            artifacts[plot_spec["key"]] = output_path

        return VisualizationResult(
            visualization="regression",
            artifacts=artifacts,
            metadata={
                "model": model_name,
                "hardware": hardware_label,
            },
            warnings=tuple(warnings),
        )


__all__ = ["RegressionVisualization"]
