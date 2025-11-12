"""Output token KDE visualization provider - reads directly from dataset."""

from __future__ import annotations

import math
from pathlib import Path

# Force non-interactive backend for headless environments
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..core.registry import VisualizationRegistry
from .base import VisualizationContext, VisualizationProvider, VisualizationResult


def _load_dataset(results_dir: Path):
    """Load the HuggingFace dataset from disk."""
    from datasets import load_from_disk

    return load_from_disk(str(results_dir))


def _extract_completion_tokens(dataset, model_name: str) -> list[float]:
    """Extract completion token counts for a given model."""
    tokens = []

    for record in dataset:
        model_metrics = record.get("model_metrics", {}).get(model_name)
        if not model_metrics:
            continue

        token_metrics = model_metrics.get("token_metrics", {})
        output_tokens = token_metrics.get("output")

        if output_tokens is not None:
            try:
                val = float(output_tokens)
                if math.isfinite(val):
                    tokens.append(val)
            except (TypeError, ValueError):
                continue

    return tokens


def _compute_kde(values: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute kernel density estimation using Gaussian kernel."""
    if len(values) < 2:
        return None

    arr = np.array(values, dtype=np.float64)
    if np.allclose(arr, arr[0]):
        return None

    # Silverman's rule of thumb for bandwidth
    std = float(np.std(arr, ddof=1))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))

    sigma = std if std > 0 else (iqr / 1.349 if iqr > 0 else None)
    if sigma is None or sigma <= 0:
        return None

    bandwidth = 0.9 * sigma * (len(arr) ** (-0.2))
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        return None

    min_val, max_val = float(np.min(arr)), float(np.max(arr))
    if math.isclose(min_val, max_val):
        return None

    xs = np.linspace(min_val, max_val, num=512)
    diffs = (xs[:, None] - arr[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs**2)
    normalization = len(arr) * bandwidth * math.sqrt(2.0 * math.pi)
    density = kernel.sum(axis=1) / normalization

    return xs, density


def _create_kde_plot(
    tokens: list[float],
    output_path: Path,
    model_name: str,
    hardware_label: str = "Unknown",
) -> bool:
    """Create and save a KDE plot of completion token distribution."""
    kde_result = _compute_kde(tokens)
    if kde_result is None:
        return False

    xs, density = kde_result
    arr = np.array(tokens, dtype=np.float64)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, density, color="#1f77b4", linewidth=2.0, label="KDE")
    ax.fill_between(xs, density, color="#1f77b4", alpha=0.2)

    ax.axvline(
        mean_val,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={mean_val:.1f}",
    )
    ax.axvline(
        median_val,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.5,
        label=f"median={median_val:.1f}",
    )

    ax.set_title(
        f"Completion Token Distribution\nModel: {model_name}  |  Hardware: {hardware_label}"
    )
    ax.set_xlabel("Completion tokens")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=280)
    plt.close(fig)
    return True


def _infer_model_name(dataset) -> str | None:
    """Infer the model name from the dataset."""
    for record in dataset:
        model_metrics = record.get("model_metrics", {})
        if model_metrics:
            # Return first model found
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


@VisualizationRegistry.register("output_kde")
class OutputTokenKDE(VisualizationProvider):
    """Generate KDE plot of completion token distribution."""

    visualization_id = "output_kde"

    def render(self, context: VisualizationContext) -> VisualizationResult:
        """Render completion token KDE plot."""
        dataset = _load_dataset(context.results_dir)

        # Get model name from options or infer
        model_name = context.options.get("model")
        if not model_name:
            model_name = _infer_model_name(dataset)
            if not model_name:
                return VisualizationResult(
                    visualization="output_kde",
                    artifacts={},
                    warnings=(
                        "No model found in dataset. Specify --model in options.",
                    ),
                )

        # Extract completion tokens
        tokens = _extract_completion_tokens(dataset, model_name)

        if not tokens or len(tokens) < 2:
            return VisualizationResult(
                visualization="output_kde",
                artifacts={},
                warnings=(
                    f"Insufficient completion token data for model '{model_name}' "
                    f"(found {len(tokens)} samples).",
                ),
            )

        # Infer hardware label
        hardware_label = _infer_hardware_label(dataset, model_name)

        # Create plot
        output_path = context.output_dir / "completion_tokens_kde.png"
        success = _create_kde_plot(tokens, output_path, model_name, hardware_label)

        if not success:
            return VisualizationResult(
                visualization="output_kde",
                artifacts={},
                warnings=("Failed to generate KDE plot (insufficient variation).",),
            )

        return VisualizationResult(
            visualization="output_kde",
            artifacts={"kde_plot": output_path},
            metadata={
                "model": model_name,
                "hardware": hardware_label,
                "sample_count": len(tokens),
                "mean_tokens": float(np.mean(tokens)),
                "median_tokens": float(np.median(tokens)),
            },
        )


__all__ = ["OutputTokenKDE"]
