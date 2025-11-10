"""Plot profiling results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

import click

from ..core.registry import VisualizationRegistry
from ._console import error, info, warning

if TYPE_CHECKING:
    from ..visualization.base import VisualizationContext


def _collect_options(ctx, param, values):
    """Parse key=value options into a dictionary."""
    collected: Dict[str, str] = {}
    for item in values:
        for piece in item.split(","):
            if not piece:
                continue
            key, _, raw = piece.partition("=")
            key = key.strip()
            if not key:
                continue
            collected[key] = raw.strip()
    return collected


@click.command(help="Generate plots from profiling results.")
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--visualization",
    "-v",
    "visualization_id",
    type=str,
    default="regression",
    help="Visualization provider to use (default: regression).",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(path_type=Path),
    help="Directory to save plots (default: <results_dir>/plots).",
)
@click.option(
    "--option",
    "options",
    multiple=True,
    callback=_collect_options,
    help="Visualization-specific options (e.g., --option model=llama3.2:1b).",
)
def plot(
    results_dir: Path,
    visualization_id: str,
    output_dir: Path | None,
    options: Dict[str, Any],
) -> None:
    """Generate visualizations from profiling results."""
    from ..visualization.base import VisualizationContext

    import trafficbench.visualization  # noqa: F401  # Register visualization providers
    # Get visualization provider
    try:
        provider_cls = VisualizationRegistry.get(visualization_id)
    except KeyError:
        available = [key for key, _ in VisualizationRegistry.items()]
        error(f"Visualization '{visualization_id}' not found.")
        if available:
            info(f"Available visualizations: {', '.join(available)}")
        raise click.Abort()

    # Determine output directory
    if output_dir is None:
        output_dir = results_dir / "plots"

    # Create context and run visualization
    context = VisualizationContext(
        results_dir=results_dir,
        output_dir=output_dir,
        options=options,
    )

    provider = provider_cls()
    result = provider.render(context)

    # Display results
    info(f"Visualization: {result.visualization}")

    if result.artifacts:
        info("\nGenerated artifacts:")
        for name, path in result.artifacts.items():
            info(f"  {name}: {path}")
    else:
        warning("No artifacts generated")

    if result.warnings:
        info("\nWarnings:")
        for warn in result.warnings:
            warning(f"  {warn}")

    if result.metadata:
        info("\nMetadata:")
        for key, value in result.metadata.items():
            info(f"  {key}: {value}")


__all__ = ["plot"]
