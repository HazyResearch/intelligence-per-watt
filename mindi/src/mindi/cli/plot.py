"""Plot profiling results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import click

from ..core.registry import VisualizationRegistry
from ..visualization.base import VisualizationContext
from ._console import console, error, info, success, warning


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

    info(f"Running visualization: {visualization_id}")
    info(f"Results directory: {results_dir}")
    info(f"Output directory: {output_dir}")

    # Create context and run visualization
    context = VisualizationContext(
        results_dir=results_dir,
        output_dir=output_dir,
        options=options,
    )

    provider = provider_cls()
    result = provider.render(context)

    # Display results
    console.print(f"\n[bold]Visualization:[/bold] {result.visualization}")

    if result.artifacts:
        console.print("\n[bold]Generated artifacts:[/bold]")
        for name, path in result.artifacts.items():
            console.print(f"  • {name}: {path}")
        success(f"Generated {len(result.artifacts)} plot(s)")
    else:
        warning("No artifacts generated")

    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warn in result.warnings:
            warning(warn)

    if result.metadata:
        console.print("\n[bold]Metadata:[/bold]")
        for key, value in result.metadata.items():
            console.print(f"  {key}: {value}")


__all__ = ["plot"]
