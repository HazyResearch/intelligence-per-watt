"""Plot profiling results."""

from __future__ import annotations

import click

from ._console import info


@click.command(help="Generate plots from profiling results.")
@click.argument("directory", type=click.Path(exists=True))
def plot(directory: str) -> None:
    """Placeholder for plot command."""
    info(f"Plot - placeholder implementation for: {directory}")


__all__ = ["plot"]
