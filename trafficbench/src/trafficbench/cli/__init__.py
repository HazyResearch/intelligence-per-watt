"""Command-line interface for the TrafficBench platform (Click-based)."""

from __future__ import annotations

import click

# Import to trigger registry decorators
from trafficbench import analysis  # noqa: F401
from trafficbench import clients  # noqa: F401
from trafficbench import datasets  # noqa: F401
from trafficbench import telemetry  # noqa: F401
from trafficbench import visualization  # noqa: F401

from .analyze import analyze
from .energy import energy
from .list import list_cmd
from .plot import plot
from .profile import profile


@click.group(help="TrafficBench development CLI tool")
def cli() -> None:
    """Top-level CLI group."""


cli.add_command(profile, "profile")
cli.add_command(analyze, "analyze")
cli.add_command(plot, "plot")
cli.add_command(list_cmd, "list")
cli.add_command(energy, "energy")


def main() -> None:
    """CLI entry point for console scripts."""
    cli()


__all__ = ["cli", "main"]
