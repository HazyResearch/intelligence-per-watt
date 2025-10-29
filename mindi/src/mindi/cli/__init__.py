"""Command-line interface for the Mindi platform (Click-based)."""

from __future__ import annotations

import click

# Import to trigger registry decorators
from mindi import clients  # noqa: F401
from mindi import datasets  # noqa: F401
from mindi import telemetry  # noqa: F401

from .analyze import analyze
from .energy import energy
from .list import list_cmd
from .plot import plot
from .profile import profile


@click.group(help="Mindi development CLI tool")
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
