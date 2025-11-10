"""Command-line interface for the TrafficBench platform (Click-based)."""

from __future__ import annotations

import click

from .analyze import analyze
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


def main() -> None:
    """CLI entry point for console scripts."""
    cli()


__all__ = ["cli", "main"]
