"""Command-line interface for the Mindi platform (Click-based)."""

from __future__ import annotations

import click

from .energy import energy
from .profile import profile


@click.group(help="Mindi development CLI tool")
def cli() -> None:
    """Top-level CLI group."""


cli.add_command(energy, "energy")
cli.add_command(profile, "profile")


def main() -> None:
    """CLI entry point for console scripts."""
    cli()


__all__ = ["cli", "main"]
