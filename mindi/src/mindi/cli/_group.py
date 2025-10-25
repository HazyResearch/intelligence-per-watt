"""Utility Click group definitions."""

from __future__ import annotations

import click


class OrderedGroup(click.Group):
    """Click group that preserves command definition order."""

    def list_commands(self, ctx):  # type: ignore[override]
        return list(self.commands.keys())


__all__ = ["OrderedGroup"]
