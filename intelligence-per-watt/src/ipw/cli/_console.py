"""Shared console helpers for CLI output."""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from ..analysis.base import AnalysisResult

console = Console(highlight=False)


def success(message: str) -> None:
    console.print(message, style="green")


def warning(message: str) -> None:
    console.print(message, style="yellow")


def error(message: str) -> None:
    console.print(message, style="red")


def info(message: str) -> None:
    console.print(message)


def _print_result(result: "AnalysisResult", *, verbose: bool = False) -> None:
    """Pretty-print an AnalysisResult in a consistent format."""
    info(f"Analysis: {result.analysis}")

    if result.summary:
        info("")
        info("Summary:")
        for key, value in result.summary.items():
            info(f"  {key}: {value}")

    if result.warnings:
        info("")
        info("Warnings:")
        for warn_msg in result.warnings:
            warning(f"  {warn_msg}")

    if result.artifacts:
        info("")
        info("Artifacts:")
        for name, path in result.artifacts.items():
            info(f"  {name}: {path}")

    if verbose and result.data:
        info("")
        info("Data:")
        info(textwrap.indent(json.dumps(dict(result.data), indent=2, default=str), "  "))

    if verbose and result.metadata:
        info("")
        info("Metadata:")
        info(
            textwrap.indent(json.dumps(dict(result.metadata), indent=2, default=str), "  ")
        )


__all__ = ["console", "success", "warning", "error", "info", "_print_result"]
