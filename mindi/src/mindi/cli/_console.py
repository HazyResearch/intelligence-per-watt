"""Shared console helpers for CLI output."""

from __future__ import annotations

from rich.console import Console

console = Console(highlight=False)


def success(message: str) -> None:
    console.print(message, style="green")


def warning(message: str) -> None:
    console.print(message, style="yellow")


def error(message: str) -> None:
    console.print(message, style="red")


def info(message: str) -> None:
    console.print(message)


__all__ = ["console", "success", "warning", "error", "info"]
