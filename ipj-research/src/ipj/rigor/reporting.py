"""Small pretty-printers shared by the fold_k5.py-family serving-run reports."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def format_pct(x: float) -> str:
    """0.751 -> '75.1%'."""
    return f"{x * 100:.1f}%"


def format_ci(ci: tuple[float, float] | list[float]) -> str:
    """[0.749, 0.788] -> '[74.9, 78.8]' (percent points, no % sign — the log-table convention)."""
    return f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def truncate_list(items: Iterable[Any], k: int = 10) -> str:
    """Join first k items with ', ', append '... (+N more)' if truncated."""
    items = list(items)
    return ", ".join(map(str, items[:k])) + (f" ... (+{len(items) - k} more)" if len(items) > k else "")
