"""Fail-closed aggregation for legacy floating-point cost fields."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypeGuard


def is_valid_cost(value: object) -> TypeGuard[int | float]:
    """Return whether ``value`` is a finite, non-negative cost observation."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def sum_costs_if_complete(
    values: Iterable[float | None],
    *,
    empty: float | None = None,
) -> float | None:
    """Sum costs only when every required component is known.

    ``None`` is unknown, while ``0.0`` is a known zero and must be preserved.
    Callers choose the meaning of an empty collection explicitly through
    ``empty``: traces use ``None``; an analysis over no actions uses ``0.0``.
    """
    items = tuple(values)
    if not items:
        return empty
    if not all(is_valid_cost(value) for value in items):
        return None
    return math.fsum(items)


__all__ = ["is_valid_cost", "sum_costs_if_complete"]
