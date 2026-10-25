"""Score-cache → escalate-set adapter (the pinned §7.3 contract).

Each router head — the embedding classifier or the LoRA-SFT head — emits the *same*
``{task_name: P(local-solves)}`` score cache. This module is the shared glue that turns
that dict into escalation decisions and traces the routed operating curve. It is genuinely
new code: ``ipj.efficiency.operating_curve`` does **not** read the score dict directly — it
consumes an ``escalate`` set plus solve/cost maps.

Convention (matches ``solvability_router.py:116-118`` and ``make_fig2.py:42-48``): rank
tasks **ascending** by ``P(local-solves)`` and escalate the lowest-``e`` (the tasks most
likely to fail locally) to the cloud. Ties break by task name for determinism.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ipj.efficiency.operating_curve import (
    SATURATED_LOCAL_COST,
    OperatingPoint,
    routed_point,
)


def ranked(preds: Mapping[str, float]) -> list[str]:
    """Task names ascending by ``P(local-solves)`` (escalate-lowest-first order).

    Ties break by task name so the ranking — and therefore every escalate set derived from
    it — is deterministic regardless of the input dict's ordering.
    """
    return sorted(preds, key=lambda t: (preds[t], t))


def escalate_set(preds: Mapping[str, float], e: int) -> set[str]:
    """The ``e`` tasks with the lowest ``P(local-solves)`` (the cloud-escalation set).

    ``e`` is clamped to ``[0, len(preds)]``: ``e=0`` escalates nothing (all-local),
    ``e=len(preds)`` escalates everything (all-cloud).
    """
    if e < 0:
        raise ValueError(f"escalate count must be >= 0, got {e}")
    return set(ranked(preds)[:e])


def curve_from_order(
    order: Sequence[str],
    local_solved: Mapping[str, bool],
    cloud_solved: Mapping[str, bool],
    cloud_cost: Mapping[str, float],
    *,
    local_cost_per_task: float = SATURATED_LOCAL_COST,
) -> list[OperatingPoint]:
    """Sweep escalation ``e=0..n`` over an **arbitrary** escalation ``order``.

    ``order`` is the sequence in which tasks are escalated to the cloud (the head decides
    the direction — ascending ``P(local-solves)`` for the solvability head, descending
    utility for the cost-aware head). At step ``e`` the first ``e`` tasks of ``order`` are
    escalated. Returns ``len(order) + 1`` points (all-local through all-escalated), each
    traced by ``operating_curve.routed_point`` on that escalate set.
    """
    n = len(order)
    return [
        routed_point(
            set(order[:e]),
            local_solved,
            cloud_solved,
            cloud_cost,
            label=f"routed+{e}",
            local_cost_per_task=local_cost_per_task,
        )
        for e in range(n + 1)
    ]


def routed_curve(
    preds: Mapping[str, float],
    local_solved: Mapping[str, bool],
    cloud_solved: Mapping[str, bool],
    cloud_cost: Mapping[str, float],
    *,
    local_cost_per_task: float = SATURATED_LOCAL_COST,
) -> list[OperatingPoint]:
    """Sweep escalation ``e=0..n`` over the ascending ``P(local-solves)`` rank.

    Thin caller of :func:`curve_from_order` over the escalate-lowest-first order. Returns
    ``len(preds) + 1`` points (all-local through all-escalated).
    """
    return curve_from_order(
        ranked(preds),
        local_solved,
        cloud_solved,
        cloud_cost,
        local_cost_per_task=local_cost_per_task,
    )


def best_under_budget(
    points: Sequence[OperatingPoint], budget: float
) -> OperatingPoint | None:
    """Max-accuracy point with ``dollars_per_task <= budget`` (``None`` if none fit).

    Ties in accuracy resolve to the **first** (fewest-escalation) point, so sweeping a
    curve produced by :func:`curve_from_order` returns the cheapest point that attains the
    best affordable accuracy.
    """
    affordable = [p for p in points if p.dollars_per_task <= budget]
    if not affordable:
        return None
    return max(affordable, key=lambda p: p.accuracy)
