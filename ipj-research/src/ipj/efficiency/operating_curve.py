"""Router operating curve for the local→cloud cascade (G1 / T1.1).

Since Wave A established that **local inference is ~free at saturation and the cascade's cost is
~100% cloud escalation**, the accuracy-vs-$ frontier is fully determined by *which* tasks get
escalated. This module traces that frontier offline from per-task solve vectors + per-task cloud
cost — no live serving (which the eval-host Docker pool can't drive at the needed concurrency).

Given a shared task set with:
  * ``local_solved[t]``  — does the local front tier resolve t?
  * ``cloud_solved[t]``  — does the cloud model resolve t?
  * ``cloud_cost[t]``    — $ the cloud model billed on t,
we build:
  * ``oracle_frontier`` — the cheapest-$-first escalation of exactly the tasks local fails and
    cloud solves: the UPPER bound on any router (accuracy per cloud-$).
  * ``random_line``     — expected accuracy/$ escalating a uniformly random subset (baseline).
  * ``routed_point``    — the (accuracy, $) of one concrete routing decision (a realized error
    signal, or a pre-route router's flagged set).
Per-task $ = ``local_cost_per_task`` (≈$0.005 at saturation) + Σ escalated cloud cost / N.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

SATURATED_LOCAL_COST = 0.005  # $/task, measured 27B on 2xH100 (throughput_dollars_per_task)


@dataclass(frozen=True)
class OperatingPoint:
    n_escalated: int
    accuracy: float
    dollars_per_task: float
    label: str = ""


def _shared(*maps: Mapping[str, object]) -> list[str]:
    s = set(maps[0])
    for m in maps[1:]:
        s &= set(m)
    return sorted(s)


def _acc_dollars(
    escalate: set[str], local_solved, cloud_solved, cloud_cost, tasks, local_cost_per_task
) -> tuple[float, float]:
    solved = sum((cloud_solved[t] if t in escalate else local_solved[t]) for t in tasks)
    esc_cost = sum(cloud_cost.get(t, 0.0) for t in escalate)
    n = len(tasks)
    return solved / n, local_cost_per_task + esc_cost / n


def oracle_frontier(
    local_solved: Mapping[str, bool], cloud_solved: Mapping[str, bool],
    cloud_cost: Mapping[str, float], *, local_cost_per_task: float = SATURATED_LOCAL_COST,
) -> list[OperatingPoint]:
    """All-local → escalate (local-fail ∧ cloud-solve) tasks, cheapest cloud-$ first. Upper bound."""
    tasks = _shared(local_solved, cloud_solved)
    base = sum(local_solved[t] for t in tasks) / len(tasks)
    pts = [OperatingPoint(0, base, local_cost_per_task, "all-local")]
    gain = sorted((t for t in tasks if not local_solved[t] and cloud_solved[t]),
                  key=lambda t: cloud_cost.get(t, 0.0))
    escalate: set[str] = set()
    for i, t in enumerate(gain, 1):
        escalate.add(t)
        acc, dol = _acc_dollars(escalate, local_solved, cloud_solved, cloud_cost, tasks, local_cost_per_task)
        pts.append(OperatingPoint(i, acc, dol, f"oracle+{i}"))
    return pts


def random_line(
    local_solved: Mapping[str, bool], cloud_solved: Mapping[str, bool],
    cloud_cost: Mapping[str, float], *, local_cost_per_task: float = SATURATED_LOCAL_COST,
) -> list[OperatingPoint]:
    """Expected (accuracy, $) escalating E uniformly-random tasks, E=0..N (a straight baseline)."""
    tasks = _shared(local_solved, cloud_solved)
    n = len(tasks)
    local_acc = sum(local_solved[t] for t in tasks) / n
    cloud_acc = sum(cloud_solved[t] for t in tasks) / n
    avg_cloud_cost = sum(cloud_cost.get(t, 0.0) for t in tasks) / n
    pts = []
    for e in range(n + 1):
        f = e / n
        acc = (1 - f) * local_acc + f * cloud_acc
        dol = local_cost_per_task + f * avg_cloud_cost
        pts.append(OperatingPoint(e, acc, dol, "random"))
    return pts


def routed_point(
    escalate: set[str], local_solved: Mapping[str, bool], cloud_solved: Mapping[str, bool],
    cloud_cost: Mapping[str, float], *, label: str = "routed",
    local_cost_per_task: float = SATURATED_LOCAL_COST,
) -> OperatingPoint:
    """(accuracy, $) for one concrete escalation set (a realized error signal or a router's picks)."""
    tasks = _shared(local_solved, cloud_solved)
    esc = escalate & set(tasks)
    acc, dol = _acc_dollars(esc, local_solved, cloud_solved, cloud_cost, tasks, local_cost_per_task)
    return OperatingPoint(len(esc), acc, dol, label)


def first_at_multiplier(
    points: list[OperatingPoint], *, reference_dollars: float, target_mult: float,
) -> OperatingPoint | None:
    """Highest-accuracy point on a frontier still at ≥ target_mult cheaper than the reference."""
    ok = [p for p in points if p.dollars_per_task > 0 and reference_dollars / p.dollars_per_task >= target_mult]
    return max(ok, key=lambda p: p.accuracy) if ok else None


def multicloud_frontier(
    local_solved: Mapping[str, bool],
    cloud_tiers: Mapping[str, tuple[Mapping[str, bool], Mapping[str, float]]],
    *,
    local_cost_per_task: float = SATURATED_LOCAL_COST,
) -> list[OperatingPoint]:
    """Multi-cloud cost-aware ORACLE: escalate each local failure to the CHEAPEST tier that solves it.

    Generalizes :func:`oracle_frontier` from one cloud model to a *menu* of cloud tiers. For every
    task the local model fails, look across all tiers and pick the **cheapest tier that actually
    solves it** (peeking at true per-tier solve + cost — this is an oracle). Tasks no tier solves are
    unrecoverable and never escalated. The recoverable failures are then escalated cheapest-first,
    tracing cumulative accuracy vs $/task. This is the tightest single-router upper bound when the
    router may choose *which* cloud to call per task.

    ``cloud_tiers`` maps ``tier_name → (solved, cost)`` where ``solved[t]`` / ``cost[t]`` are that
    tier's per-task verifier truth and billed dollars. The shared task set is the intersection of
    ``local_solved`` and every tier's ``solved``. Returns all-local plus one point per escalation.
    """
    if not cloud_tiers:
        raise ValueError("need at least one cloud tier")
    solved_maps = [s for s, _ in cloud_tiers.values()]
    tasks = _shared(local_solved, *solved_maps)
    n = len(tasks)
    if n == 0:
        raise ValueError("no shared tasks across local + cloud tiers")
    base = sum(local_solved[t] for t in tasks) / n
    pts = [OperatingPoint(0, base, local_cost_per_task, "all-local")]

    # cheapest tier that solves each local failure (skip failures no tier solves)
    cheapest: list[tuple[str, str, float]] = []  # (task, tier, cost)
    for t in tasks:
        if local_solved[t]:
            continue
        best: tuple[str, float] | None = None
        for name, (csolved, ccost) in cloud_tiers.items():
            if csolved.get(t) and (best is None or float(ccost.get(t, 0.0)) < best[1]):
                best = (name, float(ccost.get(t, 0.0)))
        if best is not None:
            cheapest.append((t, best[0], best[1]))
    cheapest.sort(key=lambda x: x[2])

    solved_ct = sum(local_solved[t] for t in tasks)
    esc_cost = 0.0
    for i, (_t, tier, c) in enumerate(cheapest, 1):
        solved_ct += 1
        esc_cost += c
        pts.append(OperatingPoint(i, solved_ct / n, local_cost_per_task + esc_cost / n, f"multicloud+{i}:{tier}"))
    return pts
