"""Approach-A cascade frontier — compute the (accuracy, $, Joules) of a difficulty-routed
local->cloud escalation ladder from per-(model, task) solve outcomes.

Each task is attempted down an ordered ladder of tiers (cheapest local model first, cloud
last); you pay for every tier you attempt (cheap attempt + escalation), and a task counts
as solved once any attempted tier resolves it. With "oracle" escalation (escalate exactly
the unresolved), this is the UPPER BOUND on what a real router — which must decide to
escalate without seeing the answer — can achieve, so it's the ceiling for the headline.

Feed it per-task outcomes from ipj.harbor.per_task.per_task_outcomes and per-task cost/J
from each cell (Cell.dollars / Cell.n_tasks, Cell.joules / Cell.n_tasks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Tier:
    name: str
    outcomes: dict[str, str]          # {task: "resolved"|"unresolved"|"errored"}
    cost_per_task: float              # $ charged per task attempted at this tier
    joules_per_task: float | None = None
    flops_per_task: float | None = None


@dataclass(frozen=True)
class CascadeResult:
    accuracy: float
    dollars: float
    joules: float | None
    flops: float | None
    n_tasks: int
    n_resolved: int
    tier_attempts: tuple[int, ...]    # tasks attempted at each tier (in ladder order)
    tier_resolved: tuple[int, ...]    # tasks first resolved at each tier


def cascade_ladder(
    tiers: list[Tier], tasks: set[str] | None = None, escalate_when: "Callable[[str], bool] | None" = None
) -> CascadeResult:
    """Route each task down the ladder, escalating per `escalate_when` (default: oracle — stop at the first resolving tier).

    Cost/Joules accumulate for every tier attempted (you pay the cheap try AND each
    escalation). Joules are reported only if every tier has joules_per_task.
    """
    if not tiers:
        raise ValueError("need at least one tier")
    if escalate_when is None:
        escalate_when = lambda s: s != "resolved"  # noqa: E731  (oracle: stop at first resolved)
    all_tasks = tasks if tasks is not None else set().union(*(set(t.outcomes) for t in tiers))
    n = len(all_tasks)
    has_j = all(t.joules_per_task is not None for t in tiers)
    has_f = all(t.flops_per_task is not None for t in tiers)
    dollars = 0.0
    joules = 0.0
    flops = 0.0
    resolved = 0
    attempts = [0] * len(tiers)
    resolved_at = [0] * len(tiers)
    for task in all_tasks:
        for i, t in enumerate(tiers):
            attempts[i] += 1
            dollars += t.cost_per_task
            if has_j:
                joules += t.joules_per_task  # type: ignore[operator]
            if has_f:
                flops += t.flops_per_task  # type: ignore[operator]
            status = t.outcomes.get(task, "errored")
            is_last = i == len(tiers) - 1
            if not escalate_when(status) or is_last:
                if status == "resolved":
                    resolved += 1
                    resolved_at[i] += 1
                break
    return CascadeResult(
        accuracy=resolved / n if n else 0.0,
        dollars=dollars,
        joules=joules if has_j else None,
        flops=flops if has_f else None,
        n_tasks=n,
        n_resolved=resolved,
        tier_attempts=tuple(attempts),
        tier_resolved=tuple(resolved_at),
    )
