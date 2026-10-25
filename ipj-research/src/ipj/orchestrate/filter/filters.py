"""Quality gates, dedup, and cost-optimal selection over verified Trajectories."""

from __future__ import annotations

from collections import Counter
from enum import Enum

from ipj.orchestrate.harness.trace import (
    Trajectory,
    is_valid_nonnegative_number,
    is_valid_token_count,
)


class DropReason(str, Enum):
    FAILED = "failed"
    EMPTY = "empty"
    NO_ANSWER = "no_answer"
    DUPLICATE = "duplicate"
    UNKNOWN_COST = "unknown_cost"
    NOT_COST_OPTIMAL = "not_cost_optimal"


def gate_trace(traj: Trajectory) -> DropReason | None:
    """None = keep. Keep iff verified-successful, non-empty, and cleanly terminated (has an answer)."""
    if traj.outcome is None or not traj.outcome.success:
        return DropReason.FAILED
    if len(traj.steps) < 1:
        return DropReason.EMPTY
    if traj.final_answer is None:
        return DropReason.NO_ANSWER
    if not is_valid_nonnegative_number(traj.outcome.total_dollars):
        return DropReason.UNKNOWN_COST
    return None


def _signature(traj: Trajectory) -> tuple[str, tuple[str, ...]]:
    return (traj.task.task_id, tuple(s.orch_text for s in traj.steps))


def dedup(trajs: list[Trajectory]) -> tuple[list[Trajectory], list[Trajectory]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    kept: list[Trajectory] = []
    dropped: list[Trajectory] = []
    for t in trajs:
        sig = _signature(t)
        if sig in seen:
            dropped.append(t)
        else:
            seen.add(sig)
            kept.append(t)
    return kept, dropped


def _cost_key(traj: Trajectory) -> tuple[float, int, int, int]:
    o = traj.outcome
    if o is None or not is_valid_nonnegative_number(o.total_dollars):
        raise ValueError("cost-optimal selection requires a known total cost")
    tokens_out_unknown = int(not is_valid_token_count(o.total_tokens_out))
    tokens_out = o.total_tokens_out if is_valid_token_count(o.total_tokens_out) else 0
    return (o.total_dollars, tokens_out_unknown, tokens_out, len(traj.steps))


def select_cost_optimal(trajs: list[Trajectory]) -> tuple[list[Trajectory], list[Trajectory]]:
    by_task: dict[str, list[Trajectory]] = {}
    for t in trajs:
        by_task.setdefault(t.task.task_id, []).append(t)
    kept: list[Trajectory] = []
    dropped: list[Trajectory] = []
    for group in by_task.values():
        eligible = [
            traj
            for traj in group
            if traj.outcome is not None
            and is_valid_nonnegative_number(traj.outcome.total_dollars)
        ]
        if not eligible:
            dropped.extend(group)
            continue
        best = min(eligible, key=_cost_key)
        kept.append(best)
        dropped.extend(t for t in group if t is not best)
    return kept, dropped


def diversity_summary(trajs: list[Trajectory]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for t in trajs:
        counter.update(t.strategy_tags)
    return dict(counter)
