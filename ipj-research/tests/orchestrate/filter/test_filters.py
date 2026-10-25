from __future__ import annotations

import pytest

from ipj.orchestrate.filter.filters import (
    DropReason,
    dedup,
    diversity_summary,
    gate_trace,
    select_cost_optimal,
)


def test_gate_keeps_successful_answered(make_traj):
    assert gate_trace(make_traj("a")) is None


def test_gate_drops_failed_empty_unanswered(make_traj):
    assert gate_trace(make_traj("a", success=False)) is DropReason.FAILED
    assert gate_trace(make_traj("a", n_steps=0)) is DropReason.EMPTY
    assert gate_trace(make_traj("a", answer=None)) is DropReason.NO_ANSWER


def test_unknown_cost_is_not_eligible_for_cost_optimal_selection(make_traj):
    unknown = make_traj("a")
    unknown.outcome.total_dollars = None
    assert gate_trace(unknown) is DropReason.UNKNOWN_COST
    kept, dropped = select_cost_optimal([unknown])
    assert kept == []
    assert dropped == [unknown]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, True, "0.1"])
def test_invalid_cost_is_treated_as_unknown(make_traj, invalid):
    trajectory = make_traj("a")
    trajectory.outcome.total_dollars = invalid
    assert gate_trace(trajectory) is DropReason.UNKNOWN_COST


def test_dedup_drops_identical(make_traj):
    a, b = make_traj("t"), make_traj("t")  # same task_id + same single orch_text
    kept, dropped = dedup([a, b])
    assert len(kept) == 1 and len(dropped) == 1


def test_select_cost_optimal_keeps_cheapest_per_task(make_traj):
    cheap = make_traj("t", dollars=0.2)
    mid = make_traj("t", dollars=0.5)
    pricey = make_traj("t", dollars=0.9)
    other = make_traj("u", dollars=1.0)
    kept, dropped = select_cost_optimal([mid, cheap, pricey, other])
    kept_ids = {(k.task.task_id, k.outcome.total_dollars) for k in kept}
    assert kept_ids == {("t", 0.2), ("u", 1.0)}
    assert len(dropped) == 2


def test_select_cost_optimal_tiebreaks_on_lower_tokens_out(make_traj):
    # Equal dollars: the (dollars, tokens_out, len) key must break the tie on tokens_out.
    fewer = make_traj("t", dollars=0.5, tokens_out=10)
    more = make_traj("t", dollars=0.5, tokens_out=20)
    kept, dropped = select_cost_optimal([more, fewer])
    assert kept == [fewer] and dropped == [more]


def test_select_cost_optimal_ranks_unknown_tokens_after_known_tokens(make_traj):
    known = make_traj("t", dollars=0.5, tokens_out=20)
    unknown = make_traj("t", dollars=0.5)
    unknown.outcome.total_tokens_out = None
    kept, dropped = select_cost_optimal([unknown, known])
    assert kept == [known]
    assert dropped == [unknown]


def test_diversity_summary_counts_tags(make_traj):
    d = diversity_summary([make_traj("a", tags=("self", "worker:cloud-frontier")), make_traj("b", tags=("self",))])
    assert d == {"self": 2, "worker:cloud-frontier": 1}
