"""Unit tests for the score-cache → escalate-set adapter (§7.3)."""

from __future__ import annotations

import pytest

from ipj.efficiency.operating_curve import routed_point
from ipj.router.scorecache import escalate_set, ranked, routed_curve

# 4 tasks: t1 local-solve; t2/t3 local-fail cloud-solve; t4 local-fail cloud-fail.
LOCAL = {"t1": True, "t2": False, "t3": False, "t4": False}
CLOUD = {"t1": True, "t2": True, "t3": True, "t4": False}
COST = {"t1": 0.5, "t2": 0.1, "t3": 0.9, "t4": 0.5}
PREDS = {"t1": 0.9, "t2": 0.1, "t3": 0.5, "t4": 0.3}


def test_ranked_is_ascending_by_p_solve():
    assert ranked(PREDS) == ["t2", "t4", "t3", "t1"]


def test_ranked_breaks_ties_by_task_name():
    preds = {"b": 0.5, "a": 0.5, "c": 0.5}
    assert ranked(preds) == ["a", "b", "c"]


def test_escalate_set_is_e_lowest_p_solve():
    assert escalate_set(PREDS, 0) == set()
    assert escalate_set(PREDS, 1) == {"t2"}
    assert escalate_set(PREDS, 2) == {"t2", "t4"}
    assert escalate_set(PREDS, 4) == {"t1", "t2", "t3", "t4"}


def test_escalate_set_clamps_beyond_n():
    assert escalate_set(PREDS, 99) == set(PREDS)


def test_escalate_set_negative_raises():
    with pytest.raises(ValueError):
        escalate_set(PREDS, -1)


def test_routed_curve_length_is_n_plus_one():
    curve = routed_curve(PREDS, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    assert len(curve) == len(PREDS) + 1


def test_routed_curve_matches_routed_point_at_each_e():
    curve = routed_curve(PREDS, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    order = ranked(PREDS)
    for e, point in enumerate(curve):
        expected = routed_point(set(order[:e]), LOCAL, CLOUD, COST, local_cost_per_task=0.0)
        assert point.n_escalated == expected.n_escalated
        assert point.accuracy == pytest.approx(expected.accuracy)
        assert point.dollars_per_task == pytest.approx(expected.dollars_per_task)


def test_routed_curve_endpoints_are_all_local_and_all_cloud():
    curve = routed_curve(PREDS, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    # e=0: all-local → accuracy = local solve rate (1/4)
    assert curve[0].n_escalated == 0
    assert curve[0].accuracy == pytest.approx(1 / 4)
    # e=n: everything escalated → accuracy = cloud solve rate (3/4)
    assert curve[-1].n_escalated == 4
    assert curve[-1].accuracy == pytest.approx(3 / 4)
