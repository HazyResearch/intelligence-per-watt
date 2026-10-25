from __future__ import annotations

import pytest

from ipj.efficiency.operating_curve import (
    first_at_multiplier,
    multicloud_frontier,
    oracle_frontier,
    random_line,
    routed_point,
)

# 4 tasks: t1 local-solve; t2 local-fail cloud-solve cheap; t3 local-fail cloud-solve pricey;
# t4 local-fail cloud-fail. Only t2,t3 are "gain" tasks.
LOCAL = {"t1": True, "t2": False, "t3": False, "t4": False}
CLOUD = {"t1": True, "t2": True, "t3": True, "t4": False}
COST = {"t1": 0.5, "t2": 0.1, "t3": 0.9, "t4": 0.5}


def test_oracle_frontier_escalates_gain_tasks_cheapest_first():
    pts = oracle_frontier(LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    # all-local accuracy = 1/4
    assert pts[0].n_escalated == 0 and pts[0].accuracy == pytest.approx(0.25)
    # +1: cheapest gain task t2 ($0.1) -> 2/4, $ = 0.1/4
    assert pts[1].accuracy == pytest.approx(0.5)
    assert pts[1].dollars_per_task == pytest.approx(0.1 / 4)
    # +2: t3 ($0.9) -> 3/4, $ = (0.1+0.9)/4
    assert pts[2].accuracy == pytest.approx(0.75)
    assert pts[2].dollars_per_task == pytest.approx(1.0 / 4)
    # never escalates t4 (no gain) -> frontier stops at 2 escalations
    assert len(pts) == 3
    # accuracy is monotone non-decreasing
    accs = [p.accuracy for p in pts]
    assert accs == sorted(accs)


def test_oracle_beats_random_at_same_escalation_count():
    orc = oracle_frontier(LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    rnd = random_line(LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    # at 2 escalations, oracle picks the 2 gain tasks -> 0.75; random averages -> lower
    orc2 = next(p for p in orc if p.n_escalated == 2)
    rnd2 = next(p for p in rnd if p.n_escalated == 2)
    assert orc2.accuracy > rnd2.accuracy


def test_random_line_endpoints():
    rnd = random_line(LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    assert rnd[0].accuracy == pytest.approx(0.25)   # all-local
    assert rnd[-1].accuracy == pytest.approx(0.75)  # all-cloud (3/4 cloud-solve)


def test_routed_point_uses_the_given_set():
    # escalate exactly {t2, t4}: t1 local-solve, t2 cloud-solve, t3 local-fail, t4 cloud-fail -> 2/4
    p = routed_point({"t2", "t4"}, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    assert p.n_escalated == 2
    assert p.accuracy == pytest.approx(0.5)
    assert p.dollars_per_task == pytest.approx((0.1 + 0.5) / 4)


def test_first_at_multiplier_picks_best_accuracy_within_budget():
    pts = oracle_frontier(LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    # reference $4.50; target 20x -> budget $0.225/task. oracle+1 = $0.025 (ok), oracle+2 = $0.25 (>budget)
    best = first_at_multiplier(pts, reference_dollars=4.50, target_mult=20.0)
    assert best.n_escalated == 1  # only the cheap escalation stays under 20x
    assert 4.50 / best.dollars_per_task >= 20.0


def test_first_at_multiplier_none_when_unreachable():
    pts = oracle_frontier(LOCAL, CLOUD, COST, local_cost_per_task=10.0)  # local alone already pricey
    assert first_at_multiplier(pts, reference_dollars=4.50, target_mult=20.0) is None


# Two cloud tiers for the multi-cloud oracle. Local solves t1 only.
# t2: cheap tier A solves ($0.1) < pricey tier B ($0.9) -> pick A.
# t3: only tier B solves ($0.3) -> pick B.
# t4: neither tier solves -> unrecoverable, never escalated.
TIER_A = ({"t1": True, "t2": True, "t3": False, "t4": False}, {"t1": 0.5, "t2": 0.1, "t3": 0.3, "t4": 0.5})
TIER_B = ({"t1": True, "t2": True, "t3": True, "t4": False}, {"t1": 0.5, "t2": 0.9, "t3": 0.3, "t4": 0.9})


def test_multicloud_frontier_picks_cheapest_solving_tier_cheapest_first():
    pts = multicloud_frontier(LOCAL, {"A": TIER_A, "B": TIER_B}, local_cost_per_task=0.0)
    # all-local = 1/4
    assert pts[0].n_escalated == 0 and pts[0].accuracy == pytest.approx(0.25)
    # cheapest recoverable failure is t2 via tier A ($0.1), then t3 via tier B ($0.3)
    assert pts[1].accuracy == pytest.approx(0.5)
    assert pts[1].dollars_per_task == pytest.approx(0.1 / 4)
    assert pts[1].label.endswith(":A")
    assert pts[2].accuracy == pytest.approx(0.75)
    assert pts[2].dollars_per_task == pytest.approx((0.1 + 0.3) / 4)
    assert pts[2].label.endswith(":B")
    # t4 unrecoverable -> frontier stops at 2 escalations
    assert len(pts) == 3
    accs = [p.accuracy for p in pts]
    assert accs == sorted(accs)


def test_multicloud_frontier_matches_single_tier_oracle_when_one_tier():
    # with a single tier the multi-cloud oracle must equal oracle_frontier exactly
    mc = multicloud_frontier(LOCAL, {"only": (CLOUD, COST)}, local_cost_per_task=0.0)
    orc = oracle_frontier(LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    assert [(p.accuracy, p.dollars_per_task) for p in mc] == [
        pytest.approx((p.accuracy, p.dollars_per_task)) for p in orc
    ]


def test_multicloud_frontier_requires_a_tier():
    with pytest.raises(ValueError):
        multicloud_frontier(LOCAL, {})
