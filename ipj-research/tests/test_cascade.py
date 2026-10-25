from __future__ import annotations

import pytest

from ipj.cascade import Tier, cascade_ladder


def test_two_tier_local_then_cloud():
    # 4 tasks: local solves t1,t2; cloud solves t3 (escalated), t4 unsolved by both.
    local = Tier("local", {"t1": "resolved", "t2": "resolved", "t3": "unresolved", "t4": "errored"},
                 cost_per_task=0.01, joules_per_task=100.0)
    cloud = Tier("cloud", {"t1": "resolved", "t2": "resolved", "t3": "resolved", "t4": "unresolved"},
                 cost_per_task=0.50, joules_per_task=0.0)
    r = cascade_ladder([local, cloud])
    assert r.n_tasks == 4
    assert r.n_resolved == 3           # t1,t2 (local) + t3 (cloud) ; t4 nobody
    assert r.accuracy == pytest.approx(0.75)
    # cost: local attempted on all 4 (0.04) + cloud on the 2 escalated t3,t4 (1.00) = 1.04
    assert r.dollars == pytest.approx(0.04 + 1.00)
    assert r.joules == pytest.approx(100.0 * 4 + 0.0 * 2)
    assert r.tier_attempts == (4, 2)   # local all 4, cloud only the 2 local missed
    assert r.tier_resolved == (2, 1)


def test_joules_none_when_a_tier_lacks_energy():
    a = Tier("a", {"t1": "resolved"}, cost_per_task=0.01, joules_per_task=100.0)
    b = Tier("b", {"t1": "resolved"}, cost_per_task=0.5)  # no joules
    assert cascade_ladder([a, b]).joules is None


def test_cheap_tier_handles_everything_no_escalation():
    local = Tier("local", {"t1": "resolved", "t2": "resolved"}, cost_per_task=0.01, joules_per_task=50.0)
    cloud = Tier("cloud", {"t1": "resolved", "t2": "resolved"}, cost_per_task=0.5, joules_per_task=0.0)
    r = cascade_ladder([local, cloud])
    assert r.accuracy == pytest.approx(1.0)
    assert r.dollars == pytest.approx(0.02)        # cloud never attempted
    assert r.tier_attempts == (2, 0)


def test_empty_tiers_raises():
    with pytest.raises(ValueError):
        cascade_ladder([])


def test_cascade_flops_accumulates_over_attempts():
    local = Tier("local", {"t1": "resolved", "t2": "unresolved"}, cost_per_task=0.01,
                 joules_per_task=100.0, flops_per_task=1e12)
    cloud = Tier("cloud", {"t1": "resolved", "t2": "resolved"}, cost_per_task=0.5,
                 joules_per_task=0.0, flops_per_task=5e12)
    r = cascade_ladder([local, cloud])
    # local attempts both (2), cloud attempts only the escalated t2 (1)
    assert r.flops == pytest.approx(2 * 1e12 + 1 * 5e12)


def test_cascade_flops_none_when_a_tier_lacks_it():
    a = Tier("a", {"t1": "resolved"}, cost_per_task=0.01, flops_per_task=1e12)
    b = Tier("b", {"t1": "resolved"}, cost_per_task=0.5)  # no flops_per_task
    assert cascade_ladder([a, b]).flops is None
