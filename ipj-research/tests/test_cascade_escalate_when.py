from __future__ import annotations

import pytest

from ipj.cascade import Tier, cascade_ladder


def _tiers():
    local = Tier("local", {"t1": "resolved", "t2": "resolved", "t3": "unresolved", "t4": "errored"},
                 cost_per_task=0.01, joules_per_task=100.0)
    cloud = Tier("cloud", {"t1": "resolved", "t2": "resolved", "t3": "resolved", "t4": "unresolved"},
                 cost_per_task=0.50, joules_per_task=0.0)
    return [local, cloud]


def test_default_escalate_when_is_oracle():
    # Unchanged behavior: escalate unless resolved -> 3 solved, cloud attempts the 2 local missed.
    r = cascade_ladder(_tiers())
    assert r.n_resolved == 3
    assert r.accuracy == pytest.approx(0.75)
    assert r.tier_attempts == (4, 2)


def test_realized_escalates_only_on_errored():
    # Realized signal: escalate ONLY errored tasks. t3 (local unresolved) is NOT escalated.
    r = cascade_ladder(_tiers(), escalate_when=lambda s: s == "errored")
    # solved = t1,t2 (local) ; t3 kept local-unresolved ; t4 escalated but cloud unresolved
    assert r.n_resolved == 2
    assert r.accuracy == pytest.approx(0.5)
    assert r.tier_attempts == (4, 1)   # cloud attempts only the 1 errored task (t4)
