from __future__ import annotations

from ipj.orchestrate.routing.signal import escalate_when_oracle, escalate_when_realized


def test_realized_escalates_only_errored():
    assert escalate_when_realized("errored") is True
    assert escalate_when_realized("unresolved") is False
    assert escalate_when_realized("resolved") is False


def test_oracle_escalates_anything_unsolved():
    assert escalate_when_oracle("errored") is True
    assert escalate_when_oracle("unresolved") is True
    assert escalate_when_oracle("resolved") is False
