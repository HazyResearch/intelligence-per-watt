from __future__ import annotations

from ipj.orchestrate.harness.tasks import ADPTask, load_smoke_tasks


def test_smoke_tasks_are_verifiable():
    tasks = load_smoke_tasks()
    assert len(tasks) >= 3
    for t in tasks:
        assert isinstance(t, ADPTask)
        assert t.meta.verifiable is True
        assert t.meta.gold is not None


def test_verifier_accepts_gold_rejects_wrong():
    t = load_smoke_tasks()[0]
    assert t.verify(t.meta.gold) is True
    assert t.verify("definitely not the answer") is False
