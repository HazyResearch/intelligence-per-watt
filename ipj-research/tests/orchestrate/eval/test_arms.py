from __future__ import annotations

from ipj.orchestrate.eval.arms import build_arm
from ipj.orchestrate.harness.workers import DEFAULT_POOL_SPEC


class _FakeClient:
    def __init__(self, base_model, adapter_dir=None, **kw):
        self.adapter_dir = adapter_dir

    def complete(self, messages, **p):  # pragma: no cover - not called here
        raise NotImplementedError


def _factory(base_model, adapter_dir=None, **kw):
    return _FakeClient(base_model, adapter_dir, **kw)


def test_trained_arm_has_full_pool_and_adapter():
    brain, dispatch, pool = build_arm(
        "trained", "Qwen/Qwen3.5-0.8B", "runs/sft/pilot-smoke", DEFAULT_POOL_SPEC, client_factory=_factory
    )
    assert len(pool.card_catalog()) == len(DEFAULT_POOL_SPEC["workers"])
    assert brain.client.adapter_dir == "runs/sft/pilot-smoke"


def test_base_solo_arm_has_empty_pool_and_no_adapter():
    brain, dispatch, pool = build_arm(
        "base_solo", "Qwen/Qwen3.5-0.8B", "runs/sft/pilot-smoke", DEFAULT_POOL_SPEC, client_factory=_factory
    )
    assert pool.card_catalog() == []          # tools-only: no workers to delegate to
    assert brain.client.adapter_dir is None   # base model, untrained
