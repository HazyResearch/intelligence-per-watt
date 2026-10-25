from __future__ import annotations

from ipj.orchestrate.harness.brain import EscalatingBrain
from tests.orchestrate.conftest import ScriptBrain


class _Ctx:
    """Minimal stand-in for OrchestratorContext: EscalatingBrain only reads ctx.history length."""

    def __init__(self, n_history: int) -> None:
        self.history = list(range(n_history))


def test_defers_to_inner_before_threshold():
    inner = ScriptBrain(['{"action":"bash","command":"ls"}'] * 5)
    eb = EscalatingBrain(inner, force_after=3, worker="cloud-frontier")
    bs = eb.propose(_Ctx(0))  # below threshold -> inner model's own action
    assert bs.action.type == "bash"


def test_forces_delegate_at_threshold():
    inner = ScriptBrain(['{"action":"bash","command":"ls"}'] * 5)
    eb = EscalatingBrain(inner, force_after=3, worker="cloud-frontier")
    bs = eb.propose(_Ctx(3))  # at/after force_after turns -> forced escalation, inner NOT called
    assert bs.action.type == "delegate_task"
    assert bs.action.target == "cloud-frontier"


def test_inherits_inner_system_prompt():
    inner = ScriptBrain([])
    inner.system_prompt = "WORKER-CARDS-SYSTEM"
    assert EscalatingBrain(inner).system_prompt == "WORKER-CARDS-SYSTEM"
