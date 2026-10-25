"""Shared fixtures and test doubles for the orchestrate suite."""

from __future__ import annotations

import pytest

from ipj.orchestrate.harness.actions import parse_action
from ipj.orchestrate.harness.brain import BrainStep
from ipj.orchestrate.harness.dispatch import WorkerResult
from ipj.orchestrate.harness.trace import Outcome, Step, TaskMeta, Trajectory


@pytest.fixture
def make_traj():
    """Factory for Trajectory fixtures spanning the knobs the filter tests vary.

    orch_text encodes both `uniq` (cross-trace identity) and the step index, so two calls with the
    same task_id/uniq/n_steps dedup as identical while a different `uniq` stays distinct. Pass
    with_outcome=False for an unscored trajectory (outcome=None), answer=None to omit the answer.
    """

    def _make(task_id="t1", *, success=True, n_steps=1, answer="x", dollars=0.1,
              tags=("self",), tokens_out=5, uniq=0, with_outcome=True):
        steps = [
            Step(index=i, action_type="finish", orch_text=f'{{"u": {uniq}, "i": {i}}}')
            for i in range(n_steps)
        ]
        outcome = Outcome(success, "verifier", dollars, None, 0.0, 0, tokens_out) if with_outcome else None
        return Trajectory(
            task=TaskMeta("swe-gym", task_id, "swe", "p", True), brain="t",
            worker_pool_snapshot={}, budget_card="low", steps=steps, outcome=outcome,
            strategy_tags=list(tags), final_answer=answer,
        )

    return _make


class ScriptBrain:
    """Emits a fixed list of action JSON strings, one per turn (7 in / 3 out tokens, $0.001 each)."""

    def __init__(self, emissions):
        self._emissions = list(emissions)
        self._i = 0

    def propose(self, ctx):
        raw = self._emissions[self._i]
        self._i += 1
        return BrainStep(action=parse_action(raw), tokens_in=7, tokens_out=3, dollars=0.001)


class FakeDispatch:
    """Worker dispatch stub: call_worker echoes alias/sub_prompt with cost+energy; delegate returns a final answer."""

    def call_worker(self, alias, sub_prompt, context):
        return WorkerResult(content=f"worker:{alias}:{sub_prompt}", dollars=0.02, gpu_energy_j=9.0)

    def delegate_task(self, alias, task_prompt):
        return WorkerResult(content="DELEGATED-ANSWER", dollars=0.5)


class FakeSession:
    """Mimics terminal-bench's TmuxSession: records the keys it is sent and returns the
    full scrollback from capture_pane (no get_incremental_output)."""

    def __init__(self, output):
        self._output = output
        self.sent = []

    def send_keys(self, keys, block=False, min_timeout_sec=0.0, max_timeout_sec=180.0):
        self.sent.append({"keys": keys, "block": block, "max_timeout_sec": max_timeout_sec})

    def capture_pane(self, capture_entire=False):
        return self._output
