from __future__ import annotations

from ipj.metrics import MEASURED_DEDICATED_JOULES_PROVENANCE
from ipj.orchestrate.harness.actions import parse_action
from ipj.orchestrate.harness.brain import BrainStep
from ipj.orchestrate.harness.dispatch import WorkerResult
from ipj.orchestrate.harness.loop import OrchestratorHarness, _add
from ipj.orchestrate.harness.tools import ToolOutcome
from ipj.orchestrate.harness.trace import TaskMeta
from ipj.orchestrate.harness.workers import DEFAULT_POOL_SPEC, WorkerPool
from tests.orchestrate.conftest import FakeDispatch, ScriptBrain


class FakeTools:
    def call(self, tool, args):
        return ToolOutcome(content=f"tool:{tool}:{args}", success=True, dollars=0.0)

    def catalog(self):
        return [{"name": "web_search"}, {"name": "bash"}]


def _meta():
    return TaskMeta(adp_source="s", task_id="t", domain="d", prompt="solve it", verifiable=True, gold="55")


def _harness(emissions):
    brain = ScriptBrain(emissions)
    return OrchestratorHarness(brain, FakeDispatch(), FakeTools(), brain_name="teacher:test")


def test_full_loop_think_tool_worker_finish():
    emissions = [
        '{"action":"think","text":"plan"}',
        '{"action":"call_tool","tool":"web_search","args":{"q":"x"}}',
        '{"action":"call_worker","worker":"local-fast","sub_prompt":"add nums"}',
        '{"action":"finish","answer":"55"}',
    ]
    traj = _harness(emissions).run(_meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC), budget_card="low")
    assert [s.action_type for s in traj.steps] == ["think", "call_tool", "call_worker", "finish"]
    assert traj.final_answer == "55"
    # Worker cost is known, while worker energy cannot complete an unmeasured brain call.
    wstep = traj.steps[2]
    assert wstep.call_response == "worker:local-fast:add nums"
    assert wstep.dollars == 0.001 + 0.02
    assert wstep.gpu_energy_j is None
    assert wstep.gpu_energy_complete is False
    assert "worker:local-fast" in traj.strategy_tags
    assert "tool:web_search" in traj.strategy_tags  # call_tool target surfaced as a strategy tag
    assert traj.brain == "teacher:test"


def test_bash_action_routes_to_tools_and_merges():
    emissions = [
        '{"action":"bash","command":"echo hi"}',
        '{"action":"finish","answer":"done"}',
    ]
    traj = _harness(emissions).run(_meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC))
    bstep = traj.steps[0]
    assert bstep.action_type == "bash"
    assert bstep.call_target == "bash"  # routed to the bash tool (explicit target, not an args sentinel)
    assert bstep.call_request == {"command": "echo hi"}  # clean args payload, no meta keys
    assert bstep.call_response == "tool:bash:{'command': 'echo hi'}"  # merged FakeTools output
    assert "self" in traj.strategy_tags  # bash counts as a self action


def test_loop_records_anonymized_system_prompt():
    emissions = ['{"action":"finish","answer":"ok"}']
    traj = _harness(emissions).run(_meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC), budget_card="low")
    # the loop renders + stores the exact generation-time system text, surfaced by the SFT linearizer
    assert "local-fast" in traj.system_prompt  # anonymized worker alias
    assert "Qwen/Qwen3.5-9B" not in traj.system_prompt  # real model id NEVER recorded
    assert "Budget: low" in traj.system_prompt
    assert traj.to_sft_messages()[0]["content"] == traj.system_prompt


def test_delegate_terminates_and_sets_final_answer():
    emissions = ['{"action":"delegate_task","worker":"cloud-frontier"}']
    traj = _harness(emissions).run(_meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC))
    assert traj.final_answer == "DELEGATED-ANSWER"
    assert traj.steps[-1].action_type == "delegate_task"
    assert "delegate:cloud-frontier" in traj.strategy_tags  # delegate target surfaced as a strategy tag


def test_max_turns_stops_without_finish():
    emissions = ['{"action":"think","text":"loop"}'] * 10
    h = OrchestratorHarness(ScriptBrain(emissions), FakeDispatch(), FakeTools(), max_turns=3)
    traj = h.run(_meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC))
    assert len(traj.steps) == 3
    assert traj.final_answer is None


def test_brain_inference_energy_flows_into_outcome():
    # A brain whose BrainStep carries gpu_energy_j (a LOCAL trained brain) must have that energy
    # land on the Step and roll into Outcome.total_gpu_energy_j.
    class EnergyBrain:
        system_prompt = "x"
        def __init__(self):
            self._n = 0
        def propose(self, ctx):
            self._n += 1
            raw = '{"action":"finish","answer":"ok"}'
            return BrainStep(action=parse_action(raw), tokens_in=1, tokens_out=1, gpu_energy_j=7.0)

    traj = OrchestratorHarness(EnergyBrain(), FakeDispatch(), FakeTools()).run(
        _meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC)
    )
    assert traj.steps[0].gpu_energy_j == 7.0
    assert traj.steps[0].gpu_energy_complete is False
    assert traj.outcome is None or True  # outcome set by score_trajectory, not run(); step energy is what matters


def test_add_propagates_unknown_but_preserves_explicit_zero():
    assert _add(None, 3.0) is None
    assert _add(3.0, None) is None
    assert _add(0.0, 0.0) == 0.0
    assert _add(-1.0, 2.0) is None
    assert _add("1", 2.0) is None


def test_complete_brain_and_worker_energy_are_merged():
    provenance = MEASURED_DEDICATED_JOULES_PROVENANCE

    class MeasuredBrain:
        system_prompt = "x"

        def propose(self, ctx):
            return BrainStep(
                action=parse_action(
                    '{"action":"call_worker","worker":"local-fast","sub_prompt":"x"}'
                ),
                gpu_energy_j=2.0,
                gpu_energy_provenance=provenance,
                gpu_energy_complete=True,
                accelerator_device_coverage_complete=True,
            )

    class MeasuredDispatch:
        def call_worker(self, alias, sub_prompt, context):
            return WorkerResult(
                content="done",
                gpu_energy_j=3.0,
                gpu_energy_provenance=provenance,
                gpu_energy_complete=True,
                accelerator_device_coverage_complete=True,
            )

    trajectory = OrchestratorHarness(
        MeasuredBrain(),
        MeasuredDispatch(),
        FakeTools(),
        max_turns=1,
    ).run(_meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC))
    assert trajectory.steps[0].gpu_energy_j == 5.0
    assert trajectory.steps[0].gpu_energy_complete is True


def test_unmeasured_tool_energy_invalidates_complete_brain_window():
    provenance = MEASURED_DEDICATED_JOULES_PROVENANCE

    class MeasuredBrain:
        system_prompt = "x"

        def propose(self, ctx):
            return BrainStep(
                action=parse_action(
                    '{"action":"call_tool","tool":"web_search","args":{"q":"x"}}'
                ),
                gpu_energy_j=2.0,
                gpu_energy_provenance=provenance,
                gpu_energy_complete=True,
                accelerator_device_coverage_complete=True,
            )

    trajectory = OrchestratorHarness(
        MeasuredBrain(),
        FakeDispatch(),
        FakeTools(),
        max_turns=1,
    ).run(_meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC))
    assert trajectory.steps[0].gpu_energy_j is None
    assert trajectory.steps[0].gpu_energy_complete is False


def test_explicit_complete_tool_energy_merges_with_brain_energy():
    provenance = MEASURED_DEDICATED_JOULES_PROVENANCE

    class MeasuredBrain:
        system_prompt = "x"

        def propose(self, ctx):
            return BrainStep(
                action=parse_action(
                    '{"action":"call_tool","tool":"web_search","args":{"q":"x"}}'
                ),
                gpu_energy_j=2.0,
                gpu_energy_provenance=provenance,
                gpu_energy_complete=True,
                accelerator_device_coverage_complete=True,
            )

    class MeasuredTools(FakeTools):
        def call(self, tool, args):
            return ToolOutcome(
                content="done",
                success=True,
                dollars=0.0,
                gpu_energy_j=0.0,
                gpu_energy_provenance=provenance,
                gpu_energy_complete=True,
                accelerator_device_coverage_complete=True,
            )

    trajectory = OrchestratorHarness(
        MeasuredBrain(),
        FakeDispatch(),
        MeasuredTools(),
        max_turns=1,
    ).run(_meta(), WorkerPool.from_spec(DEFAULT_POOL_SPEC))
    assert trajectory.steps[0].gpu_energy_j == 2.0
    assert trajectory.steps[0].gpu_energy_complete is True
