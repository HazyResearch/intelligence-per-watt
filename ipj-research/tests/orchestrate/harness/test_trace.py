from __future__ import annotations

import json

import pytest

from ipj.metrics import MEASURED_DEDICATED_JOULES_PROVENANCE
from ipj.orchestrate.harness.trace import (
    Outcome,
    Step,
    TaskMeta,
    Trajectory,
    derive_strategy_tags,
    score_trajectory,
    summarize_outcome,
)


def _meta():
    return TaskMeta(adp_source="smoke", task_id="t1", domain="arithmetic", prompt="sum 1..10", verifiable=True, gold="55")


def _traj():
    steps = [
        Step(index=0, action_type="think", orch_text='{"action":"think","text":"plan"}', dollars=0.001),
        Step(
            index=1,
            action_type="call_worker",
            orch_text='{"action":"call_worker","worker":"local-fast","sub_prompt":"add"}',
            call_target="local-fast",
            call_request={"sub_prompt": "add"},
            call_response="the answer is 55",
            dollars=0.002,
            gpu_energy_j=12.5,
        ),
        Step(index=2, action_type="finish", orch_text='{"action":"finish","answer":"55"}', dollars=0.0),
    ]
    return Trajectory(
        task=_meta(), brain="teacher:gpt", worker_pool_snapshot={}, budget_card="med", steps=steps, final_answer="55"
    )


def test_sft_messages_mask_only_orchestrator_tokens():
    msgs = _traj().to_sft_messages()
    # system + user + (assistant) + (assistant + tool) + (assistant)
    roles = [(m["role"], m["loss"]) for m in msgs]
    assert ("system", False) in roles
    assert ("user", False) in roles
    assert all(m["loss"] is True for m in msgs if m["role"] == "assistant")
    assert all(m["loss"] is False for m in msgs if m["role"] == "tool")
    # the worker output must be present but never trained on
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == "the answer is 55" and tool_msgs[0]["loss"] is False


def test_sft_system_turn_uses_full_render_when_populated():
    # When the loop records the real generation-time system text, the SFT linearizer must use it
    # (so training input == generation input), not the bare "Budget: ..." fallback.
    t = _traj()
    t.system_prompt = (
        "You are an efficiency-aware orchestrator.\n\nBudget: med\n\n"
        "Available workers:\n- local-fast: Fast small local model ($0.0/$0.0 per Mtok)\n\n"
        "Available tools: bash"
    )
    msgs = t.to_sft_messages()
    sys_msg = msgs[0]
    assert sys_msg["role"] == "system" and sys_msg["loss"] is False
    assert "local-fast" in sys_msg["content"]  # anonymized worker alias preserved
    assert "Budget: med" in sys_msg["content"]
    # loss flags unchanged: system not trained, assistant trained, tool not trained
    assert all(m["loss"] is True for m in msgs if m["role"] == "assistant")
    assert all(m["loss"] is False for m in msgs if m["role"] == "tool")


def test_to_dict_json_roundtrip():
    t = _traj()
    score_trajectory(t, verify=lambda ans: "55" in ans)
    d = t.to_dict()
    blob = json.dumps(d, default=str)  # must serialize without error
    back = json.loads(blob)
    assert back["budget_card"] == "med"
    assert back["final_answer"] == "55"
    assert back["brain"] == "teacher:gpt"
    assert len(back["steps"]) == 3
    assert back["steps"][1]["call_response"] == "the answer is 55"
    assert back["outcome"]["success"] is True


def test_derive_strategy_tags():
    tags = derive_strategy_tags(_traj().steps)
    assert "self" in tags  # think counts as self
    assert "worker:local-fast" in tags


def test_summarize_and_score():
    t = _traj()
    out = summarize_outcome(t.steps, success=True, source="verifier")
    assert out.total_dollars == 0.003
    # One observed component is not a complete trajectory measurement.
    assert out.total_gpu_energy_j is None
    assert out.gpu_energy_complete is False
    assert out.total_latency_s is None
    assert out.total_tokens_in is None
    assert out.total_tokens_out is None
    scored = score_trajectory(t, verify=lambda ans: "55" in ans)
    assert isinstance(scored.outcome, Outcome)
    assert scored.outcome.success is True


def test_score_failure():
    t = _traj()
    score_trajectory(t, verify=lambda ans: ans.strip() == "999")
    assert t.outcome.success is False


def test_from_dict_round_trips():
    t = _traj()  # existing helper in this file (a Trajectory with steps + final_answer)
    t.system_prompt = "action grammar + worker cards + budget"  # non-empty: MUST survive the round-trip
    score_trajectory(t, verify=lambda ans: True)  # populate outcome
    assert Trajectory.from_dict(t.to_dict()) == t
    assert Trajectory.from_dict(t.to_dict()).system_prompt == "action grammar + worker cards + budget"


def test_from_dict_handles_none_outcome():
    t = _traj()  # outcome is None until scored
    t.system_prompt = "ctx"
    assert t.outcome is None
    assert Trajectory.from_dict(t.to_dict()) == t


def test_summarize_preserves_unknown_and_explicit_zero():
    steps = [
        Step(index=0, action_type="think", orch_text="{}", dollars=0.0, tokens_in=0),
        Step(index=1, action_type="finish", orch_text="{}", dollars=None, tokens_in=1),
    ]
    outcome = summarize_outcome(steps, success=False, source="verifier")
    assert outcome.total_dollars is None
    assert outcome.total_tokens_in == 1


def test_summarize_requires_complete_measured_energy_for_every_step():
    steps = [
        Step(
            index=0,
            action_type="think",
            orch_text="{}",
            gpu_energy_j=2.0,
            gpu_energy_provenance=MEASURED_DEDICATED_JOULES_PROVENANCE,
            gpu_energy_complete=True,
            accelerator_device_coverage_complete=True,
        ),
        Step(
            index=1,
            action_type="finish",
            orch_text="{}",
            gpu_energy_j=0.0,
            gpu_energy_provenance=MEASURED_DEDICATED_JOULES_PROVENANCE,
            gpu_energy_complete=True,
            accelerator_device_coverage_complete=True,
        ),
    ]
    outcome = summarize_outcome(steps, success=True, source="verifier")
    assert outcome.total_gpu_energy_j == 2.0
    assert outcome.gpu_energy_complete is True
    assert outcome.gpu_energy_provenance == MEASURED_DEDICATED_JOULES_PROVENANCE


def test_old_trace_without_energy_integrity_fields_remains_loadable_but_ineligible():
    payload = _traj().to_dict()
    for step in payload["steps"]:
        step.pop("gpu_energy_complete")
        step.pop("gpu_energy_provenance")
    restored = Trajectory.from_dict(payload)
    assert all(not step.gpu_energy_complete for step in restored.steps)
    assert summarize_outcome(restored.steps, True, "verifier").total_gpu_energy_j is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, True, "1.0"])
def test_summarize_rejects_invalid_accounting_values(invalid):
    step = Step(
        index=0,
        action_type="finish",
        orch_text="{}",
        dollars=invalid,
        latency_s=invalid,
        gpu_energy_j=invalid,
        gpu_energy_provenance=MEASURED_DEDICATED_JOULES_PROVENANCE,
        gpu_energy_complete=True,
    )
    outcome = summarize_outcome([step], True, "verifier")
    assert outcome.total_dollars is None
    assert outcome.total_latency_s is None
    assert outcome.total_gpu_energy_j is None
    assert outcome.gpu_energy_complete is False
