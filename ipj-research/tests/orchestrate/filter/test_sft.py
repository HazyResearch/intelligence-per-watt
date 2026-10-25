from __future__ import annotations

import json

from ipj.orchestrate.filter.sft import to_sft_example, write_jsonl
from ipj.orchestrate.harness.trace import Outcome, Step, TaskMeta, Trajectory


def _traj():
    steps = [
        Step(index=0, action_type="call_worker", orch_text='{"action":"call_worker"}',
             call_response="worker said x"),
        Step(index=1, action_type="finish", orch_text='{"action":"finish","answer":"x"}'),
    ]
    return Trajectory(
        task=TaskMeta("swe-gym", "t1", "swe", "fix it", True), brain="teacher:gpt-5.5",
        worker_pool_snapshot={}, budget_card="low", steps=steps,
        outcome=Outcome(True, "verifier", 0.42, None, 1.0, 10, 5),
        strategy_tags=["self", "worker:cloud-frontier"], final_answer="x",
    )


def test_to_sft_example_messages_and_metadata():
    ex = to_sft_example(_traj())
    msgs = ex["messages"]
    # system & user turns are context-only, assistant turns loss-on, tool turns loss-off
    assert msgs[0]["role"] == "system" and msgs[0]["loss"] is False
    assert msgs[1]["role"] == "user" and msgs[1]["loss"] is False
    assert all(m["loss"] is True for m in msgs if m["role"] == "assistant")
    assert all(m["loss"] is False for m in msgs if m["role"] == "tool")
    md = ex["metadata"]
    assert md["task_id"] == "t1" and md["success"] is True and md["dollars"] == 0.42
    assert md["gpu_energy_provenance"] == "unknown"
    assert md["gpu_energy_complete"] is False
    assert md["strategy_tags"] == ["self", "worker:cloud-frontier"]
    assert md["budget_card"] == "low" and md["brain"] == "teacher:gpt-5.5" and md["n_turns"] == 2


def test_write_jsonl_one_line_per_example(tmp_path):
    out = tmp_path / "sub" / "sft.jsonl"
    write_jsonl([to_sft_example(_traj()), to_sft_example(_traj())], out)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["metadata"]["task_id"] == "t1"
