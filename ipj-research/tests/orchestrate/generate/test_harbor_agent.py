from __future__ import annotations

import json

import pytest

# `ipj.orchestrate.generate.harbor_agent` imports terminal_bench at top level and raises a
# helpful ImportError if it is missing. Skip the whole file at collection time when the
# `ipj[orchestrate]` extra is absent, and tag surviving tests `integration` so the fast
# tier stays deselected even when the extra IS installed.
pytest.importorskip("terminal_bench")
pytestmark = pytest.mark.integration

from ipj.orchestrate.generate.harbor_agent import IPJOrchestrator  # noqa: E402
from ipj.orchestrate.harness.dispatch import WorkerDispatch  # noqa: E402
from ipj.orchestrate.harness.tools import ToolOutcome  # noqa: E402,F401  (ensures harness import path)
from ipj.orchestrate.harness.workers import DEFAULT_POOL_SPEC, WorkerPool  # noqa: E402
from tests.orchestrate.conftest import FakeDispatch, FakeSession, ScriptBrain  # noqa: E402


def test_perform_task_runs_loop_writes_trace_returns_tokens(tmp_path):
    emissions = [
        '{"action":"bash","command":"ls"}',
        '{"action":"finish","answer":"patched"}',
    ]
    agent = IPJOrchestrator(
        brain=ScriptBrain(emissions),
        dispatch=FakeDispatch(),
        pool=WorkerPool.from_spec(DEFAULT_POOL_SPEC),
        trace_dir=tmp_path,
        brain_name="teacher:test",
        task_id="getmoto__moto-7365",
        max_turns=5,
    )
    result = agent.perform_task("fix the bug", FakeSession("listing\n__RC=0\n"))
    # AgentResult token totals come from the trajectory steps (2 turns x 7 in / 3 out)
    assert result.total_input_tokens == 14
    assert result.total_output_tokens == 6
    # the trace was written and is well-formed
    trace = json.loads((tmp_path / "getmoto__moto-7365.json").read_text())
    assert [s["action_type"] for s in trace["steps"]] == ["bash", "finish"]
    assert agent.last_trajectory is not None
    assert agent.last_trajectory.final_answer == "patched"


def test_perform_task_wires_container_delegate_runner(tmp_path):
    # The feature: delegation must ACT in this container. perform_task installs the container-acting
    # delegate runner on the live dispatch so a delegate_task hands off to a sub-agent in /testbed.
    dispatch = WorkerDispatch(WorkerPool.from_spec(DEFAULT_POOL_SPEC), server_factory=lambda b: None)
    assert dispatch.delegate_runner is None  # not wired until perform_task sees the session
    agent = IPJOrchestrator(
        brain=ScriptBrain(['{"action":"finish","answer":"ok"}']),
        dispatch=dispatch,
        pool=WorkerPool.from_spec(DEFAULT_POOL_SPEC),
        trace_dir=tmp_path,
        max_turns=5,
    )
    agent.perform_task("do it", FakeSession("ok\n__RC=0\n"))
    assert callable(dispatch.delegate_runner)  # container delegate runner now bound to the session
