from __future__ import annotations

import json
import threading
import time

import pytest

from ipj.orchestrate.generate.driver import GenResult, finalize_trace, generate
from ipj.orchestrate.generate.swe_tasks import SweTask
from ipj.orchestrate.harness.trace import Step, TaskMeta, Trajectory


def _traj(dollars):
    steps = [Step(index=0, action_type="bash", orch_text="{}", dollars=dollars)]
    return Trajectory(task=TaskMeta("swe-gym", "i", "swe", "p", True), brain="t",
                      worker_pool_snapshot={}, budget_card="low", steps=steps, final_answer="x")


def _task(iid):
    return SweTask(iid, "a/b", "c", "p", "d", ["t::a"], [], "img")


# The real converter pulls in terminal_bench (a heavy optional dep); inject a no-op that just returns
# the would-be task dir so the driver's *scheduling* logic is unit-testable in isolation.
def _fake_convert(task, tasks_root):
    return tasks_root / task.instance_id


def test_finalize_sets_outcome_from_verdict():
    t = finalize_trace(_traj(0.01), is_resolved=True)
    assert t.outcome is not None and t.outcome.success is True
    assert t.outcome.total_dollars == 0.01


def test_generate_runs_each_task_and_writes_traces(tmp_path):
    def run_one(task, task_dir):
        return _traj(0.02), task.instance_id.endswith("ok")

    results = generate([_task("x-ok"), _task("y-no")], tmp_path, run_one=run_one, convert=_fake_convert)
    assert {r.instance_id: r.resolved for r in results} == {"x-ok": True, "y-no": False}
    assert all(isinstance(r, GenResult) for r in results)
    saved = json.loads((tmp_path / "traces" / "x-ok.json").read_text())
    assert saved["outcome"]["success"] is True


def test_generate_respects_budget(tmp_path):
    dropped: list[str] = []

    def run_one(task, task_dir):
        return _traj(1.0), True

    results = generate([_task("a"), _task("b"), _task("c")], tmp_path, run_one=run_one,
                       budget=1.5, convert=_fake_convert, log=dropped.append)
    # first task costs 1.0 (cum 1.0 < 1.5 -> runs), second pushes cum to 2.0 -> stops before third
    assert [r.instance_id for r in results] == ["a", "b"]
    assert any("drop c" in m for m in dropped)  # the dropped task is logged, not silently truncated


def test_generate_never_exceeds_n_concurrent_in_flight(tmp_path):
    lock = threading.Lock()
    state = {"cur": 0, "max": 0}

    def run_one(task, task_dir):
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.02)  # hold the slot so overlap is observable
        with lock:
            state["cur"] -= 1
        return _traj(0.0), True

    tasks = [_task(f"t{i}") for i in range(12)]
    results = generate(tasks, tmp_path, run_one=run_one, n_concurrent=3, convert=_fake_convert)
    assert len(results) == 12
    assert state["max"] <= 3
    assert state["max"] >= 2  # actually ran in parallel (sanity: concurrency really happened)


def test_generate_resume_skips_completed_tasks(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir(parents=True)
    (traces / "done.json").write_text("{}")  # pretend a prior run already produced this
    ran: list[str] = []
    skipped: list[str] = []

    def run_one(task, task_dir):
        ran.append(task.instance_id)
        return _traj(0.0), True

    results = generate([_task("done"), _task("fresh")], tmp_path, run_one=run_one,
                       convert=_fake_convert, log=skipped.append)
    assert ran == ["fresh"]  # the completed task was not re-run
    assert [r.instance_id for r in results] == ["fresh"]
    assert any("skip done" in m for m in skipped)


def test_generate_budget_stops_scheduling_and_logs_all_drops(tmp_path):
    logged: list[str] = []

    def run_one(task, task_dir):
        return _traj(1.0), True

    tasks = [_task(f"t{i}") for i in range(5)]
    results = generate(tasks, tmp_path, run_one=run_one, budget=2.0, n_concurrent=1,
                       convert=_fake_convert, log=logged.append)
    # t0 -> cum 1.0 (<2 continue), t1 -> cum 2.0 (>=2 stop) => t2,t3,t4 dropped and each logged
    assert [r.instance_id for r in results] == ["t0", "t1"]
    drops = [m for m in logged if "drop" in m]
    assert len(drops) == 3 and all(any(f"t{i}" in m for m in drops) for i in (2, 3, 4))


def test_generate_preserves_unknown_cost_without_budget(tmp_path):
    def run_one(task, task_dir):
        return _traj(None), True

    results = generate([_task("a")], tmp_path, run_one=run_one, convert=_fake_convert)
    assert results[0].dollars is None


def test_generate_rejects_unknown_cost_when_enforcing_budget(tmp_path):
    def run_one(task, task_dir):
        return _traj(None), True

    with pytest.raises(ValueError, match="cannot enforce budget.*cost is unknown"):
        generate(
            [_task("a")],
            tmp_path,
            run_one=run_one,
            budget=1.0,
            convert=_fake_convert,
        )
    assert not (tmp_path / "traces" / "a.json").exists()
