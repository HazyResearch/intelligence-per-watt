from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipj.harbor.run import RunWindow
from ipj.orchestrate.routing.v1_query import run_v1


def _write_result(dirpath: Path, resolved, unresolved, errored, cost_usd=None):
    """Write a minimal Harbor result.json under <dirpath>/job/result.json."""
    job = dirpath / "job"
    job.mkdir(parents=True, exist_ok=True)
    reward = {"1.0": [f"{t}__0" for t in resolved], "0.0": [f"{t}__0" for t in unresolved]}
    evals = {"e": {"reward_stats": {"reward": reward},
                   "exception_stats": {"AgentTimeoutError": [f"{t}__0" for t in errored]}}}
    stats = {"evals": evals, "n_tasks": len(resolved) + len(unresolved) + len(errored),
             "accuracy": len(resolved) / 4.0, "tokens_in": 1000, "tokens_out": 500}
    if cost_usd is not None:
        stats["cost_usd"] = cost_usd
    (job / "result.json").write_text(json.dumps({"stats": stats}))
    return job / "result.json"


class _FakeCap:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def joules(self, start, end): return 4000.0
    def accelerator_device_coverage_complete(self, start, end): return True


def test_run_v1_reuses_cloud_baseline(tmp_path: Path):
    local_dir = tmp_path / "local"
    # Cloud baseline: GPT-5.5 solves all 4 tasks (separate fixture file).
    cloud_baseline = _write_result(tmp_path / "cloud_base",
                                   resolved=["t1", "t2", "t3", "t4"], unresolved=[], errored=[],
                                   cost_usd=4.0)

    def fake_run(cmd, *, cwd=None, env=None):
        # Only the LOCAL pass runs; cloud is reused from the baseline file.
        assert env["OPENAI_BASE_URL"] == "http://localhost:8816/v1"
        _write_result(local_dir, resolved=["t1", "t2"], unresolved=["t3"], errored=["t4"])
        return RunWindow(start=0.0, end=10.0, returncode=0)

    res = run_v1(
        local_model="Qwen/Qwen3.6-27B-FP8", dataset="terminal-bench/terminal-bench-2-1",
        output_dir=tmp_path, local_base_url="http://localhost:8816/v1",
        cloud_baseline_result=cloud_baseline, n_tasks=4,
        energy_provenance="measured_dedicated",
        run_fn=fake_run, capture_factory=_FakeCap,
    )

    assert res.escalated == ["t4"]                       # only the errored task
    assert res.realized.n_resolved == 3                  # t1,t2 local + t4 cloud ; t3 kept local-unresolved
    assert res.oracle.n_resolved == 4                    # true ceiling: t3 AND t4 escalated, cloud solves both
    assert res.realized.accuracy == pytest.approx(0.75)
    assert res.oracle.accuracy == pytest.approx(1.0)
    assert res.realized_cell.joules == pytest.approx(4000.0)   # local energy only
    assert res.realized_cell.joules_provenance == "measured_dedicated"
    assert res.realized_cell.joules_complete is False          # cloud component is unknown
    assert res.realized_cell.dollars_basis == "gpu-hours+api"


def test_run_v1_propagates_unknown_local_energy(tmp_path: Path):
    class UnavailableCap(_FakeCap):
        def joules(self, start, end):
            return None

    local_dir = tmp_path / "local"
    cloud_baseline = _write_result(
        tmp_path / "cloud_base", resolved=["t1", "t2"], unresolved=[], errored=[], cost_usd=2.0,
    )

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(local_dir, resolved=["t1"], unresolved=[], errored=["t2"])
        return RunWindow(start=0.0, end=10.0, returncode=0)

    res = run_v1(
        local_model="local", dataset="ds", output_dir=tmp_path,
        local_base_url="http://localhost/v1", cloud_baseline_result=cloud_baseline,
        n_tasks=2, energy_provenance="measured_dedicated",
        run_fn=fake_run, capture_factory=UnavailableCap,
    )

    assert res.realized_cell.joules is None
    assert res.realized_cell.joules_provenance == "unknown"
    assert res.realized_cell.joules_complete is False


def test_run_v1_restricts_to_local_task_set(tmp_path: Path):
    """Cloud baseline has 5 tasks; local only ran 3 — cascade must not count cloud-only t4,t5."""
    local_dir = tmp_path / "local"
    cloud_baseline = _write_result(
        tmp_path / "cloud_base",
        resolved=["t1", "t2", "t3", "t4", "t5"], unresolved=[], errored=[],
        cost_usd=5.0,
    )

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(local_dir, resolved=["t1"], unresolved=["t2"], errored=["t3"])
        return RunWindow(start=0.0, end=10.0, returncode=0)

    res = run_v1(
        local_model="Qwen/Qwen3.6-27B-FP8", dataset="terminal-bench/terminal-bench-2-1",
        output_dir=tmp_path, local_base_url="http://localhost:8816/v1",
        cloud_baseline_result=cloud_baseline, n_tasks=3,
        run_fn=fake_run, capture_factory=_FakeCap,
    )

    # Cloud-only tasks t4,t5 must not inflate the denominator.
    assert res.realized.n_tasks == 3
    assert res.oracle.n_tasks == 3

    # realized: t1 (local-resolved, no escalation) + t3 (errored → cloud-resolved)
    #           t2 (unresolved → kept local, not escalated in realized mode)
    assert res.realized.n_resolved == 2
    assert res.realized_cell.joules_provenance == "unknown"
    assert res.realized_cell.joules_complete is False

    # oracle: t1 (local) + t2 (unresolved → escalated → cloud-resolved) + t3 (errored → cloud)
    assert res.oracle.n_resolved == 3


def test_run_v1_rejects_used_cloud_baseline_with_unknown_cost(tmp_path: Path):
    local_dir = tmp_path / "local"
    cloud_baseline = _write_result(
        tmp_path / "cloud_base",
        resolved=["t1", "t2"],
        unresolved=[],
        errored=[],
        cost_usd=None,
    )

    def fake_run(cmd, *, cwd=None, env=None):
        _write_result(local_dir, resolved=["t1"], unresolved=[], errored=["t2"])
        return RunWindow(start=0.0, end=10.0, returncode=0)

    with pytest.raises(ValueError, match="cloud baseline has unknown cost_usd"):
        run_v1(
            local_model="local",
            dataset="ds",
            output_dir=tmp_path,
            local_base_url="http://localhost/v1",
            cloud_baseline_result=cloud_baseline,
            n_tasks=2,
            run_fn=fake_run,
            capture_factory=_FakeCap,
        )
