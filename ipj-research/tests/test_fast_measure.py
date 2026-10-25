from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipj.fast_measure import (
    build_fast_cell,
    count_completed,
    cutoff_reached,
    latest_jobs_run,
    summarize_run,
    task_status_map,
    trapezoid_joules,
)


def _mk_task(run: Path, task: str, *, reward: float | None = None, errored: bool = False,
             tin: int = 0, tout: int = 0) -> None:
    """Write a per-task result.json in Harbor 0.13.2's real shape."""
    d = run / f"{task}__abc"
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "task_name": task,
        "agent_result": {"n_input_tokens": tin, "n_output_tokens": tout, "n_cache_tokens": 0},
        "verifier_result": {"rewards": {"reward": 0.0 if reward is None else reward}},
        "exception_info": {"exception_type": "AgentTimeoutError"} if errored else None,
    }
    (d / "result.json").write_text(json.dumps(rec))


def test_count_and_latest_run(tmp_path: Path):
    run = tmp_path / "jobs" / "2026-06-20__00-00-00"
    for i in range(3):
        _mk_task(run, f"t{i}", reward=1.0)
    assert count_completed(run) == 3
    assert latest_jobs_run(tmp_path / "jobs") == run
    assert latest_jobs_run(tmp_path / "nope") is None


def test_cutoff_reached_uses_ceil(tmp_path: Path):
    run = tmp_path / "jobs" / "r"
    for i in range(3):
        _mk_task(run, f"t{i}", reward=1.0)
    assert cutoff_reached(run, n_total=4, frac=0.75) is True    # ceil(0.75*4)=3
    assert cutoff_reached(run, n_total=10, frac=0.75) is False  # need ceil(7.5)=8


def test_trapezoid_joules_constant_power():
    # constant 100 W for 10 s = 1000 J
    s = [(0.0, 100.0), (5.0, 100.0), (10.0, 100.0)]
    assert trapezoid_joules(s, 0.0, 10.0) == pytest.approx(1000.0)
    assert trapezoid_joules([], 0.0, 10.0) == 0.0


def test_summarize_scores_missing_as_unsolved_and_sums_all_tokens(tmp_path: Path):
    run = tmp_path / "jobs" / "r"
    _mk_task(run, "a", reward=1.0, tin=100, tout=10)   # resolved
    _mk_task(run, "b", reward=0.0, tin=50, tout=5)     # unresolved
    _mk_task(run, "c", errored=True, tin=30, tout=3)   # errored — tokens still count
    s = summarize_run(run, n_total=4)                  # 4th task never started
    assert s.n_resolved == 1
    assert s.accuracy == pytest.approx(0.25)           # denominator is the FULL n_total
    assert s.n_terminal == 3
    assert s.tokens_in == 180 and s.tokens_out == 18   # a+b+c, including the errored one


def test_task_status_map(tmp_path: Path):
    # {task: status} from per-task result.jsons — the cascade local-tier outcomes for --fast.
    run = tmp_path / "jobs" / "r"
    _mk_task(run, "a", reward=1.0)
    _mk_task(run, "b", reward=0.0)
    _mk_task(run, "c", errored=True)
    assert task_status_map(run) == {"a": "resolved", "b": "unresolved", "c": "errored"}


def test_task_name_is_bare_not_dataset_prefixed(tmp_path: Path):
    # Harbor stores task_name as "terminal-bench/<task>"; the cascade needs the bare name to
    # match the cloud outcomes (per_task_outcomes), else the task universe doubles.
    run = tmp_path / "jobs" / "r"
    d = run / "feal-cryptanalysis__abc"
    d.mkdir(parents=True)
    d.joinpath("result.json").write_text(json.dumps({
        "task_name": "terminal-bench/feal-cryptanalysis",
        "agent_result": {"n_input_tokens": 1, "n_output_tokens": 1},
        "verifier_result": {"rewards": {"reward": 1.0}}, "exception_info": None,
    }))
    assert task_status_map(run) == {"feal-cryptanalysis": "resolved"}


def test_build_fast_cell(tmp_path: Path):
    run = tmp_path / "jobs" / "r"
    _mk_task(run, "a", reward=1.0, tin=100, tout=10)
    # 1 of 2 tasks: this fixture exists to pin the FULL-N DENOMINATOR convention (accuracy 0.5,
    # not 1.0), so it is intentionally an incomplete run and needs the explicit opt-in added with
    # the completeness guard. See test_fast_measure_guards.py for why unflagged partial runs are
    # now refused.
    cell = build_fast_cell("openai/Qwen/Qwen3.5-27B-FP8", run, n_total=2,
                           joules=1000.0, wall_s=3600.0, cutoff_frac=0.95,
                           allow_partial=True, hardware="testsilicon-local-fast")
    assert cell.dollars == pytest.approx(2.49)         # 1 h × $2.49 × 1 GPU
    assert cell.accuracy == pytest.approx(0.5) and cell.joules == 1000.0
    assert cell.dollars_basis == "gpu-hours" and "fast" in cell.hardware
    assert "cutoff" in cell.energy_source and cell.tokens_in == 100
