from __future__ import annotations

import pytest

from ipj.efficiency.stats import (
    accuracy_ci,
    bootstrap_mean_ci,
    paired_diff_ci,
    pass_metrics,
    per_task_solve_rate,
)
from ipj.harbor.per_task import TaskOutcome


def _attempts(task: str, statuses: list[str]) -> list[TaskOutcome]:
    return [TaskOutcome(task, s, 1.0 if s == "resolved" else (0.0 if s == "unresolved" else None))
            for s in statuses]


def test_per_task_solve_rate_fraction_over_attempts():
    outs = _attempts("t1", ["resolved", "resolved", "unresolved", "errored", "resolved"])  # 3/5
    assert per_task_solve_rate(outs) == {"t1": pytest.approx(0.6)}


def test_bootstrap_mean_ci_is_deterministic_and_brackets_mean():
    vals = [0.0, 1.0] * 50  # mean 0.5
    lo1, hi1 = bootstrap_mean_ci(vals, seed=0)
    lo2, hi2 = bootstrap_mean_ci(vals, seed=0)
    assert (lo1, hi1) == (lo2, hi2)          # deterministic
    assert lo1 < 0.5 < hi1                   # brackets the mean
    assert 0.0 <= lo1 <= hi1 <= 1.0


def test_bootstrap_ci_degenerate_single_value():
    assert bootstrap_mean_ci([0.7]) == (0.7, 0.7)


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])


def test_accuracy_ci_aggregates_k_attempts_and_reports_k_mean():
    # 2 tasks, k=5 each: t1 solves 5/5, t2 solves 0/5 -> accuracy 0.5, k_mean 5
    outs = _attempts("t1", ["resolved"] * 5) + _attempts("t2", ["unresolved"] * 5)
    ci = accuracy_ci(outs, n_resamples=2000, seed=1)
    assert ci.accuracy == pytest.approx(0.5)
    assert ci.n_tasks == 2
    assert ci.k_mean == pytest.approx(5.0)
    assert ci.ci_low <= ci.accuracy <= ci.ci_high


def test_accuracy_ci_from_generator_consumed_once():
    gen = (o for o in _attempts("t1", ["resolved", "unresolved"]))  # generator, not list
    ci = accuracy_ci(gen, n_resamples=500)
    assert ci.accuracy == pytest.approx(0.5)  # would be 0 if the generator were exhausted early


def test_pass_metrics_family():
    # t1: 2/3 attempts pass (not all, but ≥1); t2: 0/3
    outs = _attempts("t1", ["resolved", "unresolved", "resolved"]) + _attempts("t2", ["unresolved"] * 3)
    m = pass_metrics(outs)
    assert m.pass_at_1 == pytest.approx(2 / 6)      # 2 of 6 attempts
    assert m.pass_hat_k == pytest.approx(0.0)        # neither task solved on ALL attempts
    assert m.any_of_k == pytest.approx(0.5)          # t1 solved at least once


def test_pass_metrics_all_pass_is_one():
    outs = _attempts("t1", ["resolved"] * 4)
    m = pass_metrics(outs)
    assert m.pass_at_1 == m.pass_hat_k == m.any_of_k == pytest.approx(1.0)


def test_paired_diff_ci_detects_significant_win():
    # A solves every task, B solves none -> diff 1.0, CI strictly positive
    a = {f"t{i}": 1.0 for i in range(30)}
    b = {f"t{i}": 0.0 for i in range(30)}
    mean, lo, hi = paired_diff_ci(a, b, n_resamples=2000, seed=2)
    assert mean == pytest.approx(1.0)
    assert lo > 0.0  # whole CI positive -> A beats B significantly


def test_paired_diff_ci_ties_include_zero():
    a = {"t1": 1.0, "t2": 0.0, "t3": 1.0, "t4": 0.0}
    b = {"t1": 0.0, "t2": 1.0, "t3": 1.0, "t4": 0.0}  # net diff 0
    mean, lo, hi = paired_diff_ci(a, b, n_resamples=2000, seed=3)
    assert lo <= 0.0 <= hi  # not significant
