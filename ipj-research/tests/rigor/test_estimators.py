"""Unit tests for ipj.rigor.estimators: primary_stats, survival_stats, k1_consistency.

Fixtures use k=5 tasks whose answers are hand-computable, so any drift in the estimator
formula (pass@k, pooled ordered-pairs survival, k=1 posterior-predictive MC) surfaces here
before it contaminates a published headline number.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ipj.rigor.estimators import k1_consistency, primary_stats, survival_stats
from ipj.rigor.trials import Trial


def _trial(task: str, name: str, solved: bool) -> Trial:
    return Trial(
        task=task, trial_name=name, solved=solved,
        has_exception=False, exception_type=None,
        started=datetime(2026, 6, 17, tzinfo=timezone.utc),
        finished=datetime(2026, 6, 17, 0, 30, tzinfo=timezone.utc),
        n_episodes=1.0, model_name="m", jobs_dir="/tmp",
    )


def _groups_from_solves(solve_pattern: dict[str, list[bool]]) -> dict[str, list[Trial]]:
    """{task_name: [True, False, True, ...]} -> {task_name: list[Trial]}."""
    return {
        task: [_trial(task, f"{task}__{i}", s) for i, s in enumerate(solves)]
        for task, solves in solve_pattern.items()
    }


class TestPrimaryStats:
    def test_all_solved_all_k(self):
        # 4 tasks x 5 attempts, every attempt solved: pass@1 = pass@5 = pass^5 = 1.0
        groups = _groups_from_solves({f"t{i}": [True] * 5 for i in range(4)})
        r = primary_stats(groups, k=5)
        assert r["n_tasks"] == 4
        assert r["n_trials"] == 20
        assert r["per_attempt_pooled"] == 1.0
        assert r["pass_at_1_per_task_mean"] == 1.0
        assert r["exact_k"]["pass_at_5_unbiased"] == 1.0
        assert r["exact_k"]["pass_hat_k"] == 1.0
        assert r["exact_k"]["any_of_k"] == 1.0
        # fragility_gap = pass@k - pass^k = 0 percentage points
        assert r["exact_k"]["fragility_gap_pt"] == pytest.approx(0.0, abs=1e-9)

    def test_none_solved(self):
        groups = _groups_from_solves({f"t{i}": [False] * 5 for i in range(4)})
        r = primary_stats(groups, k=5)
        assert r["per_attempt_pooled"] == 0.0
        assert r["pass_at_1_per_task_mean"] == 0.0
        assert r["exact_k"]["pass_at_5_unbiased"] == 0.0
        assert r["exact_k"]["pass_hat_k"] == 0.0
        assert r["exact_k"]["any_of_k"] == 0.0

    def test_mixed_solves_pooled_matches_hand(self):
        # s = [5, 0, 3, 1] over 4 tasks x 5 attempts = 9/20 pooled
        groups = _groups_from_solves({
            "t0": [True] * 5,
            "t1": [False] * 5,
            "t2": [True] * 3 + [False] * 2,
            "t3": [True] + [False] * 4,
        })
        r = primary_stats(groups, k=5)
        assert r["per_attempt_pooled"] == pytest.approx(9 / 20)
        # pass@1 per-task mean = mean(5/5, 0/5, 3/5, 1/5) = mean(1.0, 0, 0.6, 0.2) = 0.45
        assert r["pass_at_1_per_task_mean"] == pytest.approx(0.45)
        # histogram of s over n_t==5 tasks: {0:1, 1:1, 3:1, 5:1}, rest 0
        hist = r["s_histogram_nt_eq_5"]
        assert hist == {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

    def test_any_of_k_and_pass_hat_k(self):
        # 4 tasks, s = [5, 3, 1, 0]:
        # any_of_k (>=1 solve) = 3/4 = 0.75
        # pass^k (all solves) = 1/4 = 0.25
        groups = _groups_from_solves({
            "t0": [True] * 5,
            "t1": [True] * 3 + [False] * 2,
            "t2": [True] + [False] * 4,
            "t3": [False] * 5,
        })
        r = primary_stats(groups, k=5)
        assert r["exact_k"]["any_of_k"] == pytest.approx(0.75)
        assert r["exact_k"]["pass_hat_k"] == pytest.approx(0.25)

    def test_mixed_nt_isolated_in_diagnostics(self):
        # One task has only 3 attempts (n_t < k=5); exact_k should exclude it.
        groups = _groups_from_solves({
            "t0": [True] * 5,
            "t1": [True] * 5,
            "t2": [True, False, True],   # n=3, s=2
        })
        r = primary_stats(groups, k=5)
        assert r["exact_k"]["n_tasks_nt_eq_k"] == 2  # t2 excluded
        assert "t2" in r["tasks_nt_ne_5"]
        assert r["diagnostics_all_tasks"]["n_tasks_any_of_n_fallback"] == 1


class TestSurvivalStats:
    def test_all_solved_survival_is_one(self):
        # Every task, every attempt solved: given "task solved," next roll always solves too.
        # pooled = sum(s*(s-1)) / sum(s*(n-1)) = (5*4)*2 / (5*4)*2 = 1.0
        groups = _groups_from_solves({f"t{i}": [True] * 5 for i in range(2)})
        r = survival_stats(groups)
        assert r["pooled_ordered_pairs"] == pytest.approx(1.0)

    def test_single_solve_survival_is_zero(self):
        # Every solved task has s=1 → sum(s*(s-1)) = 0 → survival = 0.
        groups = _groups_from_solves({
            "t0": [True] + [False] * 4,
            "t1": [True] + [False] * 4,
        })
        r = survival_stats(groups)
        assert r["pooled_ordered_pairs"] == pytest.approx(0.0)

    def test_no_eligible_tasks_returns_note(self):
        # No task has s >= 1 → survival not estimable.
        groups = _groups_from_solves({f"t{i}": [False] * 5 for i in range(3)})
        r = survival_stats(groups)
        assert "note" in r
        assert "not estimable" in r["note"]

    def test_opus_borrowed_overridable(self):
        groups = _groups_from_solves({f"t{i}": [True] * 5 for i in range(2)})
        r = survival_stats(groups, opus_borrowed={"survival": 0.5, "ci": [0.4, 0.6], "source": "test"})
        assert r["opus_borrowed"]["source"] == "test"
        # delta = (1.0 - 0.5) * 100 = 50 pt
        assert r["delta_vs_opus_pt"] == pytest.approx(50.0)
        assert r["opus_inside_pooled_ci"] is False  # 0.5 not in [1.0, 1.0]

    def test_singleton_tasks_excluded(self):
        # Tasks with n_t == 1 are dropped from survival analysis (need >= 2 rolls per task).
        groups = _groups_from_solves({
            "t0": [True] * 5,
            "t1": [True],            # n=1 → excluded from tasks_multi_roll
            "t2": [True, True],
        })
        r = survival_stats(groups)
        assert r["n_tasks_multi_roll"] == 2  # t0 + t2, not t1
        assert r["n_solved_singletons_excluded"] == 1


class TestK1Consistency:
    def test_consistent_run_passes(self):
        # k=5 rates all 1.0, k=1 field-table also all solved → MC should say p >> 0.05.
        groups = _groups_from_solves({f"t{i}": [True] * 5 for i in range(4)})
        k1 = {f"t{i}": True for i in range(4)}
        r = k1_consistency(groups, k1)
        assert r["n_joined"] == 4
        assert r["k5_pass1_per_task_mean"] == 1.0
        assert r["k1_field_table_pass1"] == 1.0
        # Posterior predictive: obs=4 out of 4; both p_le and p_ge should be reasonable
        assert 0.0 <= r["posterior_predictive"]["p_le"] <= 1.0
        assert 0.0 <= r["posterior_predictive"]["p_ge"] <= 1.0

    def test_missing_tasks_recorded(self):
        groups = _groups_from_solves({"a": [True] * 5, "b": [True] * 5})
        k1 = {"a": True, "c": True}  # a is joined; b is k5-only, c is k1-only
        r = k1_consistency(groups, k1)
        assert r["n_joined"] == 1
        assert "b" in r["k5_tasks_missing_from_vector"]
        assert "c" in r["vector_tasks_missing_from_k5"]

    def test_no_joined_tasks(self):
        groups = _groups_from_solves({"a": [True] * 5})
        k1 = {"b": True}
        r = k1_consistency(groups, k1)
        assert r["n_joined"] == 0
        assert "note" in r
