"""Unit tests for ipj.rigor.bootstrap: pass_at_k_unbiased + cluster_boot_ci.

Every case here has an analytical answer — if a change to the estimator moves any of these,
either the change is wrong or every k>1 headline number that ships with the paper is wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ipj.rigor.bootstrap import cluster_boot_ci, pass_at_k_unbiased


class TestPassAtK:
    def test_all_solved(self):
        # n=5 attempts, s=5 solves, k=5 → pass@5 = 1.0 (trivially: any k picks of solves succeed)
        assert pass_at_k_unbiased(5, 5, k=5) == 1.0

    def test_none_solved(self):
        # n=5 attempts, s=0 solves, k=5 → pass@5 = 0.0 (trivially: no k picks succeed)
        assert pass_at_k_unbiased(5, 0, k=5) == 0.0

    def test_k_equals_1_reduces_to_rate(self):
        # pass@1 = 1 - C(n-s,1)/C(n,1) = 1 - (n-s)/n = s/n
        assert pass_at_k_unbiased(10, 3, k=1) == pytest.approx(0.3)
        assert pass_at_k_unbiased(89, 66, k=1) == pytest.approx(66 / 89)

    def test_partial_solves_matches_hand_computation(self):
        # n=5, s=3, k=5: 1 - C(2,5)/C(5,5).  C(2,5) = 0, so pass@5 = 1.0
        assert pass_at_k_unbiased(5, 3, k=5) == 1.0
        # n=5, s=3, k=3: 1 - C(2,3)/C(5,3).  C(2,3) = 0, C(5,3) = 10, so pass@3 = 1.0
        assert pass_at_k_unbiased(5, 3, k=3) == 1.0
        # n=5, s=1, k=3: 1 - C(4,3)/C(5,3) = 1 - 4/10 = 0.6
        assert pass_at_k_unbiased(5, 1, k=3) == pytest.approx(0.6)

    def test_n_lt_k_falls_back_to_any_of_n(self):
        # n=2, s=1, k=5: fallback returns 1.0 iff s>=1
        assert pass_at_k_unbiased(2, 1, k=5) == 1.0
        assert pass_at_k_unbiased(2, 0, k=5) == 0.0
        assert pass_at_k_unbiased(4, 3, k=5) == 1.0

    def test_matches_math_comb_formula(self):
        # Cross-check against the closed form for a handful of (n, s, k) triples.
        for n, s, k in [(10, 4, 5), (8, 2, 3), (20, 15, 5), (5, 2, 5)]:
            expected = 1.0 - math.comb(n - s, k) / math.comb(n, k)
            assert pass_at_k_unbiased(n, s, k=k) == pytest.approx(expected)


class TestClusterBootCI:
    def test_constant_statistic_tight_ci(self):
        # A statistic that returns 1.0 regardless of resample must have CI = (1.0, 1.0).
        lo, hi = cluster_boot_ci(lambda idx: 1.0, n_tasks=50, n_resamples=200, seed=0)
        assert lo == 1.0 and hi == 1.0

    def test_wraps_correct_percentile(self):
        # Statistic = mean of resampled binary vector (all-1s) → 1.0 → CI = (1.0, 1.0).
        arr = np.ones(30)
        lo, hi = cluster_boot_ci(lambda idx: float(arr[idx].mean()),
                                 n_tasks=len(arr), n_resamples=500, seed=0)
        assert lo == 1.0 and hi == 1.0

    def test_bernoulli_mean_ci_contains_truth(self):
        # For a large sample of coin flips with p=0.7, the 95% bootstrap CI should contain 0.7.
        rng = np.random.default_rng(20260101)
        arr = (rng.random(500) < 0.7).astype(float)
        lo, hi = cluster_boot_ci(lambda idx: float(arr[idx].mean()),
                                 n_tasks=len(arr), n_resamples=2000, seed=0)
        assert lo < 0.7 < hi, f"CI [{lo:.3f}, {hi:.3f}] must contain the true mean 0.7"

    def test_deterministic_across_calls_with_same_seed(self):
        arr = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.9, 0.1] * 5)
        a = cluster_boot_ci(lambda idx: float(arr[idx].mean()),
                            n_tasks=len(arr), n_resamples=500, seed=42)
        b = cluster_boot_ci(lambda idx: float(arr[idx].mean()),
                            n_tasks=len(arr), n_resamples=500, seed=42)
        assert a == b

    def test_nan_statistics_dropped(self):
        # When ~half the resamples return NaN, the CI is computed from the surviving half —
        # not NaN itself and lies within the finite value.
        arr = np.array([1.0, 1.0, 1.0])

        def stat(idx):
            # Return NaN on half the resamples (based on parity of first index) — nanquantile drops them
            return float(arr[idx].mean()) if idx[0] % 2 == 0 else float("nan")

        lo, hi = cluster_boot_ci(stat, n_tasks=len(arr), n_resamples=200, seed=0)
        assert lo == 1.0 and hi == 1.0
