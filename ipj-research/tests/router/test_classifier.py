"""Unit tests for the reusable solvability classifier (no network, no GPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ipj.router.classifier import auc, loo_scores


def _separable_fixture(n_per_class: int = 20, dim: int = 20, gap: float = 4.0):
    """Two well-separated Gaussian blobs → a classifier should rank them perfectly."""
    rng = np.random.default_rng(0)
    neg = rng.normal(-gap, 1.0, size=(n_per_class, dim))
    pos = rng.normal(+gap, 1.0, size=(n_per_class, dim))
    X = np.vstack([neg, pos])
    y = [False] * n_per_class + [True] * n_per_class
    tasks = [f"t{i}" for i in range(2 * n_per_class)]
    solved = dict(zip(tasks, y))
    return X, y, tasks, solved


def test_loo_scores_separable_fixture_auc_near_one():
    X, y, tasks, solved = _separable_fixture()
    preds = loo_scores(X, y, tasks)
    assert set(preds) == set(tasks)
    assert all(0.0 <= p <= 1.0 for p in preds.values())
    assert auc(preds, solved, tasks) >= 0.99


def test_loo_scores_random_labels_auc_near_half():
    rng = np.random.default_rng(1)
    n, dim = 60, 20
    X = rng.normal(0.0, 1.0, size=(n, dim))
    tasks = [f"t{i}" for i in range(n)]
    y = [bool(v) for v in rng.integers(0, 2, size=n)]
    solved = dict(zip(tasks, y))
    preds = loo_scores(X, y, tasks)
    assert abs(auc(preds, solved, tasks) - 0.5) < 0.2


def test_auc_perfect_and_ties():
    tasks = ["a", "b", "c", "d"]
    solved = {"a": True, "b": True, "c": False, "d": False}
    # perfect separation → 1.0
    assert auc({"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}, solved, tasks) == pytest.approx(1.0)
    # fully reversed → 0.0
    assert auc({"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9}, solved, tasks) == pytest.approx(0.0)
    # all tied → 0.5 (ties count 0.5)
    assert auc({t: 0.5 for t in tasks}, solved, tasks) == pytest.approx(0.5)


def test_auc_nan_when_one_class_empty():
    tasks = ["a", "b"]
    solved = {"a": True, "b": True}
    assert math.isnan(auc({"a": 0.9, "b": 0.1}, solved, tasks))


def test_loo_scores_length_mismatch_raises():
    with pytest.raises(ValueError):
        loo_scores([[0.0] * 3, [1.0] * 3], [True], ["a", "b"])
