"""Unit tests for the cost-aware router head (no network, no GPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ipj.router.costaware import (
    cost_aware_scores,
    loo_cost_model,
    rank_by_utility,
    utility,
)
from ipj.router.scorecache import (
    best_under_budget,
    curve_from_order,
    ranked,
    routed_curve,
)

# --------------------------------------------------------------------------- loo_cost_model


def _cost_linear_fixture(n: int = 40, dim: int = 20):  # dim >= PCA n_components (15)
    """cost = exp(w·x) so log-cost is exactly linear in the features — a ridge on log-cost
    should recover a high correlation while the LOO fold never sees its own label."""
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(n, dim))
    w = rng.normal(0.0, 1.0, size=dim)
    noise = rng.normal(0.0, 0.05, size=n)
    tasks = [f"t{i}" for i in range(n)]
    log_cost = X @ w + noise
    cloud_cost = {tasks[i]: float(math.exp(log_cost[i])) for i in range(n)}
    return X, cloud_cost, tasks


def test_loo_cost_model_recovers_log_correlation():
    X, cloud_cost, tasks = _cost_linear_fixture()
    cost_hat = loo_cost_model(X, cloud_cost, tasks)
    assert set(cost_hat) == set(tasks)
    assert all(v >= 1e-6 for v in cost_hat.values())
    lc = np.array([math.log(cloud_cost[t]) for t in tasks])
    lh = np.array([math.log(cost_hat[t]) for t in tasks])
    r = float(np.corrcoef(lc, lh)[0, 1])
    assert r > 0.8, f"log-cost correlation too low: {r:.3f}"


def test_loo_cost_model_does_not_leak_held_out_label():
    """A single wildly-mispriced task must not drag its own prediction toward its label:
    if the fold leaked, cost_hat for the outlier would track its (huge) true cost."""
    X, cloud_cost, tasks = _cost_linear_fixture()
    outlier = tasks[0]
    cloud_cost[outlier] = 1e6  # absurd true cost, uncorrelated with its features
    cost_hat = loo_cost_model(X, cloud_cost, tasks)
    # Held-out prediction is driven by the other 39 rows, so it stays in a sane range.
    assert cost_hat[outlier] < 1e3, f"held-out label leaked: {cost_hat[outlier]:.3g}"


def test_loo_cost_model_floors_nonpositive_cost():
    X, cloud_cost, tasks = _cost_linear_fixture()
    cloud_cost[tasks[0]] = 0.0   # non-positive costs must be floored inside log(...), not crash
    cloud_cost[tasks[1]] = -5.0
    cost_hat = loo_cost_model(X, cloud_cost, tasks)
    assert all(v >= 1e-6 and math.isfinite(v) for v in cost_hat.values())


def test_loo_cost_model_needs_two_tasks():
    with pytest.raises(ValueError):
        loo_cost_model([[0.0] * 20], {"a": 0.1}, ["a"])


# --------------------------------------------------------------------------- utility


def test_utility_value_matches_formula():
    p_local = {"t": 0.25}
    p_cloud = {"t": 0.8}
    cost_hat = {"t": 0.5}
    got = utility(p_local, p_cloud, cost_hat, ["t"])["t"]
    assert got == pytest.approx((1 - 0.25) * 0.8 / 0.5)


def test_utility_rises_as_p_local_falls():
    tasks = ["t"]
    hi = utility({"t": 0.2}, {"t": 0.7}, {"t": 0.5}, tasks)["t"]
    lo = utility({"t": 0.8}, {"t": 0.7}, {"t": 0.5}, tasks)["t"]
    assert hi > lo


def test_utility_rises_as_p_cloud_rises():
    tasks = ["t"]
    hi = utility({"t": 0.5}, {"t": 0.9}, {"t": 0.5}, tasks)["t"]
    lo = utility({"t": 0.5}, {"t": 0.1}, {"t": 0.5}, tasks)["t"]
    assert hi > lo


def test_utility_rises_as_cost_falls():
    tasks = ["t"]
    hi = utility({"t": 0.5}, {"t": 0.7}, {"t": 0.1}, tasks)["t"]
    lo = utility({"t": 0.5}, {"t": 0.7}, {"t": 2.0}, tasks)["t"]
    assert hi > lo


def test_utility_guards_nonpositive_cost():
    tasks = ["t"]
    got = utility({"t": 0.5}, {"t": 0.7}, {"t": 0.0}, tasks)["t"]
    assert math.isfinite(got) and got > 0


# --------------------------------------------------------------------------- rank_by_utility


def test_rank_by_utility_is_descending():
    util = {"a": 0.1, "b": 0.9, "c": 0.5}
    assert rank_by_utility(util) == ["b", "c", "a"]


def test_rank_by_utility_breaks_ties_by_task_name():
    util = {"b": 0.5, "a": 0.5, "c": 0.5}
    assert rank_by_utility(util) == ["a", "b", "c"]


def test_rank_by_utility_is_opposite_of_scorecache_ranked():
    """rank_by_utility escalates highest-first; scorecache.ranked escalates lowest-first."""
    util = {"a": 0.1, "b": 0.9, "c": 0.5}
    assert rank_by_utility(util) == list(reversed(ranked(util)))


# ------------------------------------------------------- cost_aware_scores round-trips


def test_cost_aware_scores_round_trips_through_ranked():
    rng = np.random.default_rng(3)
    tasks = [f"t{i}" for i in range(15)]
    p_local = {t: float(rng.uniform(0, 1)) for t in tasks}
    p_cloud = {t: float(rng.uniform(0, 1)) for t in tasks}
    cost_hat = {t: float(rng.uniform(0.01, 2.0)) for t in tasks}

    util = utility(p_local, p_cloud, cost_hat, tasks)
    scores = cost_aware_scores(p_local, p_cloud, cost_hat, tasks)
    # scorecache.ranked (ascending pseudo-P(solve)) reproduces the highest-utility-first order.
    assert ranked(scores) == rank_by_utility(util)


def test_cost_aware_scores_round_trips_with_utility_ties():
    # Two tasks with identical utility must still round-trip deterministically by name.
    p_local = {"b": 0.5, "a": 0.5, "c": 0.2}
    p_cloud = {"b": 0.6, "a": 0.6, "c": 0.9}
    cost_hat = {"b": 0.3, "a": 0.3, "c": 0.4}
    util = utility(p_local, p_cloud, cost_hat, list(p_local))
    scores = cost_aware_scores(p_local, p_cloud, cost_hat, list(p_local))
    assert ranked(scores) == rank_by_utility(util)


# --------------------------------------------------------------------------- curve helpers

# 4 tasks: t1 local-solve; t2/t3 local-fail cloud-solve; t4 local-fail cloud-fail.
LOCAL = {"t1": True, "t2": False, "t3": False, "t4": False}
CLOUD = {"t1": True, "t2": True, "t3": True, "t4": False}
COST = {"t1": 0.5, "t2": 0.1, "t3": 0.9, "t4": 0.5}


def test_curve_from_order_length_is_n_plus_one():
    order = ["t2", "t3", "t1", "t4"]
    curve = curve_from_order(order, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    assert len(curve) == len(order) + 1


def test_curve_from_order_endpoints():
    order = ["t2", "t3", "t1", "t4"]
    curve = curve_from_order(order, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    assert curve[0].n_escalated == 0
    assert curve[0].accuracy == pytest.approx(1 / 4)  # all-local: only t1 solves
    assert curve[-1].n_escalated == 4
    assert curve[-1].accuracy == pytest.approx(3 / 4)  # all-cloud: t1,t2,t3 solve


def test_routed_curve_equals_curve_from_ranked_order():
    preds = {"t1": 0.9, "t2": 0.1, "t3": 0.5, "t4": 0.3}
    a = routed_curve(preds, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    b = curve_from_order(ranked(preds), LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    assert [(p.n_escalated, p.accuracy, p.dollars_per_task) for p in a] == [
        (p.n_escalated, p.accuracy, p.dollars_per_task) for p in b
    ]


def test_best_under_budget_respects_cap_and_maximizes_accuracy():
    # Escalate cheap effective task t2 (cloud-solves, $0.1) first, then expensive t3 ($0.9).
    order = ["t2", "t3", "t1", "t4"]
    curve = curve_from_order(order, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    # Budget $0.05/task fits only e=0 (avg escalation cost 0.1/4 = 0.025 already exceeds it? no).
    # e=0 -> $0; e=1 -> 0.1/4 = 0.025; e=2 -> (0.1+0.9)/4 = 0.25.
    best = best_under_budget(curve, 0.05)
    assert best.n_escalated == 1  # escalating t2 (acc 2/4) fits; t3 (0.25) does not
    assert best.accuracy == pytest.approx(2 / 4)


def test_best_under_budget_none_when_nothing_fits():
    curve = curve_from_order(["t2", "t3"], LOCAL, CLOUD, COST, local_cost_per_task=1.0)
    assert best_under_budget(curve, 0.5) is None


def test_best_under_budget_ties_pick_fewest_escalations():
    # t4 is cloud-fail: escalating it adds cost but no accuracy, so e with and without it tie.
    order = ["t2", "t4"]
    curve = curve_from_order(order, LOCAL, CLOUD, COST, local_cost_per_task=0.0)
    best = best_under_budget(curve, 10.0)  # everything affordable
    # e=1 (t2) and e=2 (t2,t4) both give accuracy 2/4; the cheaper e=1 wins the tie.
    assert best.accuracy == pytest.approx(2 / 4)
    assert best.n_escalated == 1
