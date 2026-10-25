from __future__ import annotations

import pytest

from ipj.orchestrate.eval.aggregate import arm_metrics, compare_table, metrics_to_dict


def test_arm_metrics_accuracy_and_means(make_traj):
    trajs = [
        make_traj("t1", success=True, dollars=0.2, tokens_out=10),
        make_traj("t2", success=False, dollars=0.4, tokens_out=20),
    ]
    m = arm_metrics(trajs)
    assert m.n == 2
    assert m.accuracy == 0.5
    assert abs(m.dollars_mean - 0.3) < 1e-9
    assert m.tokens_out_mean == 15
    assert m.gpu_energy_j_mean is None  # fixture Outcome has gpu_energy_j None
    assert m.dollars_known_n == 2
    assert m.gpu_energy_complete_n == 0
    assert m.tokens_out_known_n == 2


def test_compare_table_has_a_row_per_arm_and_the_bar(make_traj):
    arms = {"trained": arm_metrics([make_traj("t1", success=True)])}
    table = compare_table(arms)
    assert "trained" in table and "cascade bar" in table and "0.76" in table
    assert "coverage" in table
    assert "not rigorous" in table
    serialized = metrics_to_dict(arms)
    assert serialized["cascade_bar"]["historical_gpu_energy_j_mean"] == 160_000.0
    assert serialized["cascade_bar"]["historical_energy_rigorous"] is False


def test_arm_metrics_fail_closed_instead_of_averaging_known_subset(make_traj):
    known = make_traj("t1", dollars=0.2, tokens_out=10)
    unknown = make_traj("t2", dollars=0.4, tokens_out=20)
    unknown.outcome.total_dollars = None
    unknown.outcome.total_tokens_out = None
    metrics = arm_metrics([known, unknown])
    assert metrics.dollars_mean is None
    assert metrics.dollars_known_n == 1
    assert metrics.tokens_out_mean is None
    assert metrics.tokens_out_known_n == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, True, "0.1"])
def test_arm_metrics_excludes_invalid_cost_from_coverage(make_traj, invalid):
    trajectory = make_traj("t1")
    trajectory.outcome.total_dollars = invalid
    metrics = arm_metrics([trajectory])
    assert metrics.dollars_mean is None
    assert metrics.dollars_known_n == 0
