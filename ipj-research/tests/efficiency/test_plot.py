from __future__ import annotations

import math
from pathlib import Path

from ipj.efficiency.plot import (
    aggregate_by_approach,
    oracle_per_metric,
    pareto_frontier,
    plot_all,
)
from ipj.efficiency.record import PerQueryRecord

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

METRICS = ["joules", "dollars", "latency_s", "input_tokens", "output_tokens", "flops"]


def _rec(
    *,
    task_id: str,
    approach: str,
    category: str,
    solved: bool,
    joules: float | None = None,
    dollars: float = 0.0,
    latency_s: float | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
    flops: float | None = None,
    reward: float | None = None,
) -> PerQueryRecord:
    return PerQueryRecord(
        task_id=task_id,
        approach=approach,
        category=category,
        hardware="H100",
        precision="FP8",
        difficulty="medium",
        tb_category="cli",
        tags=[],
        expert_time_min=10.0,
        reward=1.0 if solved else 0.0 if reward is None else reward,
        solved=solved,
        failure_mode="" if solved else "wrong_answer",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dollars=dollars,
        latency_s=latency_s,
        flops=flops,
        joules=joules,
    )


# Minimal fixture: 3 tasks, 3 approaches across 2 categories + 1 closed
RECORDS: list[PerQueryRecord] = [
    # approach A (open): solves task1 and task2, misses task3
    _rec(task_id="t1", approach="A", category="open", solved=True,
         joules=100.0, dollars=0.0, latency_s=10.0, input_tokens=200, output_tokens=100, flops=1e12),
    _rec(task_id="t2", approach="A", category="open", solved=True,
         joules=200.0, dollars=0.0, latency_s=20.0, input_tokens=300, output_tokens=150, flops=2e12),
    _rec(task_id="t3", approach="A", category="open", solved=False,
         joules=50.0, dollars=0.0, latency_s=5.0, input_tokens=100, output_tokens=50, flops=5e11),
    # approach B (cascade): solves task1, misses task2 and task3
    _rec(task_id="t1", approach="B", category="cascade", solved=True,
         joules=80.0, dollars=0.5, latency_s=8.0, input_tokens=180, output_tokens=90, flops=8e11),
    _rec(task_id="t2", approach="B", category="cascade", solved=False,
         joules=300.0, dollars=1.0, latency_s=30.0, input_tokens=400, output_tokens=200, flops=3e12),
    _rec(task_id="t3", approach="B", category="cascade", solved=False,
         joules=120.0, dollars=0.8, latency_s=12.0, input_tokens=250, output_tokens=120, flops=1.2e12),
    # approach C (closed): cloud model, joules=None on purpose
    _rec(task_id="t1", approach="C", category="closed", solved=True,
         joules=None, dollars=2.0, latency_s=3.0, input_tokens=180, output_tokens=90, flops=None),
    _rec(task_id="t2", approach="C", category="closed", solved=True,
         joules=None, dollars=3.0, latency_s=5.0, input_tokens=220, output_tokens=110, flops=None),
    _rec(task_id="t3", approach="C", category="closed", solved=True,
         joules=None, dollars=2.5, latency_s=4.0, input_tokens=200, output_tokens=100, flops=None),
]


# ---------------------------------------------------------------------------
# aggregate_by_approach
# ---------------------------------------------------------------------------

class TestAggregateByApproach:
    def test_returns_entry_for_each_approach(self):
        agg = aggregate_by_approach(RECORDS)
        assert set(agg.keys()) == {"A", "B", "C"}

    def test_accuracy_is_mean_solved(self):
        agg = aggregate_by_approach(RECORDS)
        # A: 2/3 solved
        assert math.isclose(agg["A"]["accuracy"], 2 / 3)
        # B: 1/3 solved
        assert math.isclose(agg["B"]["accuracy"], 1 / 3)
        # C: 3/3 solved
        assert math.isclose(agg["C"]["accuracy"], 1.0)

    def test_metric_is_mean_excluding_none(self):
        agg = aggregate_by_approach(RECORDS)
        # A joules: mean(100, 200, 50) = 116.667
        assert math.isclose(agg["A"]["joules"], (100 + 200 + 50) / 3)
        # A latency: mean(10, 20, 5)
        assert math.isclose(agg["A"]["latency_s"], (10 + 20 + 5) / 3)

    def test_all_none_metric_is_none(self):
        agg = aggregate_by_approach(RECORDS)
        # C (closed): joules all None
        assert agg["C"]["joules"] is None
        # C: flops all None
        assert agg["C"]["flops"] is None

    def test_category_preserved(self):
        agg = aggregate_by_approach(RECORDS)
        assert agg["A"]["category"] == "open"
        assert agg["B"]["category"] == "cascade"
        assert agg["C"]["category"] == "closed"

    def test_partial_none_metric_uses_available(self):
        # Approach A has all joules set; only test that None-skipping works via a custom record set
        recs = [
            _rec(task_id="t1", approach="X", category="open", solved=True, joules=None),
            _rec(task_id="t2", approach="X", category="open", solved=True, joules=100.0),
        ]
        agg = aggregate_by_approach(recs)
        # only one non-None joule
        assert math.isclose(agg["X"]["joules"], 100.0)


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def test_single_point_is_its_own_frontier(self):
        pts = [(5.0, 0.8)]
        assert pareto_frontier(pts) == [(5.0, 0.8)]

    def test_dominated_point_excluded(self):
        # (10, 0.5) is dominated by (5, 0.8): lower x, higher y
        pts = [(5.0, 0.8), (10.0, 0.5)]
        result = pareto_frontier(pts)
        assert (10.0, 0.5) not in result
        assert (5.0, 0.8) in result

    def test_all_pareto_optimal(self):
        # Each point has a trade-off: lower x → lower y
        pts = [(1.0, 0.5), (3.0, 0.7), (5.0, 0.9)]
        result = set(pareto_frontier(pts))
        assert result == {(1.0, 0.5), (3.0, 0.7), (5.0, 0.9)}

    def test_duplicate_dominated(self):
        # Two identical points: both on frontier (tie is not domination if we use >= / <=)
        pts = [(5.0, 0.8), (5.0, 0.8)]
        result = pareto_frontier(pts)
        # At minimum, at least one copy survives
        assert len(result) >= 1
        assert all(p == (5.0, 0.8) for p in result)

    def test_empty_returns_empty(self):
        assert pareto_frontier([]) == []

    def test_strictly_dominated_chain(self):
        # (2.0, 0.9) dominates (5.0, 0.6) which dominates (10.0, 0.3):
        # each has strictly lower x AND strictly higher accuracy, so only (2.0, 0.9) survives.
        pts = [(10.0, 0.3), (5.0, 0.6), (2.0, 0.9)]
        result = set(pareto_frontier(pts))
        assert result == {(2.0, 0.9)}

    def test_one_dominates_all(self):
        pts = [(1.0, 1.0), (5.0, 0.5), (10.0, 0.3)]
        result = pareto_frontier(pts)
        # (1.0, 1.0) dominates everyone else (lowest x, highest y)
        assert (1.0, 1.0) in result
        assert (5.0, 0.5) not in result
        assert (10.0, 0.3) not in result


# ---------------------------------------------------------------------------
# oracle_per_metric
# ---------------------------------------------------------------------------

class TestOraclePerMetric:
    def test_oracle_accuracy_excludes_oracle_category(self):
        # Tasks solved by at least one non-oracle approach: t1 and t2 by A and/or C; t3 by C
        # All 3 tasks solved by >= 1 non-oracle approach → oracle accuracy = 1.0
        acc, _ = oracle_per_metric(RECORDS, "joules")
        assert math.isclose(acc, 1.0)

    def test_oracle_joules_picks_min_per_task(self):
        # For joules: t1 solved by A(100) and B(80): min=80; t2 solved by A(200): min=200;
        # t3 solved by C(None joules) — but C's joules=None so no valid reading
        # Actually C is closed (joules=None) so task t3 is only solved by C (closed).
        # oracle_per_metric should skip approaches where the metric is None.
        # t1: A(100), B(80) → min=80
        # t2: A(200) → min=200
        # t3: C(None joules) → no valid joules pick for t3 → t3 excluded from metric mean
        # oracle metric = mean(80, 200) = 140
        acc, metric = oracle_per_metric(RECORDS, "joules")
        assert math.isclose(metric, (80.0 + 200.0) / 2)

    def test_oracle_accuracy_fractional(self):
        # Only task t1 is solvable (approaches with finite metric)
        recs = [
            _rec(task_id="t1", approach="A", category="open", solved=True, joules=50.0),
            _rec(task_id="t2", approach="A", category="open", solved=False, joules=100.0),
        ]
        acc, metric = oracle_per_metric(recs, "joules")
        # 1 task solved out of 2 total → accuracy = 0.5
        assert math.isclose(acc, 0.5)
        # metric = min joules for t1 = 50.0
        assert math.isclose(metric, 50.0)

    def test_oracle_excludes_oracle_category_from_selection(self):
        recs = [
            _rec(task_id="t1", approach="oracle_run", category="oracle", solved=True, joules=10.0),
            _rec(task_id="t1", approach="A", category="open", solved=False, joules=100.0),
        ]
        # oracle category approach should not count as "solving" the task
        acc, metric = oracle_per_metric(recs, "joules")
        assert math.isclose(acc, 0.0)

    def test_oracle_metric_none_when_no_valid_data(self):
        # All approaches have None for the metric
        recs = [
            _rec(task_id="t1", approach="C", category="closed", solved=True, joules=None),
        ]
        acc, metric = oracle_per_metric(recs, "joules")
        # t1 solved → acc=1.0 but no joule data
        assert metric is None


# ---------------------------------------------------------------------------
# Smoke test: plot_all writes 6 PNGs
# ---------------------------------------------------------------------------

EXPECTED_PNG_NAMES = {
    "accuracy_vs_joules.png",
    "accuracy_vs_dollars.png",
    "accuracy_vs_latency_s.png",
    "accuracy_vs_input_tokens.png",
    "accuracy_vs_output_tokens.png",
    "accuracy_vs_flops.png",
}


class TestPlotAll:
    def test_writes_six_pngs(self, tmp_path: Path):
        plot_all(RECORDS, tmp_path)
        written = {p.name for p in tmp_path.glob("*.png")}
        assert written == EXPECTED_PNG_NAMES

    def test_files_are_nonempty(self, tmp_path: Path):
        plot_all(RECORDS, tmp_path)
        for png in tmp_path.glob("*.png"):
            assert png.stat().st_size > 0, f"{png.name} is empty"

    def test_idempotent_overwrite(self, tmp_path: Path):
        """Calling twice should not raise and should still produce 6 PNGs."""
        plot_all(RECORDS, tmp_path)
        plot_all(RECORDS, tmp_path)
        written = {p.name for p in tmp_path.glob("*.png")}
        assert written == EXPECTED_PNG_NAMES

    def test_single_approach_no_crash(self, tmp_path: Path):
        """Edge case: only one approach — Pareto frontier of one point."""
        recs = [
            _rec(task_id="t1", approach="A", category="open", solved=True,
                 joules=100.0, dollars=1.0, latency_s=5.0,
                 input_tokens=100, output_tokens=50, flops=1e12),
        ]
        plot_all(recs, tmp_path)
        written = {p.name for p in tmp_path.glob("*.png")}
        assert written == EXPECTED_PNG_NAMES

    def test_all_none_metric_no_crash(self, tmp_path: Path):
        """If every approach has None for a metric (e.g. flops), that graph still renders."""
        recs = [
            _rec(task_id="t1", approach="A", category="open", solved=True,
                 joules=None, dollars=0.0, latency_s=None,
                 input_tokens=100, output_tokens=50, flops=None),
        ]
        plot_all(recs, tmp_path)
        written = {p.name for p in tmp_path.glob("*.png")}
        assert written == EXPECTED_PNG_NAMES
