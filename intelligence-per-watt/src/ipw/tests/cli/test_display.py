"""Tests for cli/_display.py — banner, metrics, formatting."""

from __future__ import annotations

import pytest

from ipw.cli._display import (
    _fmt,
    compute_aggregate_stats,
    compute_profile_metrics,
    compute_trace_metrics,
    print_efficiency_panel,
)


class TestFmt:
    def test_none_returns_dash(self):
        assert _fmt(None) == "\u2014"

    def test_nan_returns_dash(self):
        assert _fmt(float("nan")) == "\u2014"

    def test_inf_returns_dash(self):
        assert _fmt(float("inf")) == "\u2014"

    def test_zero(self):
        assert _fmt(0.0) == "0.00"

    def test_precision(self):
        assert _fmt(3.14159, precision=3) == "3.142"

    def test_integer(self):
        assert _fmt(42.0, precision=0) == "42"

    def test_large_number_with_commas(self):
        result = _fmt(1234567.89)
        assert "1,234,567.89" == result

    def test_negative(self):
        assert _fmt(-1.5) == "-1.50"


class TestComputeAggregateStats:
    def test_empty_list(self):
        result = compute_aggregate_stats([])
        assert result == {"avg": None, "median": None, "min": None, "max": None, "std": None}

    def test_all_none(self):
        result = compute_aggregate_stats([None, None, None])
        assert result["avg"] is None

    def test_known_values(self):
        result = compute_aggregate_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["avg"] == 3.0
        assert result["median"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["std"] == pytest.approx(1.5811, abs=0.001)

    def test_single_value(self):
        result = compute_aggregate_stats([42.0])
        assert result["avg"] == 42.0
        assert result["median"] == 42.0
        assert result["std"] == 0.0

    def test_filters_none(self):
        result = compute_aggregate_stats([None, 1.0, None, 3.0, None])
        assert result["avg"] == 2.0
        assert result["min"] == 1.0
        assert result["max"] == 3.0


class TestComputeProfileMetrics:
    def _make_record(self, model: str, energy: float, tokens_in: int, tokens_out: int):
        """Create a mock ProfilingRecord."""
        from ipw.execution.types import (
            DerivedEfficiencyMetrics,
            EnergyMetrics,
            LatencyMetrics,
            MetricStats,
            ModelMetrics,
            PowerComponentMetrics,
            PowerMetrics,
            ProfilingRecord,
            TokenMetrics,
        )

        mm = ModelMetrics(
            energy_metrics=EnergyMetrics(per_query_joules=energy),
            latency_metrics=LatencyMetrics(
                total_query_seconds=1.5,
                throughput_tokens_per_sec=100.0,
            ),
            token_metrics=TokenMetrics(input=tokens_in, output=tokens_out),
            power_metrics=PowerMetrics(
                gpu=PowerComponentMetrics(per_query_watts=MetricStats(avg=200.0)),
            ),
            efficiency=DerivedEfficiencyMetrics(throughput_per_watt=0.5),
        )
        return ProfilingRecord(
            problem="test",
            answer="test",
            model_metrics={model: mm},
        )

    def test_returns_rows(self):
        records = [
            self._make_record("m", 10.0, 100, 50),
            self._make_record("m", 20.0, 200, 100),
        ]
        rows = compute_profile_metrics(records, "m")
        assert len(rows) > 0
        labels = [r.label for r in rows]
        assert "GPU Energy" in labels
        assert "Throughput" in labels

    def test_energy_avg(self):
        records = [
            self._make_record("m", 10.0, 100, 50),
            self._make_record("m", 20.0, 200, 100),
        ]
        rows = compute_profile_metrics(records, "m")
        energy_row = next(r for r in rows if r.label == "GPU Energy")
        assert energy_row.avg == pytest.approx(15.0)

    def test_empty_records(self):
        assert compute_profile_metrics([], "m") == []

    def test_fallback_to_first_model_key(self):
        records = [self._make_record("other-model", 5.0, 50, 25)]
        rows = compute_profile_metrics(records, "nonexistent")
        energy_row = next(r for r in rows if r.label == "GPU Energy")
        assert energy_row.avg == pytest.approx(5.0)


class TestComputeTraceMetrics:
    def _make_trace(
        self,
        wall_s: float,
        gpu_energy: float | None,
        completed: bool,
        gpu_power: float | None = None,
        cpu_power: float | None = None,
        cost: float | None = None,
    ):
        """Create a mock QueryTrace."""
        from ipw.execution.trace import QueryTrace, TurnTrace

        turns = [TurnTrace(
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
            wall_clock_s=wall_s,
            gpu_power_avg_watts=gpu_power,
            cpu_power_avg_watts=cpu_power,
            cost_usd=cost,
        )]
        return QueryTrace(
            query_id="q1",
            workload_type="test",
            turns=turns,
            total_wall_clock_s=wall_s,
            completed=completed,
            query_gpu_energy_joules=gpu_energy,
        )

    def test_returns_rows(self):
        traces = [
            self._make_trace(1.0, 5.0, True, gpu_power=200.0),
            self._make_trace(2.0, 10.0, False, gpu_power=300.0),
        ]
        rows = compute_trace_metrics(traces)
        labels = [r.label for r in rows]
        assert "Wall Clock" in labels
        assert "GPU Energy" in labels
        assert "GPU Power" in labels
        assert "CPU Power" in labels
        assert "Total Tokens" in labels
        assert "Throughput" in labels
        assert "Energy/Token" in labels
        assert "Cost" in labels
        assert "Completed" in labels

    def test_completed_count(self):
        traces = [
            self._make_trace(1.0, 5.0, True),
            self._make_trace(2.0, 10.0, True),
            self._make_trace(3.0, None, False),
        ]
        rows = compute_trace_metrics(traces)
        completed_row = next(r for r in rows if r.label == "Completed")
        assert completed_row.avg == 2.0
        assert completed_row.unit == "2/3"

    def test_empty_traces(self):
        assert compute_trace_metrics([]) == []

    def test_wall_clock_avg(self):
        traces = [
            self._make_trace(2.0, None, True),
            self._make_trace(4.0, None, True),
        ]
        rows = compute_trace_metrics(traces)
        wc_row = next(r for r in rows if r.label == "Wall Clock")
        assert wc_row.avg == pytest.approx(3.0)

    def test_gpu_power_row(self):
        traces = [
            self._make_trace(1.0, 5.0, True, gpu_power=200.0),
            self._make_trace(2.0, 10.0, True, gpu_power=300.0),
        ]
        rows = compute_trace_metrics(traces)
        power_row = next(r for r in rows if r.label == "GPU Power")
        assert power_row.avg == pytest.approx(250.0)
        assert power_row.unit == "W"

    def test_throughput_row(self):
        traces = [
            self._make_trace(1.0, None, True),
            self._make_trace(2.0, None, True),
        ]
        rows = compute_trace_metrics(traces)
        tp_row = next(r for r in rows if r.label == "Throughput")
        # Trace 1: 50 output / 1.0 s = 50, Trace 2: 50 output / 2.0 s = 25
        assert tp_row.avg == pytest.approx(37.5)
        assert tp_row.unit == "tok/s"

    def test_cost_row(self):
        traces = [
            self._make_trace(1.0, None, True, cost=0.01),
            self._make_trace(2.0, None, True, cost=0.03),
        ]
        rows = compute_trace_metrics(traces)
        cost_row = next(r for r in rows if r.label == "Cost")
        assert cost_row.avg == pytest.approx(0.02)
        assert cost_row.unit == "$"


class TestPrintEfficiencyPanel:
    """Test print_efficiency_panel context lines and IPJ/IPW output."""

    def _make_trace(self, wall_s: float, gpu_energy: float | None, completed: bool, is_resolved: bool | None = None):
        from ipw.execution.trace import QueryTrace, TurnTrace

        turns = [TurnTrace(turn_index=0, input_tokens=100, output_tokens=50, wall_clock_s=wall_s)]
        return QueryTrace(
            query_id="q1",
            workload_type="test",
            turns=turns,
            total_wall_clock_s=wall_s,
            completed=completed,
            query_gpu_energy_joules=gpu_energy,
            is_resolved=is_resolved,
        )

    def test_panel_shows_accuracy_context(self):
        """Accuracy, Total Energy, and Avg Power appear in panel output."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False, width=120)

        traces = [
            self._make_trace(2.0, 10.0, True),
            self._make_trace(3.0, 15.0, True),
        ]
        print_efficiency_panel(con, traces=traces, accuracy=0.8)
        output = buf.getvalue()
        assert "80.0%" in output
        assert "25.00" in output  # total energy 10+15=25
        assert "IPJ" in output
        assert "IPW" in output

    def test_panel_with_accuracy_none_falls_back_to_completed(self):
        """When accuracy is None, the panel falls back to completion rate."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False, width=120)

        traces = [
            self._make_trace(1.0, 5.0, True),
            self._make_trace(2.0, 10.0, False),
        ]
        print_efficiency_panel(con, traces=traces)
        output = buf.getvalue()
        # acc = 1/2 = 50%
        assert "50.0%" in output

    def test_panel_precision_six_decimals(self):
        """IPJ/IPW values use 6 decimal places."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False, width=120)

        traces = [self._make_trace(10.0, 1000.0, True)]
        print_efficiency_panel(con, traces=traces, accuracy=0.5)
        output = buf.getvalue()
        # IPJ = 0.5/1000 = 0.000500 — should have 6 decimal places
        assert "0.000500" in output

    def test_panel_no_ipj_ipw_when_no_energy(self):
        """IPJ/IPW are absent when there is no energy data, but accuracy still shows."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        con = Console(file=buf, highlight=False, width=120)

        traces = [self._make_trace(1.0, None, True)]
        print_efficiency_panel(con, traces=traces, accuracy=0.5)
        output = buf.getvalue()
        assert "50.0%" in output  # accuracy context line still appears
        assert "IPJ" not in output
        assert "IPW" not in output


class TestFLOPsModels:
    """Verify new model entries resolve correctly."""

    def test_qwen3_lookup(self):
        from ipw.compute.flops import estimate_flops

        total, per_token = estimate_flops("Qwen/Qwen3-0.6B", 100, 200)
        assert total > 0
        assert per_token > 0

    def test_deepseek_r1_lookup(self):
        from ipw.compute.flops import estimate_flops

        total, per_token = estimate_flops("deepseek-r1", 100, 200)
        assert total > 0

    def test_phi4_lookup(self):
        from ipw.compute.flops import estimate_flops

        total, per_token = estimate_flops("phi-4", 100, 200)
        assert total > 0

    def test_gemma3_lookup(self):
        from ipw.compute.flops import estimate_flops

        total, per_token = estimate_flops("google/gemma-3-27b", 100, 200)
        assert total > 0

    def test_llama4_scout(self):
        from ipw.compute.flops import estimate_flops

        total, per_token = estimate_flops("meta-llama/Llama-4-Scout", 100, 200)
        assert total > 0

    def test_qwen3_normalization(self):
        from ipw.compute.flops import normalize_model_name

        assert "qwen-3" in normalize_model_name("Qwen/Qwen3-8B-Instruct")
