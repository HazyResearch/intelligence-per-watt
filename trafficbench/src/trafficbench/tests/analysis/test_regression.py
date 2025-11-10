"""Tests for regression analysis utilities."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from trafficbench.analysis.regression import (
    RegressionSample,
    build_zero_warnings,
    create_regression_containers,
    derive_total_tokens,
    finalize_regressions,
    register_regression_sample,
    to_float,
    RegressionAnalysis,
    _compute_regression,
    _compute_average,
    _filter_none_regressions,
)
from trafficbench.analysis.base import AnalysisContext


class TestToFloat:
    """Test to_float conversion utility."""

    def test_converts_valid_number(self) -> None:
        assert to_float(42) == 42.0
        assert to_float(3.14) == 3.14
        assert to_float("2.5") == 2.5

    def test_returns_none_for_none(self) -> None:
        assert to_float(None) is None

    def test_returns_none_for_invalid_types(self) -> None:
        assert to_float("invalid") is None
        assert to_float([1, 2, 3]) is None
        assert to_float({"key": "value"}) is None


class TestDeriveTotalTokens:
    """Test total token derivation."""

    def test_sums_both_when_present(self) -> None:
        assert derive_total_tokens(100.0, 50.0) == 150.0

    def test_handles_none_prompt_tokens(self) -> None:
        assert derive_total_tokens(None, 50.0) == 50.0

    def test_handles_none_completion_tokens(self) -> None:
        assert derive_total_tokens(100.0, None) == 100.0

    def test_returns_none_when_both_none(self) -> None:
        assert derive_total_tokens(None, None) is None

    def test_treats_none_as_zero_in_sum(self) -> None:
        # When one is None, it's treated as 0
        assert derive_total_tokens(None, 100.0) == 100.0
        assert derive_total_tokens(100.0, None) == 100.0


class TestRegressionContainers:
    """Test regression container creation."""

    def test_creates_expected_keys(self) -> None:
        regressions, zero_counts = create_regression_containers()

        assert "input_tokens_vs_ttft" in regressions
        assert "total_tokens_vs_energy" in regressions
        assert "total_tokens_vs_latency" in regressions
        assert "total_tokens_vs_power" in regressions

        assert "energy" in zero_counts
        assert "power" in zero_counts
        assert "ttft" in zero_counts
        assert "latency" in zero_counts

    def test_initializes_empty_lists(self) -> None:
        regressions, zero_counts = create_regression_containers()

        for key in regressions:
            assert isinstance(regressions[key], list)
            assert len(regressions[key]) == 0

    def test_initializes_zero_counts(self) -> None:
        regressions, zero_counts = create_regression_containers()

        for key in zero_counts:
            assert zero_counts[key] == 0


class TestRegisterRegressionSample:
    """Test regression sample registration."""

    def test_registers_ttft_sample(self) -> None:
        regressions, zero_counts = create_regression_containers()

        register_regression_sample(
            regressions,
            zero_counts,
            prompt_tokens=100.0,
            completion_tokens=50.0,
            total_tokens=150.0,
            ttft_seconds=0.5,
            total_latency_seconds=2.0,
            per_query_joules=10.0,
            per_query_watts=5.0,
        )

        assert len(regressions["input_tokens_vs_ttft"]) == 1
        sample = regressions["input_tokens_vs_ttft"][0]
        assert sample.x == 100.0
        assert sample.y == 0.5

    def test_registers_energy_sample(self) -> None:
        regressions, zero_counts = create_regression_containers()

        register_regression_sample(
            regressions,
            zero_counts,
            prompt_tokens=100.0,
            completion_tokens=50.0,
            total_tokens=150.0,
            ttft_seconds=0.5,
            total_latency_seconds=2.0,
            per_query_joules=10.0,
            per_query_watts=5.0,
        )

        assert len(regressions["total_tokens_vs_energy"]) == 1
        sample = regressions["total_tokens_vs_energy"][0]
        assert sample.x == 150.0
        assert sample.y == 10.0

    def test_skips_sample_when_values_missing(self) -> None:
        regressions, zero_counts = create_regression_containers()

        register_regression_sample(
            regressions,
            zero_counts,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            ttft_seconds=None,
            total_latency_seconds=None,
            per_query_joules=None,
            per_query_watts=None,
        )

        for key in regressions:
            assert len(regressions[key]) == 0

    def test_counts_zero_energy(self) -> None:
        regressions, zero_counts = create_regression_containers()

        register_regression_sample(
            regressions,
            zero_counts,
            prompt_tokens=100.0,
            completion_tokens=50.0,
            total_tokens=150.0,
            ttft_seconds=0.5,
            total_latency_seconds=2.0,
            per_query_joules=0.0,
            per_query_watts=5.0,
        )

        assert zero_counts["energy"] == 1

    def test_counts_zero_power(self) -> None:
        regressions, zero_counts = create_regression_containers()

        register_regression_sample(
            regressions,
            zero_counts,
            prompt_tokens=100.0,
            completion_tokens=50.0,
            total_tokens=150.0,
            ttft_seconds=0.5,
            total_latency_seconds=2.0,
            per_query_joules=10.0,
            per_query_watts=0.0,
        )

        assert zero_counts["power"] == 1

    def test_excludes_zero_power_from_regression(self) -> None:
        regressions, zero_counts = create_regression_containers()

        register_regression_sample(
            regressions,
            zero_counts,
            prompt_tokens=100.0,
            completion_tokens=50.0,
            total_tokens=150.0,
            ttft_seconds=0.5,
            total_latency_seconds=2.0,
            per_query_joules=10.0,
            per_query_watts=0.0,
        )

        # Should not add to power regression when power is ~0
        assert len(regressions["total_tokens_vs_power"]) == 0


class TestComputeRegression:
    """Test regression computation."""

    def test_returns_none_for_empty_samples(self) -> None:
        result = _compute_regression([])
        assert result["count"] == 0
        assert result["slope"] is None
        assert result["intercept"] is None
        assert result["r2"] is None

    def test_returns_none_for_single_sample(self) -> None:
        samples = [RegressionSample(x=1.0, y=2.0)]
        result = _compute_regression(samples)
        assert result["count"] == 1
        assert result["slope"] is None

    def test_computes_perfect_linear_fit(self) -> None:
        samples = [
            RegressionSample(x=1.0, y=2.0),
            RegressionSample(x=2.0, y=4.0),
            RegressionSample(x=3.0, y=6.0),
        ]
        result = _compute_regression(samples)

        assert result["count"] == 3
        assert result["slope"] is not None
        assert result["intercept"] is not None
        assert result["r2"] is not None
        assert math.isclose(result["slope"], 2.0, abs_tol=1e-10)
        assert math.isclose(result["intercept"], 0.0, abs_tol=1e-10)
        assert math.isclose(result["r2"], 1.0, abs_tol=1e-10)

    def test_filters_infinite_values(self) -> None:
        samples = [
            RegressionSample(x=1.0, y=2.0),
            RegressionSample(x=float("inf"), y=4.0),
            RegressionSample(x=3.0, y=6.0),
        ]
        result = _compute_regression(samples)

        assert result["count"] == 2

    def test_filters_nan_values(self) -> None:
        samples = [
            RegressionSample(x=1.0, y=2.0),
            RegressionSample(x=2.0, y=float("nan")),
            RegressionSample(x=3.0, y=6.0),
        ]
        result = _compute_regression(samples)

        assert result["count"] == 2

    def test_log_x_filters_negative_and_zero(self) -> None:
        samples = [
            RegressionSample(x=-1.0, y=2.0),
            RegressionSample(x=0.0, y=3.0),
            RegressionSample(x=1.0, y=4.0),
            RegressionSample(x=2.0, y=5.0),
        ]
        result = _compute_regression(samples, log_x=True)

        assert result["count"] == 2

    def test_log_y_filters_negative_and_zero(self) -> None:
        samples = [
            RegressionSample(x=1.0, y=-1.0),
            RegressionSample(x=2.0, y=0.0),
            RegressionSample(x=3.0, y=4.0),
            RegressionSample(x=4.0, y=5.0),
        ]
        result = _compute_regression(samples, log_y=True)

        assert result["count"] == 2

    def test_handles_constant_x_values(self) -> None:
        samples = [
            RegressionSample(x=5.0, y=1.0),
            RegressionSample(x=5.0, y=2.0),
            RegressionSample(x=5.0, y=3.0),
        ]
        result = _compute_regression(samples)

        assert result["slope"] is None
        assert result["intercept"] is None
        assert result["r2"] is None


class TestComputeAverage:
    """Test average computation."""

    def test_returns_none_for_empty(self) -> None:
        assert _compute_average([]) is None

    def test_computes_mean(self) -> None:
        samples = [
            RegressionSample(x=0.0, y=2.0),
            RegressionSample(x=0.0, y=4.0),
            RegressionSample(x=0.0, y=6.0),
        ]
        avg = _compute_average(samples)
        assert avg == 4.0


class TestFinalizeRegressions:
    """Test regression finalization."""

    def test_includes_all_regression_keys(self) -> None:
        regressions, _ = create_regression_containers()
        result = finalize_regressions(regressions)

        assert "input_tokens_vs_ttft" in result
        assert "total_tokens_vs_energy" in result
        assert "total_tokens_vs_latency" in result
        assert "total_tokens_vs_power" in result

    def test_includes_power_log_by_default(self) -> None:
        regressions, _ = create_regression_containers()
        result = finalize_regressions(regressions)

        assert "total_tokens_vs_power_log" in result

    def test_excludes_power_log_when_disabled(self) -> None:
        regressions, _ = create_regression_containers()
        result = finalize_regressions(regressions, include_power_log=False)

        assert "total_tokens_vs_power_log" not in result

    def test_includes_avg_y_in_results(self) -> None:
        regressions, _ = create_regression_containers()
        regressions["input_tokens_vs_ttft"] = [
            RegressionSample(x=1.0, y=2.0),
            RegressionSample(x=2.0, y=4.0),
        ]
        result = finalize_regressions(regressions)

        assert "avg_y" in result["input_tokens_vs_ttft"]
        assert result["input_tokens_vs_ttft"]["avg_y"] == 3.0


class TestBuildZeroWarnings:
    """Test zero warning generation."""

    def test_returns_empty_for_no_zeroes(self) -> None:
        zero_counts = {
            "energy": 0,
            "power": 0,
            "ttft": 0,
            "latency": 0,
            "output_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        warnings = build_zero_warnings(zero_counts)
        assert warnings == []

    def test_generates_warning_for_zero_energy(self) -> None:
        zero_counts = {
            "energy": 5,
            "power": 0,
            "ttft": 0,
            "latency": 0,
            "output_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        warnings = build_zero_warnings(zero_counts)
        assert len(warnings) == 1
        assert "5" in warnings[0]
        assert "energy" in warnings[0]

    def test_includes_context_in_warnings(self) -> None:
        zero_counts = {
            "energy": 1,
            "power": 0,
            "ttft": 0,
            "latency": 0,
            "output_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        warnings = build_zero_warnings(zero_counts, context=" in test")
        assert " in test" in warnings[0]


class TestFilterNoneRegressions:
    """Test regression filtering."""

    def test_filters_regressions_with_none_values(self) -> None:
        regressions = {
            "good": {"slope": 1.0, "intercept": 0.0, "r2": 0.9, "avg_y": 5.0},
            "missing_slope": {"slope": None, "intercept": 0.0, "r2": 0.9, "avg_y": 5.0},
            "missing_r2": {"slope": 1.0, "intercept": 0.0, "r2": None, "avg_y": 5.0},
        }
        result = _filter_none_regressions(regressions)

        assert "good" in result
        assert "missing_slope" not in result
        assert "missing_r2" not in result

    def test_preserves_all_complete_regressions(self) -> None:
        regressions = {
            "reg1": {"slope": 1.0, "intercept": 0.0, "r2": 0.9, "avg_y": 5.0},
            "reg2": {"slope": 2.0, "intercept": 1.0, "r2": 0.8, "avg_y": 10.0},
        }
        result = _filter_none_regressions(regressions)

        assert len(result) == 2
        assert "reg1" in result
        assert "reg2" in result


class TestRegressionAnalysis:
    """Test the full RegressionAnalysis provider."""

    def test_requires_dataset_directory(self, tmp_path: Path) -> None:
        context = AnalysisContext(results_dir=tmp_path, options={})
        analysis = RegressionAnalysis()

        # Should raise because no dataset exists at the path
        with pytest.raises((RuntimeError, FileNotFoundError)):
            analysis.run(context)

    @patch("trafficbench.analysis.regression.load_metrics_dataset")
    @patch("trafficbench.analysis.regression.resolve_model_name")
    @patch("trafficbench.analysis.regression.iter_model_entries")
    def test_raises_on_no_entries(
        self,
        mock_iter: Mock,
        mock_resolve: Mock,
        mock_load: Mock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = Mock()
        mock_resolve.return_value = "test-model"
        mock_iter.return_value = []

        context = AnalysisContext(results_dir=tmp_path, options={})
        analysis = RegressionAnalysis()

        with pytest.raises(RuntimeError, match="No usable metrics"):
            analysis.run(context)

    @patch("trafficbench.analysis.regression.load_metrics_dataset")
    @patch("trafficbench.analysis.regression.resolve_model_name")
    @patch("trafficbench.analysis.regression.iter_model_entries")
    def test_creates_analysis_artifact(
        self,
        mock_iter: Mock,
        mock_resolve: Mock,
        mock_load: Mock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = Mock()
        mock_resolve.return_value = "test-model"
        mock_iter.return_value = [
            {
                "token_metrics": {"input": 100, "output": 50, "total": 150},
                "latency_metrics": {
                    "time_to_first_token_seconds": 0.5,
                    "total_query_seconds": 2.0,
                },
                "energy_metrics": {"per_query_joules": 10.0},
                "power_metrics": {"gpu": {"per_query_watts": {"avg": 5.0}}},
            }
        ]

        context = AnalysisContext(results_dir=tmp_path, options={})
        analysis = RegressionAnalysis()
        result = analysis.run(context)

        assert result.analysis == "regression"
        assert result.summary["total_samples"] == 1
        assert "report" in result.artifacts

        # Check artifact was written
        artifact_path = tmp_path / "analysis" / "regression.json"
        assert artifact_path.exists()

    @patch("trafficbench.analysis.regression.load_metrics_dataset")
    @patch("trafficbench.analysis.regression.resolve_model_name")
    @patch("trafficbench.analysis.regression.iter_model_entries")
    def test_skip_zeroes_filters_incomplete_regressions(
        self,
        mock_iter: Mock,
        mock_resolve: Mock,
        mock_load: Mock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = Mock()
        mock_resolve.return_value = "test-model"
        # Single sample won't produce valid regression
        mock_iter.return_value = [
            {
                "token_metrics": {"input": 100, "output": 50, "total": 150},
                "latency_metrics": {
                    "time_to_first_token_seconds": 0.5,
                    "total_query_seconds": 2.0,
                },
                "energy_metrics": {"per_query_joules": 10.0},
                "power_metrics": {"gpu": {"per_query_watts": {"avg": 5.0}}},
            }
        ]

        context = AnalysisContext(results_dir=tmp_path, options={"skip_zeroes": True})
        analysis = RegressionAnalysis()
        result = analysis.run(context)

        # With skip_zeroes, incomplete regressions should be filtered
        assert "regressions" in result.data
        # Single sample produces None slope/intercept/r2, so should be filtered
        assert len(result.data["regressions"]) == 0
