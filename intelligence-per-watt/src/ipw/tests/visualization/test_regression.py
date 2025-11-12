"""Tests for regression visualization provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from ipw.visualization.regression import (
    RegressionVisualization,
    _extract_regression_samples,
    _generate_linear_fit,
    _generate_log_fit,
    _infer_hardware_label,
    _infer_model_name,
    _load_regression_data,
    _safe_get,
)
from ipw.visualization.base import VisualizationContext


class TestSafeGet:
    """Test safe value extraction."""

    def test_extracts_from_dict(self) -> None:
        obj = {"key": "value"}
        assert _safe_get(obj, "key") == "value"

    def test_extracts_from_object(self) -> None:
        obj = Mock()
        obj.key = "value"
        assert _safe_get(obj, "key") == "value"

    def test_returns_none_for_missing_key(self) -> None:
        obj = {"other": "value"}
        assert _safe_get(obj, "key") is None

    def test_returns_none_for_missing_attribute(self) -> None:
        obj = Mock(spec=[])
        assert _safe_get(obj, "key") is None


class TestInferModelName:
    """Test model name inference."""

    def test_returns_first_model_name(self) -> None:
        dataset = Dataset.from_list([
            {"model_metrics": {"llama": {"tokens": 100}}},
        ])
        model = _infer_model_name(dataset)
        assert model == "llama"

    def test_returns_none_when_no_models(self) -> None:
        dataset = Dataset.from_list([
            {"other_field": "value"},
        ])
        model = _infer_model_name(dataset)
        assert model is None

    def test_skips_empty_model_metrics(self) -> None:
        dataset = Dataset.from_list([
            {"model_metrics": {}},
            {"model_metrics": {"gpt4": {"tokens": 100}}},
        ])
        model = _infer_model_name(dataset)
        assert model == "gpt4"


class TestInferHardwareLabel:
    """Test hardware label inference."""

    def test_extracts_gpu_name(self) -> None:
        dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {"gpu_info": {"name": "NVIDIA RTX 3090"}}
                }
            },
        ])
        label = _infer_hardware_label(dataset, "llama")
        assert label == "NVIDIA RTX 3090"

    def test_falls_back_to_cpu_brand(self) -> None:
        dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {"system_info": {"cpu_brand": "Intel i9"}}
                }
            },
        ])
        label = _infer_hardware_label(dataset, "llama")
        assert label == "Intel i9"

    def test_returns_unknown_when_no_info(self) -> None:
        dataset = Dataset.from_list([
            {"model_metrics": {"llama": {}}},
        ])
        label = _infer_hardware_label(dataset, "llama")
        assert label == "Unknown"

    def test_skips_unknown_cpu(self) -> None:
        dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {"system_info": {"cpu_brand": "Unknown CPU"}}
                }
            },
        ])
        label = _infer_hardware_label(dataset, "llama")
        assert label == "Unknown"


class TestExtractRegressionSamples:
    """Test sample extraction from dataset."""

    def test_extracts_simple_path(self) -> None:
        dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {
                        "token_metrics": {"total": 100},
                        "latency_metrics": {"total_query_seconds": 2.0},
                    }
                }
            },
            {
                "model_metrics": {
                    "llama": {
                        "token_metrics": {"total": 200},
                        "latency_metrics": {"total_query_seconds": 4.0},
                    }
                }
            },
        ])

        xs, ys = _extract_regression_samples(
            dataset,
            "llama",
            ["token_metrics", "total"],
            ["latency_metrics", "total_query_seconds"],
        )

        assert xs == [100.0, 200.0]
        assert ys == [2.0, 4.0]

    def test_derives_total_tokens_when_missing(self) -> None:
        dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {
                        "token_metrics": {"input": 80, "output": 20},
                        "latency_metrics": {"total_query_seconds": 2.0},
                    }
                }
            },
        ])

        xs, ys = _extract_regression_samples(
            dataset,
            "llama",
            ["token_metrics", "total"],
            ["latency_metrics", "total_query_seconds"],
        )

        assert xs == [100.0]

    def test_skips_none_values(self) -> None:
        dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {
                        "token_metrics": {"total": 100},
                        "latency_metrics": {"total_query_seconds": None},
                    }
                }
            },
            {
                "model_metrics": {
                    "llama": {
                        "token_metrics": {"total": 200},
                        "latency_metrics": {"total_query_seconds": 4.0},
                    }
                }
            },
        ])

        xs, ys = _extract_regression_samples(
            dataset,
            "llama",
            ["token_metrics", "total"],
            ["latency_metrics", "total_query_seconds"],
        )

        assert xs == [200.0]
        assert ys == [4.0]

    def test_skips_infinite_values(self) -> None:
        dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {
                        "token_metrics": {"total": float("inf")},
                        "latency_metrics": {"total_query_seconds": 2.0},
                    }
                }
            },
        ])

        xs, ys = _extract_regression_samples(
            dataset,
            "llama",
            ["token_metrics", "total"],
            ["latency_metrics", "total_query_seconds"],
        )

        assert xs == []
        assert ys == []

    def test_extracts_nested_path(self) -> None:
        dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {
                        "token_metrics": {"total": 100},
                        "power_metrics": {
                            "gpu": {"per_query_watts": {"avg": 50.0}}
                        },
                    }
                }
            },
        ])

        xs, ys = _extract_regression_samples(
            dataset,
            "llama",
            ["token_metrics", "total"],
            ["power_metrics", "gpu", "per_query_watts", "avg"],
        )

        assert xs == [100.0]
        assert ys == [50.0]


class TestGenerateLinearFit:
    """Test linear fit generation."""

    def test_generates_line_for_valid_data(self) -> None:
        xs = [1.0, 2.0, 3.0]
        slope = 2.0
        intercept = 1.0

        result = _generate_linear_fit(xs, slope, intercept)
        assert result is not None

        x_line, y_line = result
        assert len(x_line) == 200
        assert len(y_line) == 200

    def test_returns_none_for_empty_data(self) -> None:
        result = _generate_linear_fit([], 1.0, 0.0)
        assert result is None

    def test_returns_none_for_single_point(self) -> None:
        result = _generate_linear_fit([1.0], 1.0, 0.0)
        assert result is None

    def test_returns_none_for_constant_x(self) -> None:
        result = _generate_linear_fit([5.0, 5.0, 5.0], 1.0, 0.0)
        assert result is None


class TestGenerateLogFit:
    """Test log fit generation."""

    def test_generates_line_for_valid_data(self) -> None:
        xs = [1.0, 2.0, 3.0]
        slope = 2.0
        intercept = 1.0

        result = _generate_log_fit(xs, slope, intercept)
        assert result is not None

        x_line, y_line = result
        assert len(x_line) == 200
        assert len(y_line) == 200

    def test_returns_none_for_empty_data(self) -> None:
        result = _generate_log_fit([], 1.0, 0.0)
        assert result is None

    def test_filters_non_positive_values(self) -> None:
        xs = [-1.0, 0.0, 1.0, 2.0]
        slope = 2.0
        intercept = 1.0

        result = _generate_log_fit(xs, slope, intercept)
        assert result is not None

    def test_returns_none_when_all_non_positive(self) -> None:
        xs = [-1.0, 0.0, -2.0]
        slope = 2.0
        intercept = 1.0

        result = _generate_log_fit(xs, slope, intercept)
        assert result is None


class TestLoadRegressionData:
    """Test regression data loading."""

    def test_loads_regression_json(self, tmp_path: Path) -> None:
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()

        regression_file = analysis_dir / "regression.json"
        regression_file.write_text(json.dumps({"data": {"key": "value"}}))

        data = _load_regression_data(tmp_path)
        assert data["data"]["key"] == "value"

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Regression analysis not found"):
            _load_regression_data(tmp_path)


class TestRegressionVisualization:
    """Test RegressionVisualization provider."""

    @patch("ipw.visualization.regression._load_dataset")
    @patch("ipw.visualization.regression._load_regression_data")
    @patch("ipw.visualization.regression._create_scatter_plot")
    def test_renders_plots(
        self,
        mock_create_plot: Mock,
        mock_load_regression: Mock,
        mock_load_dataset: Mock,
        tmp_path: Path,
    ) -> None:
        # Setup mock data
        mock_load_regression.return_value = {
            "data": {
                "regressions": {
                    "input_tokens_vs_ttft": {
                        "count": 10,
                        "slope": 0.001,
                        "intercept": 0.1,
                        "r2": 0.9,
                    },
                    "total_tokens_vs_energy": {
                        "count": 10,
                        "slope": 0.05,
                        "intercept": 1.0,
                        "r2": 0.85,
                    },
                }
            }
        }

        mock_dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "llama": {
                        "token_metrics": {"input": 100, "output": 50, "total": 150},
                        "latency_metrics": {"time_to_first_token_seconds": 0.5},
                        "energy_metrics": {"per_query_joules": 10.0},
                        "gpu_info": {"name": "RTX 3090"},
                    }
                }
            },
        ])
        mock_load_dataset.return_value = mock_dataset

        context = VisualizationContext(
            results_dir=tmp_path,
            output_dir=tmp_path / "plots",
            options={},
        )

        viz = RegressionVisualization()
        result = viz.render(context)

        assert result.visualization == "regression"
        # Should have created plots
        assert mock_create_plot.call_count >= 2

    @patch("ipw.visualization.regression._load_dataset")
    @patch("ipw.visualization.regression._load_regression_data")
    def test_skips_plots_with_no_data(
        self,
        mock_load_regression: Mock,
        mock_load_dataset: Mock,
        tmp_path: Path,
    ) -> None:
        mock_load_regression.return_value = {
            "data": {
                "regressions": {
                    "input_tokens_vs_ttft": {"count": 0},
                }
            }
        }

        mock_dataset = Dataset.from_list([
            {"model_metrics": {"llama": {}}},
        ])
        mock_load_dataset.return_value = mock_dataset

        context = VisualizationContext(
            results_dir=tmp_path,
            output_dir=tmp_path / "plots",
            options={},
        )

        viz = RegressionVisualization()
        result = viz.render(context)

        # Should have warnings about skipped plots
        assert len(result.warnings) > 0

    @patch("ipw.visualization.regression._load_dataset")
    @patch("ipw.visualization.regression._load_regression_data")
    def test_uses_model_from_options(
        self,
        mock_load_regression: Mock,
        mock_load_dataset: Mock,
        tmp_path: Path,
    ) -> None:
        mock_load_regression.return_value = {"data": {"regressions": {}}}

        mock_dataset = Dataset.from_list([
            {
                "model_metrics": {
                    "custom-model": {
                        "gpu_info": {"name": "RTX 3090"},
                    }
                }
            },
        ])
        mock_load_dataset.return_value = mock_dataset

        context = VisualizationContext(
            results_dir=tmp_path,
            output_dir=tmp_path / "plots",
            options={"model": "custom-model"},
        )

        viz = RegressionVisualization()
        result = viz.render(context)

        assert result.metadata["model"] == "custom-model"

