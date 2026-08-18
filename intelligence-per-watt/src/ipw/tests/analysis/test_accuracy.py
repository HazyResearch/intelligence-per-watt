"""Tests for the accuracy analysis."""

from __future__ import annotations

from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest

from ipw.analysis.accuracy import AccuracyAnalysis
from ipw.analysis.base import AnalysisContext


@patch.object(AccuracyAnalysis, "_needs_evaluation", return_value=False)
@patch("ipw.analysis.accuracy.resolve_model_name")
@patch("ipw.analysis.accuracy.load_metrics_dataset")
def test_computes_intelligence_metrics_from_energy_counters(
    mock_load: mock.MagicMock,
    mock_resolve: mock.MagicMock,
    _mock_needs_eval: mock.MagicMock,
    tmp_path: Path,
) -> None:
    mock_resolve.return_value = "test-model"
    mock_load.return_value = [
        {
            "problem": "p1",
            "answer": "a1",
            "model_answers": {"test-model": "a1"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": True},
                    "energy_metrics": {"per_query_joules": 10.0},
                    "latency_metrics": {"total_query_seconds": 2.0},
                }
            },
        },
        {
            "problem": "p2",
            "answer": "a2",
            "model_answers": {"test-model": "wrong"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": False},
                    "energy_metrics": {"per_query_joules": 20.0},
                    "latency_metrics": {"total_query_seconds": 4.0},
                }
            },
        },
    ]

    context = AnalysisContext(results_dir=tmp_path, options={})
    result = AccuracyAnalysis().run(context)

    assert result.summary["accuracy"] == pytest.approx(0.5)
    assert result.summary["intelligence_per_joule"] == pytest.approx(1 / 30)
    assert result.summary["intelligence_per_watt"] == pytest.approx(0.1)
    efficiency = result.data["efficiency"]["test-model"]
    assert efficiency["energy"]["count"] == 2
    assert efficiency["power"]["derived_power_samples"] == 2
    assert efficiency["power"]["power_metric_samples"] == 0
    assert "analysis/accuracy.json" in str(result.artifacts["report"])


@patch.object(AccuracyAnalysis, "_needs_evaluation", return_value=False)
@patch("ipw.analysis.accuracy.resolve_model_name")
@patch("ipw.analysis.accuracy.load_metrics_dataset")
def test_handles_missing_energy_and_uses_power_metrics(
    mock_load: mock.MagicMock,
    mock_resolve: mock.MagicMock,
    _mock_needs_eval: mock.MagicMock,
    tmp_path: Path,
) -> None:
    mock_resolve.return_value = "test-model"
    mock_load.return_value = [
        {
            "problem": "p1",
            "answer": "a1",
            "model_answers": {"test-model": "a1"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": True},
                    "power_metrics": {"gpu": {"per_query_watts": {"avg": 5.0}}},
                }
            },
        },
        {
            "problem": "p2",
            "answer": "a2",
            "model_answers": {"test-model": "wrong"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": False},
                    "power_metrics": {"gpu": {"per_query_watts": {"avg": 5.0}}},
                }
            },
        },
    ]

    context = AnalysisContext(results_dir=tmp_path, options={})
    result = AccuracyAnalysis().run(context)

    assert result.summary["intelligence_per_joule"] is None
    assert result.summary["energy_sample_count"] == 0
    assert result.summary["intelligence_per_watt"] == pytest.approx(0.1)
    efficiency = result.data["efficiency"]["test-model"]
    assert efficiency["power"]["power_metric_samples"] == 2
    assert any("No per-query energy measurements" in msg for msg in result.warnings)


@patch.object(AccuracyAnalysis, "_needs_evaluation", return_value=False)
@patch("ipw.analysis.accuracy.resolve_model_name")
@patch("ipw.analysis.accuracy.load_metrics_dataset")
def test_intelligence_metrics_use_overall_accuracy(
    mock_load: mock.MagicMock,
    mock_resolve: mock.MagicMock,
    _mock_needs_eval: mock.MagicMock,
    tmp_path: Path,
) -> None:
    mock_resolve.return_value = "test-model"
    mock_load.return_value = [
        {
            "problem": "p1",
            "answer": "a1",
            "model_answers": {"test-model": "a1"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": True},
                    "energy_metrics": {"per_query_joules": 10.0},
                    "latency_metrics": {"total_query_seconds": 2.0},
                }
            },
        },
        {
            "problem": "p2",
            "answer": "a2",
            "model_answers": {"test-model": "wrong"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": False},
                }
            },
        },
    ]

    context = AnalysisContext(results_dir=tmp_path, options={})
    result = AccuracyAnalysis().run(context)

    # Overall accuracy is 0.5; energy/power averages are from the single measured query.
    assert result.summary["accuracy"] == pytest.approx(0.5)
    assert result.summary["intelligence_per_joule"] == pytest.approx(0.5 / 10.0)
    assert result.summary["intelligence_per_watt"] == pytest.approx(0.5 / 5.0)


@patch.object(AccuracyAnalysis, "_needs_evaluation", return_value=False)
@patch("ipw.analysis.accuracy.resolve_model_name")
@patch("ipw.analysis.accuracy.load_metrics_dataset")
def test_zero_energy_imputed_from_power_and_latency(
    mock_load: mock.MagicMock,
    mock_resolve: mock.MagicMock,
    _mock_needs_eval: mock.MagicMock,
    tmp_path: Path,
) -> None:
    mock_resolve.return_value = "test-model"
    mock_load.return_value = [
        {
            "problem": "p1",
            "answer": "a1",
            "model_answers": {"test-model": "a1"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": True},
                    "energy_metrics": {"per_query_joules": 0.0},
                    "latency_metrics": {"total_query_seconds": 2.0},
                    "power_metrics": {"gpu": {"per_query_watts": {"avg": 4.0}}},
                }
            },
        },
        {
            "problem": "p2",
            "answer": "a2",
            "model_answers": {"test-model": "wrong"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": False},
                    "energy_metrics": {"per_query_joules": 10.0},
                    "latency_metrics": {"total_query_seconds": 2.0},
                }
            },
        },
    ]

    context = AnalysisContext(results_dir=tmp_path, options={})
    result = AccuracyAnalysis().run(context)

    # Imputed energy uses per-record power * latency: 4.0 * 2.0 = 8.0
    # Combined energies = [10.0, 8.0] => avg 9.0
    # Avg power = (4.0 + 5.0) / 2 = 4.5 (second query derives power from energy/latency)
    assert result.summary["accuracy"] == pytest.approx(0.5)
    assert result.summary["avg_per_query_energy_joules"] == pytest.approx(9.0)
    assert result.summary["intelligence_per_joule"] == pytest.approx(0.5 / 9.0)
    assert result.summary["intelligence_per_watt"] == pytest.approx(0.5 / 4.5)
    eff = result.data["efficiency"]["test-model"]["energy"]
    assert eff["imputed_from_power"] == pytest.approx(8.0)
    assert eff["imputed_count"] == 1
    assert "imputed_energy_from_power" not in result.summary
    assert any("Imputed energy" in msg for msg in result.warnings)


def test_needs_evaluation_retries_failed_until_limit() -> None:
    analysis = AccuracyAnalysis()
    model_name = "test-model"

    retryable = [
        {
            "model_metrics": {
                model_name: {
                    "evaluation": {
                        "is_correct": None,
                        "metadata": {"evaluation_failed": True, "evaluation_attempts": 1},
                    }
                }
            }
        }
    ]
    assert analysis._needs_evaluation(retryable, model_name) is True

    exhausted = [
        {
            "model_metrics": {
                model_name: {
                    "evaluation": {
                        "is_correct": None,
                        "metadata": {
                            "evaluation_failed": True,
                            "evaluation_attempts": analysis.MAX_EVALUATION_ATTEMPTS,
                        },
                    }
                }
            }
        }
    ]
    assert analysis._needs_evaluation(exhausted, model_name) is False


class TestApplySoCBasis:
    """IPJ/IPW must aggregate the same rails the runner derived its metrics from.

    On Apple Silicon powermetrics splits GPU, CPU and ANE into separate rails
    and the GPU rail is near-idle for an ANE-resident model, so aggregating GPU
    alone reported energies ~3 orders of magnitude too low.
    """

    def _row(self, model: str, metrics: dict) -> dict:
        return {
            "problem": "p",
            "answer": "a",
            "model_answers": {model: "a"},
            "model_metrics": {model: {"evaluation": {"is_correct": True}, **metrics}},
        }

    def _run(self, tmp_path: Path, metrics: dict):
        with patch.object(
            AccuracyAnalysis, "_needs_evaluation", return_value=False
        ), patch("ipw.analysis.accuracy.resolve_model_name") as mock_resolve, patch(
            "ipw.analysis.accuracy.load_metrics_dataset"
        ) as mock_load:
            mock_resolve.return_value = "test-model"
            mock_load.return_value = [self._row("test-model", metrics)]
            return AccuracyAnalysis().run(
                AnalysisContext(results_dir=tmp_path, options={})
            )

    APPLE_METRICS = {
        "energy_metrics": {
            "per_query_joules": 0.06,
            "soc_per_query_joules": 240.0,
            "basis": "soc",
        },
        "power_metrics": {
            "gpu": {"per_query_watts": {"avg": 0.002}},
            "soc": {"per_query_watts": {"avg": 8.0}},
            "basis": "soc",
        },
        "latency_metrics": {"total_query_seconds": 30.0},
    }

    def test_recorded_soc_basis_is_used(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path, self.APPLE_METRICS).summary

        assert summary["energy_basis"] == "soc"
        assert summary["avg_per_query_energy_joules"] == pytest.approx(240.0)
        assert summary["intelligence_per_joule"] == pytest.approx(1 / 240.0)

    def test_apple_vendor_infers_soc_basis_without_recorded_field(
        self, tmp_path: Path
    ) -> None:
        # Datasets profiled before EnergyMetrics.basis existed.
        metrics = {
            "energy_metrics": {"per_query_joules": 0.06, "soc_per_query_joules": 240.0},
            "power_metrics": {
                "gpu": {"per_query_watts": {"avg": 0.002}},
                "soc": {"per_query_watts": {"avg": 8.0}},
            },
            "latency_metrics": {"total_query_seconds": 30.0},
            "gpu_info": {"name": "Apple GPU", "vendor": "Apple"},
        }

        summary = self._run(tmp_path, metrics).summary

        assert summary["energy_basis"] == "soc"
        assert summary["avg_per_query_energy_joules"] == pytest.approx(240.0)

    def test_discrete_gpu_keeps_gpu_basis(self, tmp_path: Path) -> None:
        metrics = {
            "energy_metrics": {
                "per_query_joules": 100.0,
                # The runner sums CPU into soc_* on every platform; a discrete
                # GPU host must not silently start counting it.
                "soc_per_query_joules": 175.0,
                "basis": "gpu",
            },
            "power_metrics": {
                "gpu": {"per_query_watts": {"avg": 50.0}},
                "soc": {"per_query_watts": {"avg": 87.5}},
                "basis": "gpu",
            },
            "latency_metrics": {"total_query_seconds": 2.0},
            "gpu_info": {"name": "NVIDIA H100", "vendor": "NVIDIA"},
        }

        summary = self._run(tmp_path, metrics).summary

        assert summary["energy_basis"] == "gpu"
        assert summary["avg_per_query_energy_joules"] == pytest.approx(100.0)

    def test_soc_basis_falls_back_to_gpu_when_soc_absent(self, tmp_path: Path) -> None:
        metrics = {
            "energy_metrics": {"per_query_joules": 12.0, "basis": "soc"},
            "power_metrics": {
                "gpu": {"per_query_watts": {"avg": 4.0}},
                "basis": "soc",
            },
            "latency_metrics": {"total_query_seconds": 3.0},
        }

        summary = self._run(tmp_path, metrics).summary

        assert summary["avg_per_query_energy_joules"] == pytest.approx(12.0)
