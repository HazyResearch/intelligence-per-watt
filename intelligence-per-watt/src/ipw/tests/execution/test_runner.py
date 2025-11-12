"""Tests for profiler runner orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from ipw.core.types import (
    ChatUsage,
    DatasetRecord,
    ProfilerConfig,
    Response,
    TelemetryReading,
)
from ipw.execution.runner import ProfilerRunner, _slugify_model, _stat_summary
from ipw.execution.telemetry_session import TelemetrySample


class TestStatSummary:
    """Test statistical summary computation."""

    def test_computes_stats_from_values(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _stat_summary(values)

        assert stats.avg == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_filters_none_values(self) -> None:
        values = [1.0, None, 3.0, None, 5.0]
        stats = _stat_summary(values)

        assert stats.avg == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_returns_none_stats_for_empty(self) -> None:
        values = []
        stats = _stat_summary(values)

        assert stats.avg is None
        assert stats.min is None
        assert stats.max is None
        assert stats.median is None

    def test_returns_none_stats_for_all_none(self) -> None:
        values = [None, None, None]
        stats = _stat_summary(values)

        assert stats.avg is None
        assert stats.min is None
        assert stats.max is None
        assert stats.median is None

    def test_handles_single_value(self) -> None:
        values = [42.0]
        stats = _stat_summary(values)

        assert stats.avg == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.median == 42.0


class TestSlugifyModel:
    """Test model name slugification."""

    def test_replaces_special_chars_with_underscores(self) -> None:
        assert _slugify_model("llama-3.2:1b") == "llama_3_2_1b"

    def test_strips_leading_trailing_underscores(self) -> None:
        assert _slugify_model("_model_") == "model"

    def test_preserves_alphanumeric(self) -> None:
        assert _slugify_model("llama32") == "llama32"

    def test_returns_model_for_empty_string(self) -> None:
        assert _slugify_model("") == "model"

    def test_returns_model_for_all_special_chars(self) -> None:
        assert _slugify_model("!!!") == "model"


class TestProfilerRunner:
    """Test ProfilerRunner orchestration."""

    def test_initializes_with_config(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="ollama",
            dataset_id="ipw",
        )
        runner = ProfilerRunner(config)
        assert runner._config == config

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    @patch("ipw.execution.runner.EnergyMonitorCollector")
    @patch("ipw.execution.runner.TelemetrySession")
    @patch("ipw.execution.runner.Dataset")
    def test_run_creates_output_directory(
        self,
        mock_dataset_class: Mock,
        mock_session: Mock,
        mock_collector: Mock,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
        tmp_path: Path,
    ) -> None:
        # Setup mocks
        mock_dataset = MagicMock()
        mock_dataset.size.return_value = 1
        mock_dataset.__iter__.return_value = iter([
            DatasetRecord(problem="test", answer="answer", subject="math")
        ])
        mock_dataset.dataset_id = "test"
        mock_dataset.dataset_name = "Test Dataset"
        mock_dataset_registry.get.return_value = Mock(return_value=mock_dataset)

        mock_client = Mock()
        mock_client.health.return_value = True
        mock_client.stream_chat_completion.return_value = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            time_to_first_token_ms=100.0,
        )
        mock_client_registry.get.return_value = Mock(return_value=mock_client)

        mock_collector_instance = Mock()
        mock_collector.return_value = mock_collector_instance

        mock_telemetry = Mock()
        mock_telemetry.window.return_value = []
        mock_session.return_value.__enter__.return_value = mock_telemetry

        # Mock Dataset.from_list to return a mock with save_to_disk that creates the directory
        mock_hf_dataset = Mock()
        
        def mock_save_to_disk(path: str):
            Path(path).mkdir(parents=True, exist_ok=True)
        
        mock_hf_dataset.save_to_disk = mock_save_to_disk
        mock_dataset_class.from_list.return_value = mock_hf_dataset

        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="test-dataset",
            output_dir=tmp_path,
        )
        runner = ProfilerRunner(config)
        runner.run()

        # Check that output directory was created
        assert (tmp_path / "profile_UNKNOWN_HW_test_model").exists()
        # Check that summary.json was written
        assert (tmp_path / "profile_UNKNOWN_HW_test_model" / "summary.json").exists()

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    def test_raises_on_unknown_dataset(
        self,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
    ) -> None:
        mock_dataset_registry.get.side_effect = KeyError("unknown")

        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="unknown",
        )
        runner = ProfilerRunner(config)

        with pytest.raises(RuntimeError, match="Unknown dataset"):
            runner.run()

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    def test_raises_on_unknown_client(
        self,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
    ) -> None:
        mock_dataset_registry.get.return_value = Mock(return_value=Mock())
        mock_client_registry.get.side_effect = KeyError("unknown")

        config = ProfilerConfig(
            model="test-model",
            client_id="unknown",
            dataset_id="test-dataset",
        )
        runner = ProfilerRunner(config)

        with pytest.raises(RuntimeError, match="Unknown client"):
            runner.run()

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    @patch("ipw.execution.runner.EnergyMonitorCollector")
    def test_raises_when_client_unhealthy(
        self,
        mock_collector: Mock,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
    ) -> None:
        mock_dataset = Mock()
        mock_dataset_registry.get.return_value = Mock(return_value=mock_dataset)

        mock_client = Mock()
        mock_client.health.return_value = False
        mock_client.client_name = "test-client"
        mock_client_registry.get.return_value = Mock(return_value=mock_client)

        mock_collector.return_value = Mock()

        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="test-dataset",
        )
        runner = ProfilerRunner(config)

        with pytest.raises(RuntimeError, match="unavailable"):
            runner.run()

    def test_compute_energy_metrics_handles_empty_readings(self) -> None:
        config = ProfilerConfig(
            model="test",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        metrics = runner._compute_energy_metrics([])
        assert metrics.per_query_joules is None
        assert metrics.total_joules is None

    def test_compute_energy_metrics_handles_first_query(self) -> None:
        config = ProfilerConfig(
            model="test",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        readings = [
            TelemetryReading(energy_joules=100.0),
            TelemetryReading(energy_joules=150.0),
        ]
        metrics = runner._compute_energy_metrics(readings)

        assert metrics.per_query_joules == 50.0
        assert metrics.total_joules == 50.0

    def test_compute_energy_metrics_handles_subsequent_queries(self) -> None:
        config = ProfilerConfig(
            model="test",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        # First query
        readings1 = [
            TelemetryReading(energy_joules=100.0),
            TelemetryReading(energy_joules=150.0),
        ]
        runner._compute_energy_metrics(readings1)

        # Second query
        readings2 = [
            TelemetryReading(energy_joules=150.0),
            TelemetryReading(energy_joules=200.0),
        ]
        metrics = runner._compute_energy_metrics(readings2)

        assert metrics.per_query_joules == 50.0

    def test_compute_energy_metrics_handles_counter_reset(self) -> None:
        config = ProfilerConfig(
            model="test",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        # First query
        readings1 = [
            TelemetryReading(energy_joules=100.0),
            TelemetryReading(energy_joules=150.0),
        ]
        runner._compute_energy_metrics(readings1)

        # Counter reset (goes backward)
        readings2 = [
            TelemetryReading(energy_joules=50.0),
            TelemetryReading(energy_joules=100.0),
        ]
        metrics = runner._compute_energy_metrics(readings2)

        # Should handle reset gracefully
        assert metrics.per_query_joules is None

    def test_compute_energy_metrics_filters_infinite(self) -> None:
        config = ProfilerConfig(
            model="test",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        readings = [
            TelemetryReading(energy_joules=float("inf")),
        ]
        metrics = runner._compute_energy_metrics(readings)

        assert metrics.per_query_joules is None

    def test_compute_energy_metrics_filters_negative(self) -> None:
        config = ProfilerConfig(
            model="test",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        readings = [
            TelemetryReading(energy_joules=-100.0),
        ]
        metrics = runner._compute_energy_metrics(readings)

        assert metrics.per_query_joules is None

    def test_build_record_creates_model_metrics(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            time_to_first_token_ms=100.0,
        )
        samples = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(
                    energy_joules=100.0,
                    power_watts=50.0,
                ),
            ),
            TelemetrySample(
                timestamp=2.0,
                reading=TelemetryReading(
                    energy_joules=150.0,
                    power_watts=50.0,
                ),
            ),
        ]

        result = runner._build_record(0, record, response, samples, 0.0, 2.0)

        assert result is not None
        assert "test-model" in result.model_metrics
        metrics = result.model_metrics["test-model"]
        assert metrics.token_metrics.input == 10
        assert metrics.token_metrics.output == 5
        assert metrics.token_metrics.total == 15

    def test_build_record_handles_zero_completion_tokens(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
            time_to_first_token_ms=0.0,
        )
        samples = []

        result = runner._build_record(0, record, response, samples, 0.0, 1.0)

        assert result is not None
        metrics = result.model_metrics["test-model"]
        assert metrics.latency_metrics.per_token_ms is None
        assert metrics.latency_metrics.throughput_tokens_per_sec is None

    def test_build_record_computes_throughput(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            time_to_first_token_ms=100.0,
        )
        samples = []

        # 10 tokens in 1 second = 10 tokens/sec
        result = runner._build_record(0, record, response, samples, 0.0, 1.0)

        assert result is not None
        metrics = result.model_metrics["test-model"]
        assert metrics.latency_metrics.throughput_tokens_per_sec == 10.0
        assert metrics.latency_metrics.per_token_ms == 100.0

    def test_get_output_path_includes_hardware_and_model(self, tmp_path: Path) -> None:
        config = ProfilerConfig(
            model="llama-3.2:1b",
            client_id="test",
            dataset_id="test",
            output_dir=tmp_path,
        )
        runner = ProfilerRunner(config)
        runner._hardware_label = "RTX3090"

        path = runner._get_output_path()
        assert "RTX3090" in str(path)
        assert "llama_3_2_1b" in str(path)
