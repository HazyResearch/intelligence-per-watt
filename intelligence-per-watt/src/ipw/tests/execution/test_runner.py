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
from ipw.execution.runner import (
    ProfilerRunner,
    _lookup_gpu_peak_tflops,
    _percentile,
    _slugify_model,
    _stat_summary,
    _sum_optional,
)
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
        mock_dataset.__iter__.return_value = iter(
            [DatasetRecord(problem="test", answer="answer", subject="math")]
        )
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
        mock_telemetry.readings.return_value = []
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
            warmup_queries=0,
        )
        runner = ProfilerRunner(config)
        runner.run()

        # Check that output directory was created
        expected_dir = tmp_path / "profile_UNKNOWN_HW_test_model_Test Dataset"
        assert expected_dir.exists()
        # Check that summary.json was written
        summary_path = expected_dir / "summary.json"
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text())
        assert summary["profiler_config"]["model"] == "test-model"
        assert "versions" in summary

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

    def test_compute_energy_metrics_handles_counter_reset_between_queries(self) -> None:
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

        # Should measure per-query window delta despite reset while idle
        assert metrics.per_query_joules == 50.0

    def test_compute_energy_metrics_ignores_idle_energy_between_queries(self) -> None:
        config = ProfilerConfig(
            model="test",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        # First query establishes baseline
        readings1 = [
            TelemetryReading(energy_joules=100.0),
            TelemetryReading(energy_joules=150.0),
        ]
        runner._compute_energy_metrics(readings1)

        # Second query starts after unrelated energy consumption
        readings2 = [
            TelemetryReading(energy_joules=250.0),
            TelemetryReading(energy_joules=260.0),
        ]
        metrics = runner._compute_energy_metrics(readings2)

        assert metrics.per_query_joules == 10.0

    def test_compute_energy_metrics_handles_counter_reset_within_query(self) -> None:
        config = ProfilerConfig(
            model="test",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        # First query establishes baseline
        readings1 = [
            TelemetryReading(energy_joules=100.0),
            TelemetryReading(energy_joules=150.0),
        ]
        runner._compute_energy_metrics(readings1)

        # Counter resets while the query is running
        readings2 = [
            TelemetryReading(energy_joules=200.0),
            TelemetryReading(energy_joules=10.0),
        ]
        metrics = runner._compute_energy_metrics(readings2)

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

    @patch.object(ProfilerRunner, "_process_records", autospec=True, side_effect=RuntimeError("boom"))
    @patch.object(ProfilerRunner, "_ensure_client_ready")
    @patch("ipw.execution.runner.TelemetrySession")
    @patch("ipw.execution.runner.EnergyMonitorCollector")
    def test_run_closes_client_on_error(
        self,
        mock_collector: Mock,
        mock_session: Mock,
        _mock_ensure_ready: Mock,
        _mock_process_records: Mock,
    ) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)
        client = Mock()
        client.close = Mock()
        dataset = Mock()

        mock_session.return_value.__enter__.return_value = Mock()
        mock_collector.return_value = Mock()

        with patch.object(runner, "_resolve_dataset", return_value=dataset), patch.object(
            runner, "_resolve_client", return_value=client
        ):
            with pytest.raises(RuntimeError):
                runner.run()

        client.close.assert_called_once()

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
        assert path == tmp_path / "profile_RTX3090_llama_3_2_1b_test"

    def test_build_record_computes_per_token_energy(self) -> None:
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
                reading=TelemetryReading(energy_joules=100.0, power_watts=50.0),
            ),
            TelemetrySample(
                timestamp=2.0,
                reading=TelemetryReading(energy_joules=150.0, power_watts=50.0),
            ),
        ]

        result = runner._build_record(0, record, response, samples, 0.0, 2.0)
        assert result is not None
        metrics = result.model_metrics["test-model"]
        # 50J / 5 output tokens = 10 J/token
        assert metrics.energy_metrics.energy_per_output_token_joules == 10.0
        # 50J / 15 total tokens = 3.333... J/token
        assert metrics.energy_metrics.energy_per_total_token_joules == pytest.approx(50.0 / 15)

    def test_build_record_per_token_energy_none_when_no_energy(self) -> None:
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

        result = runner._build_record(0, record, response, [], 0.0, 1.0)
        assert result is not None
        metrics = result.model_metrics["test-model"]
        assert metrics.energy_metrics.energy_per_output_token_joules is None
        assert metrics.energy_metrics.energy_per_total_token_joules is None

    def test_build_record_computes_itl_percentiles(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        # Simulate 11 tokens with 10ms inter-token latency (in seconds)
        timestamps = [1.0 + i * 0.010 for i in range(11)]
        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="hello world test",
            usage=ChatUsage(prompt_tokens=5, completion_tokens=11, total_tokens=16),
            time_to_first_token_ms=50.0,
            token_timestamps=timestamps,
        )

        result = runner._build_record(0, record, response, [], 0.0, 1.0)
        assert result is not None
        metrics = result.model_metrics["test-model"]
        # All inter-token latencies are 10ms
        assert metrics.latency_metrics.median_itl_ms == pytest.approx(10.0, abs=0.1)
        assert metrics.latency_metrics.p90_itl_ms == pytest.approx(10.0, abs=0.1)
        assert metrics.latency_metrics.p95_itl_ms == pytest.approx(10.0, abs=0.1)
        assert metrics.latency_metrics.p99_itl_ms == pytest.approx(10.0, abs=0.1)
        assert metrics.latency_metrics.itl_std_ms == pytest.approx(0.0, abs=0.1)

    def test_build_record_itl_none_with_single_token(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="hi",
            usage=ChatUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
            time_to_first_token_ms=50.0,
            token_timestamps=[1.0],
        )

        result = runner._build_record(0, record, response, [], 0.0, 1.0)
        assert result is not None
        metrics = result.model_metrics["test-model"]
        assert metrics.latency_metrics.median_itl_ms is None
        assert metrics.latency_metrics.p90_itl_ms is None

    def test_build_record_itl_none_with_no_timestamps(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            time_to_first_token_ms=50.0,
        )

        result = runner._build_record(0, record, response, [], 0.0, 1.0)
        assert result is not None
        metrics = result.model_metrics["test-model"]
        assert metrics.latency_metrics.median_itl_ms is None

    def test_build_record_computes_throughput_per_watt(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=100, total_tokens=110),
            time_to_first_token_ms=100.0,
        )
        # 100W average power, 100 tokens in 2 seconds = 50 tok/s
        # throughput_per_watt = 50 / 100 = 0.5
        samples = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(energy_joules=100.0, power_watts=100.0),
            ),
            TelemetrySample(
                timestamp=3.0,
                reading=TelemetryReading(energy_joules=300.0, power_watts=100.0),
            ),
        ]

        result = runner._build_record(0, record, response, samples, 0.0, 2.0)
        assert result is not None
        metrics = result.model_metrics["test-model"]
        assert metrics.efficiency.throughput_per_watt == pytest.approx(0.5)

    def test_build_record_efficiency_none_without_power(self) -> None:
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

        result = runner._build_record(0, record, response, [], 0.0, 1.0)
        assert result is not None
        metrics = result.model_metrics["test-model"]
        assert metrics.efficiency.throughput_per_watt is None

    def test_build_record_computes_flops_for_known_model(self) -> None:
        config = ProfilerConfig(
            model="llama3.2:1b",
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

        result = runner._build_record(0, record, response, [], 0.0, 1.0)
        assert result is not None
        metrics = result.model_metrics["llama3.2:1b"]
        assert metrics.compute_metrics.total_flops is not None
        assert metrics.compute_metrics.total_flops > 0
        assert metrics.compute_metrics.flops_per_token is not None
        assert metrics.compute_metrics.flops_per_request is not None

    def test_build_record_flops_zero_for_unknown_model(self) -> None:
        config = ProfilerConfig(
            model="unknown-model-xyz",
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

        result = runner._build_record(0, record, response, [], 0.0, 1.0)
        assert result is not None
        metrics = result.model_metrics["unknown-model-xyz"]
        # Unknown model → 0 flops → ComputeMetrics stays empty
        assert metrics.compute_metrics.total_flops is None

    def test_build_record_computes_phase_metrics(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        # TTFT of 500ms, request_start_time = 1.0 → ttft_epoch = 1.5
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            time_to_first_token_ms=500.0,
            request_start_time=1.0,
        )
        samples = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(energy_joules=100.0, power_watts=200.0),
            ),
            TelemetrySample(
                timestamp=1.3,
                reading=TelemetryReading(energy_joules=120.0, power_watts=200.0),
            ),
            TelemetrySample(
                timestamp=1.6,
                reading=TelemetryReading(energy_joules=150.0, power_watts=100.0),
            ),
            TelemetrySample(
                timestamp=2.0,
                reading=TelemetryReading(energy_joules=180.0, power_watts=100.0),
            ),
        ]

        result = runner._build_record(0, record, response, samples, 1.0, 2.0)
        assert result is not None
        metrics = result.model_metrics["test-model"]
        phase = metrics.phase_metrics
        # Prefill: samples at 1.0 and 1.3 (both <= 1.5), energy delta = 120-100 = 20
        assert phase.prefill_energy_j == 20.0
        # Decode: samples at 1.6 and 2.0 (both > 1.5), energy delta = 180-150 = 30
        assert phase.decode_energy_j == 30.0
        assert phase.prefill_duration_ms == 500.0
        assert phase.prefill_energy_per_input_token_j == pytest.approx(20.0 / 10)
        assert phase.decode_energy_per_output_token_j == pytest.approx(30.0 / 5)

    def test_build_record_phase_metrics_none_short_prefill(self) -> None:
        """PhaseMetrics gracefully returns None when prefill is too short for samples."""
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        # Very short TTFT → no prefill samples
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            time_to_first_token_ms=5.0,
            request_start_time=1.0,
        )
        # All samples are after TTFT epoch (1.005)
        samples = [
            TelemetrySample(
                timestamp=1.05,
                reading=TelemetryReading(energy_joules=100.0, power_watts=100.0),
            ),
            TelemetrySample(
                timestamp=1.1,
                reading=TelemetryReading(energy_joules=110.0, power_watts=100.0),
            ),
        ]

        result = runner._build_record(0, record, response, samples, 1.0, 1.1)
        assert result is not None
        phase = result.model_metrics["test-model"].phase_metrics
        # No prefill samples → prefill energy is None
        assert phase.prefill_energy_j is None
        # Decode should still work
        assert phase.decode_energy_j == 10.0

    def test_build_record_mfu_populated_when_gpu_known(self) -> None:
        config = ProfilerConfig(
            model="llama3.2:1b",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)
        from ipw.core.types import GpuInfo
        runner._gpu_info = GpuInfo(name="NVIDIA H100")

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            time_to_first_token_ms=100.0,
        )

        result = runner._build_record(0, record, response, [], 0.0, 1.0)
        assert result is not None
        metrics = result.model_metrics["llama3.2:1b"]
        assert metrics.hardware_utilization.derived.mfu is not None
        assert metrics.hardware_utilization.derived.mfu > 0
        assert metrics.hardware_utilization.derived.mfu < 1.0  # Should be well below 100%


class TestWarmupQueries:
    """Test warmup query exclusion."""

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    @patch("ipw.execution.runner.EnergyMonitorCollector")
    @patch("ipw.execution.runner.TelemetrySession")
    @patch("ipw.execution.runner.Dataset")
    def test_warmup_queries_excluded_from_records(
        self,
        mock_dataset_class: Mock,
        mock_session: Mock,
        mock_collector: Mock,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
        tmp_path: Path,
    ) -> None:
        # Setup dataset with 6 records (3 warmup + 3 measured)
        records = [
            DatasetRecord(problem=f"q{i}", answer=f"a{i}", subject="math")
            for i in range(6)
        ]
        mock_dataset = MagicMock()
        mock_dataset.size.return_value = 6
        mock_dataset.__iter__.return_value = iter(records)
        mock_dataset.dataset_id = "test"
        mock_dataset.dataset_name = "Test"
        mock_dataset_registry.get.return_value = Mock(return_value=mock_dataset)

        mock_client = Mock()
        mock_client.health.return_value = True
        mock_client.stream_chat_completion.return_value = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            time_to_first_token_ms=100.0,
        )
        mock_client_registry.get.return_value = Mock(return_value=mock_client)

        mock_collector.return_value = Mock()
        mock_telemetry = Mock()
        mock_telemetry.window.return_value = []
        mock_telemetry.readings.return_value = []
        mock_session.return_value.__enter__.return_value = mock_telemetry

        mock_hf_dataset = Mock()
        mock_hf_dataset.save_to_disk = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
        mock_dataset_class.from_list.return_value = mock_hf_dataset

        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
            output_dir=tmp_path,
            warmup_queries=3,
            max_queries=3,
        )
        runner = ProfilerRunner(config)
        runner.run()

        # 3 warmup + 3 measured = 6 total calls to client
        assert mock_client.stream_chat_completion.call_count == 6
        # But only 3 records stored
        assert len(runner._records) == 3

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    @patch("ipw.execution.runner.EnergyMonitorCollector")
    @patch("ipw.execution.runner.TelemetrySession")
    @patch("ipw.execution.runner.Dataset")
    def test_zero_warmup_queries(
        self,
        mock_dataset_class: Mock,
        mock_session: Mock,
        mock_collector: Mock,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
        tmp_path: Path,
    ) -> None:
        records = [
            DatasetRecord(problem=f"q{i}", answer=f"a{i}", subject="math")
            for i in range(3)
        ]
        mock_dataset = MagicMock()
        mock_dataset.size.return_value = 3
        mock_dataset.__iter__.return_value = iter(records)
        mock_dataset.dataset_id = "test"
        mock_dataset.dataset_name = "Test"
        mock_dataset_registry.get.return_value = Mock(return_value=mock_dataset)

        mock_client = Mock()
        mock_client.health.return_value = True
        mock_client.stream_chat_completion.return_value = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            time_to_first_token_ms=100.0,
        )
        mock_client_registry.get.return_value = Mock(return_value=mock_client)

        mock_collector.return_value = Mock()
        mock_telemetry = Mock()
        mock_telemetry.window.return_value = []
        mock_telemetry.readings.return_value = []
        mock_session.return_value.__enter__.return_value = mock_telemetry

        mock_hf_dataset = Mock()
        mock_hf_dataset.save_to_disk = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
        mock_dataset_class.from_list.return_value = mock_hf_dataset

        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
            output_dir=tmp_path,
            warmup_queries=0,
            max_queries=3,
        )
        runner = ProfilerRunner(config)
        runner.run()

        # 0 warmup → all 3 are measured
        assert mock_client.stream_chat_completion.call_count == 3
        assert len(runner._records) == 3


class TestPercentile:
    """Test _percentile helper."""

    def test_median_of_sorted_list(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 50) == 3.0

    def test_p90_of_10_values(self) -> None:
        values = list(range(1, 11))  # 1..10
        result = _percentile([float(v) for v in values], 90)
        assert result == pytest.approx(9.1)

    def test_p99_interpolates_for_small_list(self) -> None:
        values = [1.0, 2.0]
        assert _percentile(values, 99) == pytest.approx(1.99)

    def test_single_value(self) -> None:
        assert _percentile([42.0], 50) == 42.0


class TestGpuPeakTflopsLookup:
    """Test GPU peak TFLOPS lookup."""

    def test_finds_h100(self) -> None:
        assert _lookup_gpu_peak_tflops("NVIDIA H100") == 989.5

    def test_finds_a100(self) -> None:
        assert _lookup_gpu_peak_tflops("NVIDIA A100-SXM4-80GB") == 312.0

    def test_finds_rtx_3090(self) -> None:
        assert _lookup_gpu_peak_tflops("NVIDIA GeForce RTX 3090") == 71.0

    def test_returns_none_for_unknown(self) -> None:
        assert _lookup_gpu_peak_tflops("Unknown GPU") is None

    def test_returns_none_for_none(self) -> None:
        assert _lookup_gpu_peak_tflops(None) is None


class TestSumOptional:
    """Combining power/energy rails that are each absent on some platforms."""

    def test_sums_present_values(self) -> None:
        assert _sum_optional(1.0, 2.0, 3.0) == 6.0

    def test_ignores_missing_rails(self) -> None:
        assert _sum_optional(10.0, None, 5.0) == 15.0

    def test_all_missing_is_none(self) -> None:
        assert _sum_optional(None, None) is None

    def test_no_args_is_none(self) -> None:
        assert _sum_optional() is None


class TestSocEnergyMetrics:
    """Whole-SoC energy: needed because macOS reports GPU-only in the headline.

    On Apple Silicon powermetrics reports CPU/GPU/ANE separately and
    `per_query_joules` carries the GPU rail alone, so an ANE-resident model such
    as Apple Foundation Models would appear nearly free.
    """

    def _runner(self) -> ProfilerRunner:
        return ProfilerRunner(
            ProfilerConfig(model="afm-3-core-advanced", client_id="afm", dataset_id="test")
        )

    def test_soc_sums_all_three_rails(self) -> None:
        readings = [
            TelemetryReading(
                energy_joules=100.0, cpu_energy_joules=50.0, ane_energy_joules=10.0
            ),
            TelemetryReading(
                energy_joules=110.0, cpu_energy_joules=70.0, ane_energy_joules=25.0
            ),
        ]
        metrics = self._runner()._compute_energy_metrics(readings)

        assert metrics.per_query_joules == 10.0  # GPU only, unchanged semantics
        assert metrics.cpu_per_query_joules == 20.0
        assert metrics.ane_per_query_joules == 15.0
        assert metrics.soc_per_query_joules == 45.0
        assert metrics.soc_total_joules == 45.0

    def test_soc_exceeds_gpu_only_for_ane_workload(self) -> None:
        """The AFM case: almost all energy on the ANE rail."""
        readings = [
            TelemetryReading(
                energy_joules=100.0, cpu_energy_joules=20.0, ane_energy_joules=200.0
            ),
            TelemetryReading(
                energy_joules=100.4, cpu_energy_joules=22.0, ane_energy_joules=260.0
            ),
        ]
        metrics = self._runner()._compute_energy_metrics(readings)

        assert metrics.per_query_joules == pytest.approx(0.4)
        assert metrics.soc_per_query_joules == pytest.approx(62.4)
        assert metrics.soc_per_query_joules > metrics.per_query_joules

    def test_soc_falls_back_to_available_rails(self) -> None:
        # Discrete-GPU hosts report no ANE at all.
        readings = [
            TelemetryReading(energy_joules=100.0, cpu_energy_joules=50.0),
            TelemetryReading(energy_joules=140.0, cpu_energy_joules=60.0),
        ]
        metrics = self._runner()._compute_energy_metrics(readings)

        assert metrics.ane_per_query_joules is None
        assert metrics.soc_per_query_joules == 50.0

    def test_soc_is_none_without_any_energy(self) -> None:
        readings = [TelemetryReading(power_watts=10.0), TelemetryReading(power_watts=11.0)]
        metrics = self._runner()._compute_energy_metrics(readings)

        assert metrics.soc_per_query_joules is None


class TestDerivedMetricBasis:
    """macOS derives efficiency from SoC energy; other platforms stay GPU-only."""

    MODEL = "afm-3-core"

    def _metrics(self, platform_name: str):
        """Build one record and return its ModelMetrics."""
        runner = ProfilerRunner(
            ProfilerConfig(model=self.MODEL, client_id="afm", dataset_id="test")
        )
        readings = [
            TelemetryReading(
                power_watts=1.0,
                cpu_power_watts=3.0,
                ane_power_watts=6.0,
                energy_joules=100.0,
                cpu_energy_joules=50.0,
                ane_energy_joules=10.0,
                platform=platform_name,
            ),
            TelemetryReading(
                power_watts=1.0,
                cpu_power_watts=3.0,
                ane_power_watts=6.0,
                energy_joules=101.0,
                cpu_energy_joules=53.0,
                ane_energy_joules=16.0,
                platform=platform_name,
            ),
        ]
        samples = [TelemetrySample(timestamp=float(i), reading=r) for i, r in enumerate(readings)]
        response = Response(
            content="hi",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            time_to_first_token_ms=5.0,
        )
        record = DatasetRecord(problem="q", answer="a", subject="test")
        built = runner._build_record(0, record, response, samples, 0.0, 1.0)
        assert built is not None
        return built.model_metrics[self.MODEL]

    def test_macos_uses_soc_energy(self) -> None:
        metrics = self._metrics("macos")
        # GPU delta is 1.0 J; SoC delta is 1 + 3 + 6 = 10.0 J over 10 tokens.
        assert metrics.energy_metrics.per_query_joules == pytest.approx(1.0)
        assert metrics.energy_metrics.soc_per_query_joules == pytest.approx(10.0)
        assert metrics.energy_metrics.energy_per_output_token_joules == pytest.approx(1.0)
        # 10 tok/s over SoC power 10 W.
        assert metrics.efficiency.throughput_per_watt == pytest.approx(1.0)

    def test_non_macos_keeps_gpu_only_basis(self) -> None:
        metrics = self._metrics("linux")
        # GPU-only: 1.0 J over 10 tokens, 10 tok/s over 1 W.
        assert metrics.energy_metrics.energy_per_output_token_joules == pytest.approx(0.1)
        assert metrics.efficiency.throughput_per_watt == pytest.approx(10.0)

    def test_ane_and_soc_power_stats_recorded(self) -> None:
        metrics = self._metrics("macos")
        assert metrics.power_metrics.ane.per_query_watts.avg == pytest.approx(6.0)
        assert metrics.power_metrics.soc.per_query_watts.avg == pytest.approx(10.0)
        # GPU stats keep their original meaning.
        assert metrics.power_metrics.gpu.per_query_watts.avg == pytest.approx(1.0)


class TestPhaseEnergyBasis:
    """Phase energy must use the same rail basis as the per-query metrics."""

    def test_macos_phase_energy_includes_ane(self) -> None:
        readings = [
            TelemetryReading(
                energy_joules=10.0,
                cpu_energy_joules=5.0,
                ane_energy_joules=100.0,
                platform="macos",
            ),
            TelemetryReading(
                energy_joules=10.5,
                cpu_energy_joules=7.0,
                ane_energy_joules=160.0,
                platform="macos",
            ),
        ]
        runner = ProfilerRunner(
            ProfilerConfig(model="afm-3-core", client_id="afm", dataset_id="test")
        )
        # 0.5 GPU + 2.0 CPU + 60.0 ANE
        assert runner._compute_phase_energy(readings) == pytest.approx(62.5)

    def test_non_macos_phase_energy_stays_gpu_only(self) -> None:
        readings = [
            TelemetryReading(energy_joules=10.0, cpu_energy_joules=5.0, platform="linux"),
            TelemetryReading(energy_joules=14.0, cpu_energy_joules=9.0, platform="linux"),
        ]
        runner = ProfilerRunner(
            ProfilerConfig(model="m", client_id="vllm", dataset_id="test")
        )
        assert runner._compute_phase_energy(readings) == pytest.approx(4.0)

    def test_phase_power_basis(self) -> None:
        mac = TelemetryReading(
            power_watts=1.0, cpu_power_watts=2.0, ane_power_watts=7.0, platform="macos"
        )
        linux = TelemetryReading(
            power_watts=300.0, cpu_power_watts=90.0, platform="linux"
        )
        assert ProfilerRunner._phase_power(mac) == pytest.approx(10.0)
        assert ProfilerRunner._phase_power(linux) == pytest.approx(300.0)
