"""Tests for cli/bench.py — energy helper functions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipw.cli.bench import _compute_energy_delta, _compute_energy_metrics, _safe_max, _safe_mean


class TestSafeMean:
    def test_valid_values(self) -> None:
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty_list(self) -> None:
        assert _safe_mean([]) is None

    def test_all_none(self) -> None:
        assert _safe_mean([None, None]) is None

    def test_mixed_none_and_values(self) -> None:
        assert _safe_mean([None, 4.0, None, 6.0]) == pytest.approx(5.0)

    def test_filters_nan(self) -> None:
        assert _safe_mean([1.0, float("nan"), 3.0]) == pytest.approx(2.0)

    def test_filters_inf(self) -> None:
        assert _safe_mean([float("inf"), 2.0, 4.0]) == pytest.approx(3.0)

    def test_single_value(self) -> None:
        assert _safe_mean([7.5]) == pytest.approx(7.5)


class TestSafeMax:
    def test_valid_values(self) -> None:
        assert _safe_max([1.0, 5.0, 3.0]) == pytest.approx(5.0)

    def test_empty_list(self) -> None:
        assert _safe_max([]) is None

    def test_all_none(self) -> None:
        assert _safe_max([None, None]) is None

    def test_mixed_none_and_values(self) -> None:
        assert _safe_max([None, 4.0, None, 6.0]) == pytest.approx(6.0)

    def test_filters_nan(self) -> None:
        assert _safe_max([float("nan"), 3.0, 1.0]) == pytest.approx(3.0)

    def test_filters_inf(self) -> None:
        assert _safe_max([float("inf"), 2.0]) == pytest.approx(2.0)

    def test_single_value(self) -> None:
        assert _safe_max([7.5]) == pytest.approx(7.5)


class TestComputeEnergyDelta:
    def test_normal_delta(self) -> None:
        assert _compute_energy_delta([10.0, 15.0, 20.0]) == pytest.approx(10.0)

    def test_single_value_returns_none(self) -> None:
        assert _compute_energy_delta([10.0]) is None

    def test_empty_list(self) -> None:
        assert _compute_energy_delta([]) is None

    def test_all_none(self) -> None:
        assert _compute_energy_delta([None, None]) is None

    def test_negative_delta_returns_none(self) -> None:
        assert _compute_energy_delta([20.0, 10.0]) is None

    def test_filters_none_values(self) -> None:
        assert _compute_energy_delta([None, 10.0, None, 30.0]) == pytest.approx(20.0)

    def test_filters_negative_values(self) -> None:
        assert _compute_energy_delta([-5.0, 10.0, 30.0]) == pytest.approx(20.0)

    def test_filters_nan(self) -> None:
        assert _compute_energy_delta([float("nan"), 10.0, 20.0]) == pytest.approx(10.0)

    def test_filters_inf(self) -> None:
        assert _compute_energy_delta([float("inf"), 10.0, 20.0]) == pytest.approx(10.0)

    def test_zero_delta(self) -> None:
        assert _compute_energy_delta([10.0, 10.0]) == pytest.approx(0.0)


def _make_reading(**kwargs):
    """Create a mock reading with defaults for all fields."""
    defaults = {
        "energy_joules": None,
        "cpu_energy_joules": None,
        "power_watts": None,
        "cpu_power_watts": None,
        "gpu_memory_bandwidth_utilization_pct": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_sample(reading):
    """Wrap a reading in a sample namespace."""
    return SimpleNamespace(reading=reading)


class TestComputeEnergyMetrics:
    def test_basic_energy_computation(self) -> None:
        samples = [
            _make_sample(_make_reading(energy_joules=100.0, power_watts=200.0)),
            _make_sample(_make_reading(energy_joules=150.0, power_watts=250.0)),
            _make_sample(_make_reading(energy_joules=200.0, power_watts=300.0)),
        ]
        result = _compute_energy_metrics(samples, start_time=0.0, end_time=10.0)
        assert result["gpu_energy_joules"] == pytest.approx(100.0)
        assert result["avg_gpu_power_watts"] == pytest.approx(250.0)
        assert result["max_gpu_power_watts"] == pytest.approx(300.0)
        assert result["duration_seconds"] == pytest.approx(10.0)

    def test_empty_samples(self) -> None:
        result = _compute_energy_metrics([], start_time=0.0, end_time=10.0)
        assert result["gpu_energy_joules"] is None
        assert result["cpu_energy_joules"] is None
        assert result["avg_gpu_power_watts"] is None
        assert result["avg_mbu_pct"] is None
        assert result["telemetry_samples"] == 0

    def test_mbu_extraction(self) -> None:
        samples = [
            _make_sample(_make_reading(
                energy_joules=10.0,
                gpu_memory_bandwidth_utilization_pct=40.0,
            )),
            _make_sample(_make_reading(
                energy_joules=20.0,
                gpu_memory_bandwidth_utilization_pct=60.0,
            )),
        ]
        result = _compute_energy_metrics(samples, start_time=0.0, end_time=5.0)
        assert result["avg_mbu_pct"] == pytest.approx(50.0)
        assert result["max_mbu_pct"] == pytest.approx(60.0)

    def test_mbu_filters_negative(self) -> None:
        samples = [
            _make_sample(_make_reading(gpu_memory_bandwidth_utilization_pct=-1.0)),
            _make_sample(_make_reading(gpu_memory_bandwidth_utilization_pct=50.0)),
        ]
        result = _compute_energy_metrics(samples, start_time=0.0, end_time=1.0)
        assert result["avg_mbu_pct"] == pytest.approx(50.0)
        assert result["max_mbu_pct"] == pytest.approx(50.0)

    def test_mbu_none_when_all_unavailable(self) -> None:
        samples = [
            _make_sample(_make_reading(gpu_memory_bandwidth_utilization_pct=None)),
            _make_sample(_make_reading(gpu_memory_bandwidth_utilization_pct=-1.0)),
        ]
        result = _compute_energy_metrics(samples, start_time=0.0, end_time=1.0)
        assert result["avg_mbu_pct"] is None
        assert result["max_mbu_pct"] is None

    def test_cpu_energy(self) -> None:
        samples = [
            _make_sample(_make_reading(cpu_energy_joules=50.0, cpu_power_watts=80.0)),
            _make_sample(_make_reading(cpu_energy_joules=70.0, cpu_power_watts=90.0)),
        ]
        result = _compute_energy_metrics(samples, start_time=0.0, end_time=5.0)
        assert result["cpu_energy_joules"] == pytest.approx(20.0)
        assert result["avg_cpu_power_watts"] == pytest.approx(85.0)

    def test_total_energy_combines_gpu_and_cpu(self) -> None:
        samples = [
            _make_sample(_make_reading(energy_joules=100.0, cpu_energy_joules=50.0)),
            _make_sample(_make_reading(energy_joules=200.0, cpu_energy_joules=80.0)),
        ]
        result = _compute_energy_metrics(samples, start_time=0.0, end_time=5.0)
        assert result["total_energy_joules"] == pytest.approx(130.0)  # 100 + 30

    def test_total_energy_none_when_zero(self) -> None:
        samples = [
            _make_sample(_make_reading()),
        ]
        result = _compute_energy_metrics(samples, start_time=0.0, end_time=1.0)
        assert result["total_energy_joules"] is None

    def test_duration_non_negative(self) -> None:
        result = _compute_energy_metrics([], start_time=10.0, end_time=5.0)
        assert result["duration_seconds"] == 0.0

    def test_telemetry_samples_count(self) -> None:
        samples = [_make_sample(_make_reading()) for _ in range(7)]
        result = _compute_energy_metrics(samples, start_time=0.0, end_time=1.0)
        assert result["telemetry_samples"] == 7
