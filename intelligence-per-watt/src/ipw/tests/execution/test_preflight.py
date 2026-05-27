"""Tests for execution/preflight.py — startup hardware baseline check."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ipw.execution.preflight import run_preflight


class TestPreflight:
    def test_clean_hardware_returns_dirty_false(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.5, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=2.0):
            r = run_preflight(sample_seconds=0.0)
        assert r.shared_device_baseline_dirty is False
        assert r.gpu_util_pct_avg == pytest.approx(0.5)
        assert r.cpu_util_pct_avg == pytest.approx(2.0)
        assert r.foreign_gpu_pids == []

    def test_gpu_util_above_threshold_sets_dirty(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(25.0, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            r = run_preflight(sample_seconds=0.0, gpu_util_threshold_pct=5.0)
        assert r.shared_device_baseline_dirty is True

    def test_cpu_util_above_threshold_sets_dirty(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=50.0):
            r = run_preflight(sample_seconds=0.0, cpu_util_threshold_pct=10.0)
        assert r.shared_device_baseline_dirty is True

    def test_foreign_pids_set_dirty(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [12345])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            r = run_preflight(sample_seconds=0.0)
        assert r.shared_device_baseline_dirty is True
        assert r.foreign_gpu_pids == [12345]

    def test_strict_mode_raises_when_dirty(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(50.0, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            with pytest.raises(RuntimeError, match="Shared-device contamination"):
                run_preflight(sample_seconds=0.0, strict=True)

    def test_nvml_unavailable_degrades_gracefully(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", side_effect=RuntimeError("no nvml")), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            r = run_preflight(sample_seconds=0.0)
        assert r.gpu_util_pct_avg is None
        assert r.shared_device_baseline_dirty is False
        assert any("NVML" in w for w in r.warnings)

    def test_psutil_unavailable_degrades_gracefully(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", side_effect=RuntimeError("no psutil")):
            r = run_preflight(sample_seconds=0.0)
        assert r.cpu_util_pct_avg is None
        assert r.shared_device_baseline_dirty is False
        assert any("psutil" in w for w in r.warnings)

    def test_strict_does_not_raise_when_clean(self) -> None:
        with patch("ipw.execution.preflight._sample_gpu_util", return_value=(0.1, [])), \
             patch("ipw.execution.preflight._sample_cpu_util", return_value=1.0):
            # Should not raise
            r = run_preflight(sample_seconds=0.0, strict=True)
        assert r.shared_device_baseline_dirty is False
