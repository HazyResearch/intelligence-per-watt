"""Tier 3 integration tests for NVIDIA GPU telemetry.

These tests require actual NVIDIA GPU hardware and drivers.
They are skipped unless the environment provides NVIDIA tooling.
"""

from __future__ import annotations

import shutil

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.nvidia,
    pytest.mark.skipif(
        shutil.which("nvidia-smi") is None,
        reason="nvidia-smi not available (no NVIDIA GPU)",
    ),
]


def test_nvidia_smi_available() -> None:
    """Verify nvidia-smi is accessible."""
    assert shutil.which("nvidia-smi") is not None


def test_nvml_telemetry_collection() -> None:
    """Placeholder: collect NVML telemetry readings from a real GPU."""
    pytest.skip("Full NVML integration test not yet implemented")


def test_gpu_energy_counter_monotonic() -> None:
    """Placeholder: verify GPU energy counter is monotonically increasing."""
    pytest.skip("Full NVML energy counter test not yet implemented")
