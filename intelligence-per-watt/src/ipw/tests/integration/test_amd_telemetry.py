"""Tier 3 integration tests for AMD GPU telemetry.

These tests require actual AMD GPU hardware with ROCm drivers.
They are skipped unless rocm-smi is available.
"""

from __future__ import annotations

import shutil

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.amd,
    pytest.mark.skipif(
        shutil.which("rocm-smi") is None,
        reason="rocm-smi not available (no AMD GPU with ROCm)",
    ),
]


def test_rocm_smi_available() -> None:
    """Verify rocm-smi is accessible."""
    assert shutil.which("rocm-smi") is not None


def test_amd_telemetry_collection() -> None:
    """Placeholder: collect AMD GPU telemetry readings."""
    pytest.skip("Full AMD integration test not yet implemented")
