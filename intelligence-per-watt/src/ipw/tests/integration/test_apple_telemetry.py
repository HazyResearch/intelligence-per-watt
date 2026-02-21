"""Tier 3 integration tests for Apple Silicon telemetry.

These tests require macOS with Apple Silicon and powermetrics access.
They are skipped on non-macOS platforms.
"""

from __future__ import annotations

import platform
import shutil

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.apple,
    pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="Apple Silicon telemetry only available on macOS",
    ),
]


def test_powermetrics_available() -> None:
    """Verify powermetrics is accessible on macOS."""
    assert shutil.which("powermetrics") is not None


def test_apple_ane_telemetry() -> None:
    """Placeholder: collect Apple ANE (Neural Engine) telemetry."""
    pytest.skip("Full Apple ANE integration test not yet implemented")
