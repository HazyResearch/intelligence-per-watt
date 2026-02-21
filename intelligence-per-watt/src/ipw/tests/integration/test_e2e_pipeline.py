"""Tier 3 end-to-end integration tests for the full profiling pipeline.

These tests run the full pipeline including:
- Energy monitor subprocess
- gRPC telemetry stream
- Inference client (Ollama or vLLM)
- Dataset iteration
- Profiling record construction
- Analysis and export

They require a running inference server and energy monitor binary.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("IPW_E2E_TESTS"),
        reason="Set IPW_E2E_TESTS=1 to run end-to-end tests",
    ),
]


def test_profile_pipeline_ollama() -> None:
    """Placeholder: full profile run with Ollama backend."""
    pytest.skip("Full e2e pipeline test not yet implemented")


def test_agentic_run_pipeline() -> None:
    """Placeholder: full agentic run with agent + dataset + telemetry."""
    pytest.skip("Full e2e agentic pipeline test not yet implemented")


def test_export_roundtrip() -> None:
    """Placeholder: export JSONL + HF dataset, reload and verify."""
    pytest.skip("Full e2e export roundtrip test not yet implemented")


def test_analysis_from_run() -> None:
    """Placeholder: run analysis (accuracy, regression) on profiling output."""
    pytest.skip("Full e2e analysis test not yet implemented")
