"""Tests for AgenticRunner preflight integration."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestAgenticRunnerPreflight:
    def test_preflight_runs_once_before_first_query(self) -> None:
        from ipw.execution.agentic_runner import AgenticRunner

        with patch("ipw.execution.agentic_runner.run_preflight") as mock_pf:
            mock_pf.return_value.shared_device_baseline_dirty = False
            mock_pf.return_value.warnings = []
            runner = AgenticRunner.__new__(AgenticRunner)
            runner._preflight_done = False
            runner._preflight_baseline_dirty = False
            runner._run_preflight_if_needed(strict=False)
            assert mock_pf.call_count == 1
            runner._run_preflight_if_needed(strict=False)
            assert mock_pf.call_count == 1  # idempotent — only runs once

    def test_preflight_strict_raises(self) -> None:
        from ipw.execution.agentic_runner import AgenticRunner

        with patch("ipw.execution.agentic_runner.run_preflight", side_effect=RuntimeError("dirty")):
            runner = AgenticRunner.__new__(AgenticRunner)
            runner._preflight_done = False
            runner._preflight_baseline_dirty = False
            with pytest.raises(RuntimeError, match="dirty"):
                runner._run_preflight_if_needed(strict=True)

    def test_preflight_dirty_sets_baseline_flag(self) -> None:
        from ipw.execution.agentic_runner import AgenticRunner

        mock_result = type("R", (), {"shared_device_baseline_dirty": True, "warnings": ["foo"]})()
        with patch("ipw.execution.agentic_runner.run_preflight", return_value=mock_result):
            runner = AgenticRunner.__new__(AgenticRunner)
            runner._preflight_done = False
            runner._preflight_baseline_dirty = False
            runner._run_preflight_if_needed(strict=False)
            assert runner._preflight_baseline_dirty is True

    def test_preflight_attributes_initialized_in_constructor(self) -> None:
        """Constructor should initialize _preflight_done and _preflight_baseline_dirty."""
        from ipw.agents.base import BaseAgent
        from ipw.datasets.base import DatasetProvider
        from ipw.execution.agentic_runner import AgenticRunner

        # Build a minimal valid runner — these are abstract, mock minimally
        class _StubAgent(BaseAgent):
            async def run(self, *args, **kwargs):
                return None

        class _StubDataset(DatasetProvider):
            workload_type = "test"
            def iter_records(self): return iter([])
            def size(self): return 0

        runner = AgenticRunner(agent=_StubAgent(), dataset=_StubDataset())
        assert runner._preflight_done is False
        assert runner._preflight_baseline_dirty is False
