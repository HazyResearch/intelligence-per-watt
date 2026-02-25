"""Tests for datasets/terminalbench_native.py — TerminalBenchNativeDataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord


class TestTerminalBenchNativeDataset:
    """Test TerminalBenchNativeDataset without requiring terminal-bench."""

    def test_create_task_env_returns_context_manager(self) -> None:
        """Verify create_task_env returns a TerminalBenchTaskEnv."""
        from ipw.execution.terminalbench_env import TerminalBenchTaskEnv

        # We import and test the method directly by mocking the dataset init
        with patch("ipw.datasets.terminalbench_native._check_terminal_bench", return_value=True), \
             patch("ipw.datasets.terminalbench_native.TBDataset", create=True):

            # Build a mock dataset via patching
            from ipw.datasets.terminalbench_native import TerminalBenchNativeDataset

            with patch.object(TerminalBenchNativeDataset, "__init__", lambda self, **kw: None):
                ds = TerminalBenchNativeDataset.__new__(TerminalBenchNativeDataset)

                record = DatasetRecord(
                    problem="Do a task",
                    answer="",
                    subject="test",
                    dataset_metadata={
                        "task": MagicMock(),
                        "task_paths": MagicMock(),
                        "task_id": "test.1",
                    },
                )

                env = ds.create_task_env(record)
                assert isinstance(env, TerminalBenchTaskEnv)

    def test_score_reads_is_resolved_from_metadata(self) -> None:
        """Verify score() extracts is_resolved from dataset_metadata."""
        with patch("ipw.datasets.terminalbench_native._check_terminal_bench", return_value=True), \
             patch("ipw.datasets.terminalbench_native.TBDataset", create=True):

            from ipw.datasets.terminalbench_native import TerminalBenchNativeDataset

            with patch.object(TerminalBenchNativeDataset, "__init__", lambda self, **kw: None):
                ds = TerminalBenchNativeDataset.__new__(TerminalBenchNativeDataset)

                record = DatasetRecord(
                    problem="Task",
                    answer="",
                    subject="test",
                    dataset_metadata={
                        "is_resolved": True,
                        "test_results": {"test_1": "PASSED"},
                    },
                )
                is_correct, details = ds.score(record, "output")
                assert is_correct is True
                assert details["match_type"] == "test_script"

    def test_score_returns_none_when_no_results(self) -> None:
        """score() returns (None, ...) when is_resolved is not in metadata."""
        with patch("ipw.datasets.terminalbench_native._check_terminal_bench", return_value=True), \
             patch("ipw.datasets.terminalbench_native.TBDataset", create=True):

            from ipw.datasets.terminalbench_native import TerminalBenchNativeDataset

            with patch.object(TerminalBenchNativeDataset, "__init__", lambda self, **kw: None):
                ds = TerminalBenchNativeDataset.__new__(TerminalBenchNativeDataset)

                record = DatasetRecord(
                    problem="Task",
                    answer="",
                    subject="test",
                    dataset_metadata={},
                )
                is_correct, details = ds.score(record, "output")
                assert is_correct is None
                assert details["reason"] == "no_test_results"

    def test_score_false_when_not_resolved(self) -> None:
        """score() returns (False, ...) when is_resolved is False."""
        with patch("ipw.datasets.terminalbench_native._check_terminal_bench", return_value=True), \
             patch("ipw.datasets.terminalbench_native.TBDataset", create=True):

            from ipw.datasets.terminalbench_native import TerminalBenchNativeDataset

            with patch.object(TerminalBenchNativeDataset, "__init__", lambda self, **kw: None):
                ds = TerminalBenchNativeDataset.__new__(TerminalBenchNativeDataset)

                record = DatasetRecord(
                    problem="Task",
                    answer="",
                    subject="test",
                    dataset_metadata={
                        "is_resolved": False,
                        "test_results": {"test_1": "FAILED"},
                    },
                )
                is_correct, details = ds.score(record, "output")
                assert is_correct is False


class TestTerminalBenchNativeEvalHandler:
    """Test the terminalbench-native evaluation handler."""

    def test_evaluate_with_resolved(self) -> None:
        from ipw.evaluation.terminalbench_native import TerminalBenchNativeHandler

        handler = TerminalBenchNativeHandler()
        is_correct, details = handler.evaluate(
            problem="Q",
            reference="",
            model_answer="output",
            metadata={"is_resolved": True, "test_results": {}},
        )
        assert is_correct is True
        assert details["match_type"] == "test_script"

    def test_evaluate_without_results(self) -> None:
        from ipw.evaluation.terminalbench_native import TerminalBenchNativeHandler

        handler = TerminalBenchNativeHandler()
        is_correct, details = handler.evaluate(
            problem="Q",
            reference="",
            model_answer="output",
            metadata={},
        )
        assert is_correct is None
        assert details["reason"] == "no_test_results"

    def test_evaluate_not_resolved(self) -> None:
        from ipw.evaluation.terminalbench_native import TerminalBenchNativeHandler

        handler = TerminalBenchNativeHandler()
        is_correct, details = handler.evaluate(
            problem="Q",
            reference="",
            model_answer="output",
            metadata={"is_resolved": False},
        )
        assert is_correct is False


__all__ = ["TestTerminalBenchNativeDataset", "TestTerminalBenchNativeEvalHandler"]
