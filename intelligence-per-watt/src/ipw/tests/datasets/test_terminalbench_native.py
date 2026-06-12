"""Tests for datasets/terminalbench_native.py — TerminalBenchNativeDataset."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

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

    def test_init_sorts_tasks_before_limiting(self, tmp_path, monkeypatch) -> None:
        """n_tasks should select a deterministic sorted task prefix."""
        task_dirs = []
        for name in ["z-task", "a-task", "m-task"]:
            task_dir = tmp_path / name
            task_dir.mkdir()
            task_dirs.append(task_dir)

        created_kwargs = {}

        class FakeTBDataset:
            def __init__(self, **kwargs):
                created_kwargs.update(kwargs)
                self.tasks = task_dirs

        class FakeTaskPaths:
            def __init__(self, task_dir):
                self.task_config_path = task_dir / "task.yaml"
                self.run_tests_path = task_dir / "run-tests.sh"
                self.test_dir = task_dir / "tests"

        class FakeTask:
            @staticmethod
            def from_yaml(path):
                return types.SimpleNamespace(
                    instruction=path.parent.name,
                    max_agent_timeout_sec=60,
                    max_test_timeout_sec=60,
                    parser_name="pytest",
                    category="cat",
                    difficulty=types.SimpleNamespace(value="easy"),
                    run_tests_in_same_shell=False,
                    disable_asciinema=True,
                )

        terminal_bench = types.ModuleType("terminal_bench")
        dataset_mod = types.ModuleType("terminal_bench.dataset")
        dataset_mod.Dataset = FakeTBDataset
        handlers_mod = types.ModuleType("terminal_bench.handlers")
        trial_mod = types.ModuleType("terminal_bench.handlers.trial_handler")
        trial_mod.Task = FakeTask
        trial_mod.TaskPaths = FakeTaskPaths

        monkeypatch.setitem(sys.modules, "terminal_bench", terminal_bench)
        monkeypatch.setitem(sys.modules, "terminal_bench.dataset", dataset_mod)
        monkeypatch.setitem(sys.modules, "terminal_bench.handlers", handlers_mod)
        monkeypatch.setitem(
            sys.modules, "terminal_bench.handlers.trial_handler", trial_mod
        )

        from ipw.datasets.terminalbench_native import TerminalBenchNativeDataset

        ds = TerminalBenchNativeDataset(path=str(tmp_path), n_tasks=2)

        assert created_kwargs["n_tasks"] is None
        assert [record.dataset_metadata["task_id"] for record in ds.iter_records()] == [
            "a-task",
            "m-task",
        ]


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
