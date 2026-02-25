"""Tests for execution/terminalbench_env.py — TerminalBenchTaskEnv."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from ipw.execution.terminalbench_env import TerminalBenchTaskEnv


def _ensure_terminal_bench_mocks():
    """Pre-populate sys.modules with mock terminal_bench submodules.

    When the full test suite runs, ``terminal_bench`` may already be
    partially imported with broken transitive deps (litellm→openai).
    Injecting mocks here ensures the lazy ``from terminal_bench.…``
    imports inside ``TerminalBenchTaskEnv`` resolve cleanly.
    """
    modules = {
        "terminal_bench": MagicMock(),
        "terminal_bench.terminal": MagicMock(),
        "terminal_bench.terminal.terminal": MagicMock(),
        "terminal_bench.terminal.docker_compose_manager": MagicMock(),
        "terminal_bench.parsers": MagicMock(),
        "terminal_bench.parsers.base_parser": MagicMock(),
        "terminal_bench.parsers.parser_factory": MagicMock(),
    }
    return modules


class TestTerminalBenchTaskEnv:
    """Test TerminalBenchTaskEnv context manager lifecycle."""

    def _make_metadata(self) -> dict:
        """Build minimal metadata for TerminalBenchTaskEnv."""
        task = MagicMock()
        task.disable_asciinema = True
        task.max_agent_timeout_sec = 60
        task.max_test_timeout_sec = 30
        task.run_tests_in_same_shell = False
        task.parser_name = "default"

        task_paths = MagicMock()
        task_paths.docker_compose_path = "/fake/docker-compose.yml"
        task_paths.run_tests_path = MagicMock()
        task_paths.run_tests_path.name = "run_tests.sh"
        task_paths.test_dir = MagicMock()
        task_paths.test_dir.exists.return_value = False

        return {
            "task": task,
            "task_paths": task_paths,
            "task_id": "test.task.1",
        }

    def _setup_spin_up_mock(self):
        """Create a mock spin_up_terminal that returns a proper context manager."""
        mock_terminal = MagicMock()
        mock_session = MagicMock()
        mock_terminal.create_session.return_value = mock_session

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_terminal)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_spin_up = MagicMock(return_value=mock_cm)
        return mock_spin_up, mock_terminal, mock_session

    def test_enter_populates_metadata(self) -> None:
        """Verify __enter__ populates terminal, session, container in metadata."""
        metadata = self._make_metadata()
        mock_spin_up, mock_terminal, mock_session = self._setup_spin_up_mock()

        mock_modules = _ensure_terminal_bench_mocks()
        mock_modules["terminal_bench.terminal.terminal"].spin_up_terminal = mock_spin_up

        with patch.dict(sys.modules, mock_modules):
            env = TerminalBenchTaskEnv(metadata)
            env.__enter__()

            assert metadata["terminal"] is mock_terminal
            assert metadata["session"] is mock_session
            assert "container" in metadata

            env.__exit__(None, None, None)

    def test_exit_cleans_up_metadata(self) -> None:
        """Verify __exit__ removes terminal, session, container from metadata."""
        metadata = self._make_metadata()
        mock_spin_up, mock_terminal, _ = self._setup_spin_up_mock()

        mock_modules = _ensure_terminal_bench_mocks()
        mock_modules["terminal_bench.terminal.terminal"].spin_up_terminal = mock_spin_up

        with patch.dict(sys.modules, mock_modules):
            env = TerminalBenchTaskEnv(metadata)
            env.__enter__()

            assert "terminal" in metadata
            assert "session" in metadata
            assert "container" in metadata

            env.__exit__(None, None, None)

            assert "terminal" not in metadata
            assert "session" not in metadata
            assert "container" not in metadata

    def test_missing_task_raises_value_error(self) -> None:
        """Verify __enter__ raises if 'task' is missing from metadata."""
        mock_modules = _ensure_terminal_bench_mocks()
        with patch.dict(sys.modules, mock_modules):
            metadata: dict = {"task_paths": MagicMock()}
            env = TerminalBenchTaskEnv(metadata)
            with pytest.raises(ValueError, match="task"):
                env.__enter__()

    def test_missing_task_paths_raises_value_error(self) -> None:
        """Verify __enter__ raises if 'task_paths' is missing from metadata."""
        mock_modules = _ensure_terminal_bench_mocks()
        with patch.dict(sys.modules, mock_modules):
            metadata: dict = {"task": MagicMock()}
            env = TerminalBenchTaskEnv(metadata)
            with pytest.raises(ValueError, match="task_paths"):
                env.__enter__()

    def test_run_tests_without_terminal_returns_false(self) -> None:
        """run_tests() returns (False, ...) when terminal is not running."""
        metadata = self._make_metadata()

        mock_modules = _ensure_terminal_bench_mocks()
        with patch.dict(sys.modules, mock_modules):
            env = TerminalBenchTaskEnv(metadata)
            # _terminal is None by default
            is_resolved, results = env.run_tests()
            assert is_resolved is False
            assert results["error"] == "terminal_not_running"
            assert metadata["is_resolved"] is False

    def test_run_tests_writes_results_to_metadata(self) -> None:
        """Verify run_tests populates is_resolved and test_results in metadata."""
        metadata = self._make_metadata()
        mock_spin_up, mock_terminal, mock_session = self._setup_spin_up_mock()

        # Set up test session separately
        mock_test_session = MagicMock()
        mock_terminal.create_session.side_effect = [mock_session, mock_test_session]
        mock_test_session.capture_pane.return_value = "ALL TESTS PASSED"

        # Mock UnitTestStatus enum value
        mock_passed = MagicMock()
        mock_passed.value = "PASSED"

        mock_parser = MagicMock()
        mock_parser.parse.return_value = {"test_1": mock_passed}

        mock_modules = _ensure_terminal_bench_mocks()
        mock_modules["terminal_bench.terminal.terminal"].spin_up_terminal = mock_spin_up

        # Set up DockerComposeManager and parser mocks
        mock_dcm = MagicMock()
        mock_dcm.CONTAINER_TEST_DIR = MagicMock()
        mock_dcm.CONTAINER_TEST_DIR.__truediv__ = MagicMock(
            return_value="/tests/run_tests.sh"
        )
        mock_modules["terminal_bench.terminal.docker_compose_manager"].DockerComposeManager = mock_dcm
        mock_modules["terminal_bench.parsers.base_parser"].UnitTestStatus.PASSED = mock_passed
        mock_modules["terminal_bench.parsers.parser_factory"].ParserFactory.get_parser.return_value = mock_parser

        with patch.dict(sys.modules, mock_modules):
            env = TerminalBenchTaskEnv(metadata)
            env.__enter__()

            is_resolved, results = env.run_tests()

            assert is_resolved is True
            assert metadata["is_resolved"] is True
            assert "test_results" in metadata

            env.__exit__(None, None, None)


__all__ = ["TestTerminalBenchTaskEnv"]
