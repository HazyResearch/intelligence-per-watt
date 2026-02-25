"""TerminalBench native dataset — loads task definitions directly from the terminal-bench package.

Unlike the HuggingFace-based ``terminalbench`` dataset, this provider reads
per-task YAML configs from the local ``terminal_bench`` installation so that
IPW can manage Docker containers and collect host-side energy telemetry.

The companion :class:`~ipw.execution.terminalbench_env.TerminalBenchTaskEnv`
handles per-task Docker lifecycle and test execution, allowing any agent to
work with this dataset.
"""

from __future__ import annotations

import logging
import shutil
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, List, MutableMapping, Optional, Tuple

from ..core.registry import DatasetRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

LOGGER = logging.getLogger(__name__)


def _check_terminal_bench():
    """Lazily verify that terminal_bench is importable."""
    try:
        import terminal_bench  # noqa: F401

        return True
    except ImportError:
        return False


@DatasetRegistry.register("terminalbench-native")
class TerminalBenchNativeDataset(DatasetProvider):
    """Load TerminalBench tasks from the local terminal-bench package.

    Each record carries full task metadata (task_path, Dockerfile, test scripts,
    timeouts, parser config) in ``dataset_metadata``.  The runner wraps each
    task in a :class:`~ipw.execution.terminalbench_env.TerminalBenchTaskEnv`
    context manager (via :meth:`create_task_env`) so that any agent can work
    with TerminalBench tasks.

    Requires either a local ``path`` to a dataset directory, or ``name`` +
    ``version`` to download from the TerminalBench registry.
    """

    dataset_id = "terminalbench-native"
    dataset_name = "TerminalBench (native)"
    evaluation_method = "terminalbench-native"

    # No LLM judge needed — scoring uses test scripts inside Docker.
    eval_client: str | None = None
    eval_base_url: str | None = None
    eval_model: str | None = None

    def __init__(
        self,
        *,
        name: Optional[str] = "terminal-bench-core",
        version: Optional[str] = "0.1.1",
        path: Optional[str] = None,
        task_ids: Optional[list[str]] = None,
        n_tasks: Optional[int] = None,
    ) -> None:
        if not _check_terminal_bench():
            raise ImportError(
                "The 'terminal-bench' package is required for terminalbench-native. "
                "Install with: pip install terminal-bench"
            )

        from terminal_bench.dataset import Dataset as TBDataset

        if path is not None:
            self._tb_dataset = TBDataset(path=Path(path), task_ids=task_ids, n_tasks=n_tasks)
        else:
            self._tb_dataset = TBDataset(
                name=name, version=version, task_ids=task_ids, n_tasks=n_tasks
            )
        self._records: tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        if not _check_terminal_bench():
            issues.append(
                "Missing 'terminal-bench' package. Install with: pip install terminal-bench"
            )
        if not shutil.which("docker"):
            issues.append(
                "Docker CLI not found on PATH. Docker is required to run TerminalBench tasks."
            )
        return issues

    def create_task_env(
        self, record: DatasetRecord
    ) -> Optional[AbstractContextManager]:
        """Return a :class:`TerminalBenchTaskEnv` for the given record."""
        from ..execution.terminalbench_env import TerminalBenchTaskEnv

        return TerminalBenchTaskEnv(record.dataset_metadata)

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client=None,
    ) -> Tuple[Optional[bool], dict[str, object]]:
        """Extract pre-computed test results from dataset_metadata.

        :class:`~ipw.execution.terminalbench_env.TerminalBenchTaskEnv` writes
        ``is_resolved`` and ``test_results`` into the mutable
        ``dataset_metadata`` dict after running tests.
        """
        meta = record.dataset_metadata
        is_resolved = meta.get("is_resolved")
        details: dict[str, object] = {
            "match_type": "test_script",
            "test_results": meta.get("test_results"),
        }
        if is_resolved is None:
            return None, {**details, "reason": "no_test_results"}
        return bool(is_resolved), details

    # ------------------------------------------------------------------
    # Record building
    # ------------------------------------------------------------------

    def _build_records(self) -> List[DatasetRecord]:
        from terminal_bench.handlers.trial_handler import Task, TaskPaths

        records: list[DatasetRecord] = []

        # TB Dataset.tasks is a list[Path] of task directories
        for task_dir in self._tb_dataset.tasks:
            try:
                task_paths = TaskPaths(task_dir)
                task = Task.from_yaml(task_paths.task_config_path)

                metadata: MutableMapping[str, Any] = {
                    "dataset_name": self.dataset_name,
                    "task_id": task_dir.name,
                    "task_path": str(task_dir),
                    "task_paths": task_paths,
                    "task": task,
                    "timeout": task.max_agent_timeout_sec,
                    "parser_name": task.parser_name,
                    "category": task.category,
                    "difficulty": task.difficulty.value,
                    "run_tests_in_same_shell": task.run_tests_in_same_shell,
                    "disable_asciinema": task.disable_asciinema,
                }

                records.append(
                    DatasetRecord(
                        problem=task.instruction,
                        answer="",  # Scoring is test-based, not text comparison
                        subject=task.category,
                        dataset_metadata=metadata,
                    )
                )
            except Exception:
                LOGGER.exception("Failed to load task from %s", task_dir)

        LOGGER.info("Loaded %d TerminalBench tasks", len(records))
        return records


__all__ = ["TerminalBenchNativeDataset"]
