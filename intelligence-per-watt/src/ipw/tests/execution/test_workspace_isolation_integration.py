"""End-to-end test that agent-driven tool calls cannot pollute the project root."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from ipw.agents.base import ToolUsingAgent
from ipw.execution.executor import ExecutorContext, ToolCall, TurnOutput
from ipw.tools.shell_exec import ShellExecTool

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # repo root (above intelligence-per-watt/)


class _ShellAbuseAgent(ToolUsingAgent):
    """Agent that issues a shell_exec call that would create files,
    then returns final on the next turn."""

    name = "shell_abuse"

    def __init__(self, command: str, event_recorder=None) -> None:
        super().__init__(event_recorder=event_recorder)
        self.tools = [ShellExecTool()]
        self._task = None
        self._command = command
        self._step_count = 0

    def set_task(self, task: str) -> None:
        self._task = task

    async def step(self, context: ExecutorContext) -> TurnOutput:
        self._step_count += 1
        if self._step_count == 1:
            return TurnOutput(final_answer=None, tool_calls=[
                ToolCall(name="shell_exec", input={"command": self._command},
                         correlation_id="c1"),
            ])
        return TurnOutput(final_answer="done", tool_calls=[])


def _project_files_snapshot() -> set:
    """Return a set of (relative) file paths under the project root."""
    return {
        str(p.relative_to(PROJECT_ROOT))
        for p in PROJECT_ROOT.rglob("*")
        if p.is_file()
        and ".git/" not in str(p)
        and ".venv/" not in str(p)
        and "__pycache__/" not in str(p)
        and "node_modules/" not in str(p)
    }


class TestWorkspaceIsolationIntegration:
    @pytest.mark.integration
    def test_shell_exec_cannot_pollute_project_root_via_agent_dispatch(self) -> None:
        """The hardened runner path keeps shell_exec calls inside a temp dir."""
        from ipw.core.types import DatasetRecord
        from ipw.execution.agentic_runner import AgenticRunner
        from ipw.telemetry.events import EventRecorder

        before = _project_files_snapshot()

        # Agent issues a shell_exec that would create files in cwd if not isolated
        agent = _ShellAbuseAgent(command="touch leaked_file_marker.txt")
        runner = AgenticRunner.__new__(AgenticRunner)
        runner._agent = agent
        runner._event_recorder = EventRecorder()
        runner._traces = []
        runner._records = []
        runner._max_attempts = 3
        runner._max_turns = 10

        record = DatasetRecord(
            problem="run touch", answer="", subject="t",
            dataset_metadata={"workload_type": "test"},
        )

        asyncio.run(runner._run_with_executor(
            index=0, record=record, model="x",
            agent=agent, event_recorder=runner._event_recorder,
        ))

        after = _project_files_snapshot()
        new_files = after - before

        # Filter pytest's own cache writes (which can race during test runs)
        new_files = {f for f in new_files if not f.startswith(".pytest_cache")}

        assert not new_files, (
            f"workspace isolation broken — agent created files in project root: {new_files}"
        )

    @pytest.mark.integration
    def test_git_branch_unchanged_after_agent_run(self) -> None:
        """An agent shell_exec cannot leave us on a different git branch."""
        from ipw.core.types import DatasetRecord
        from ipw.execution.agentic_runner import AgenticRunner
        from ipw.telemetry.events import EventRecorder

        # Capture starting branch
        before_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).stdout.strip()

        # Agent issues a command that would try to switch branches
        # (would normally affect the worktree if not isolated)
        agent = _ShellAbuseAgent(command="git checkout -b should_not_exist_branch")
        runner = AgenticRunner.__new__(AgenticRunner)
        runner._agent = agent
        runner._event_recorder = EventRecorder()
        runner._traces = []
        runner._records = []
        runner._max_attempts = 3
        runner._max_turns = 10

        record = DatasetRecord(
            problem="git checkout", answer="", subject="t",
            dataset_metadata={"workload_type": "test"},
        )

        asyncio.run(runner._run_with_executor(
            index=0, record=record, model="x",
            agent=agent, event_recorder=runner._event_recorder,
        ))

        after_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).stdout.strip()

        assert before_branch == after_branch, (
            f"agent shell_exec changed branch: {before_branch} → {after_branch}"
        )

        # Also: should_not_exist_branch should NOT have been created
        list_branches = subprocess.run(
            ["git", "branch", "-l", "should_not_exist_branch"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).stdout.strip()
        assert not list_branches, (
            f"agent created branch in project repo: {list_branches}"
        )
