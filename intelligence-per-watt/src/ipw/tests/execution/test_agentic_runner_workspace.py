"""Tests for AgenticRunner per-query workspace isolation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ipw.agents.base import ToolUsingAgent
from ipw.execution.executor import ExecutorContext, TurnOutput
from ipw.tools.base import BaseTool, ToolResult, ToolSpec


class _WorkspaceCapturingTool(BaseTool):
    """Records the workspace path it saw during each invocation."""
    spec = ToolSpec(name="ws_capture", description="ws", parameters={})

    def __init__(self, bus=None) -> None:
        super().__init__(bus=bus)
        self.workspaces_seen: list = []

    async def run(self, **kwargs) -> ToolResult:
        self.workspaces_seen.append(self._default_cwd)
        return ToolResult(content="ok", success=True)


class _SingleToolAgent(ToolUsingAgent):
    """Agent that calls the workspace-capturing tool once then returns final."""
    name = "single_tool"

    def __init__(self, tool: _WorkspaceCapturingTool, event_recorder=None) -> None:
        super().__init__(event_recorder=event_recorder)
        self.tools = [tool]
        self._task = None
        self._step_count = 0
        self._tool = tool

    def set_task(self, task: str) -> None:
        self._task = task

    async def step(self, context: ExecutorContext) -> TurnOutput:
        from ipw.execution.executor import ToolCall
        self._step_count += 1
        if self._step_count == 1:
            return TurnOutput(final_answer=None, tool_calls=[
                ToolCall(name="ws_capture", input={}, correlation_id="c1"),
            ])
        return TurnOutput(final_answer="done", tool_calls=[])


class TestRunnerWorkspaceIsolation:
    def test_tool_sees_per_query_workspace(self) -> None:
        """Each query's tool invocation sees the workspace dir set by the runner."""
        from ipw.core.types import DatasetRecord
        from ipw.execution.agentic_runner import AgenticRunner
        from ipw.telemetry.events import EventRecorder

        tool = _WorkspaceCapturingTool()
        agent = _SingleToolAgent(tool)
        runner = AgenticRunner.__new__(AgenticRunner)
        runner._agent = agent
        runner._event_recorder = EventRecorder()
        runner._traces = []
        runner._records = []
        runner._max_attempts = 3
        runner._max_turns = 10

        record = DatasetRecord(
            problem="Test task", answer="", subject="test",
            dataset_metadata={"workload_type": "test"},
        )

        asyncio.run(runner._run_with_executor(
            index=0, record=record, model="x",
            agent=agent, event_recorder=runner._event_recorder,
        ))

        # Tool was called once and saw a workspace path (not None)
        assert len(tool.workspaces_seen) == 1
        assert tool.workspaces_seen[0] is not None
        # The workspace path actually exists during the call
        # (TemporaryDirectory is cleaned up after, so we can't assert existence now —
        # but we CAN assert the path looked like a tmp dir)
        assert "tmp" in tool.workspaces_seen[0].lower() or "/var/folders" in tool.workspaces_seen[0]

    def test_workspace_cleared_after_query(self) -> None:
        """After _run_with_executor returns, tools' _default_cwd is reset to None."""
        from ipw.core.types import DatasetRecord
        from ipw.execution.agentic_runner import AgenticRunner
        from ipw.telemetry.events import EventRecorder

        tool = _WorkspaceCapturingTool()
        agent = _SingleToolAgent(tool)
        runner = AgenticRunner.__new__(AgenticRunner)
        runner._agent = agent
        runner._event_recorder = EventRecorder()
        runner._traces = []
        runner._records = []
        runner._max_attempts = 3
        runner._max_turns = 10

        record = DatasetRecord(
            problem="x", answer="", subject="t",
            dataset_metadata={"workload_type": "test"},
        )

        asyncio.run(runner._run_with_executor(
            index=0, record=record, model="x",
            agent=agent, event_recorder=runner._event_recorder,
        ))

        # After the query, tool's workspace should be cleared
        assert tool._default_cwd is None

    def test_workspace_dir_cleaned_up_after_query(self) -> None:
        """The temp workspace dir is removed after the query (TemporaryDirectory exit)."""
        from ipw.core.types import DatasetRecord
        from ipw.execution.agentic_runner import AgenticRunner
        from ipw.telemetry.events import EventRecorder

        tool = _WorkspaceCapturingTool()
        agent = _SingleToolAgent(tool)
        runner = AgenticRunner.__new__(AgenticRunner)
        runner._agent = agent
        runner._event_recorder = EventRecorder()
        runner._traces = []
        runner._records = []
        runner._max_attempts = 3
        runner._max_turns = 10

        record = DatasetRecord(
            problem="x", answer="", subject="t",
            dataset_metadata={"workload_type": "test"},
        )

        asyncio.run(runner._run_with_executor(
            index=0, record=record, model="x",
            agent=agent, event_recorder=runner._event_recorder,
        ))

        ws_path = tool.workspaces_seen[0]
        assert ws_path is not None
        assert not Path(ws_path).exists(), f"workspace {ws_path} should be cleaned up"
