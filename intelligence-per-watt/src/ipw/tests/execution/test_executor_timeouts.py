"""Tests for Executor step + tool timeouts."""

from __future__ import annotations

import asyncio
from typing import List

from ipw.execution.executor import Executor, ExecutorContext, ToolCall, TurnOutput
from ipw.telemetry.eventbus import EventBus
from ipw.tools.base import BaseTool, ToolResult, ToolSpec


class _HangingTool(BaseTool):
    spec = ToolSpec(name="hang", description="hangs forever", parameters={})

    async def run(self, **kwargs):
        await asyncio.sleep(3600)  # never returns in test time
        return ToolResult(content="never", success=True)


class _SlowStepAgent:
    """Agent whose step() hangs longer than the step timeout."""
    tools: List[BaseTool] = []

    async def step(self, context: ExecutorContext) -> TurnOutput:
        await asyncio.sleep(3600)
        return TurnOutput(final_answer="never", tool_calls=[])


class _HangingToolAgent:
    """Agent that calls a hanging tool, then would finalize."""
    def __init__(self) -> None:
        self.tools = [_HangingTool()]
        self._n = 0

    async def step(self, context: ExecutorContext) -> TurnOutput:
        self._n += 1
        if self._n == 1:
            return TurnOutput(final_answer=None, tool_calls=[
                ToolCall(name="hang", input={}, correlation_id="c1"),
            ])
        return TurnOutput(final_answer="done", tool_calls=[])


class TestStepTimeout:
    def test_slow_step_times_out_and_fails(self) -> None:
        """A step() that exceeds step_timeout_s is retried then fails (not hang)."""
        bus = EventBus()
        executor = Executor(
            bus=bus, base_backoff_s=0.0,
            step_timeout_s=0.2, max_attempts_per_turn=2,
        )
        result = asyncio.run(executor.execute(
            _SlowStepAgent(), task_id="t1", max_turns=5,
        ))
        assert result.status == "failed"
        # It must complete quickly, not hang for 3600s
        # (the test itself would time out if the executor didn't bound the step)

    def test_default_step_timeout_is_set(self) -> None:
        """Executor has a default step timeout (not None/unbounded)."""
        executor = Executor(bus=EventBus())
        assert executor._step_timeout_s is not None
        assert executor._step_timeout_s > 0


class TestToolTimeout:
    def test_hanging_tool_times_out_recorded_as_error(self) -> None:
        """A tool that hangs beyond tool_timeout_s is recorded as a failed
        tool call, and the agent run continues (does not hang)."""
        bus = EventBus()
        executor = Executor(
            bus=bus, base_backoff_s=0.0, tool_timeout_s=0.2,
        )
        result = asyncio.run(executor.execute(
            _HangingToolAgent(), task_id="t1", max_turns=5,
        ))
        # The hanging tool's call is recorded with an error; the agent
        # finalizes on the next turn → success
        assert result.status == "success"
        # First turn's tool record should carry a timeout error
        first_turn = result.history[0]
        assert first_turn.tool_records[0].error is not None
        assert "timeout" in first_turn.tool_records[0].error.lower()

    def test_default_tool_timeout_is_set(self) -> None:
        executor = Executor(bus=EventBus())
        assert executor._tool_timeout_s is not None
        assert executor._tool_timeout_s > 0
