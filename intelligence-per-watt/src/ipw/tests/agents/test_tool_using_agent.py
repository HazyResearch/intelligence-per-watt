"""Tests for ToolUsingAgent — the base class for Executor-driven native agents."""

from __future__ import annotations

import asyncio

import pytest

from ipw.agents.base import BaseAgent, ToolUsingAgent
from ipw.execution.executor import ExecutorContext, TurnOutput
from ipw.tools.base import ToolCallMode


class _MinimalToolUsingAgent(ToolUsingAgent):
    name = "minimal"
    tools: list = []
    tool_mode = ToolCallMode.STRUCTURED_TEXT

    def __init__(self, answer: str = "ok", event_recorder=None) -> None:
        super().__init__(event_recorder=event_recorder)
        self._answer = answer

    async def step(self, context: ExecutorContext) -> TurnOutput:
        return TurnOutput(final_answer=self._answer, tool_calls=[])


class TestToolUsingAgent:
    def test_subclasses_base_agent(self) -> None:
        agent = _MinimalToolUsingAgent()
        assert isinstance(agent, BaseAgent)

    def test_default_tool_mode_attribute_exists(self) -> None:
        # ToolUsingAgent declares tool_mode at class level (subclasses set it)
        assert hasattr(ToolUsingAgent, "tool_mode")

    def test_default_tools_is_empty_list(self) -> None:
        # ToolUsingAgent declares tools at class level (subclasses populate)
        assert hasattr(ToolUsingAgent, "tools")
        assert ToolUsingAgent.tools == []

    def test_step_not_implemented_on_base(self) -> None:
        """Direct ToolUsingAgent without override must raise NotImplementedError."""
        agent = ToolUsingAgent()
        ctx = ExecutorContext(task_id="t")
        with pytest.raises(NotImplementedError):
            asyncio.run(agent.step(ctx))

    def test_subclass_step_returns_turn_output(self) -> None:
        agent = _MinimalToolUsingAgent(answer="42")
        ctx = ExecutorContext(task_id="t")
        result = asyncio.run(agent.step(ctx))
        assert isinstance(result, TurnOutput)
        assert result.final_answer == "42"
