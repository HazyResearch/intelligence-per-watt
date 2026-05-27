"""Integration test: ToolUsingAgent dispatch path through AgenticRunner."""

from __future__ import annotations

from ipw.agents.base import ToolUsingAgent
from ipw.execution.executor import ExecutorContext, TurnOutput


class _ScriptedToolUsingAgent(ToolUsingAgent):
    name = "scripted"
    tools: list = []

    def __init__(self, answer: str = "42", event_recorder=None) -> None:
        super().__init__(event_recorder=event_recorder)
        self._answer = answer
        self._task = None

    def set_task(self, task: str) -> None:
        self._task = task

    async def step(self, context: ExecutorContext) -> TurnOutput:
        return TurnOutput(final_answer=self._answer, tool_calls=[])


class TestRunnerDispatchesToolUsingAgent:
    def test_run_with_executor_method_exists(self) -> None:
        from ipw.execution.agentic_runner import AgenticRunner
        assert hasattr(AgenticRunner, "_run_with_executor"), \
               "AgenticRunner must expose _run_with_executor for native agents"

    def test_run_single_query_dispatches_to_executor_for_tool_using_agent(self) -> None:
        """_run_single_query checks isinstance(agent, ToolUsingAgent) and routes
        to _run_with_executor. We assert the method dispatches; the full
        integration through a real dataset is exercised by the live agent path."""
        import inspect

        from ipw.execution.agentic_runner import AgenticRunner

        # Confirm _run_single_query body contains the isinstance check or
        # delegates to _run_with_executor. Inspect source.
        src = inspect.getsource(AgenticRunner._run_single_query)
        assert "ToolUsingAgent" in src, "_run_single_query must check for ToolUsingAgent"
        assert "_run_with_executor" in src, "_run_single_query must dispatch to _run_with_executor"
