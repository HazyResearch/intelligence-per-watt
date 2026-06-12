"""Tests for execution/executor.py — Executor, retry, parallel dispatch."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

from ipw.execution.executor import Executor, ExecutorContext, ToolCall, TurnOutput
from ipw.telemetry.eventbus import Event, EventBus
from ipw.telemetry.events import EventType
from ipw.tools.base import BaseTool, ToolResult, ToolSpec


class _EchoTool(BaseTool):
    spec = ToolSpec(name="echo", description="echo", parameters={})

    async def run(self, **kwargs):
        return ToolResult(content=str(kwargs.get("text", "")), success=True)


class _SlowTool(BaseTool):
    spec = ToolSpec(name="slow", description="sleeps", parameters={})

    async def run(self, **kwargs):
        await asyncio.sleep(kwargs.get("seconds", 0.01))
        return ToolResult(content="done", success=True)


class _ConflictingTool(BaseTool):
    spec = ToolSpec(name="conflict", description="serial only", parameters={},
                    side_effect_conflict=True)

    async def run(self, **kwargs):
        return ToolResult(content="serial", success=True)


@dataclass
class _ScriptedAgent:
    turns: List[TurnOutput]
    tools: List[BaseTool]
    tool_mode: object = None

    async def step(self, context: ExecutorContext) -> TurnOutput:
        idx = len(context.history)
        if idx >= len(self.turns):
            raise RuntimeError(f"agent exhausted at turn {idx}")
        return self.turns[idx]


class TestExecutorSingleTurn:
    def test_returns_final_on_first_turn(self) -> None:
        bus = EventBus()
        agent = _ScriptedAgent(
            turns=[TurnOutput(final_answer="42", tool_calls=[])],
            tools=[],
        )
        executor = Executor(bus=bus)
        result = asyncio.run(executor.execute(agent, task_id="t1", max_turns=5))
        assert result.status == "success"
        assert result.final_answer == "42"
        assert result.n_turns == 1


class TestExecutorMultiTurn:
    def test_tool_call_then_final(self) -> None:
        bus = EventBus()
        agent = _ScriptedAgent(
            turns=[
                TurnOutput(final_answer=None, tool_calls=[
                    ToolCall(name="echo", input={"text": "hi"}, correlation_id="c1"),
                ]),
                TurnOutput(final_answer="done", tool_calls=[]),
            ],
            tools=[_EchoTool(bus=bus)],
        )
        executor = Executor(bus=bus)
        result = asyncio.run(executor.execute(agent, task_id="t1", max_turns=5))
        assert result.status == "success"
        assert result.n_turns == 2
        assert result.final_answer == "done"


class TestExecutorRetry:
    def test_retries_on_retryable_error(self) -> None:
        bus = EventBus()
        attempts = {"n": 0}

        class _FlakyAgent:
            tools: List[BaseTool] = []

            async def step(self, context: ExecutorContext) -> TurnOutput:
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise TimeoutError("transient")
                return TurnOutput(final_answer="ok", tool_calls=[])

        executor = Executor(bus=bus, base_backoff_s=0.0)
        result = asyncio.run(executor.execute(_FlakyAgent(), task_id="t1", max_turns=5))
        assert result.status == "success"
        assert attempts["n"] == 3

    def test_fails_after_max_retries(self) -> None:
        bus = EventBus()

        class _AlwaysTimeout:
            tools: List[BaseTool] = []

            async def step(self, context: ExecutorContext) -> TurnOutput:
                raise TimeoutError("always")

        executor = Executor(bus=bus, base_backoff_s=0.0)
        result = asyncio.run(executor.execute(_AlwaysTimeout(), task_id="t1", max_turns=5))
        assert result.status == "failed"
        assert "always" in str(result.error or "")

    def test_retried_to_exhaustion_emits_retry_attempt_events(self) -> None:
        """Distinguishes 'fatal from attempt 1' from 'retried N times' via
        RETRY_ATTEMPT bus events. RetryExhaustedError surfaces in the result."""

        bus = EventBus()
        retry_events: List[Event] = []
        bus.subscribe(EventType.RETRY_ATTEMPT, retry_events.append)

        class _AlwaysTimeout:
            tools: List[BaseTool] = []

            async def step(self, context: ExecutorContext) -> TurnOutput:
                raise TimeoutError("always")

        executor = Executor(bus=bus, base_backoff_s=0.0)
        result = asyncio.run(executor.execute(_AlwaysTimeout(), task_id="t1", max_turns=5))
        assert result.status == "failed"
        # All 3 attempts emitted RETRY_ATTEMPT events with error_class="retryable"
        assert len(retry_events) == 3
        assert all(e.payload["error_class"] == "retryable" for e in retry_events)
        assert [e.payload["attempt"] for e in retry_events] == [1, 2, 3]

    def test_fatal_error_does_not_retry(self) -> None:
        bus = EventBus()
        attempts = {"n": 0}

        class _FatalAgent:
            tools: List[BaseTool] = []

            async def step(self, context: ExecutorContext) -> TurnOutput:
                attempts["n"] += 1
                raise AssertionError("structural bug")

        executor = Executor(bus=bus, base_backoff_s=0.0)
        result = asyncio.run(executor.execute(_FatalAgent(), task_id="t1", max_turns=5))
        assert result.status == "failed"
        assert attempts["n"] == 1


class TestExecutorParallelDispatch:
    def test_concurrent_tool_calls_run_in_parallel(self) -> None:
        bus = EventBus()
        agent = _ScriptedAgent(
            turns=[
                TurnOutput(final_answer=None, tool_calls=[
                    ToolCall(name="slow", input={"seconds": 0.05}, correlation_id="a"),
                    ToolCall(name="slow", input={"seconds": 0.05}, correlation_id="b"),
                ]),
                TurnOutput(final_answer="ok", tool_calls=[]),
            ],
            tools=[_SlowTool(bus=bus)],
        )
        executor = Executor(bus=bus)

        import time
        start = time.monotonic()
        result = asyncio.run(executor.execute(agent, task_id="t1", max_turns=5))
        elapsed = time.monotonic() - start

        # Two 50ms tools in parallel should take ~50ms, not 100ms
        # Generous bound to tolerate CI/loaded-host variance
        assert elapsed < 0.15, f"parallel dispatch too slow: {elapsed:.3f}s"
        assert result.status == "success"

    def test_side_effect_conflict_tool_runs_sequentially(self) -> None:
        bus = EventBus()
        agent = _ScriptedAgent(
            turns=[
                TurnOutput(final_answer=None, tool_calls=[
                    ToolCall(name="conflict", input={}, correlation_id="a"),
                    ToolCall(name="conflict", input={}, correlation_id="b"),
                ]),
                TurnOutput(final_answer="ok", tool_calls=[]),
            ],
            tools=[_ConflictingTool(bus=bus)],
        )
        executor = Executor(bus=bus)
        result = asyncio.run(executor.execute(agent, task_id="t1", max_turns=5))
        assert result.status == "success"


class TestExecutorMaxTurns:
    def test_max_turns_exhausted_returns_failed(self) -> None:
        bus = EventBus()
        agent = _ScriptedAgent(
            turns=[
                TurnOutput(final_answer=None, tool_calls=[
                    ToolCall(name="echo", input={"text": "x"}, correlation_id="c"),
                ])
            ] * 10,
            tools=[_EchoTool(bus=bus)],
        )
        executor = Executor(bus=bus)
        result = asyncio.run(executor.execute(agent, task_id="t1", max_turns=2))
        assert result.status == "max_turns_exhausted"
        assert result.n_turns == 2


class TestExecutorContextTurnBudget:
    """Verify context.turn_index and context.max_turns are set before each step."""

    def test_context_turn_index_and_max_turns_updated_each_step(self) -> None:
        """A recording agent sees monotonically increasing turn_index and constant max_turns."""
        bus = EventBus()
        recorded: List[dict] = []

        class _RecordingAgent:
            tools: List[BaseTool] = []

            async def step(self, context: ExecutorContext) -> TurnOutput:
                recorded.append({
                    "turn_index": context.turn_index,
                    "max_turns": context.max_turns,
                })
                # Return final on the 3rd turn.
                if context.turn_index >= 2:
                    return TurnOutput(final_answer="done", tool_calls=[])
                return TurnOutput(final_answer=None, tool_calls=[
                    ToolCall(name="echo", input={"text": "x"}, correlation_id="c"),
                ])

        executor = Executor(bus=bus)
        result = asyncio.run(executor.execute(
            _RecordingAgent(), task_id="t", max_turns=5,
        ))
        assert result.status == "success"

        # Should have had 3 steps (indices 0, 1, 2).
        assert len(recorded) == 3
        assert [r["turn_index"] for r in recorded] == [0, 1, 2]
        assert all(r["max_turns"] == 5 for r in recorded)

    def test_context_max_turns_exhausted_still_sets_fields(self) -> None:
        """Even when all turns are used, each step sees the correct index."""
        bus = EventBus()
        recorded: List[dict] = []

        class _NeverFinalAgent:
            tools: List[BaseTool] = []

            async def step(self, context: ExecutorContext) -> TurnOutput:
                recorded.append({
                    "turn_index": context.turn_index,
                    "max_turns": context.max_turns,
                })
                return TurnOutput(final_answer=None, tool_calls=[])

        executor = Executor(bus=bus)
        result = asyncio.run(executor.execute(
            _NeverFinalAgent(), task_id="t", max_turns=3,
        ))
        assert result.status == "max_turns_exhausted"
        assert [r["turn_index"] for r in recorded] == [0, 1, 2]
        assert all(r["max_turns"] == 3 for r in recorded)


class TestExecutorEvents:
    def test_publishes_turn_start_and_end(self) -> None:
        bus = EventBus()
        seen: List[Event] = []
        bus.subscribe(EventType.TURN_START, seen.append)
        bus.subscribe(EventType.TURN_END, seen.append)
        agent = _ScriptedAgent(
            turns=[TurnOutput(final_answer="x", tool_calls=[])],
            tools=[],
        )
        executor = Executor(bus=bus)
        asyncio.run(executor.execute(agent, task_id="t1", max_turns=5))
        types = [e.event_type for e in seen]
        assert EventType.TURN_START in types
        assert EventType.TURN_END in types

    def test_publishes_agent_start_and_end(self) -> None:
        bus = EventBus()
        seen: List[Event] = []
        bus.subscribe(EventType.AGENT_START, seen.append)
        bus.subscribe(EventType.AGENT_END, seen.append)
        agent = _ScriptedAgent(
            turns=[TurnOutput(final_answer="x", tool_calls=[])],
            tools=[],
        )
        executor = Executor(bus=bus)
        asyncio.run(executor.execute(agent, task_id="t1", max_turns=5))
        types = [e.event_type for e in seen]
        assert EventType.AGENT_START in types
        assert EventType.AGENT_END in types
