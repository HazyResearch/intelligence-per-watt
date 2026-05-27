"""Tests that AgenticRunner threads max_attempts / max_turns down into Executor.

These tests verify the contract:
  - AgenticRunner(max_attempts=N) → Executor(max_attempts_per_turn=N)
  - AgenticRunner(max_turns=M)    → executor.execute(..., max_turns=M)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from ipw.agents.base import ToolUsingAgent
from ipw.core.types import DatasetRecord
from ipw.execution.executor import ExecutorContext, ExecutorResult, TurnOutput
from ipw.telemetry.eventbus import EventBus
from ipw.telemetry.events import EventRecorder

# ---------------------------------------------------------------------------
# Minimal stub agent — step() returns a final answer immediately
# ---------------------------------------------------------------------------

class _StubAgent(ToolUsingAgent):
    """Stub ToolUsingAgent whose step() always returns a final TurnOutput."""

    tools: list = []

    async def step(self, context: ExecutorContext) -> TurnOutput:
        return TurnOutput(final_answer="stub_answer", tool_calls=[])


# ---------------------------------------------------------------------------
# Recording Executor double
# ---------------------------------------------------------------------------

class _RecordingExecutor:
    """Drop-in double for Executor that records construction kwargs and execute() calls."""

    # Captures across all instances within a test
    instances: List["_RecordingExecutor"] = []

    def __init__(self, bus: EventBus, **kwargs: Any) -> None:
        self.init_kwargs: Dict[str, Any] = kwargs
        self.execute_calls: List[Dict[str, Any]] = []
        _RecordingExecutor.instances.append(self)

    async def execute(
        self,
        agent: Any,
        *,
        task_id: str,
        max_turns: int = 10,
        agent_name: Optional[str] = None,
    ) -> ExecutorResult:
        self.execute_calls.append({"task_id": task_id, "max_turns": max_turns, "agent_name": agent_name})
        return ExecutorResult(
            status="success",
            final_answer="stub_answer",
            n_turns=1,
            n_tool_calls=0,
            n_retries=0,
        )


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_dataset_record() -> DatasetRecord:
    return DatasetRecord(
        problem="What is 2+2?",
        answer="4",
        subject="math",
        dataset_metadata={"dataset_name": "test"},
    )


def _make_mock_dataset(record: DatasetRecord):
    """Minimal dataset stub supporting __iter__ and size()."""
    ds = MagicMock()
    ds.__iter__ = MagicMock(return_value=iter([record]))
    ds.size.return_value = 1
    return ds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecutorConfigThreading:
    def setup_method(self) -> None:
        # Clear recording double instance list before each test
        _RecordingExecutor.instances.clear()

    def test_max_attempts_threaded_to_executor_max_attempts(self) -> None:
        """AgenticRunner(max_attempts=5) passes max_attempts_per_turn=5 to Executor."""
        from ipw.execution.agentic_runner import AgenticRunner

        record = _make_dataset_record()
        agent = _StubAgent()
        dataset = _make_mock_dataset(record)
        recorder = EventRecorder()

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=None,
            config={"model": "stub-model"},
            event_recorder=recorder,
            max_attempts=5,
            max_turns=10,
        )

        with patch("ipw.execution.agentic_runner.Executor", _RecordingExecutor):
            asyncio.run(runner._run_with_executor(0, record, "stub-model", agent, recorder))

        assert len(_RecordingExecutor.instances) == 1
        init_kwargs = _RecordingExecutor.instances[0].init_kwargs
        assert init_kwargs.get("max_attempts_per_turn") == 5, (
            f"Expected max_attempts_per_turn=5, got {init_kwargs!r}"
        )

    def test_max_turns_threaded_to_executor_execute(self) -> None:
        """AgenticRunner(max_turns=7) passes max_turns=7 to executor.execute()."""
        from ipw.execution.agentic_runner import AgenticRunner

        record = _make_dataset_record()
        agent = _StubAgent()
        dataset = _make_mock_dataset(record)
        recorder = EventRecorder()

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=None,
            config={"model": "stub-model"},
            event_recorder=recorder,
            max_attempts=3,
            max_turns=7,
        )

        with patch("ipw.execution.agentic_runner.Executor", _RecordingExecutor):
            asyncio.run(runner._run_with_executor(0, record, "stub-model", agent, recorder))

        assert len(_RecordingExecutor.instances) == 1
        execute_calls = _RecordingExecutor.instances[0].execute_calls
        assert len(execute_calls) == 1
        assert execute_calls[0]["max_turns"] == 7, (
            f"Expected max_turns=7, got {execute_calls[0]!r}"
        )

    def test_defaults_are_preserved(self) -> None:
        """AgenticRunner() with no explicit max_attempts/max_turns uses default values."""
        from ipw.execution.agentic_runner import AgenticRunner

        record = _make_dataset_record()
        agent = _StubAgent()
        dataset = _make_mock_dataset(record)
        recorder = EventRecorder()

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=None,
            config={"model": "stub-model"},
            event_recorder=recorder,
        )

        with patch("ipw.execution.agentic_runner.Executor", _RecordingExecutor):
            asyncio.run(runner._run_with_executor(0, record, "stub-model", agent, recorder))

        assert len(_RecordingExecutor.instances) == 1
        inst = _RecordingExecutor.instances[0]
        # Default max_attempts=3 → max_attempts_per_turn=3
        assert inst.init_kwargs.get("max_attempts_per_turn") == 3
        # Default max_turns=10
        assert inst.execute_calls[0]["max_turns"] == 10

    def test_both_params_threaded_simultaneously(self) -> None:
        """Both max_attempts=5 and max_turns=7 are correctly threaded simultaneously."""
        from ipw.execution.agentic_runner import AgenticRunner

        record = _make_dataset_record()
        agent = _StubAgent()
        dataset = _make_mock_dataset(record)
        recorder = EventRecorder()

        runner = AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=None,
            config={"model": "stub-model"},
            event_recorder=recorder,
            max_attempts=5,
            max_turns=7,
        )

        with patch("ipw.execution.agentic_runner.Executor", _RecordingExecutor):
            asyncio.run(runner._run_with_executor(0, record, "stub-model", agent, recorder))

        inst = _RecordingExecutor.instances[0]
        assert inst.init_kwargs.get("max_attempts_per_turn") == 5
        assert inst.execute_calls[0]["max_turns"] == 7
