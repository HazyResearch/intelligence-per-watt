"""Tests for execution/agentic_runner.py — AgenticRunner."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import AgentRunResult, DatasetRecord
from ipw.execution.agentic_runner import AgenticRunner
from ipw.execution.trace import QueryTrace, TurnTrace
from ipw.telemetry.events import EventRecorder


class TestAgenticRunner:
    """Test AgenticRunner with mocks."""

    def _make_runner(
        self,
        agent: MagicMock | None = None,
        dataset: MagicMock | None = None,
        event_recorder: EventRecorder | None = None,
    ) -> AgenticRunner:
        if agent is None:
            agent = MagicMock()
            agent.run.return_value = AgentRunResult(
                content="mock answer",
                tool_calls_attempted=1,
                tool_calls_succeeded=1,
                num_turns=1,
                input_tokens=50,
                output_tokens=25,
            )
        if dataset is None:
            dataset = MagicMock()
            records = [
                DatasetRecord(
                    problem="Q1", answer="A1", subject="math",
                    dataset_metadata={"dataset_name": "test"},
                )
            ]
            dataset.__iter__ = MagicMock(return_value=iter(records))
            dataset.size.return_value = 1
        return AgenticRunner(
            agent=agent,
            dataset=dataset,
            telemetry_session=None,
            config={"model": "test-model"},
            event_recorder=event_recorder or EventRecorder(),
        )

    def test_run_returns_traces(self) -> None:
        runner = self._make_runner()
        traces = asyncio.run(runner.run())
        assert len(traces) == 1
        assert isinstance(traces[0], QueryTrace)
        assert traces[0].completed is True

    def test_agent_invocation_returns_content(self) -> None:
        runner = self._make_runner()
        traces = asyncio.run(runner.run())
        assert traces[0].response_text == "mock answer"

    def test_traces_property(self) -> None:
        runner = self._make_runner()
        asyncio.run(runner.run())
        traces = runner.traces
        assert len(traces) == 1
        assert traces[0].query_id == "q0000"

    def test_records_property(self) -> None:
        runner = self._make_runner()
        asyncio.run(runner.run())
        records = runner.records
        assert len(records) == 1
        assert records[0].problem == "Q1"
        assert "test-model" in records[0].model_metrics

    def test_profiling_record_constructed(self) -> None:
        runner = self._make_runner()
        asyncio.run(runner.run())
        record = runner.records[0]
        assert "test-model" in record.model_metrics
        # Token metrics come from trace turns, not AgentRunResult directly.
        # With no events recorded, there are no turns, so tokens are 0.
        metrics = record.model_metrics["test-model"]
        assert metrics.lm_response == "mock answer"

    def test_agent_failure_creates_incomplete_trace(self) -> None:
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("agent error")
        runner = self._make_runner(agent=agent)
        traces = asyncio.run(runner.run())
        assert len(traces) == 1
        assert traces[0].completed is False
        assert "agent error" in traces[0].response_text

    def test_max_queries_limits_processing(self) -> None:
        agent = MagicMock()
        agent.run.return_value = AgentRunResult(content="ok")

        dataset = MagicMock()
        records = [
            DatasetRecord(problem=f"Q{i}", answer=f"A{i}", subject="s")
            for i in range(10)
        ]
        dataset.__iter__ = MagicMock(return_value=iter(records))
        dataset.size.return_value = 10

        runner = self._make_runner(agent=agent, dataset=dataset)
        traces = asyncio.run(runner.run(max_queries=3))
        assert len(traces) == 3

    def test_multi_turn_trace_building(self) -> None:
        """Verify _build_turn_traces correctly parses events into turns."""
        from ipw.telemetry.events import AgentEvent, EventType

        runner = self._make_runner()
        now = 1000.0

        events = [
            AgentEvent(event_type=EventType.LM_INFERENCE_START, timestamp=now),
            AgentEvent(
                event_type=EventType.TOOL_CALL_START,
                timestamp=now + 0.1,
                metadata={"tool": "calc"},
            ),
            AgentEvent(
                event_type=EventType.TOOL_CALL_END,
                timestamp=now + 0.5,
                metadata={"tool": "calc"},
            ),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_END,
                timestamp=now + 1.0,
                metadata={"prompt_tokens": 50, "completion_tokens": 20},
            ),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_START,
                timestamp=now + 1.1,
            ),
            AgentEvent(
                event_type=EventType.LM_INFERENCE_END,
                timestamp=now + 1.5,
                metadata={"prompt_tokens": 30, "completion_tokens": 10},
            ),
        ]

        turns = runner._build_turn_traces(events, readings=[])

        assert len(turns) == 2
        assert turns[0].tools_called == ["calc"]
        assert turns[0].input_tokens == 50
        assert turns[0].output_tokens == 20
        assert turns[1].input_tokens == 30
        assert turns[1].output_tokens == 10

    def test_event_recorder_integration_with_runner(self) -> None:
        """Verify events recorded during agent.run() flow into traces.

        Note: AgenticRunner.__init__ uses ``event_recorder or EventRecorder()``
        which evaluates an empty recorder as falsy (because __len__ == 0).
        We work around this by assigning the recorder directly after init.
        """
        from ipw.telemetry.events import EventType

        recorder = EventRecorder()
        agent = MagicMock()

        def run_with_events(prompt: str, **kwargs) -> AgentRunResult:
            recorder.record(EventType.LM_INFERENCE_START)
            recorder.record(EventType.TOOL_CALL_START, tool="calc")
            recorder.record(EventType.TOOL_CALL_END, tool="calc")
            recorder.record(
                EventType.LM_INFERENCE_END,
                prompt_tokens=50,
                completion_tokens=20,
            )
            recorder.record(EventType.LM_INFERENCE_START)
            recorder.record(
                EventType.LM_INFERENCE_END,
                prompt_tokens=30,
                completion_tokens=10,
            )
            return AgentRunResult(content="final answer")

        agent.run.side_effect = run_with_events

        runner = self._make_runner(agent=agent)
        # Directly inject the recorder to bypass the falsy-empty-container
        # issue in AgenticRunner.__init__ (event_recorder or EventRecorder())
        runner._event_recorder = recorder

        traces = asyncio.run(runner.run())

        assert len(traces) == 1
        trace = traces[0]
        assert trace.num_turns == 2
        assert trace.turns[0].tools_called == ["calc"]
        assert trace.turns[0].input_tokens == 50
        assert trace.turns[0].output_tokens == 20
        assert trace.turns[1].input_tokens == 30
        assert trace.turns[1].output_tokens == 10
