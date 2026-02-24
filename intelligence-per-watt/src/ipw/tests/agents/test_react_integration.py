"""Integration tests for the ReAct agent harness."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import AgentRunResult
from ipw.telemetry.events import EventRecorder, EventType


class TestReactIntegration:
    """Integration tests for the ReAct agent with mocked Agno backend."""

    @patch("ipw.agents.react.Agent")
    def test_initializes_with_model(self, MockAgent: MagicMock) -> None:
        from ipw.agents.react import React

        model = MagicMock()
        agent = React(model=model)
        assert agent.model is model
        MockAgent.assert_called_once()

    @patch("ipw.agents.react.Agent")
    def test_run_returns_agent_run_result(self, MockAgent: MagicMock) -> None:
        from ipw.agents.react import React

        mock_result = MagicMock()
        mock_result.content = "The answer is 42."
        mock_result.metrics = None
        MockAgent.return_value.run.return_value = mock_result

        model = MagicMock()
        agent = React(model=model)
        result = agent.run("What is the answer?")

        assert isinstance(result, AgentRunResult)
        assert result.content == "The answer is 42."

    @patch("ipw.agents.react.Agent")
    def test_token_metrics_captured(self, MockAgent: MagicMock) -> None:
        from ipw.agents.react import React

        mock_result = MagicMock()
        mock_result.content = "Answer"
        mock_metrics = MagicMock()
        mock_metrics.input_tokens = 100
        mock_metrics.output_tokens = 50
        mock_metrics.total_tokens = 150
        mock_result.metrics = mock_metrics
        MockAgent.return_value.run.return_value = mock_result

        model = MagicMock()
        agent = React(model=model)
        result = agent.run("Question")

        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @patch("ipw.agents.react.Agent")
    def test_tool_instrumentation_emits_events(self, MockAgent: MagicMock) -> None:
        from ipw.agents.react import React

        mock_result = MagicMock()
        mock_result.content = "Done"
        mock_result.metrics = None

        # Make the Agent constructor capture tools and call them during run
        captured_tools = []

        def agent_init(**kwargs):
            if "tools" in kwargs:
                captured_tools.extend(kwargs["tools"])

        MockAgent.side_effect = lambda **kw: (agent_init(**kw) or MagicMock(run=MagicMock(return_value=mock_result)))

        def my_tool(x: str) -> str:
            return f"result: {x}"

        recorder = EventRecorder()
        model = MagicMock()
        agent = React(model=model, tools=[my_tool], event_recorder=recorder)

        # Manually invoke the instrumented tool to verify events
        if captured_tools:
            captured_tools[0]("test_input")

        tool_events = [
            e for e in recorder.get_events()
            if e.event_type in (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END)
        ]
        assert len(tool_events) == 2
        assert tool_events[0].event_type == EventType.TOOL_CALL_START
        assert tool_events[0].metadata["tool"] == "my_tool"
        assert tool_events[1].event_type == EventType.TOOL_CALL_END

    @patch("ipw.agents.react.Agent")
    def test_lm_inference_events_recorded(self, MockAgent: MagicMock) -> None:
        from ipw.agents.react import React

        mock_result = MagicMock()
        mock_result.content = "Answer"
        mock_result.metrics = None
        MockAgent.return_value.run.return_value = mock_result

        recorder = EventRecorder()
        model = MagicMock()
        agent = React(model=model, event_recorder=recorder)
        agent.run("Question")

        events = recorder.get_events()
        event_types = [e.event_type for e in events]
        assert EventType.LM_INFERENCE_START in event_types
        assert EventType.LM_INFERENCE_END in event_types

    @patch("ipw.agents.react.Agent")
    def test_exception_still_records_end_event(self, MockAgent: MagicMock) -> None:
        from ipw.agents.react import React

        MockAgent.return_value.run.side_effect = RuntimeError("model error")

        recorder = EventRecorder()
        model = MagicMock()
        agent = React(model=model, event_recorder=recorder)

        with pytest.raises(RuntimeError, match="model error"):
            agent.run("Question")

        events = recorder.get_events()
        event_types = [e.event_type for e in events]
        assert EventType.LM_INFERENCE_START in event_types
        assert EventType.LM_INFERENCE_END in event_types
