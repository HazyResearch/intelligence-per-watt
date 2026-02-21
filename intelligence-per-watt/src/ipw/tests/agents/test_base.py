"""Tests for agents/base.py — BaseAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ipw.agents.base import BaseAgent
from ipw.telemetry.events import EventRecorder


class TestBaseAgent:
    """Test BaseAgent construction and interface."""

    def test_run_raises_not_implemented(self) -> None:
        agent = BaseAgent()
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            agent.run("test input")

    def test_default_attributes(self) -> None:
        agent = BaseAgent()
        assert agent.mcp_tools == {}
        assert agent.event_recorder is None

    def test_accepts_mcp_tools(self) -> None:
        mock_tool = MagicMock()
        agent = BaseAgent(mcp_tools={"calc": mock_tool})
        assert "calc" in agent.mcp_tools
        assert agent.mcp_tools["calc"] is mock_tool

    def test_accepts_event_recorder(self) -> None:
        recorder = EventRecorder()
        agent = BaseAgent(event_recorder=recorder)
        assert agent.event_recorder is recorder

    def test_record_event_with_recorder(self) -> None:
        recorder = EventRecorder()
        agent = BaseAgent(event_recorder=recorder)
        agent._record_event("tool_call_start", tool="calc")
        events = recorder.get_events()
        assert len(events) == 1
        assert events[0].event_type == "tool_call_start"
        assert events[0].metadata["tool"] == "calc"

    def test_record_event_without_recorder_is_noop(self) -> None:
        agent = BaseAgent()
        # Should not raise
        agent._record_event("tool_call_start", tool="calc")

    def test_subclass_can_override_run(self) -> None:
        class CustomAgent(BaseAgent):
            def run(self, input: str, **kwargs):
                return f"processed: {input}"

        agent = CustomAgent()
        result = agent.run("hello")
        assert result == "processed: hello"
