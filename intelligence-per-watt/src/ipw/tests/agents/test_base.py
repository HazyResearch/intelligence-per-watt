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

    def test_set_task_metadata_default_is_noop(self) -> None:
        """Default set_task_metadata does nothing and raises no errors."""
        agent = BaseAgent()
        metadata: dict = {"session": "mock", "terminal": "mock"}
        agent.set_task_metadata(metadata)
        # No exception, metadata unchanged
        assert metadata == {"session": "mock", "terminal": "mock"}

    def test_set_task_metadata_can_be_overridden(self) -> None:
        """Subclass can override set_task_metadata to store metadata."""

        class MetadataAgent(BaseAgent):
            def __init__(self) -> None:
                super().__init__()
                self.stored_metadata = None

            def set_task_metadata(self, metadata):
                self.stored_metadata = metadata

            def run(self, input: str, **kwargs):
                return f"session={self.stored_metadata.get('session')}"

        agent = MetadataAgent()
        agent.set_task_metadata({"session": "tmux-1"})
        assert agent.stored_metadata is not None
        assert agent.stored_metadata["session"] == "tmux-1"

    def test_artifact_dir_property(self) -> None:
        """artifact_dir property returns the value set at init."""
        from pathlib import Path

        agent = BaseAgent(artifact_dir=Path("/tmp/artifacts"))
        assert agent.artifact_dir == Path("/tmp/artifacts")

    def test_artifact_dir_default_none(self) -> None:
        agent = BaseAgent()
        assert agent.artifact_dir is None
