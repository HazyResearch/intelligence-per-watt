"""Tests for terminus-tb agent token count population."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_terminus2():
    """Patch Terminus2 so we can instantiate TerminusTB without a real LLM."""
    mock_cls = MagicMock()
    with patch(
        "terminal_bench.agents.terminus_2.Terminus2", mock_cls
    ):
        yield mock_cls


def _make_agent(mock_terminus2_cls):
    """Create a TerminusTB agent with mocked Terminus2."""
    from ipw.agents.terminus_tb import TerminusTB
    return TerminusTB(model="test-model")


class TestTerminusTBTokenCounts:
    def test_token_counts_populated_from_agent_result(self, mock_terminus2):
        """Token counts from Terminus2 result should appear in AgentRunResult."""
        agent_result = SimpleNamespace(
            total_input_tokens=500,
            total_output_tokens=200,
        )
        mock_terminus2.return_value.perform_task.return_value = agent_result

        agent = _make_agent(mock_terminus2)

        session = MagicMock()
        session.capture_pane.return_value = "terminal output"
        task = SimpleNamespace(instruction="do something", max_agent_timeout_sec=60)
        metadata = {"session": session, "task": task, "task_id": "test-1"}
        agent.set_task_metadata(metadata)

        result = agent.run("test input")

        assert result.input_tokens == 500
        assert result.output_tokens == 200

    def test_token_counts_zero_when_agent_fails(self, mock_terminus2):
        """When perform_task raises, token counts should be 0."""
        mock_terminus2.return_value.perform_task.side_effect = RuntimeError("boom")

        agent = _make_agent(mock_terminus2)

        session = MagicMock()
        session.capture_pane.return_value = ""
        task = SimpleNamespace(instruction="do something", max_agent_timeout_sec=60)
        metadata = {"session": session, "task": task, "task_id": "test-2"}
        agent.set_task_metadata(metadata)

        result = agent.run("test input")

        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_token_counts_zero_when_attrs_missing(self, mock_terminus2):
        """When agent_result lacks token attrs, defaults to 0."""
        agent_result = SimpleNamespace()  # no token attributes
        mock_terminus2.return_value.perform_task.return_value = agent_result

        agent = _make_agent(mock_terminus2)

        session = MagicMock()
        session.capture_pane.return_value = "output"
        task = SimpleNamespace(instruction="do something", max_agent_timeout_sec=60)
        metadata = {"session": session, "task": task, "task_id": "test-3"}
        agent.set_task_metadata(metadata)

        result = agent.run("test input")

        assert result.input_tokens == 0
        assert result.output_tokens == 0
