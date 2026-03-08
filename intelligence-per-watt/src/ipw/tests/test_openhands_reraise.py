"""Tests for OpenHands TB agent failure re-raising."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ipw.core.registry import AgentRegistry


def _get_openhands_cls():
    """Get or import the OpenHands class, handling registry conflicts."""
    if AgentRegistry.has("openhands"):
        return AgentRegistry.get("openhands")
    from ipw.agents.openhands import OpenHands
    return OpenHands


def _make_openhands_agent():
    """Create an OpenHands agent with mocked SDK dependencies."""
    # Mock the openhands SDK imports that happen inside __init__
    mock_sdk = MagicMock()
    mock_agent_cls = MagicMock()
    mock_sdk.Agent = mock_agent_cls
    mock_sdk.LLMSummarizingCondenser = MagicMock()
    mock_sdk.LocalConversation = MagicMock()

    with patch.dict("sys.modules", {
        "openhands.sdk": mock_sdk,
        "openhands.sdk.event": MagicMock(),
        "openhands.sdk.event.llm_convertible": MagicMock(),
        "openhands.sdk.event.llm_convertible.action": MagicMock(),
        "openhands.sdk.event.llm_convertible.observation": MagicMock(),
    }):
        cls = _get_openhands_cls()
        model = MagicMock()
        model.__str__ = lambda self: "test-model"
        model.model = "test-model"
        model.base_url = "http://localhost:8000/v1"
        model.api_key = "test-key"
        model.metrics = MagicMock()
        model.metrics.accumulated_token_usage = None
        model.metrics.accumulated_cost = 0.0
        agent = cls(model=model)
    return agent


class TestTerminalbenchReRaise:
    def test_reraises_on_perform_task_failure(self):
        """When perform_task raises, the exception should propagate,
        but lm_inference_end should still fire."""
        agent = _make_openhands_agent()

        session = MagicMock()
        terminal = MagicMock()
        task = SimpleNamespace(instruction="do something")
        metadata = {
            "session": session,
            "terminal": terminal,
            "task": task,
            "task_id": "test-fail",
        }
        agent.set_task_metadata(metadata)

        # Mock the TB agent import and make perform_task raise
        mock_tb_module = MagicMock()
        mock_tb_agent = MagicMock()
        mock_tb_agent.perform_task.side_effect = RuntimeError("agent crashed")
        mock_tb_module.OpenHandsAgent.return_value = mock_tb_agent

        with patch.dict("sys.modules", {
            "terminal_bench": MagicMock(),
            "terminal_bench.agents": MagicMock(),
            "terminal_bench.agents.installed_agents": MagicMock(),
            "terminal_bench.agents.installed_agents.openhands": MagicMock(),
            "terminal_bench.agents.installed_agents.openhands.openhands_agent": mock_tb_module,
        }):
            with pytest.raises(RuntimeError, match="agent crashed"):
                agent.run("test input")

    def test_survives_capture_failure(self):
        """When perform_task succeeds but capture_pane fails, result is still returned."""
        agent = _make_openhands_agent()

        session = MagicMock()
        session.capture_pane.side_effect = RuntimeError("capture failed")
        session.container = MagicMock()
        session.container.exec_run.return_value = MagicMock(exit_code=1, output=b"")
        terminal = MagicMock()
        task = SimpleNamespace(instruction="do something")
        metadata = {
            "session": session,
            "terminal": terminal,
            "task": task,
            "task_id": "test-capture",
        }
        agent.set_task_metadata(metadata)

        mock_tb_module = MagicMock()
        mock_tb_agent = MagicMock()
        mock_tb_agent.perform_task.return_value = None
        mock_tb_module.OpenHandsAgent.return_value = mock_tb_agent

        with patch.dict("sys.modules", {
            "terminal_bench": MagicMock(),
            "terminal_bench.agents": MagicMock(),
            "terminal_bench.agents.installed_agents": MagicMock(),
            "terminal_bench.agents.installed_agents.openhands": MagicMock(),
            "terminal_bench.agents.installed_agents.openhands.openhands_agent": mock_tb_module,
        }):
            result = agent.run("test input")

        assert result is not None
        assert result.content == ""  # capture_pane failed, so empty
