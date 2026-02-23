"""Integration tests for the OpenHands agent harness."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import AgentRunResult
from ipw.telemetry.events import EventRecorder, EventType


@pytest.fixture()
def _mock_openhands():
    """Patch openhands SDK imports used by the OpenHands agent."""
    mock_agent_cls = MagicMock()
    mock_conversation_cls = MagicMock()
    mock_condenser_cls = MagicMock()
    mock_event = MagicMock()
    mock_action_event = type("ActionEvent", (), {})
    mock_observation_event = type("ObservationEvent", (), {})
    mock_llm_convertible = type("LLMConvertibleEvent", (), {})

    modules = {
        "openhands": MagicMock(),
        "openhands.sdk": MagicMock(
            Agent=mock_agent_cls,
            Event=mock_event,
            LLMConvertibleEvent=mock_llm_convertible,
            LLMSummarizingCondenser=mock_condenser_cls,
            LocalConversation=mock_conversation_cls,
        ),
        "openhands.sdk.event": MagicMock(),
        "openhands.sdk.event.llm_convertible": MagicMock(),
        "openhands.sdk.event.llm_convertible.action": MagicMock(
            ActionEvent=mock_action_event,
        ),
        "openhands.sdk.event.llm_convertible.observation": MagicMock(
            ObservationEvent=mock_observation_event,
        ),
        "openhands.sdk.conversation": MagicMock(),
        "openhands.sdk.conversation.response_utils": MagicMock(
            get_agent_final_response=MagicMock(return_value="The final answer"),
        ),
    }

    with patch.dict("sys.modules", modules):
        yield {
            "Agent": mock_agent_cls,
            "LocalConversation": mock_conversation_cls,
            "Condenser": mock_condenser_cls,
            "ActionEvent": mock_action_event,
            "ObservationEvent": mock_observation_event,
            "get_agent_final_response": modules["openhands.sdk.conversation.response_utils"].get_agent_final_response,
        }


class TestOpenHandsIntegration:
    """Integration tests for the OpenHands agent with mocked SDK."""

    def test_initializes_with_model(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        model = MagicMock()
        agent = OpenHands(model=model)
        assert agent.model is model
        _mock_openhands["Agent"].assert_called_once()

    def test_run_returns_agent_run_result(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        _mock_openhands["get_agent_final_response"].return_value = "42"
        conv_mock = MagicMock()
        _mock_openhands["LocalConversation"].return_value = conv_mock

        model = MagicMock()
        agent = OpenHands(model=model)
        result = agent.run("What is 6 * 7?")

        assert isinstance(result, AgentRunResult)
        assert "42" in result.content
        conv_mock.send_message.assert_called_once_with("What is 6 * 7?")
        conv_mock.run.assert_called()

    def test_max_turns_respected(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        model = MagicMock()
        agent = OpenHands(model=model, max_turns=5)
        assert agent._max_turns == 5

    def test_callback_records_tool_events(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        recorder = EventRecorder()
        model = MagicMock()
        agent = OpenHands(model=model, event_recorder=recorder)

        # Simulate an ActionEvent callback
        action_event = _mock_openhands["ActionEvent"]()
        action_event.tool_name = "bash"
        agent._instrumented_callback(action_event)

        # Simulate an ObservationEvent callback
        obs_event = _mock_openhands["ObservationEvent"]()
        obs_event.tool_name = "bash"
        agent._instrumented_callback(obs_event)

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == EventType.TOOL_CALL_START
        assert events[0].metadata["tool"] == "bash"
        assert events[1].event_type == EventType.TOOL_CALL_END

    def test_lm_events_recorded_on_run(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        _mock_openhands["get_agent_final_response"].return_value = "done"
        conv_mock = MagicMock()
        _mock_openhands["LocalConversation"].return_value = conv_mock

        recorder = EventRecorder()
        model = MagicMock()
        agent = OpenHands(model=model, event_recorder=recorder)
        agent.run("test")

        events = recorder.get_events()
        event_types = [e.event_type for e in events]
        assert EventType.LM_INFERENCE_START in event_types
        assert EventType.LM_INFERENCE_END in event_types

    def test_conversation_closed_after_run(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        _mock_openhands["get_agent_final_response"].return_value = "done"
        conv_mock = MagicMock()
        _mock_openhands["LocalConversation"].return_value = conv_mock

        model = MagicMock()
        agent = OpenHands(model=model)
        agent.run("test")

        conv_mock.close.assert_called_once()

    def test_set_workspace(self, _mock_openhands: dict) -> None:
        from ipw.agents.openhands import OpenHands

        model = MagicMock()
        agent = OpenHands(model=model)
        agent.set_workspace("/tmp/workspace")
        assert agent._workspace == "/tmp/workspace"
