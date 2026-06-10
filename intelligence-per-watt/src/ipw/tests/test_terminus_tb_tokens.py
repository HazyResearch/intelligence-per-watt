"""Tests for terminus-tb agent token count population."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ipw.telemetry.events import EventRecorder, EventType


@pytest.fixture()
def mock_terminus2():
    """Patch Terminus2 so we can instantiate TerminusTB without terminal-bench installed."""
    mock_cls = MagicMock()
    mock_module = MagicMock()
    mock_module.Terminus2 = mock_cls
    with patch.dict("sys.modules", {
        "terminal_bench": MagicMock(),
        "terminal_bench.agents": MagicMock(),
        "terminal_bench.agents.terminus_2": mock_module,
    }):
        yield mock_cls


def _make_agent(mock_terminus2_cls):
    """Create a TerminusTB agent with mocked Terminus2."""
    from ipw.agents.terminus_tb import TerminusTB
    return TerminusTB(model="test-model")


class TestTerminusTBTokenCounts:
    def test_agent_result_token_counts_do_not_populate_trace_metrics(self, mock_terminus2):
        """Terminus2 local count_tokens values are not provider usage metrics."""
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

        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.metadata["token_source"] == "missing"

    def test_token_counts_none_when_agent_fails(self, mock_terminus2):
        """When perform_task raises, token counts should be unknown."""
        mock_terminus2.return_value.perform_task.side_effect = RuntimeError("boom")

        agent = _make_agent(mock_terminus2)

        session = MagicMock()
        session.capture_pane.return_value = ""
        task = SimpleNamespace(instruction="do something", max_agent_timeout_sec=60)
        metadata = {"session": session, "task": task, "task_id": "test-2"}
        agent.set_task_metadata(metadata)

        result = agent.run("test input")

        assert result.input_tokens is None
        assert result.output_tokens is None

    def test_token_counts_none_when_attrs_missing(self, mock_terminus2):
        """When agent_result lacks provider usage, token counts stay unknown."""
        agent_result = SimpleNamespace()  # no token attributes
        mock_terminus2.return_value.perform_task.return_value = agent_result

        agent = _make_agent(mock_terminus2)

        session = MagicMock()
        session.capture_pane.return_value = "output"
        task = SimpleNamespace(instruction="do something", max_agent_timeout_sec=60)
        metadata = {"session": session, "task": task, "task_id": "test-3"}
        agent.set_task_metadata(metadata)

        result = agent.run("test input")

        assert result.input_tokens is None
        assert result.output_tokens is None

    def test_session_send_keys_records_tool_events(self, mock_terminus2):
        """TerminalBench tmux actions should appear as tool-call events."""
        from ipw.agents.terminus_tb import TerminusTB

        recorder = EventRecorder()
        agent_result = SimpleNamespace(total_input_tokens=1, total_output_tokens=1)

        def _perform_task(_instruction, *, session, time_limit_seconds):
            session.send_keys("ls -la")
            return agent_result

        mock_terminus2.return_value.perform_task.side_effect = _perform_task
        agent = TerminusTB(model="test-model", event_recorder=recorder)

        session = MagicMock()
        session.capture_pane.return_value = "terminal output"
        task = SimpleNamespace(instruction="do something", max_agent_timeout_sec=60)
        metadata = {"session": session, "task": task, "task_id": "test-tools"}
        agent.set_task_metadata(metadata)

        agent.run("test input")

        event_types = [event.event_type for event in recorder.get_events()]
        assert EventType.TOOL_CALL_START.value in event_types
        assert EventType.TOOL_CALL_END.value in event_types

    def test_llm_call_kwargs_are_applied_to_terminus_llm(self, mock_terminus2):
        """Terminus-TB should pass IPW LiteLLM defaults through to LLM calls."""
        from ipw.agents.terminus_tb import TerminusTB

        fake_llm = SimpleNamespace()
        original_call = MagicMock(return_value="ok")
        fake_llm.call = original_call
        mock_terminus2.return_value._llm = fake_llm

        agent = TerminusTB(
            model="test-model",
            llm_call_kwargs={"timeout": 1000000},
        )

        agent._terminus._llm.call("prompt", temperature=0.2)

        original_call.assert_called_once_with(
            "prompt",
            timeout=1000000,
            temperature=0.2,
        )

    def test_context_limit_overrides_terminus_model_context(self, mock_terminus2):
        """Terminus-TB should use the server context limit when provided."""
        from ipw.agents.terminus_tb import TerminusTB

        fake_terminus = mock_terminus2.return_value
        fake_terminus._llm = SimpleNamespace(call=MagicMock(return_value="ok"))

        agent = TerminusTB(
            model="test-model",
            context_limit=65536,
            context_buffer_tokens=512,
        )

        assert agent._terminus._get_model_context_limit() == 65024

    def test_terminal_output_limit_can_be_lowered(self, mock_terminus2):
        """Terminus-TB should cap terminal output prompt bytes when configured."""
        from ipw.agents.terminus_tb import TerminusTB

        fake_terminus = mock_terminus2.return_value
        fake_terminus._llm = SimpleNamespace(call=MagicMock(return_value="ok"))

        def _limit_output_length(output: str, max_bytes: int = 10000) -> str:
            return output[:max_bytes]

        fake_terminus._limit_output_length = _limit_output_length

        agent = TerminusTB(model="test-model", terminal_output_max_bytes=32)

        assert agent._terminus._limit_output_length("x" * 100, max_bytes=10000) == "x" * 32
