"""Integration tests for the Terminus agent harness."""

from __future__ import annotations

import shutil
import sys
from unittest.mock import MagicMock, patch

import pytest

from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult
from ipw.telemetry.events import EventRecorder, EventType

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="Docker not available",
    ),
]


class _ContextLengthExceededError(Exception):
    pass


class _OutputLengthExceededError(Exception):
    pass


def _mock_terminus_modules(mock_docker, MockTerminus2):
    mock_llm = MockTerminus2.return_value._llm
    mock_llm._ipw_terminus_instrumented = False
    mock_llm.count_tokens.return_value = 1
    mock_llm.call.return_value = "mock response"
    return {
        "docker": mock_docker,
        "terminal_bench": MagicMock(),
        "terminal_bench.agents": MagicMock(),
        "terminal_bench.agents.terminus_2": MagicMock(Terminus2=MockTerminus2),
        "terminal_bench.llms": MagicMock(),
        "terminal_bench.llms.base_llm": MagicMock(
            ContextLengthExceededError=_ContextLengthExceededError,
            OutputLengthExceededError=_OutputLengthExceededError,
        ),
    }


@pytest.fixture(autouse=True)
def _clean_terminus_registration():
    """Ensure the AgentRegistry 'terminus' entry is cleared between tests."""
    yield
    AgentRegistry._entries().pop("terminus", None)
    sys.modules.pop("ipw.agents.terminus", None)


class TestTerminusIntegrationMocked:
    """Tests for Terminus agent with mocked Docker/terminal-bench dependencies."""

    def test_initializes_with_model(self) -> None:
        mock_docker = MagicMock()
        MockTerminus2 = MagicMock()
        with patch.dict("sys.modules", _mock_terminus_modules(mock_docker, MockTerminus2)):
            from ipw.agents.terminus import Terminus

            Terminus(model="gpt-4o")
            MockTerminus2.assert_called_once_with(model_name="openai/gpt-4o")

    def test_lm_events_recorded(self) -> None:
        mock_docker = MagicMock()
        MockTerminus2 = MagicMock()
        modules = _mock_terminus_modules(mock_docker, MockTerminus2)
        modules.update({
            "terminal_bench.terminal": MagicMock(),
            "terminal_bench.terminal.tmux_session": MagicMock(),
        })
        with patch.dict("sys.modules", modules):
            from ipw.agents.terminus import Terminus

            recorder = EventRecorder()
            agent = Terminus(model="gpt-4o", event_recorder=recorder)
            agent.agent._llm.call(prompt="test")

            # Mock container and session
            mock_container = MagicMock()
            mock_container.status = "running"
            agent._container = mock_container

            mock_session = MagicMock()
            mock_session.capture_pane.return_value = "output text"

            with patch.object(agent, "get_session", return_value=mock_session):
                result = agent.run("ls -la")

            assert isinstance(result, AgentRunResult)
            assert result.content == "output text"

            events = recorder.get_events()
            event_types = [e.event_type for e in events]
            assert EventType.LM_INFERENCE_START in event_types
            assert EventType.LM_INFERENCE_END in event_types

    def test_get_session_starts_tmux_session(self) -> None:
        mock_docker = MagicMock()
        MockTerminus2 = MagicMock()

        class MockTmuxSession:
            instances = []

            def __init__(self, *args, **kwargs):
                self.start = MagicMock()
                self.is_session_alive = MagicMock(return_value=True)
                MockTmuxSession.instances.append(self)

        modules = _mock_terminus_modules(mock_docker, MockTerminus2)
        modules.update({
            "terminal_bench.terminal": MagicMock(),
            "terminal_bench.terminal.tmux_session": MagicMock(TmuxSession=MockTmuxSession),
        })
        with patch.dict("sys.modules", modules):
            from ipw.agents.terminus import Terminus

            agent = Terminus(model="gpt-4o")

            mock_container = MagicMock()
            mock_container.status = "running"
            agent._container = mock_container

            session = agent.get_session()

            assert isinstance(session, MockTmuxSession)
            session.start.assert_called_once()
            session.is_session_alive.assert_called_once()

    def test_set_workspace_mounts_container_workspace(self, tmp_path) -> None:
        mock_docker = MagicMock()
        mock_docker.errors.NotFound = type("NotFound", (Exception,), {})
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.containers.get.side_effect = mock_docker.errors.NotFound()
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_container.exec_run.return_value = (0, b"/usr/bin/tmux")
        MockTerminus2 = MagicMock()

        with patch.dict("sys.modules", _mock_terminus_modules(mock_docker, MockTerminus2)):
            from ipw.agents.terminus import Terminus

            agent = Terminus(model="gpt-4o")
            agent.set_workspace(str(tmp_path))
            agent._get_or_create_container()

            _, kwargs = mock_client.containers.run.call_args
            assert kwargs["volumes"] == {
                str(tmp_path.resolve()): {"bind": "/workspace", "mode": "rw"}
            }
            assert kwargs["working_dir"] == "/workspace"

    def test_run_raises_on_tmux_capture_error(self) -> None:
        mock_docker = MagicMock()
        MockTerminus2 = MagicMock()
        modules = _mock_terminus_modules(mock_docker, MockTerminus2)
        modules.update({
            "terminal_bench.terminal": MagicMock(),
            "terminal_bench.terminal.tmux_session": MagicMock(),
        })
        with patch.dict("sys.modules", modules):
            from ipw.agents.terminus import Terminus

            agent = Terminus(model="gpt-4o")
            mock_session = MagicMock()
            mock_session.is_session_alive.return_value = True
            mock_session.capture_pane.return_value = (
                "error connecting to /tmp/tmux-0/default (No such file or directory)"
            )

            with patch.object(agent, "get_session", return_value=mock_session):
                with pytest.raises(RuntimeError, match="failed to capture tmux pane"):
                    agent.run("ls -la")

    def test_cleanup_removes_container(self) -> None:
        mock_docker = MagicMock()
        MockTerminus2 = MagicMock()
        with patch.dict("sys.modules", _mock_terminus_modules(mock_docker, MockTerminus2)):
            from ipw.agents.terminus import Terminus

            agent = Terminus(model="gpt-4o")
            mock_container = MagicMock()
            agent._container = mock_container
            agent._owns_container = True

            agent.cleanup()

            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()
            assert agent._container is None
