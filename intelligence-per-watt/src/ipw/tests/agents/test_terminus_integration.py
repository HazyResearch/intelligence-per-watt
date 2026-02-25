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
        with patch.dict("sys.modules", {
            "docker": mock_docker,
            "terminal_bench": MagicMock(),
            "terminal_bench.agents": MagicMock(),
            "terminal_bench.agents.terminus_2": MagicMock(Terminus2=MockTerminus2),
        }):
            from ipw.agents.terminus import Terminus

            agent = Terminus(model="gpt-4o")
            MockTerminus2.assert_called_once_with(model_name="gpt-4o")

    def test_lm_events_recorded(self) -> None:
        mock_docker = MagicMock()
        MockTerminus2 = MagicMock()
        with patch.dict("sys.modules", {
            "docker": mock_docker,
            "terminal_bench": MagicMock(),
            "terminal_bench.agents": MagicMock(),
            "terminal_bench.agents.terminus_2": MagicMock(Terminus2=MockTerminus2),
            "terminal_bench.terminal": MagicMock(),
            "terminal_bench.terminal.tmux_session": MagicMock(),
        }):
            from ipw.agents.terminus import Terminus

            recorder = EventRecorder()
            agent = Terminus(model="gpt-4o", event_recorder=recorder)

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

    def test_cleanup_removes_container(self) -> None:
        with patch.dict("sys.modules", {
            "docker": MagicMock(),
            "terminal_bench": MagicMock(),
            "terminal_bench.agents": MagicMock(),
            "terminal_bench.agents.terminus_2": MagicMock(Terminus2=MagicMock()),
        }):
            from ipw.agents.terminus import Terminus

            agent = Terminus(model="gpt-4o")
            mock_container = MagicMock()
            agent._container = mock_container
            agent._owns_container = True

            agent.cleanup()

            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()
            assert agent._container is None
