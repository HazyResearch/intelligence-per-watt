"""Base class for all agents with optional MCP tool and telemetry support."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, MutableMapping, Optional

if TYPE_CHECKING:
    from ipw.agents.mcp.base import BaseMCPServer
    from ipw.telemetry.events import EventRecorder


class BaseAgent:
    """Base class for all agents with optional MCP tool and telemetry support."""

    def __init__(
        self,
        mcp_tools: Optional[dict[str, "BaseMCPServer"]] = None,
        event_recorder: Optional["EventRecorder"] = None,
        artifact_dir: Optional["Path"] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            mcp_tools: Optional dictionary of MCP server instances for tool access.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            artifact_dir: Optional directory for agent file artifacts.
        """
        self.mcp_tools = mcp_tools or {}
        self.event_recorder = event_recorder
        self._artifact_dir = artifact_dir
        self._task_metadata: Optional[MutableMapping[str, Any]] = None

    @property
    def artifact_dir(self) -> Optional[Path]:
        """Directory for storing file artifacts produced during agent runs."""
        return self._artifact_dir

    def _record_event(self, event_type: str, **metadata: Any) -> None:
        """Record an event if a recorder is attached.

        Args:
            event_type: Type of event (e.g., 'tool_call_start', 'lm_inference_end')
            **metadata: Additional metadata to attach to the event
        """
        if self.event_recorder is not None:
            self.event_recorder.record(event_type, **metadata)

    def set_task_metadata(self, metadata: MutableMapping[str, Any]) -> None:
        """Receive per-task metadata before ``run()``.

        Override in agents that need access to task-level information such as
        a tmux session for TerminalBench environments.
        """
        self._task_metadata = metadata

    def _terminal_session(self) -> Any | None:
        if not self._task_metadata:
            return None
        return self._task_metadata.get("session")

    def _execute_terminal_session_command(self, command: str) -> str:
        """Send a command to a TerminalBench tmux session and return pane text."""
        session = self._terminal_session()
        if session is None:
            raise RuntimeError("No TerminalBench session is available")

        command_text = str(command).strip()
        if not command_text:
            return "Error: no terminal command provided"

        try:
            session.send_keys([command_text, "Enter"], block=False)
        except TypeError:
            session.send_keys(command_text, block=False)

        settle_seconds = float(os.getenv("IPW_TERMINAL_TOOL_SETTLE_SECONDS", "1.0"))
        if settle_seconds > 0:
            time.sleep(settle_seconds)

        output = session.capture_pane(capture_entire=True)
        max_chars = int(os.getenv("IPW_TERMINAL_TOOL_MAX_CHARS", "2500"))
        if max_chars and len(output) > max_chars:
            return output[-max_chars:] + "\n... (terminal pane truncated to latest output)"
        return output

    def run(self, input: str, **kwargs: Any) -> Any:
        """Run the agent.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments.

        Returns:
            The output from the agent.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the run method")
