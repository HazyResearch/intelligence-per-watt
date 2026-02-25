"""Base class for all agents with optional MCP tool and telemetry support."""

from __future__ import annotations

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
        a tmux session for TerminalBench environments.  Default is a no-op.
        """

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
