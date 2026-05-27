"""Tool base classes.

Every tool subclasses BaseTool and declares a ToolSpec at the class level.
BaseTool.__call__ wraps run() with TOOL_CALL_START/END bus events so
subclasses never write per-tool telemetry boilerplate.
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ipw.telemetry.eventbus import EventBus


class ToolCallMode(str, Enum):
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_TEXT = "structured_text"


@dataclass
class ToolSpec:
    """Static description of a tool — used to build prompts and function-call schemas."""

    name: str
    description: str
    parameters: Dict[str, Any]
    side_effect_conflict: bool = False
    requires_docker: bool = False
    requires_network: bool = False
    sandboxed: bool = False


@dataclass
class ToolResult:
    """Result of a single tool invocation."""

    content: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool:
    """Base class for all tools.

    Subclasses declare `spec` at class level and implement `async def run(**kwargs)`.
    The __call__ wrapper handles bus telemetry — subclasses never touch it.
    """

    spec: ToolSpec

    def __init__(self, bus: Optional["EventBus"] = None) -> None:
        self._bus = bus
        self._default_cwd: Optional[str] = None

    def set_workspace(self, workspace_dir: Optional[str]) -> None:
        """Set the per-tool-instance default working directory.

        When the agent invokes the tool without an explicit `cwd` argument,
        the tool falls back to this directory instead of os.getcwd(). The
        runner calls this per query with a temp dir to keep agent-driven
        commands isolated from the project root.

        Pass None to clear (reverts to os.getcwd() fallback).
        """
        self._default_cwd = workspace_dir

    async def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    async def __call__(self, correlation_id: Optional[str] = None, **kwargs: Any) -> ToolResult:
        from ipw.telemetry.eventbus import Event
        from ipw.telemetry.events import EventType

        cid = correlation_id or str(uuid.uuid4())
        start_ns = time.time_ns()
        if self._bus is not None:
            self._bus.publish(Event(
                event_type=EventType.TOOL_CALL_START,
                timestamp_ns=start_ns,
                correlation_id=cid,
                payload={"tool": self.spec.name, "args": dict(kwargs)},
            ))

        try:
            result = await self.run(**kwargs)
        except Exception as exc:
            end_ns = time.time_ns()
            if self._bus is not None:
                self._bus.publish(Event(
                    event_type=EventType.TOOL_CALL_END,
                    timestamp_ns=end_ns,
                    correlation_id=cid,
                    payload={
                        "tool": self.spec.name,
                        "status": "error",
                        "duration_ns": end_ns - start_ns,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                ))
            raise

        end_ns = time.time_ns()
        if self._bus is not None:
            self._bus.publish(Event(
                event_type=EventType.TOOL_CALL_END,
                timestamp_ns=end_ns,
                correlation_id=cid,
                payload={
                    "tool": self.spec.name,
                    "status": "ok" if result.success else "error",
                    "duration_ns": end_ns - start_ns,
                    "error": result.error,
                },
            ))
        return result


__all__ = ["BaseTool", "ToolCallMode", "ToolResult", "ToolSpec"]
