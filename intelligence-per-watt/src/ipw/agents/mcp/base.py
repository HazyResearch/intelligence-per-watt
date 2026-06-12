"""Base MCP server with telemetry integration.

All MCP servers (local models, cloud APIs, tools) inherit from BaseMCPServer,
which automatically captures energy, power, cost, and latency metrics around
every execution.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Try to import telemetry components (optional)
try:
    from ipw.execution.telemetry_session import TelemetrySample, TelemetrySession
    from ipw.telemetry import EnergyMonitorCollector
    HAS_TELEMETRY = True
except ImportError:
    # Telemetry not available - create stub types
    EnergyMonitorCollector = None
    TelemetrySession = None
    HAS_TELEMETRY = False

    @dataclass
    class TelemetrySample:
        """Stub telemetry sample when ipw telemetry is not available."""
        timestamp: float = 0.0
        power_watts: float = 0.0
        energy_joules: float = 0.0

# Try to import event recorder (optional)
try:
    from ipw.telemetry.events import EventRecorder
    HAS_EVENT_RECORDER = True
except ImportError:
    EventRecorder = None
    HAS_EVENT_RECORDER = False


@dataclass
class MCPToolResult:
    """Result from MCP tool execution with telemetry."""

    content: str
    """Response text from tool/model"""

    usage: Dict[str, Optional[int]] = field(default_factory=dict)
    """Token counts: prompt_tokens, completion_tokens, total_tokens"""

    cost_usd: Optional[float] = None
    """API cost in USD (for cloud APIs)"""

    telemetry_samples: List[Any] = field(default_factory=list)
    """Energy/power/memory readings during execution"""

    latency_seconds: float = 0.0
    """Wall-clock execution time"""

    ttft_seconds: Optional[float] = None
    """Time to first token (for streaming APIs)"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional tool-specific metadata"""


class BaseMCPServer(ABC):
    """Base class for all MCP servers with automatic telemetry capture.

    All subclasses must implement _execute_impl() which performs the actual
    tool invocation. The base class wraps this with telemetry collection.

    Example:
        class MyTool(BaseMCPServer):
            def _execute_impl(self, prompt: str, **params) -> MCPToolResult:
                response = self.api.call(prompt)
                return MCPToolResult(
                    content=response.text,
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                    cost_usd=0.001
                )
    """

    def __init__(
        self,
        name: str,
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
    ):
        """Initialize MCP server.

        Args:
            name: Tool name for logging/tracking
            telemetry_collector: Energy monitor collector. If None, runs without telemetry.
            event_recorder: EventRecorder for per-action tracking. If None, no events recorded.
        """
        self.name = name
        self.telemetry_collector = telemetry_collector
        self.event_recorder = event_recorder

    def execute(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute tool with automatic telemetry capture.

        Args:
            prompt: Input prompt/query for tool
            **params: Additional tool-specific parameters

        Returns:
            MCPToolResult with content, usage, cost, and telemetry samples
        """
        start_time = time.time()

        # Get model info for event recording (subclasses can override these)
        model_id = getattr(self, "model_path", None) or getattr(self, "model_name", self.name)
        model_alias = getattr(self, "model_name", self.name)
        backend = self._get_backend()

        # Record start event
        if self.event_recorder is not None:
            self.event_recorder.record(
                "submodel_call_start",
                model_id=model_id,
                model_alias=model_alias,
                backend=backend,
                tool_name=self.name,
            )

        # Execute with telemetry if available
        if HAS_TELEMETRY and self.telemetry_collector is not None and TelemetrySession is not None:
            with TelemetrySession(self.telemetry_collector) as session:
                result = self._execute_impl(prompt, **params)
                end_time = time.time()
                result.telemetry_samples = list(session.window(start_time, end_time))
                result.latency_seconds = end_time - start_time
        else:
            # Execute without telemetry
            result = self._execute_impl(prompt, **params)
            end_time = time.time()
            result.latency_seconds = end_time - start_time

        # Record end event
        if self.event_recorder is not None:
            self.event_recorder.record(
                "submodel_call_end",
                model_id=model_id,
                model_alias=model_alias,
                backend=backend,
                tool_name=self.name,
                total_tokens=result.usage.get("total_tokens", 0),
                prompt_tokens=result.usage.get("prompt_tokens", 0),
                completion_tokens=result.usage.get("completion_tokens", 0),
                cost_usd=result.cost_usd,
                latency_seconds=result.latency_seconds,
            )

        return result

    def _get_backend(self) -> str:
        """Get the backend type for this server.

        Returns:
            Backend identifier (e.g., 'vllm', 'ollama', 'openai')
        """
        # Default: extract from name if it contains a prefix like "vllm:" or "ollama:"
        if ":" in self.name:
            return self.name.split(":")[0]
        return "unknown"

    @abstractmethod
    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Implement tool execution logic.

        Subclasses must override this to provide actual tool functionality.

        Args:
            prompt: Input prompt/query
            **params: Tool-specific parameters

        Returns:
            MCPToolResult (telemetry_samples and latency_seconds will be added by base class)
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Check if tool is available and healthy.

        Returns:
            True if tool is operational, False otherwise
        """
        try:
            result = self.execute("test", timeout=5)
            return result.content is not None
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
