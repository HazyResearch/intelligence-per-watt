"""React agent implementation using the Agno framework."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder


@AgentRegistry.register("react")
class React(BaseAgent):
    """React agent that uses the Agno Agent framework for tool-augmented reasoning."""

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: Any,
        tools: List[Callable] | None = None,
        instructions: str | None = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the React agent.

        Args:
            model: The Agno Model instance to use.
            tools: Optional list of callable tools/functions for the agent to use.
            instructions: Optional custom instructions for the agent.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional keyword arguments passed to the Agent constructor.
        """
        super().__init__(event_recorder=event_recorder)

        # Lazy import: agno is optional
        try:
            from agno.agent import Agent
        except ImportError:
            raise ImportError(
                "agno package is required for React agent. "
                "Install with: pip install agno"
            )

        self.model = model
        self._original_tools = tools or []
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS

        # Instrument tools if event_recorder is provided
        if event_recorder is not None and self._original_tools:
            self.tools = self._instrument_tools(self._original_tools)
        else:
            self.tools = self._original_tools

        agent_kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            **kwargs,
        }
        if self.tools:
            agent_kwargs["tools"] = self.tools
            agent_kwargs["tool_choice"] = "auto"

        self.agent = Agent(**agent_kwargs)

    def _instrument_tools(self, tools: List[Callable]) -> List[Callable]:
        """Wrap tools to emit start/end events for energy tracking."""
        instrumented = []
        for tool in tools:
            tool_name = getattr(tool, "__name__", str(tool))

            @functools.wraps(tool)
            def wrapper(
                *args: Any,
                __tool: Callable = tool,
                __name: str = tool_name,
                **kwargs: Any,
            ) -> Any:
                self._record_event("tool_call_start", tool=__name)
                try:
                    return __tool(*args, **kwargs)
                finally:
                    self._record_event("tool_call_end", tool=__name)

            instrumented.append(wrapper)
        return instrumented

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        """Run the React agent.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments passed to agent.run().

        Returns:
            AgentRunResult with the agent's response.
        """
        self._record_event("lm_inference_start", model=str(self.model))
        result = None
        try:
            result = self.agent.run(input, **kwargs)
            # Extract token metrics from the result if available
            end_metadata: dict[str, Any] = {"model": str(self.model)}
            input_tokens = 0
            output_tokens = 0
            if result is not None and hasattr(result, "metrics") and result.metrics is not None:
                metrics = result.metrics
                input_tokens = getattr(metrics, "input_tokens", 0)
                output_tokens = getattr(metrics, "output_tokens", 0)
                end_metadata["prompt_tokens"] = input_tokens
                end_metadata["completion_tokens"] = output_tokens
                end_metadata["total_tokens"] = getattr(metrics, "total_tokens", 0)
            self._record_event("lm_inference_end", **end_metadata)

            # Extract content from result
            content = ""
            if result is not None:
                if hasattr(result, "content"):
                    content = str(result.content)
                else:
                    content = str(result)

            return AgentRunResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception:
            self._record_event("lm_inference_end", model=str(self.model))
            raise
