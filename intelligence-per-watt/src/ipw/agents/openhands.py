"""OpenHands agent implementation with per-tool energy tracking."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.agents.mcp.base import BaseMCPServer
    from ipw.telemetry.events import EventRecorder

logger = logging.getLogger(__name__)


def _register_mcp_tools(mcp_tools: Dict[str, "BaseMCPServer"]) -> list:
    """Register MCP servers as OpenHands tools and return Tool specs.

    Each MCP server is wrapped as a ToolDefinition and registered in the
    OpenHands tool registry so the Agent can resolve it by name.

    Args:
        mcp_tools: Mapping of tool name to BaseMCPServer instance.

    Returns:
        List of Tool specs that can be passed to Agent(tools=...).
    """
    from openhands.sdk import Tool, register_tool
    from openhands.sdk.tool.schema import Action, Observation
    from openhands.sdk.tool.tool import ToolAnnotations, ToolDefinition, ToolExecutor

    class IPWMCPAction(Action):
        query: str

    class IPWMCPObservation(Observation):
        pass

    class MCPToolExecutor(ToolExecutor):
        def __init__(self, mcp_server: "BaseMCPServer") -> None:
            self._server = mcp_server

        def __call__(self, action: IPWMCPAction, conversation: Any = None) -> IPWMCPObservation:
            result = self._server.execute(action.query)
            content = result.content if hasattr(result, "content") else str(result)
            return IPWMCPObservation.from_text(text=content)

    class _MCPToolDef(ToolDefinition[IPWMCPAction, IPWMCPObservation]):
        @classmethod
        def create(cls, *args: Any, **kwargs: Any) -> Sequence["_MCPToolDef"]:
            return []

    tool_specs: list = []

    for name, server in mcp_tools.items():
        oh_name = f"mcp_{name}"

        spec = getattr(server, "_spec", None)
        description = (
            spec.description
            if spec and hasattr(spec, "description")
            else f"Execute the {name} tool"
        )

        executor = MCPToolExecutor(server)

        def _make_factory(_oh_name: str, _desc: str, _executor: MCPToolExecutor):
            def factory(conv_state: Any = None, **kwargs: Any) -> Sequence[ToolDefinition]:
                tool_def = _MCPToolDef(
                    description=_desc,
                    action_type=IPWMCPAction,
                    observation_type=IPWMCPObservation,
                    executor=_executor,
                    annotations=ToolAnnotations(title=_oh_name),
                )
                object.__setattr__(tool_def, "name", _oh_name)
                return [tool_def]

            return factory

        register_tool(oh_name, _make_factory(oh_name, description, executor))
        tool_specs.append(Tool(name=oh_name))
        logger.info(f"Registered OpenHands MCP tool: {oh_name}")

    return tool_specs


@AgentRegistry.register("openhands")
class OpenHands(BaseAgent):
    """OpenHands agent using the OpenHands SDK with energy telemetry."""

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: Any,
        tools: list | None = None,
        mcp_tools: Optional[Dict[str, "BaseMCPServer"]] = None,
        event_recorder: Optional["EventRecorder"] = None,
        max_turns: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenHands agent.

        Args:
            model: The LLM model instance to use.
            tools: List of OpenHands Tool specs.
            mcp_tools: Optional dict mapping tool name to BaseMCPServer instance.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            max_turns: Maximum iterations per run (default 20).
            **kwargs: Additional keyword arguments passed to the Agent constructor.
        """
        super().__init__(event_recorder=event_recorder)

        # Lazy imports: openhands-sdk is optional
        try:
            from openhands.sdk import Agent, Event, LLMConvertibleEvent, LLMSummarizingCondenser, LocalConversation
            from openhands.sdk.event.llm_convertible.action import ActionEvent
            from openhands.sdk.event.llm_convertible.observation import ObservationEvent
        except ImportError:
            raise ImportError(
                "openhands-sdk package is required for OpenHands agent. "
                "Install with: pip install openhands-sdk"
            )

        self.model = model
        self.tools = tools
        self._pending_tool: Optional[str] = None
        self._tool_names_used: List[str] = []
        self._num_turns: int = 0

        # Store references for use in callbacks
        self._ActionEvent = ActionEvent
        self._ObservationEvent = ObservationEvent
        self._LLMConvertibleEvent = LLMConvertibleEvent

        # Context condenser
        condenser = LLMSummarizingCondenser(
            llm=model,
            max_tokens=24000,
            keep_first=2,
        )

        agent_kwargs = {"llm": model, "condenser": condenser}

        if tools:
            agent_kwargs["tools"] = tools
        elif mcp_tools:
            extra_tool_specs = _register_mcp_tools(mcp_tools)
            agent_kwargs["tools"] = extra_tool_specs

        self.agent = Agent(**agent_kwargs)

        self.conversation = LocalConversation(
            agent=self.agent,
            callbacks=[self._instrumented_callback],
            workspace=os.getcwd(),
            max_iteration_per_run=max_turns,
        )
        self.current_result = ""

    def _instrumented_callback(self, event: Any) -> None:
        """Instrumented callback that emits telemetry events for tool calls."""
        if isinstance(event, self._ActionEvent):
            tool_name = event.tool_name
            self._pending_tool = tool_name
            self._tool_names_used.append(tool_name)
            self._num_turns += 1
            self._record_event("tool_call_start", tool=tool_name)
        elif isinstance(event, self._ObservationEvent):
            tool_name = self._pending_tool or event.tool_name
            self._record_event("tool_call_end", tool=tool_name)
            self._pending_tool = None

        if isinstance(event, self._LLMConvertibleEvent):
            self.current_result = event.to_llm_message()

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Extract plain text from an OpenHands Message or fallback to str()."""
        if hasattr(message, "content") and isinstance(message.content, (list, tuple)):
            try:
                from openhands.sdk.llm.message import TextContent
                parts = [
                    item.text for item in message.content if isinstance(item, TextContent)
                ]
                if parts:
                    text = "\n".join(parts)
                else:
                    text = str(message)
            except ImportError:
                text = str(message)
        elif isinstance(message, str):
            text = message
        else:
            text = str(message)

        # Strip <think>...</think> blocks (extended thinking output)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r".*</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        """Run the OpenHands agent with telemetry.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentRunResult with content, tool_names_used, and num_turns.
        """
        from openhands.sdk.conversation.response_utils import get_agent_final_response

        # Reset per-run tracking
        self._tool_names_used = []
        self._num_turns = 0

        self._record_event("lm_inference_start", model=str(self.model))
        try:
            self.conversation.send_message(input)
            self.conversation.run()

            result = get_agent_final_response(self.conversation.state.events)
            if not result:
                # Agent hit the turn cap without calling FinishTool.
                logger.info("No FinishTool call detected, sending synthesis nudge (2 turns)")
                saved_limit = self.conversation.max_iteration_per_run
                self.conversation.max_iteration_per_run = 2
                self.conversation.send_message(
                    "You have run out of turns. Please provide your final answer now. "
                    "Use the finish tool to submit your answer."
                )
                self.conversation.run()
                self.conversation.max_iteration_per_run = saved_limit
                result = get_agent_final_response(self.conversation.state.events)

            if not result:
                result = self._extract_text(self.current_result)
                logger.warning("get_agent_final_response() returned empty after nudge, using callback fallback")

            result = self._extract_text(result)
            self.current_result = ""
            return AgentRunResult(
                content=result,
                tool_calls_attempted=len(self._tool_names_used),
                tool_calls_succeeded=len(self._tool_names_used),
                tool_names_used=list(self._tool_names_used),
                num_turns=self._num_turns,
            )
        finally:
            self._record_event("lm_inference_end", model=str(self.model))
            try:
                self.conversation.close()
            except Exception as e:
                logger.warning(f"Error closing conversation: {e}")
