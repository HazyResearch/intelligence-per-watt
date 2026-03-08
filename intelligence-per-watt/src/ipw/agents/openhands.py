"""OpenHands agent implementation with per-tool energy tracking."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Sequence

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.agents.mcp.base import BaseMCPServer
    from ipw.telemetry.events import EventRecorder

logger = logging.getLogger(__name__)

_docker_host_ip: str | None = None


def _get_docker_host_ip() -> str:
    """Return an IP address the host is reachable at from inside Docker containers.

    On Linux, ``host.docker.internal`` does not resolve by default (it requires
    ``--add-host`` at container creation time).  We therefore resolve the Docker
    bridge gateway IP via ``docker network inspect bridge`` and cache it for the
    lifetime of the process.  Falls back to ``172.17.0.1`` (the default bridge
    gateway) if detection fails.
    """
    global _docker_host_ip
    if _docker_host_ip is not None:
        return _docker_host_ip

    try:
        out = subprocess.check_output(
            ["docker", "network", "inspect", "bridge",
             "--format", "{{range .IPAM.Config}}{{.Gateway}}{{end}}"],
            text=True, timeout=5,
        ).strip()
        if out:
            _docker_host_ip = out
            logger.debug("Detected Docker bridge gateway IP: %s", out)
            return _docker_host_ip
    except Exception:
        logger.debug("Could not detect Docker bridge gateway, using 172.17.0.1")

    _docker_host_ip = "172.17.0.1"
    return _docker_host_ip


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


def _parse_openhands_stats(output: str) -> dict[str, Any]:
    """Parse OpenHands conversation stats from captured terminal output.

    Looks for common OpenHands log patterns such as:
    - ``Accumulated cost: $1.23``
    - ``Total tokens: 12345`` or ``input_tokens: 1000, output_tokens: 500``
    - ``Number of turns: 5``
    - ``Metrics: {...}`` JSON blob

    Returns a dict with keys: input_tokens, output_tokens, cost, num_turns.
    Missing fields default to 0.
    """
    result: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "num_turns": 0,
    }

    if not output:
        return result

    # Try to find a Metrics JSON blob (OpenHands >= 0.14)
    metrics_match = re.search(
        r"(?:Metrics|metrics)\s*[:=]\s*(\{[^}]+\})", output
    )
    if metrics_match:
        try:
            import json

            metrics = json.loads(metrics_match.group(1))
            result["input_tokens"] = int(metrics.get("accumulated_input_tokens", 0))
            result["output_tokens"] = int(metrics.get("accumulated_output_tokens", 0))
            result["cost"] = float(metrics.get("accumulated_cost", 0.0))
            result["num_turns"] = int(metrics.get("num_turns", 0))
            return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Accumulated cost
    cost_match = re.search(
        r"[Aa]ccumulated\s+cost\s*[:=]\s*\$?([\d.]+)", output
    )
    if cost_match:
        try:
            result["cost"] = float(cost_match.group(1))
        except ValueError:
            pass

    # Token counts
    for pattern, key in [
        (r"input_tokens\s*[:=]\s*(\d+)", "input_tokens"),
        (r"output_tokens\s*[:=]\s*(\d+)", "output_tokens"),
        (r"[Tt]otal\s+tokens\s*[:=]\s*(\d+)", "_total_tokens"),
    ]:
        m = re.search(pattern, output)
        if m:
            try:
                val = int(m.group(1))
                if key == "_total_tokens" and result["input_tokens"] == 0:
                    # Rough split when only total is available
                    result["input_tokens"] = int(val * 0.7)
                    result["output_tokens"] = val - result["input_tokens"]
                else:
                    result[key] = val
            except ValueError:
                pass

    # Number of turns / iterations
    turns_match = re.search(
        r"(?:[Nn]umber\s+of\s+turns|[Ii]terations?|num_turns)\s*[:=]\s*(\d+)",
        output,
    )
    if turns_match:
        try:
            result["num_turns"] = int(turns_match.group(1))
        except ValueError:
            pass

    return result


def _read_openhands_trajectory(session: Any) -> dict[str, Any]:
    """Read token/cost metrics from OpenHands trajectory files inside Docker.

    OpenHands saves trajectory JSON to ``/agent-logs/`` inside the container
    (via ``SAVE_TRAJECTORY_PATH``).  The trajectory ``metrics`` dict contains
    ``accumulated_input_tokens``, ``accumulated_output_tokens``,
    ``accumulated_cost``, and ``num_turns``.

    Returns a dict with keys: input_tokens, output_tokens, cost, num_turns.
    """
    result: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "num_turns": 0,
    }
    try:
        # OpenHands saves trajectory to SAVE_TRAJECTORY_PATH.  If the path is
        # a directory, it writes <session_id>.json inside; otherwise it writes
        # directly to the path as a file.  Check both cases.
        traj_path = "/agent-logs"

        # First check if /agent-logs is a regular file (OpenHands wrote directly)
        test = session.container.exec_run(["test", "-f", traj_path])
        if test.exit_code == 0:
            logger.info("Trajectory saved as file at %s", traj_path)
            cat = session.container.exec_run(["cat", traj_path])
        else:
            # Directory case: search for .json files inside
            find = session.container.exec_run(
                ["find", traj_path, "-name", "*.json", "-type", "f"]
            )
            if find.exit_code != 0 or not find.output.strip():
                logger.warning("No trajectory files found in %s (exit=%d, output=%r)",
                               traj_path, find.exit_code,
                               find.output[:200] if find.output else b"")
                return result
            files = find.output.decode().strip().split("\n")
            logger.info("Found %d trajectory file(s) in %s: %s",
                        len(files), traj_path, files[-1])
            cat = session.container.exec_run(["cat", files[-1]])
        if cat.exit_code != 0:
            logger.warning("Failed to cat trajectory file %s", files[-1])
            return result
        trajectory = json.loads(cat.output.decode())

        # OpenHands trajectory can be:
        #   - a dict {"metrics": {...}, ...} (older format)
        #   - a list of event dicts (headless mode, v1.4+)
        #     where the last event has "llm_metrics" with token/cost data
        metrics: dict[str, Any] | None = None
        if isinstance(trajectory, dict) and "metrics" in trajectory:
            metrics = trajectory["metrics"]
        elif isinstance(trajectory, list):
            # v1.4+ format: last event has "llm_metrics" dict
            for event in reversed(trajectory):
                if isinstance(event, dict):
                    if "llm_metrics" in event:
                        metrics = event["llm_metrics"]
                        break
                    if "metrics" in event:
                        metrics = event["metrics"]
                        break
                    extras = event.get("extras", {})
                    if isinstance(extras, dict) and "metrics" in extras:
                        metrics = extras["metrics"]
                        break

        if metrics is not None:
            # OpenHands llm_metrics dict structure (v1.4+):
            #   accumulated_cost: float
            #   accumulated_token_usage: {prompt_tokens, completion_tokens, ...}
            #   token_usages: [{prompt_tokens, completion_tokens, ...}, ...]
            #   costs: [...]
            # Older format uses: accumulated_input_tokens, accumulated_output_tokens

            # Token counts from accumulated_token_usage (v1.4+)
            token_usage = metrics.get("accumulated_token_usage", {})
            if not isinstance(token_usage, dict):
                token_usage = {}
            input_tok = (
                token_usage.get("prompt_tokens")
                or metrics.get("accumulated_input_tokens")
                or 0
            )
            output_tok = (
                token_usage.get("completion_tokens")
                or metrics.get("accumulated_output_tokens")
                or 0
            )

            # If accumulated_token_usage was empty, sum from token_usages list
            if input_tok == 0 and output_tok == 0:
                for tu in metrics.get("token_usages", []):
                    if isinstance(tu, dict):
                        input_tok += tu.get("prompt_tokens", 0)
                        output_tok += tu.get("completion_tokens", 0)

            cost = metrics.get("accumulated_cost", 0.0) or 0.0
            # num_turns = number of token_usages entries (each is one LLM call)
            turns = metrics.get("num_turns", 0)
            if not turns:
                turns = len(metrics.get("token_usages", []))

            result["input_tokens"] = int(input_tok)
            result["output_tokens"] = int(output_tok)
            result["cost"] = float(cost)
            result["num_turns"] = int(turns)
            logger.info("Extracted from trajectory: in=%d out=%d cost=%.4f turns=%d",
                        result["input_tokens"], result["output_tokens"],
                        result["cost"], result["num_turns"])
        else:
            desc = f"list[{len(trajectory)}]" if isinstance(trajectory, list) else str(type(trajectory))
            logger.warning("No 'metrics' found in trajectory (%s)", desc)
    except Exception:
        logger.warning("Failed to read OH trajectory", exc_info=True)
    return result


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
            from openhands.sdk import (  # noqa: F401
                Agent,
                Event,
                LLMConvertibleEvent,
                LLMSummarizingCondenser,
                LocalConversation,
            )
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
        self._workspace: str = os.getcwd()
        self._max_turns = max_turns

        # Store references for use in callbacks
        self._ActionEvent = ActionEvent
        self._ObservationEvent = ObservationEvent
        self._LLMConvertibleEvent = LLMConvertibleEvent

        # Store SDK classes for lazy conversation creation
        self._LocalConversation = LocalConversation
        self._LLMSummarizingCondenser = LLMSummarizingCondenser

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
        self.conversation: Optional[Any] = None
        self.current_result = ""
        self._task_metadata: Optional[MutableMapping[str, Any]] = None

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

    def set_task_metadata(self, metadata: MutableMapping[str, Any]) -> None:
        self._task_metadata = metadata

    def set_workspace(self, workspace_path: str) -> None:
        """Set the workspace directory for the next agent run."""
        self._workspace = workspace_path

    def _create_conversation(self) -> Any:
        """Create a fresh LocalConversation for the next run."""
        return self._LocalConversation(
            agent=self.agent,
            callbacks=[self._instrumented_callback],
            workspace=self._workspace,
            max_iteration_per_run=self._max_turns,
        )

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

        Dispatches to TerminalBench mode when task metadata contains a
        ``session`` (set by :class:`~ipw.execution.terminalbench_env.TerminalBenchTaskEnv`),
        otherwise runs locally via ``LocalConversation``.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentRunResult with content, tool_names_used, and num_turns.
        """
        if self._task_metadata and self._task_metadata.get("session"):
            return self._run_terminalbench(input)

        from openhands.sdk.conversation.response_utils import get_agent_final_response

        # Reset per-run tracking
        self._tool_names_used = []
        self._num_turns = 0

        # Create a fresh conversation for each run (previous one is closed
        # in the finally block, so we need a new one each time).
        self.conversation = self._create_conversation()

        # Snapshot LLM token metrics before this run to compute per-query delta
        _pre_input = 0
        _pre_output = 0
        _pre_cost = 0.0
        try:
            _m = self.model.metrics
            if _m.accumulated_token_usage is not None:
                _pre_input = _m.accumulated_token_usage.prompt_tokens or 0
                _pre_output = _m.accumulated_token_usage.completion_tokens or 0
            _pre_cost = _m.accumulated_cost or 0.0
        except Exception:
            pass

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

            # Extract per-query token usage as delta from pre-run snapshot
            input_tokens = 0
            output_tokens = 0
            cost_usd = 0.0
            try:
                metrics = self.model.metrics
                if metrics.accumulated_token_usage is not None:
                    input_tokens = (metrics.accumulated_token_usage.prompt_tokens or 0) - _pre_input
                    output_tokens = (metrics.accumulated_token_usage.completion_tokens or 0) - _pre_output
                cost_usd = (metrics.accumulated_cost or 0.0) - _pre_cost
            except Exception:
                pass

            return AgentRunResult(
                content=result,
                tool_calls_attempted=len(self._tool_names_used),
                tool_calls_succeeded=len(self._tool_names_used),
                tool_names_used=list(self._tool_names_used),
                num_turns=self._num_turns,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        finally:
            self._record_event("lm_inference_end", model=str(self.model))
            try:
                self.conversation.close()
            except Exception as e:
                logger.warning(f"Error closing conversation: {e}")

    # ------------------------------------------------------------------
    # TerminalBench mode
    # ------------------------------------------------------------------

    def _run_terminalbench(self, input: str) -> AgentRunResult:
        """Run OpenHands inside a TerminalBench Docker container.

        Uses TB's ``OpenHandsAgent`` (an ``AbstractInstalledAgent``) which
        installs OpenHands inside the container and runs it headless.  The
        host LLM endpoint is forwarded via ``host.docker.internal``.
        """
        from terminal_bench.agents.installed_agents.openhands.openhands_agent import (
            OpenHandsAgent as TBOpenHandsAgent,
        )

        assert self._task_metadata is not None
        session = self._task_metadata["session"]
        _terminal = self._task_metadata["terminal"]  # noqa: F841
        task = self._task_metadata["task"]
        task_id = self._task_metadata.get("task_id", "unknown")

        # TB's OpenHandsAgent reads LLM_MODEL, LLM_BASE_URL, LLM_API_KEY from
        # os.environ (via its _env property) and writes them into a setup-env.sh
        # sourced inside the Docker container.  We need to:
        #   1. Extract the plain model-id string (self.model is an LLM object)
        #   2. Translate localhost URLs → Docker bridge gateway IP
        #   3. Set os.environ so TB's _env picks up the translated values

        # --- model name ---
        # self.model is openhands.sdk.LLM; .model gives the litellm model string
        model_str = getattr(self.model, "model", None) or str(self.model)

        # --- base URL ---
        # Prefer the URL from the LLM object; fall back to env vars
        llm_base_url = getattr(self.model, "base_url", None) or ""
        if not llm_base_url:
            llm_base_url = os.environ.get("LLM_BASE_URL", "")
        if not llm_base_url:
            llm_base_url = os.environ.get("IPW_CLIENT_BASE_URL", "")
        if llm_base_url:
            # Translate localhost/127.0.0.1 to the Docker bridge gateway IP so
            # containers can reach the host-side vLLM server.
            docker_ip = _get_docker_host_ip()
            llm_base_url = llm_base_url.replace("localhost", docker_ip)
            llm_base_url = llm_base_url.replace("127.0.0.1", docker_ip)
            os.environ["LLM_BASE_URL"] = llm_base_url

        # Ensure the model string has a litellm provider prefix
        _KNOWN_PREFIXES = (
            "openai/", "anthropic/", "gemini/", "google/",
            "azure/", "bedrock/", "vertex_ai/", "ollama/",
        )
        if llm_base_url and not model_str.startswith(_KNOWN_PREFIXES):
            model_str = f"openai/{model_str}"

        # --- API key ---
        llm_api_key = getattr(self.model, "api_key", None)
        # api_key may be a SecretStr; extract the raw value
        if llm_api_key is not None and hasattr(llm_api_key, "get_secret_value"):
            llm_api_key = llm_api_key.get_secret_value()
        if not llm_api_key:
            llm_api_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        if llm_api_key:
            os.environ["LLM_API_KEY"] = llm_api_key

        logger.debug(
            "TB env setup: LLM_BASE_URL=%s, LLM_MODEL=%s, LLM_API_KEY=%s",
            os.environ.get("LLM_BASE_URL", "<unset>"),
            model_str,
            "***" if llm_api_key else "<unset>",
        )

        terminal_output = ""
        extracted_input_tokens = 0
        extracted_output_tokens = 0
        extracted_cost: float = 0.0
        extracted_num_turns = 0

        _agent_exc: Exception | None = None
        self._record_event("lm_inference_start", model=model_str)
        try:
            tb_agent = TBOpenHandsAgent(model_name=model_str)

            logger.info(
                "Running OpenHands (TB installed agent) on task %s", task_id
            )
            tb_agent.perform_task(
                instruction=task.instruction,
                session=session,
            )

            try:
                terminal_output = session.capture_pane(capture_entire=True)
            except Exception:
                logger.warning("Failed to capture terminal output")

            # Primary: read trajectory file from container
            stats = _read_openhands_trajectory(session)
            # Fallback: parse terminal output if trajectory returned nothing
            if stats["input_tokens"] == 0 and stats["output_tokens"] == 0:
                stats = _parse_openhands_stats(terminal_output)
            extracted_input_tokens = stats.get("input_tokens", 0)
            extracted_output_tokens = stats.get("output_tokens", 0)
            extracted_cost = stats.get("cost", 0.0)
            extracted_num_turns = stats.get("num_turns", 0)

        except Exception as exc:
            logger.exception("OpenHands TB agent failed on task %s", task_id)
            _agent_exc = exc
        finally:
            self._record_event("lm_inference_end", model=model_str)

        if _agent_exc is not None:
            raise _agent_exc

        return AgentRunResult(
            content=terminal_output,
            input_tokens=extracted_input_tokens,
            output_tokens=extracted_output_tokens,
            num_turns=extracted_num_turns,
            cost_usd=extracted_cost,
            metadata={"task_id": task_id},
        )
