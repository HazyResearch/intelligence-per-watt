"""OpenHands agent implementation with per-tool energy tracking."""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Sequence

from ipw.agents.base import BaseAgent
from ipw.agents.openai_usage_proxy import OpenAIUsageProxy
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.agents.mcp.base import BaseMCPServer
    from ipw.telemetry.events import EventRecorder

logger = logging.getLogger(__name__)

_docker_host_ip: str | None = None


def _usage_counts(model: Any) -> tuple[int | None, int | None, float | None]:
    """Return accumulated prompt/output tokens and cost from an OpenHands LLM."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    try:
        metrics = model.metrics
        if metrics.accumulated_token_usage is not None:
            prompt_tokens = metrics.accumulated_token_usage.prompt_tokens
            completion_tokens = metrics.accumulated_token_usage.completion_tokens
            if isinstance(prompt_tokens, (int, float)):
                input_tokens = int(prompt_tokens)
            if isinstance(completion_tokens, (int, float)):
                output_tokens = int(completion_tokens)
        accumulated_cost = metrics.accumulated_cost
        if isinstance(accumulated_cost, (int, float)):
            cost_usd = float(accumulated_cost)
    except Exception:
        pass
    return input_tokens, output_tokens, cost_usd


_CALL_METHODS = {
    "call",
    "chat",
    "completion",
    "acompletion",
    "complete",
    "acomplete",
    "invoke",
    "ainvoke",
}


_MCP_TOOL_TYPES: tuple[type, type, type, type] | None = None


def _get_mcp_tool_types() -> tuple[type, type, type, type]:
    """Create stable OpenHands MCP tool classes without requiring OpenHands at import."""
    global _MCP_TOOL_TYPES
    if _MCP_TOOL_TYPES is not None:
        return _MCP_TOOL_TYPES

    from openhands.sdk.tool.schema import Action, Observation
    from openhands.sdk.tool.tool import ToolDefinition, ToolExecutor

    class IPWMCPAction(Action):
        query: Optional[str] = None
        command: Optional[str] = None
        path: Optional[str] = None
        content: Optional[str] = None
        mode: Optional[str] = None
        old_str: Optional[str] = None
        new_str: Optional[str] = None
        timeout: Optional[int] = None
        working_dir: Optional[str] = None

    class IPWMCPObservation(Observation):
        pass

    class MCPToolExecutor(ToolExecutor):
        def __init__(self, mcp_server: "BaseMCPServer") -> None:
            self._server = mcp_server

        def __call__(self, action: IPWMCPAction, conversation: Any = None) -> IPWMCPObservation:
            payload = action.model_dump(exclude_none=True)
            prompt = (
                payload.pop("query", None)
                or payload.pop("command", None)
                or payload.pop("path", None)
                or payload.pop("content", None)
                or ""
            )
            result = self._server.execute(str(prompt), **payload)
            content = result.content if hasattr(result, "content") else str(result)
            return IPWMCPObservation.from_text(text=content)

    class MCPToolDefinition(ToolDefinition[IPWMCPAction, IPWMCPObservation]):
        @classmethod
        def create(cls, *args: Any, **kwargs: Any) -> Sequence["MCPToolDefinition"]:
            return []

    for cls in (IPWMCPAction, IPWMCPObservation, MCPToolExecutor, MCPToolDefinition):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls

    _MCP_TOOL_TYPES = (IPWMCPAction, IPWMCPObservation, MCPToolExecutor, MCPToolDefinition)
    return _MCP_TOOL_TYPES


def _instrument_openhands_llm(
    model: Any,
    recorder: Optional["EventRecorder"],
) -> Any:
    """Record every OpenHands LLM call while preserving the LLM instance type."""
    if recorder is None or getattr(model, "_ipw_instrumented", False) is True:
        return model

    def _record_event(event_type: str, **metadata: Any) -> None:
        recorder.record(event_type, **metadata)

    def _record_lm_end(
        pre_input: int | None,
        pre_output: int | None,
        pre_cost: float | None,
    ) -> None:
        post_input, post_output, post_cost = _usage_counts(model)
        metadata: dict[str, Any] = {"model": str(model)}
        input_delta = (
            post_input - pre_input
            if post_input is not None and pre_input is not None
            else None
        )
        output_delta = (
            post_output - pre_output
            if post_output is not None and pre_output is not None
            else None
        )
        cost_delta = (
            post_cost - pre_cost
            if post_cost is not None and pre_cost is not None
            else None
        )
        if input_delta is not None and input_delta > 0:
            metadata["prompt_tokens"] = input_delta
        if output_delta is not None and output_delta > 0:
            metadata["completion_tokens"] = output_delta
        if cost_delta:
            metadata["cost_usd"] = cost_delta
        _record_event("lm_inference_end", **metadata)

    def _wrap_call(func: Any) -> Any:
        if inspect.iscoroutinefunction(func):
            async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
                pre_input, pre_output, pre_cost = _usage_counts(model)
                _record_event("lm_inference_start", model=str(model))
                try:
                    return await func(*args, **kwargs)
                finally:
                    _record_lm_end(pre_input, pre_output, pre_cost)

            return _async_wrapped

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            pre_input, pre_output, pre_cost = _usage_counts(model)
            _record_event("lm_inference_start", model=str(model))
            try:
                return func(*args, **kwargs)
            finally:
                _record_lm_end(pre_input, pre_output, pre_cost)

        return _wrapped

    for name in _CALL_METHODS:
        attr = getattr(model, name, None)
        if callable(attr):
            object.__setattr__(model, name, _wrap_call(attr))

    object.__setattr__(model, "_ipw_instrumented", True)
    return model


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
    from openhands.sdk.tool.tool import ToolAnnotations, ToolDefinition

    (
        IPWMCPAction,
        IPWMCPObservation,
        MCPToolExecutor,
        MCPToolDefinition,
    ) = _get_mcp_tool_types()

    tool_specs: list = []

    for name, server in mcp_tools.items():
        oh_name = getattr(server, "openhands_name", None) or (
            name if name in {"bash", "str_replace_editor"} else f"mcp_{name}"
        )

        spec = getattr(server, "_spec", None)
        description = (
            spec.description
            if spec and hasattr(spec, "description")
            else f"Execute the {name} tool"
        )

        executor = MCPToolExecutor(server)

        def _make_factory(_oh_name: str, _desc: str, _executor: MCPToolExecutor):
            def factory(conv_state: Any = None, **kwargs: Any) -> Sequence[ToolDefinition]:
                tool_def = MCPToolDefinition(
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
    Missing token fields stay ``None``; this parser never estimates a prompt /
    completion split from a total-only token count.
    """
    result: dict[str, Any] = {
        "input_tokens": None,
        "output_tokens": None,
        "cost": None,
        "num_turns": 0,
        "token_source": "missing",
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
            if metrics.get("accumulated_input_tokens") is not None:
                result["input_tokens"] = int(metrics["accumulated_input_tokens"])
            if metrics.get("accumulated_output_tokens") is not None:
                result["output_tokens"] = int(metrics["accumulated_output_tokens"])
            if metrics.get("accumulated_cost") is not None:
                result["cost"] = float(metrics["accumulated_cost"])
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
    ]:
        m = re.search(pattern, output)
        if m:
            try:
                val = int(m.group(1))
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
        "input_tokens": None,
        "output_tokens": None,
        "cost": None,
        "num_turns": 0,
        "token_source": "missing",
    }

    def _read_conversation_stats() -> dict[str, Any]:
        script = r"""
import base64
import glob
import json
import pickle

paths = sorted(glob.glob('/root/.openhands/sessions/*/conversation_stats.pkl'))
if not paths:
    print(json.dumps({}))
    raise SystemExit(0)

with open(paths[-1], 'r') as f:
    encoded = f.read()
metrics_by_service = pickle.loads(base64.b64decode(encoded))

prompt = 0
completion = 0
cost = 0.0
turns = 0
for metrics in metrics_by_service.values():
    usage = getattr(metrics, 'accumulated_token_usage', None)
    if usage is not None:
        prompt += int(getattr(usage, 'prompt_tokens', 0) or 0)
        completion += int(getattr(usage, 'completion_tokens', 0) or 0)
    cost += float(getattr(metrics, 'accumulated_cost', 0.0) or 0.0)
    token_usages = getattr(metrics, 'token_usages', None) or []
    turns += len(token_usages)

print(json.dumps({
    'input_tokens': prompt,
    'output_tokens': completion,
    'cost': cost,
    'num_turns': turns,
}))
"""
        cmd = ["/opt/openhands-venv/bin/python", "-c", script]
        stats = session.container.exec_run(cmd)
        if stats.exit_code != 0 or not stats.output.strip():
            return result
        parsed = json.loads(stats.output.decode())
        input_tokens = int(parsed.get("input_tokens") or 0)
        output_tokens = int(parsed.get("output_tokens") or 0)
        if input_tokens == 0 and output_tokens == 0:
            return result
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": float(parsed.get("cost") or 0.0),
            "num_turns": int(parsed.get("num_turns") or 0),
            "token_source": "openhands_conversation_stats",
        }

    def _recursive_token_metrics(value: Any) -> dict[str, Any]:
        accumulated: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                usage = node.get("accumulated_token_usage")
                if isinstance(usage, dict):
                    accumulated.append(
                        {
                            "input_tokens": int(usage.get("prompt_tokens") or 0),
                            "output_tokens": int(usage.get("completion_tokens") or 0),
                            "cost": float(node.get("accumulated_cost") or 0.0),
                            "num_turns": len(node.get("token_usages") or []),
                        }
                    )
                token_usages = node.get("token_usages")
                if isinstance(token_usages, list):
                    for usage_item in token_usages:
                        if isinstance(usage_item, dict):
                            calls.append(
                                {
                                    "prompt_tokens": int(
                                        usage_item.get("prompt_tokens") or 0
                                    ),
                                    "completion_tokens": int(
                                        usage_item.get("completion_tokens") or 0
                                    ),
                                }
                            )
                elif (
                    "prompt_tokens" in node
                    or "completion_tokens" in node
                ):
                    calls.append(
                        {
                            "prompt_tokens": int(node.get("prompt_tokens") or 0),
                            "completion_tokens": int(
                                node.get("completion_tokens") or 0
                            ),
                        }
                    )
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)
        best = max(
            accumulated,
            key=lambda item: item["input_tokens"] + item["output_tokens"],
            default=None,
        )
        if best and (best["input_tokens"] or best["output_tokens"]):
            best["token_source"] = "openhands_trajectory"
            return best
        input_tokens = sum(item["prompt_tokens"] for item in calls)
        output_tokens = sum(item["completion_tokens"] for item in calls)
        if input_tokens or output_tokens:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": 0.0,
                "num_turns": len(calls),
                "token_source": "openhands_trajectory",
            }
        return result

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
                stats_result = _read_conversation_stats()
                if stats_result["token_source"] != "missing":
                    return stats_result
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
                    if isinstance(extras, dict) and "llm_metrics" in extras:
                        metrics = extras["llm_metrics"]
                        break
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
            result["token_source"] = "openhands_trajectory"
            logger.info("Extracted from trajectory: in=%d out=%d cost=%.4f turns=%d",
                        result["input_tokens"], result["output_tokens"],
                        result["cost"], result["num_turns"])
        else:
            recursive_result = _recursive_token_metrics(trajectory)
            if recursive_result["token_source"] != "missing":
                result.update(recursive_result)
                logger.info(
                    "Extracted recursive trajectory metrics: in=%d out=%d turns=%d",
                    result["input_tokens"],
                    result["output_tokens"],
                    result["num_turns"],
                )
            else:
                stats_result = _read_conversation_stats()
                if stats_result["token_source"] != "missing":
                    result.update(stats_result)
                    logger.info(
                        "Extracted fallback conversation metrics: in=%d out=%d turns=%d",
                        result["input_tokens"],
                        result["output_tokens"],
                        result["num_turns"],
                    )
                else:
                    desc = (
                        f"list[{len(trajectory)}]"
                        if isinstance(trajectory, list)
                        else str(type(trajectory))
                    )
                    logger.warning("No actual token metrics found in trajectory (%s)", desc)
    except Exception:
        logger.warning("Failed to read OH trajectory", exc_info=True)
    return result


@AgentRegistry.register("openhands")
class OpenHands(BaseAgent):
    """OpenHands agent using the OpenHands SDK with energy telemetry."""

    DEFAULT_MAX_TURNS = 20

    DEFAULT_INSTRUCTIONS = (
        "You operate autonomously with two tools: `terminal` (bash) and "
        "`file_editor`. NEVER ask the user questions — make every decision "
        "yourself. To produce a deliverable, write a script with "
        "`file_editor`, RUN it via `terminal` (e.g. `python script.py`), "
        "and verify the output files exist with `ls`. Saving a script "
        "without running it accomplishes nothing. Install any missing "
        "libraries via `terminal` (e.g. `pip install openpyxl pdfplumber`). "
        "Call `finish` exactly once, only after the deliverable files are "
        "on disk."
    )

    def __init__(
        self,
        model: Any,
        tools: list | None = None,
        mcp_tools: Optional[Dict[str, "BaseMCPServer"]] = None,
        event_recorder: Optional["EventRecorder"] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenHands agent.

        Args:
            model: The LLM model instance to use.
            tools: List of OpenHands Tool specs.
            mcp_tools: Optional dict mapping tool name to BaseMCPServer instance.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            max_turns: Maximum iterations per run.
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
                Tool,
            )
            from openhands.sdk.event.llm_convertible.action import ActionEvent
            from openhands.sdk.event.llm_convertible.observation import ObservationEvent
        except ImportError:
            raise ImportError(
                "openhands-sdk package is required for OpenHands agent. "
                "Install with: pip install openhands-sdk"
            )

        # Importing openhands.tools registers the standard tools (terminal,
        # file_editor, task_tracker, etc.) in the SDK's tool registry. This
        # is optional — if it isn't installed the agent will fall back to
        # the SDK builtins (finish/think only).
        _have_openhands_tools = False
        try:
            import openhands.tools  # noqa: F401
            _have_openhands_tools = True
        except ImportError:
            logger.warning(
                "openhands-tools not installed; agent will only have finish/"
                "think available. Install with: pip install openhands-tools"
            )

        self.model = model
        self._instrumented_model = _instrument_openhands_llm(model, event_recorder)
        self.tools = tools
        self._mcp_tools = mcp_tools or {}
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
            llm=self._instrumented_model,
            max_tokens=24000,
            keep_first=2,
        )

        # Build agent_kwargs up front but DEFER Agent construction to per-run.
        #
        # Why: openhands.sdk.AgentBase._initialize() builds the tools map
        # exactly once and refuses to re-init. The first conversation's
        # workspace becomes baked into the TerminalExecutor and reused for
        # every subsequent conversation, so task N+1's `python build.py`
        # writes outputs into task N's workspace. Building a fresh Agent
        # per run() call forces _initialize to re-run with the current
        # conversation's state (and the correct per-task workspace).
        agent_kwargs: dict[str, Any] = {
            "llm": self._instrumented_model,
            "condenser": condenser,
        }

        if tools:
            agent_kwargs["tools"] = tools
        elif mcp_tools:
            extra_tool_specs = _register_mcp_tools(mcp_tools)
            agent_kwargs["tools"] = extra_tool_specs
        elif _have_openhands_tools:
            # Default tools: terminal + file_editor only. Skipping
            # task_tracker so small models don't spend turns building todo
            # lists. terminal_type="subprocess" avoids the tmux pane pool
            # (also a source of cross-conversation state leakage).
            agent_kwargs["tools"] = [
                Tool(name="terminal", params={"terminal_type": "subprocess"}),
                Tool(name="file_editor"),
            ]

        self._Agent = Agent
        self._agent_kwargs = agent_kwargs
        self.agent: Optional[Any] = None  # rebuilt in each run()
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
        # GDPval: tell the rubric judge where the agent's deliverables live.
        # OpenHands writes outputs directly into the workspace root (not a
        # subdirectory), so point the judge at the workspace itself.
        if metadata is not None and self._workspace and not metadata.get("session"):
            metadata.setdefault("gdpval_outputs_dir", self._workspace)
        # GDPval: materialize reference files into the workspace so the agent
        # can read them via its file tools. Skipped in TerminalBench mode
        # (handled separately via Docker copy).
        inputs_dir = metadata.get("gdpval_inputs_dir") if metadata else None
        if inputs_dir and not metadata.get("session"):
            try:
                dest = Path(self._workspace) / "inputs"
                dest.mkdir(parents=True, exist_ok=True)
                for src in Path(inputs_dir).iterdir():
                    target = dest / src.name
                    if not target.exists():
                        shutil.copy2(src, target)
            except Exception:
                logger.warning("Failed to stage gdpval inputs", exc_info=True)

    def set_workspace(self, workspace_path: str) -> None:
        """Set the workspace directory for the next agent run."""
        self._workspace = workspace_path
        for server in self._mcp_tools.values():
            if hasattr(server, "working_dir"):
                server.working_dir = workspace_path
            if getattr(server, "_ipw_dynamic_allowed_dirs", False) and hasattr(
                server, "allowed_dirs"
            ):
                server.allowed_dirs = [workspace_path]

    def _create_conversation(self) -> Any:
        """Create a fresh Agent + LocalConversation for the next run.

        Building a fresh Agent forces openhands.sdk.AgentBase._initialize()
        to re-run with the current conversation's state, which in turn
        creates a fresh TerminalExecutor bound to this task's workspace.
        Without this, the executor's working_dir is frozen at the first
        task's workspace and every subsequent task's outputs land there.
        """
        self.agent = self._Agent(**self._agent_kwargs)
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
        _pre_input, _pre_output, _pre_cost = _usage_counts(self.model)

        try:
            self.conversation.send_message(input)
            self.conversation.run()

            result = get_agent_final_response(self.conversation.state.events)
            if not result:
                # If the SDK did not emit a finish event, ask for a final answer
                # without lowering the iteration budget. Query/client timeouts are
                # the operational guardrail for long-running model behavior.
                logger.info("No FinishTool call detected, sending final-answer nudge")
                saved_limit = self.conversation.max_iteration_per_run
                self.conversation.max_iteration_per_run = saved_limit + 1
                self.conversation.send_message(
                    "Please provide your final answer now. "
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
            post_input, post_output, post_cost = _usage_counts(self.model)
            input_tokens = (
                post_input - _pre_input
                if post_input is not None and _pre_input is not None
                else None
            )
            output_tokens = (
                post_output - _pre_output
                if post_output is not None and _pre_output is not None
                else None
            )
            cost_usd = (
                post_cost - _pre_cost
                if post_cost is not None and _pre_cost is not None
                else None
            )
            token_source = (
                "openhands_litellm_metrics"
                if (input_tokens is not None and input_tokens > 0)
                or (output_tokens is not None and output_tokens > 0)
                else "missing"
            )

            return AgentRunResult(
                content=result,
                tool_calls_attempted=len(self._tool_names_used),
                tool_calls_succeeded=len(self._tool_names_used),
                tool_names_used=list(self._tool_names_used),
                num_turns=self._num_turns,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                metadata={"token_source": token_source},
            )
        finally:
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
        openhands_version = os.environ.get(
            "IPW_TERMINALBENCH_OPENHANDS_VERSION",
            "0.59.0",
        )

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
        usage_proxy: OpenAIUsageProxy | None = None
        proxy_stats: dict[str, Any] = {}

        if llm_base_url:
            original_llm_base_url = llm_base_url
            try:
                usage_proxy = OpenAIUsageProxy(
                    original_llm_base_url,
                    model=model_str,
                    event_callback=self._record_event,
                )
                usage_proxy.start()
                llm_base_url = usage_proxy.base_url_for_client(_get_docker_host_ip())
            except Exception:
                logger.warning(
                    "Failed to start OpenAI usage proxy for TerminalBench OpenHands",
                    exc_info=True,
                )
                usage_proxy = None

        if llm_base_url and usage_proxy is None:
            # Translate localhost/127.0.0.1 to the Docker bridge gateway IP so
            # containers can reach the host-side vLLM server.
            docker_ip = _get_docker_host_ip()
            llm_base_url = llm_base_url.replace("localhost", docker_ip)
            llm_base_url = llm_base_url.replace("127.0.0.1", docker_ip)

        if llm_base_url:
            os.environ["LLM_BASE_URL"] = llm_base_url
            # TerminalBench's installed OpenHands wrapper forwards LLM_* and
            # also forwards OPENHANDS_* after stripping the prefix. Different
            # OpenHands/litellm versions honor different base-url names, so set
            # the non-secret aliases to the same local/proxy endpoint.
            os.environ["OPENHANDS_LLM_BASE_URL"] = llm_base_url
            os.environ["OPENHANDS_OPENAI_BASE_URL"] = llm_base_url
            os.environ["OPENHANDS_OPENAI_API_BASE"] = llm_base_url
            os.environ["OPENHANDS_LITELLM_API_BASE"] = llm_base_url
            os.environ.setdefault("OPENHANDS_LLM_TIMEOUT", "120")
            os.environ.setdefault("OPENHANDS_TIMEOUT", "120")
            os.environ.setdefault("OPENHANDS_LLM_NUM_RETRIES", "10")
            os.environ.setdefault("OPENHANDS_NUM_RETRIES", "10")
        else:
            # Cloud model path: remove stale LLM_BASE_URL so litellm routes
            # to the provider's API endpoint directly.
            os.environ.pop("LLM_BASE_URL", None)
            os.environ.pop("OPENHANDS_LLM_BASE_URL", None)
            os.environ.pop("OPENHANDS_OPENAI_BASE_URL", None)
            os.environ.pop("OPENHANDS_OPENAI_API_BASE", None)
            os.environ.pop("OPENHANDS_LITELLM_API_BASE", None)
            os.environ.pop("OPENHANDS_LLM_TIMEOUT", None)
            os.environ.pop("OPENHANDS_TIMEOUT", None)
            os.environ.pop("OPENHANDS_LLM_NUM_RETRIES", None)
            os.environ.pop("OPENHANDS_NUM_RETRIES", None)

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
            # Check provider-specific env vars based on model prefix
            if model_str.startswith("anthropic/"):
                llm_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            elif model_str.startswith(("gemini/", "google/")):
                llm_api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
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
        extracted_input_tokens: int | None = None
        extracted_output_tokens: int | None = None
        extracted_cost: float | None = None
        extracted_num_turns = 0

        _agent_exc: Exception | None = None
        records_lm_calls = usage_proxy is not None
        if not records_lm_calls:
            self._record_event("lm_inference_start", model=model_str)
        try:
            tb_agent = TBOpenHandsAgent(
                model_name=model_str,
                version=openhands_version,
            )

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
            if usage_proxy is not None:
                proxy_stats = usage_proxy.snapshot()
                if proxy_stats.get("token_source") == "openai_api_usage":
                    stats = proxy_stats
            extracted_input_tokens = stats.get("input_tokens")
            extracted_output_tokens = stats.get("output_tokens")
            extracted_cost = stats.get("cost")
            extracted_num_turns = stats.get("num_turns", 0)
            token_source = stats.get("token_source", "missing")

        except Exception as exc:
            logger.exception("OpenHands TB agent failed on task %s", task_id)
            _agent_exc = exc
        finally:
            if not records_lm_calls:
                self._record_event("lm_inference_end", model=model_str)
            if usage_proxy is not None:
                try:
                    proxy_stats = usage_proxy.snapshot()
                finally:
                    usage_proxy.stop()

        if _agent_exc is not None:
            raise _agent_exc

        return AgentRunResult(
            content=terminal_output,
            input_tokens=extracted_input_tokens,
            output_tokens=extracted_output_tokens,
            num_turns=extracted_num_turns,
            cost_usd=extracted_cost,
            metadata={
                "task_id": task_id,
                "token_source": token_source,
                "openhands_version": openhands_version,
                "usage_proxy": proxy_stats,
            },
        )
