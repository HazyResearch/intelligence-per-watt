"""Stirrup agent adapter for IPW telemetry runs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
from time import perf_counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, MutableMapping, Optional

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder

LOGGER = logging.getLogger(__name__)


def _run_async_from_sync(coro):
    """Run a coroutine from sync code, including when an event loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive bridge
            error["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error["error"]
    return result.get("value")


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return {
        key: getattr(obj, key)
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def _flatten_history(history: Any) -> list[Any]:
    if not history:
        return []
    flat: list[Any] = []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
    return flat


def _token_attr(token_usage: Any, *names: str) -> int:
    for name in names:
        value = getattr(token_usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _turns_from_stirrup_history(history: Any) -> list[TurnTrace]:
    from ipw.execution.trace import TurnTrace

    turns: list[TurnTrace] = []
    for msg in _flatten_history(history):
        if not hasattr(msg, "token_usage"):
            continue
        token_usage = getattr(msg, "token_usage", None)
        tool_calls = getattr(msg, "tool_calls", None) or []
        tools_called: list[str] = []
        for call in tool_calls:
            name = getattr(call, "name", None)
            if name is None and isinstance(call, dict):
                name = call.get("name")
            if name:
                tools_called.append(str(name))

        start = getattr(msg, "request_start_time", None)
        end = getattr(msg, "request_end_time", None)
        wall_clock = 0.0
        if start is not None and end is not None:
            try:
                wall_clock = max(0.0, float(end) - float(start))
            except (TypeError, ValueError):
                wall_clock = 0.0

        turns.append(
            TurnTrace(
                turn_index=len(turns),
                input_tokens=_token_attr(token_usage, "input", "prompt", "prompt_tokens", "input_tokens"),
                output_tokens=_token_attr(token_usage, "output", "answer", "completion_tokens", "output_tokens"),
                tools_called=tools_called,
                wall_clock_s=wall_clock,
            )
        )
    return turns


@AgentRegistry.register("stirrup")
class StirrupAgent(BaseAgent):
    """Run tasks with Artificial Analysis' Stirrup harness inside IPW."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        backend: str = "local",
        max_turns: int = 60,
        max_tokens: int = 128_000,
        context_window: int = 32_768,
        context_summarization_cutoff: float = 0.9,
        use_litellm: bool = False,
        include_view_image: bool = True,
        require_tool_calls: bool = True,
        turns_remaining_warning_threshold: int = 15,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(event_recorder=event_recorder)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.backend = backend
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.context_summarization_cutoff = context_summarization_cutoff
        self.use_litellm = use_litellm
        self.include_view_image = include_view_image
        self.require_tool_calls = require_tool_calls
        self.turns_remaining_warning_threshold = turns_remaining_warning_threshold
        self.agent_kwargs = kwargs
        self._workspace: str | None = None
        self._task_metadata: MutableMapping[str, Any] | None = None

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = workspace_path

    def set_task_metadata(self, metadata: MutableMapping[str, Any]) -> None:
        self._task_metadata = metadata
        if self._workspace:
            Path(self._workspace).mkdir(parents=True, exist_ok=True)
            metadata["gdpval_outputs_dir"] = self._workspace

    def _make_client(self) -> Any:
        try:
            if self.use_litellm:
                from stirrup.clients.litellm_client import LiteLLMClient

                return LiteLLMClient(model_slug=self.model, max_tokens=self.max_tokens)
            from stirrup.clients.chat_completions_client import ChatCompletionsClient
            from stirrup.clients.utils import to_openai_messages, to_openai_tools
            from stirrup.core.models import AssistantMessage, Reasoning, TokenUsage, ToolCall
            from openai import BadRequestError

            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.context_window,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key:
                kwargs["api_key"] = self.api_key

            class QwenResilientChatCompletionsClient(ChatCompletionsClient):
                async def generate(self, messages, tools):  # type: ignore[no-untyped-def]
                    request_kwargs: dict[str, Any] = {
                        "model": self._model,
                        "messages": _sanitize_openai_tool_arguments(to_openai_messages(messages)),
                        "max_completion_tokens": getattr(
                            self,
                            "_ipw_completion_max_tokens",
                            self._max_tokens,
                        ),
                        **self._kwargs,
                    }
                    if tools:
                        request_kwargs["tools"] = to_openai_tools(tools)
                        request_kwargs["tool_choice"] = (
                            "required"
                            if getattr(self, "_ipw_require_tool_calls", False)
                            else "auto"
                        )
                    if self._reasoning_effort:
                        request_kwargs["reasoning_effort"] = self._reasoning_effort

                    request_start_time = perf_counter()
                    try:
                        response = await self._client.chat.completions.create(**request_kwargs)
                    except BadRequestError as exc:
                        message = str(exc)
                        if "Unterminated string" in message or "maximum context length" in message:
                            return AssistantMessage(
                                content=(
                                    "The previous model step produced malformed tool JSON or exceeded "
                                    "the context limit. Retry the next action concisely with valid tool "
                                    "arguments. Use short shell commands and relative file paths only."
                                ),
                                token_usage=TokenUsage(input=0, answer=0, reasoning=0),
                                request_start_time=request_start_time,
                                request_end_time=perf_counter(),
                            )
                        raise
                    request_end_time = perf_counter()

                    choice = response.choices[0]
                    msg = choice.message
                    reasoning: Reasoning | None = None
                    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                        reasoning = Reasoning(content=msg.reasoning_content)
                    tool_calls = [
                        ToolCall(
                            tool_call_id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments or "",
                        )
                        for tc in (msg.tool_calls or [])
                    ]
                    usage = response.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0
                    reasoning_tokens = 0
                    if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                        reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                    answer_tokens = output_tokens - reasoning_tokens

                    content = msg.content or ""
                    if choice.finish_reason in ("max_tokens", "length") and not tool_calls:
                        content = (
                            f"{content}\n\n"
                            "The response was truncated. Continue concisely with the next required "
                            "tool call or finish call using valid JSON and relative file paths."
                        ).strip()

                    return AssistantMessage(
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_calls,
                        token_usage=TokenUsage(
                            input=input_tokens,
                            answer=answer_tokens,
                            reasoning=reasoning_tokens,
                        ),
                        request_start_time=request_start_time,
                        request_end_time=request_end_time,
                    )
            client = QwenResilientChatCompletionsClient(**kwargs)
            client._ipw_completion_max_tokens = self.max_tokens
            client._ipw_require_tool_calls = self.require_tool_calls
            return client
        except ImportError as exc:
            raise ImportError(
                "stirrup is required for the 'stirrup' agent. Install with: "
                "pip install 'stirrup[litellm,e2b,docker]'"
            ) from exc

    def _make_tools(self) -> list[Any]:
        try:
            from stirrup.tools import ViewImageToolProvider, WebToolProvider
        except ImportError as exc:
            raise ImportError(
                "stirrup tools are required for the 'stirrup' agent."
            ) from exc

        backend = self.backend.lower().strip()
        if backend == "e2b":
            from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider

            exec_provider = E2BCodeExecToolProvider()
        elif backend == "docker":
            from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

            image = self.agent_kwargs.pop("docker_image", "python:3.12-slim")
            exec_provider = DockerCodeExecToolProvider.from_image(image)
        elif backend == "local":
            exec_provider = _IPWLocalCodeExecToolProvider(
                description=(
                    "Execute one bash command in the task working directory. "
                    "The current directory already contains uploaded inputs. "
                    "Pass arguments as JSON like {\"cmd\":\"ls -la\"}. Use only "
                    "relative paths; do not cd to /home/user, /home/ubuntu, /tmp, "
                    "or ~/. Create deliverables in the current directory or a "
                    "relative subdirectory such as outputs/."
                )
            )
        else:
            raise ValueError("backend must be one of: local, docker, e2b")

        tools: list[Any] = [exec_provider, WebToolProvider()]
        if self.include_view_image:
            tools.append(ViewImageToolProvider())
        return tools

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        return _run_async_from_sync(self.run_async(input, **kwargs))

    async def run_async(self, input: str, **kwargs: Any) -> AgentRunResult:
        try:
            from stirrup import Agent
            from stirrup.utils.logging import AgentLogger
        except ImportError as exc:
            raise ImportError(
                "stirrup is required for the 'stirrup' agent. Install with: "
                "pip install 'stirrup[litellm,e2b,docker]'"
            ) from exc

        _patch_stirrup_summarization()
        _patch_stirrup_run_tool()

        output_dir = Path(self._workspace or tempfile_workspace())
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files: str | None = None
        if self._task_metadata:
            inputs_dir = self._task_metadata.get("gdpval_inputs_dir")
            if inputs_dir:
                input_files = str(inputs_dir)

        self._record_event("lm_inference_start", model=self.model, agent="stirrup")
        finish_params = None
        history = None
        metadata = None
        session_obj = None
        try:
            agent = Agent(
                client=self._make_client(),
                name="ipw_stirrup",
                tools=self._make_tools(),
                finish_tool=_make_tolerant_finish_tool(),
                max_turns=self.max_turns,
                context_summarization_cutoff=self.context_summarization_cutoff,
                turns_remaining_warning_threshold=self.turns_remaining_warning_threshold,
                logger=AgentLogger(show_spinner=False, level=logging.WARNING),
            )
            async with agent.session(output_dir=output_dir, input_files=input_files) as session:
                session_obj = session
                finish_params, history, metadata = await session.run(input)
        finally:
            self._record_event("lm_inference_end", model=self.model, agent="stirrup")

        finish_dict = _as_dict(finish_params)
        turns = _turns_from_stirrup_history(history)
        total_input = sum(t.input_tokens for t in turns)
        total_output = sum(t.output_tokens for t in turns)

        local_files = sorted(str(p) for p in output_dir.rglob("*") if p.is_file())
        finish_paths = [str(p) for p in (finish_dict.get("paths") or [])]
        transferred_paths = [str(p) for p in (getattr(session_obj, "_transferred_paths", None) or [])]
        submitted_paths = _submitted_files(output_dir, finish_paths, transferred_paths, local_files)
        if self._task_metadata is not None:
            self._task_metadata["gdpval_outputs_dir"] = str(output_dir)
            self._task_metadata["gdpval_submitted_files"] = submitted_paths
            self._task_metadata["stirrup_finish_params"] = finish_dict

        did_finish = bool(finish_dict)
        if self._task_metadata is not None:
            self._task_metadata["stirrup_finished"] = did_finish

        summary = str(
            finish_dict.get("reason")
            or finish_dict.get("summary")
            or finish_dict
            or "Stirrup run ended without a finish call."
        )
        from ipw.execution.trace import QueryTrace

        trace = QueryTrace(query_id="", workload_type="stirrup", turns=turns)
        return AgentRunResult(
            content=summary,
            tool_calls_attempted=sum(len(t.tools_called) for t in turns),
            tool_calls_succeeded=sum(len(t.tools_called) for t in turns),
            tool_names_used=sorted({name for t in turns for name in t.tools_called}),
            num_turns=len(turns),
            input_tokens=total_input,
            output_tokens=total_output,
            trace=trace,
            metadata={
                "finish_params": finish_dict,
                "stirrup_metadata": metadata,
                "submitted_paths": finish_paths,
                "transferred_paths": transferred_paths,
                "local_output_files": local_files,
                "output_dir": str(output_dir),
                "gdpval_outputs_dir": str(output_dir),
                "gdpval_submitted_files": submitted_paths,
                "stirrup_finish_params": finish_dict,
                "stirrup_finished": did_finish,
            },
        )


class _IPWLocalCodeExecToolProvider:
    """Local Stirrup code backend with path normalization for weaker tool callers."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from stirrup.tools import LocalCodeExecToolProvider

        class Provider(LocalCodeExecToolProvider):  # type: ignore[misc, valid-type]
            async def run_command(self, cmd: str, *, timeout: int | None = None) -> Any:
                return await super().run_command(_sanitize_shell_command(cmd, self.temp_dir), timeout=timeout)

        return Provider(*args, **kwargs)


def _sanitize_shell_command(cmd: str, temp_dir: Path | None = None) -> str:
    """Rewrite common absolute-path mistakes back into the local exec cwd."""
    sanitized = str(cmd or "").strip()
    if temp_dir:
        temp = str(temp_dir)
        sanitized = sanitized.replace(temp + "/", "./").replace(temp, ".")

    # The local backend already runs in the task cwd. Drop unsafe/irrelevant cd
    # prefixes that Qwen often invents after seeing Linux paths in errors.
    sanitized = re.sub(
        r"^\s*cd\s+(?:/home/user|/home/ubuntu|/tmp(?:/local_exec_env_[A-Za-z0-9_]+)?|~)\s*(?:&&|;)\s*",
        "",
        sanitized,
    )

    # Keep file operations inside cwd if the model embeds common fake roots.
    sanitized = re.sub(r"(?<![\w.-])/(?:home/user|home/ubuntu)/(?:[^'\"\s;&|<>]*/)*([^'\"\s;&|<>/]+)", r"\1", sanitized)
    sanitized = re.sub(r"(?<![\w.-])/tmp/local_exec_env_[A-Za-z0-9_]+/", "./", sanitized)
    return sanitized


def _submitted_files(
    output_dir: Path,
    finish_paths: list[str],
    transferred_paths: list[str],
    local_files: list[str],
) -> list[str]:
    """Resolve Stirrup finish paths to persisted files in the IPW artifact dir."""
    candidates: list[Path] = []
    for raw in [*transferred_paths, *finish_paths]:
        path = Path(raw)
        if path.is_file():
            candidates.append(path)
        basename_path = output_dir / path.name
        if basename_path.is_file():
            candidates.append(basename_path)
    candidates.extend(Path(p) for p in local_files)

    out: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen and path.is_file():
            seen.add(key)
            out.append(str(path))
    return out


def _make_tolerant_finish_tool() -> Any:
    """Create a finish tool that turns bad paths into retryable tool errors.

    Stirrup's default finish validator lets some execution-environment path
    errors escape the tool call and fail the whole prompt. GDPval agents often
    hallucinate absolute paths like /home/user/file.docx; this validator first
    tries the basename in the execution directory and otherwise returns a
    normal failed tool result so the model can retry with a valid relative path.
    """
    from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
    from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
    from stirrup.tools.finish import FinishParams

    async def _finish_executor(params: FinishParams) -> ToolResult[ToolUseCountMetadata]:
        from stirrup.core.agent import _SESSION_STATE

        if not params.paths:
            return ToolResult(
                content=(
                    "ERROR: finish requires at least one deliverable file path. "
                    "Create the requested output file in the current working directory, "
                    "verify it with ls -la, then call finish with paths like "
                    "['report.docx'] or ['outputs/chart.png']."
                ),
                metadata=ToolUseCountMetadata(),
                success=False,
            )

        try:
            state = _SESSION_STATE.get(None)
            exec_env = state.exec_env if state else None
        except LookupError:
            exec_env = None

        if exec_env and params.paths:
            normalized_paths: list[str] = []
            missing: list[str] = []
            for raw in params.paths:
                raw_str = str(raw)
                path = Path(raw_str)
                candidates = [raw_str]
                if path.is_absolute() and path.name:
                    candidates.append(path.name)

                found: str | None = None
                for candidate in candidates:
                    try:
                        if await exec_env.file_exists(candidate):
                            found = candidate
                            break
                    except Exception:
                        continue
                if found is None:
                    missing.append(raw_str)
                else:
                    normalized_paths.append(found)

            if missing:
                return ToolResult(
                    content=(
                        "ERROR: Files do not exist inside the execution environment: "
                        f"{missing}. Use relative file paths from the current working "
                        "directory, not /home/user, /home/ubuntu, /tmp, or folders."
                    ),
                    metadata=ToolUseCountMetadata(),
                    success=False,
                )
            params.paths = normalized_paths

        return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)

    return Tool[FinishParams, ToolUseCountMetadata](
        name=DEFAULT_FINISH_TOOL_NAME,
        description=(
            "Signal task completion with a reason and the relative paths of output "
            "files created in the execution environment. The paths list must contain "
            "at least one file. Do not submit directories or absolute paths such as "
            "/home/user, /home/ubuntu, /tmp, or ~/..."
        ),
        parameters=FinishParams,
        executor=_finish_executor,
    )


def _sanitize_openai_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure historical assistant tool-call arguments are valid JSON strings."""
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                function["arguments"] = json.dumps(arguments or {})
                continue
            if not arguments.strip():
                function["arguments"] = "{}"
                continue
            try:
                json.loads(arguments)
            except Exception:
                function["arguments"] = json.dumps({
                    "invalid_tool_arguments": arguments[:2000],
                })
    return messages


def _coerce_tool_arguments(tool_name: str, raw: str, parameters: Any) -> Any | None:
    """Best-effort recovery for malformed Qwen XML tool-call arguments."""
    raw = (raw or "").strip()
    if not raw:
        return None

    candidates: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            candidates.append(parsed)

    if tool_name == "code_exec":
        cmd = None
        parsed = candidates[0] if candidates else None
        if parsed and isinstance(parsed.get("cmd"), str):
            cmd = parsed["cmd"]
        if cmd is None:
            cmd = _extract_string_field(raw, "cmd")
        if cmd is None and not raw.lstrip().startswith("{"):
            cmd = raw
        if cmd:
            candidates.append({"cmd": _sanitize_shell_command(cmd)})

    elif tool_name == "finish":
        parsed = candidates[0] if candidates else {}
        reason = parsed.get("reason") if isinstance(parsed.get("reason"), str) else None
        paths = parsed.get("paths") if isinstance(parsed.get("paths"), list) else None
        if reason is None:
            reason = _extract_string_field(raw, "reason") or "Task completed."
        if paths is None:
            paths = _extract_paths(raw)
        candidates.append({"reason": reason, "paths": paths})

    for candidate in candidates:
        with contextlib.suppress(Exception):
            return parameters.model_validate(candidate)
    return None


def _extract_string_field(raw: str, field: str) -> str | None:
    match = re.search(rf'["\']?{re.escape(field)}["\']?\s*:\s*', raw)
    if not match:
        return None
    tail = raw[match.end():].lstrip()
    if not tail:
        return None
    quote = tail[0] if tail[0] in {"'", '"'} else None
    if quote:
        tail = tail[1:]
        pieces: list[str] = []
        escaped = False
        for char in tail:
            if escaped:
                pieces.append("\\" + char)
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote:
                break
            pieces.append(char)
        value = "".join(pieces)
    else:
        value = re.split(r"[,}]\s*$", tail, maxsplit=1)[0]
    with contextlib.suppress(Exception):
        return bytes(value, "utf-8").decode("unicode_escape")
    return value.strip()


def _extract_paths(raw: str) -> list[str]:
    parsed_paths: list[str] = []
    paths_field = re.search(r'["\']?paths["\']?\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if paths_field:
        parsed_paths.extend(re.findall(r'["\']([^"\']+)["\']', paths_field.group(1)))
    if not parsed_paths:
        parsed_paths.extend(
            match.group(0)
            for match in re.finditer(
                r"[\w./-]+\.(?:docx|xlsx|pptx|pdf|png|jpg|jpeg|csv|txt|mp3|wav|zip|html|json)",
                raw,
                re.IGNORECASE,
            )
        )
    clean: list[str] = []
    seen: set[str] = set()
    for path in parsed_paths:
        normalized = Path(path).name if Path(path).is_absolute() else path
        if normalized and normalized not in seen:
            clean.append(normalized)
            seen.add(normalized)
    return clean


def _patch_stirrup_run_tool() -> None:
    """Patch Stirrup tool execution to recover malformed JSON arguments."""
    try:
        import anyio
        import inspect as inspect_mod
        import stirrup.core.agent as agent_mod
        from pydantic import ValidationError
        from stirrup.core.models import ToolMessage, ToolResult
    except Exception:
        return

    if getattr(agent_mod.Agent.run_tool, "_ipw_qwen_tolerant", False):
        return

    async def run_tool(self: Any, tool_call: Any, run_metadata: dict[str, list[Any]]) -> Any:
        tool = self._active_tools.get(tool_call.name)
        args_valid = True

        if tool_call.name not in run_metadata:
            run_metadata[tool_call.name] = []

        tool_start_time = perf_counter()
        if tool:
            try:
                args = tool_call.arguments if tool_call.arguments and tool_call.arguments.strip() else "{}"
                try:
                    params = tool.parameters.model_validate_json(args)
                except ValidationError:
                    params = _coerce_tool_arguments(tool_call.name, args, tool.parameters)
                    if params is None:
                        raise
                    tool_call.arguments = params.model_dump_json()
                    args_valid = False

                prev_depth = agent_mod._PARENT_DEPTH.set(self._logger.depth)
                try:
                    if inspect_mod.iscoroutinefunction(tool.executor):
                        result = await tool.executor(params)
                    elif self._run_sync_in_thread:
                        result = await anyio.to_thread.run_sync(tool.executor, params)
                    else:
                        result = tool.executor(params)
                finally:
                    agent_mod._PARENT_DEPTH.reset(prev_depth)

                if result.metadata is not None:
                    run_metadata[tool_call.name].append(result.metadata)
            except ValidationError:
                LOGGER.debug(
                    "LLMClient tried to use the tool %s but the tool arguments are not valid: %r",
                    tool_call.name,
                    tool_call.arguments,
                )
                result = ToolResult(content="Tool arguments are not valid. Retry with a compact valid JSON object.", success=False)
                args_valid = False
        else:
            LOGGER.debug("LLMClient tried to use the tool %s which is not in the tools list", tool_call.name)
            result = ToolResult(content=f"{tool_call.name} is not a valid tool", success=False)

        tool_end_time = perf_counter()
        return ToolMessage(
            content=result.content,
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            args_was_valid=args_valid,
            success=result.success,
            tool_start_time=tool_start_time,
            tool_end_time=tool_end_time,
        )

    setattr(run_tool, "_ipw_qwen_tolerant", True)
    agent_mod.Agent.run_tool = run_tool


def _patch_stirrup_summarization() -> None:
    """Patch Stirrup summarization to avoid tool-call parsing during summaries."""
    try:
        import stirrup.core.agent as agent_mod
        from itertools import takewhile
        from stirrup.core.models import AssistantMessage, ChatMessage, SummaryMessage, UserMessage
    except Exception:
        return

    if getattr(agent_mod.Agent.summarize_messages, "_ipw_qwen_safe", False):
        return

    async def summarize_messages(self: Any, messages: list["ChatMessage"]) -> list["ChatMessage"]:
        task_context: list[ChatMessage] = list(
            takewhile(lambda m: not isinstance(m, (AssistantMessage, SummaryMessage)), messages)
        )
        summary_prompt = [*messages, UserMessage(content=agent_mod.MESSAGE_SUMMARIZER)]
        summary = await self._client.generate(summary_prompt, {})
        summary_bridge_prompt = agent_mod.MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE.format(summary=summary.content)
        summary_bridge = SummaryMessage(content=summary_bridge_prompt)
        acknowledgement_msg = UserMessage(content="Got it, thanks!")
        summary_content = summary.content if isinstance(summary.content, str) else str(summary.content)
        self._logger.context_summarization_complete(summary_content, summary_bridge_prompt)
        return [*task_context, summary_bridge, acknowledgement_msg]

    setattr(summarize_messages, "_ipw_qwen_safe", True)
    agent_mod.Agent.summarize_messages = summarize_messages


def tempfile_workspace() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="ipw_stirrup_")


__all__ = ["StirrupAgent"]
