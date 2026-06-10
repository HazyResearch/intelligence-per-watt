"""Terminus agent implementation for terminal-based tasks."""

from __future__ import annotations

import logging
import os
import time
import types
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult
from ipw.cost.pricing import calculate_cost

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder

# Default Docker image with tmux pre-installed
DEFAULT_DOCKER_IMAGE = "ubuntu:22.04"
_LITELLM_PREFIXES = (
    "openai/",
    "anthropic/",
    "gemini/",
    "google/",
    "azure/",
    "bedrock/",
    "vertex_ai/",
    "ollama/",
)
LOGGER = logging.getLogger(__name__)


def _cloud_litellm_model_name(model: str) -> str:
    """Return a LiteLLM provider-qualified model name for cloud calls."""
    if model.startswith(_LITELLM_PREFIXES):
        return model
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    if model.startswith("gemini-"):
        return f"gemini/{model}"
    if model.startswith(("gpt-", "o1", "o3", "o4")) and not model.startswith("gpt-oss"):
        return f"openai/{model}"
    return f"openai/{model}"


def _pricing_provider_model(model: str) -> tuple[str | None, str]:
    """Map a LiteLLM model name to the IPW pricing table provider/model."""
    normalized = _cloud_litellm_model_name(model)
    if "/" not in normalized:
        return None, normalized
    provider, priced_model = normalized.split("/", 1)
    if provider == "google":
        provider = "gemini"
    if provider not in {"openai", "anthropic", "gemini"}:
        return None, priced_model
    return provider, priced_model


@AgentRegistry.register("terminus")
class Terminus(BaseAgent):
    """Terminus agent for terminal-based task execution in Docker containers."""

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: str,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        container_name: str | None = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Terminus agent.

        Args:
            model: The model name to use (e.g., "gpt-4o").
            docker_image: Docker image to use for the container. Must have tmux installed.
            container_name: Optional name for the Docker container.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional keyword arguments passed to Terminus2.
        """
        super().__init__(event_recorder=event_recorder)
        self._docker_image = docker_image
        self._container_name = container_name or f"terminus-container-{uuid.uuid4().hex[:8]}"
        self._docker_client = None
        self._container = None
        self._owns_container = False
        self._workspace: Path | None = None

        # Lazy imports: docker and terminal-bench are optional
        try:
            import docker as _docker_mod  # noqa: F401
        except ImportError:
            raise ImportError(
                "docker package is required for Terminus agent. "
                "Install with: pip install docker"
            )

        try:
            from terminal_bench.agents.terminus_2 import Terminus2
            from terminal_bench.llms.base_llm import (
                ContextLengthExceededError,
                OutputLengthExceededError,
            )
        except ImportError:
            raise ImportError(
                "terminal-bench package is required for Terminus agent. "
                "Install with: pip install terminal-bench"
            )

        context_limit = kwargs.pop("context_limit", None)
        max_output_tokens = kwargs.pop("max_output_tokens", None)
        max_turns = kwargs.pop("max_turns", None)
        context_buffer_tokens = int(kwargs.pop("context_buffer_tokens", 512))
        terminal_output_max_bytes = kwargs.pop("terminal_output_max_bytes", None)
        llm_call_kwargs = kwargs.pop("llm_call_kwargs", None) or {}
        if max_turns is not None:
            kwargs.setdefault("max_episodes", max(1, int(max_turns)))
        self._usage_calls: list[dict[str, Any]] = []
        local_openai_endpoint = bool(kwargs.get("api_base") or kwargs.get("base_url"))
        if local_openai_endpoint:
            os.environ.setdefault("OPENAI_API_KEY", os.getenv("VLLM_API_KEY", "EMPTY"))
        if local_openai_endpoint:
            litellm_model = f"openai/{model.removeprefix('openai/')}"
        else:
            litellm_model = _cloud_litellm_model_name(model)
        self.agent = Terminus2(model_name=litellm_model, **kwargs)
        if terminal_output_max_bytes is not None:
            output_limit = int(terminal_output_max_bytes)
            original_limit_output_length = self.agent._limit_output_length

            def _ipw_limit_output_length(
                _self: Any,
                output: str,
                max_bytes: int = 10000,
            ) -> str:
                return original_limit_output_length(
                    output,
                    max_bytes=min(max_bytes, output_limit),
                )

            self.agent._limit_output_length = types.MethodType(
                _ipw_limit_output_length,
                self.agent,
            )
        if context_limit is not None:
            server_context_limit = int(context_limit)
            effective_context_limit = max(1, server_context_limit - context_buffer_tokens)

            def _forced_context_limit(_self: Any) -> int:
                return effective_context_limit

            self.agent._get_model_context_limit = types.MethodType(  # type: ignore[attr-defined]
                _forced_context_limit,
                self.agent,
            )

        configured_max_output_tokens = (
            int(max_output_tokens) if max_output_tokens is not None else None
        )
        llm = self.agent._llm  # type: ignore[attr-defined]
        if not getattr(llm, "_ipw_terminus_instrumented", False):
            original_call = llm.call

            def _call_messages(
                call_args: tuple[Any, ...],
                call_kwargs: dict[str, Any],
            ) -> list[dict[str, Any]]:
                prompt = call_kwargs.get("prompt")
                message_history = call_kwargs.get("message_history", [])
                messages_arg = call_kwargs.get("messages")
                if messages_arg is not None:
                    return list(messages_arg or [])
                if prompt is None and call_args:
                    prompt = call_args[0]
                messages = list(message_history or [])
                if prompt is not None:
                    messages.append({"role": "user", "content": prompt})
                return messages

            def _trim_message_history(
                _llm: Any,
                call_args: tuple[Any, ...],
                call_kwargs: dict[str, Any],
            ) -> None:
                if context_limit is None:
                    return
                history = list(call_kwargs.get("message_history") or [])
                if not history:
                    return
                prompt = call_kwargs.get("prompt")
                if prompt is None and call_args:
                    prompt = call_args[0]
                prompt_message = (
                    [{"role": "user", "content": prompt}]
                    if prompt is not None
                    else []
                )
                target_tokens = max(1, int(context_limit) - context_buffer_tokens)
                trimmed = history
                while trimmed and _approx_context_tokens(
                    _llm,
                    [*trimmed, *prompt_message],
                ) > target_tokens:
                    trimmed = trimmed[2:] if len(trimmed) >= 2 else []
                if len(trimmed) != len(history):
                    call_kwargs["message_history"] = trimmed

            def _approx_context_tokens(_llm: Any, messages: list[dict[str, Any]]) -> int:
                try:
                    token_count = int(_llm.count_tokens(messages))
                except Exception:
                    token_count = 0

                # Some LiteLLM provider/model pairs undercount unknown local models.
                # Keep a conservative character-based lower bound so max_tokens does
                # not push the OpenAI-compatible server over its total context limit.
                char_count = 0
                for message in messages:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        char_count += len(content)
                    else:
                        char_count += len(str(content))
                char_estimate = max(1, char_count // 3)
                return max(token_count, char_estimate)

            def _bounded_max_tokens(
                _llm: Any,
                call_args: tuple[Any, ...],
                call_kwargs: dict[str, Any],
                input_tokens: int,
            ) -> Optional[int]:
                existing_max_tokens = call_kwargs.get("max_tokens")
                existing_max_completion_tokens = call_kwargs.get("max_completion_tokens")
                if (
                    configured_max_output_tokens is None
                    and existing_max_tokens is None
                    and existing_max_completion_tokens is None
                ):
                    return None
                configured_limits = [
                    value
                    for value in (
                        configured_max_output_tokens,
                        existing_max_tokens,
                        existing_max_completion_tokens,
                    )
                    if value is not None
                ]
                available = min(int(value) for value in configured_limits)
                if context_limit is not None:
                    available = min(
                        available,
                        max(1, int(context_limit) - input_tokens - context_buffer_tokens),
                    )
                return available

            def _is_context_limit_error(exc: Exception) -> bool:
                text = str(exc).lower()
                return (
                    "context length" in text
                    or "context window" in text
                    or "maximum context" in text
                    or "longer than the model" in text
                    or ("input" in text and "longer" in text)
                )

            def _is_token_limit_error(exc: Exception) -> bool:
                text = str(exc).lower()
                return (
                    "badrequest" in text
                    or "bad request" in text
                    or "context" in text
                    or "max_tokens" in text
                    or "maximum context length" in text
                )

            def _apply_output_bound(
                call_kwargs: dict[str, Any],
                available: Optional[int],
            ) -> None:
                if available is None:
                    return
                existing_max_tokens = call_kwargs.get("max_tokens")
                existing_max_completion_tokens = call_kwargs.get("max_completion_tokens")
                if existing_max_tokens is None and existing_max_completion_tokens is None:
                    call_kwargs["max_tokens"] = available
                elif existing_max_tokens is not None:
                    call_kwargs["max_tokens"] = min(int(existing_max_tokens), available)
                elif existing_max_completion_tokens is not None:
                    call_kwargs["max_completion_tokens"] = min(
                        int(existing_max_completion_tokens),
                        available,
                    )

            def _call_with_retry(
                args: tuple[Any, ...],
                call_kwargs: dict[str, Any],
                available: Optional[int],
            ) -> tuple[str, dict[str, Any]]:
                captured_usage: dict[str, Any] = {}

                def _coerce_int(value: Any) -> int:
                    try:
                        return int(value or 0)
                    except Exception:
                        return 0

                def _response_model(response: Any) -> str:
                    if isinstance(response, dict):
                        return str(response.get("model") or litellm_model)
                    return str(getattr(response, "model", None) or litellm_model)

                def _extract_usage(response: Any) -> dict[str, Any]:
                    usage = None
                    if isinstance(response, dict):
                        usage = response.get("usage")
                    else:
                        usage = getattr(response, "usage", None)
                    if usage is None:
                        return {}
                    if isinstance(usage, dict):
                        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
                        completion = usage.get("completion_tokens") or usage.get("output_tokens")
                        total = usage.get("total_tokens")
                    else:
                        prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                        completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
                        total = getattr(usage, "total_tokens", None)
                    prompt_i = _coerce_int(prompt)
                    completion_i = _coerce_int(completion)
                    total_i = _coerce_int(total) or prompt_i + completion_i
                    if prompt_i == 0 and completion_i == 0:
                        return {}
                    provider, priced_model = _pricing_provider_model(
                        _response_model(response)
                    )
                    cost_usd = (
                        calculate_cost(provider, priced_model, prompt_i, completion_i)
                        if provider
                        else None
                    )
                    if cost_usd == 0.0:
                        cost_usd = None
                    return {
                        "prompt_tokens": prompt_i,
                        "completion_tokens": completion_i,
                        "total_tokens": total_i,
                        "cost_usd": cost_usd,
                    }

                def _capture_litellm_usage() -> Any:
                    import litellm

                    original_completion = litellm.completion

                    def _completion_with_usage(*c_args: Any, **c_kwargs: Any) -> Any:
                        response = original_completion(*c_args, **c_kwargs)
                        usage = _extract_usage(response)
                        if usage:
                            captured_usage.update(usage)
                        return response

                    litellm.completion = _completion_with_usage
                    try:
                        return original_call(*args, **call_kwargs)
                    finally:
                        litellm.completion = original_completion

                try:
                    return _capture_litellm_usage(), dict(captured_usage)
                except Exception as exc:
                    if isinstance(exc, OutputLengthExceededError):
                        raise
                    if not _is_token_limit_error(exc) or available is None:
                        if _is_context_limit_error(exc):
                            raise ContextLengthExceededError(str(exc)) from exc
                        raise
                    retry_kwargs = dict(call_kwargs)
                    retry_limit = min(
                        retry_kwargs.get(
                            "max_tokens",
                            configured_max_output_tokens or available,
                        ),
                        max(1, available // 2),
                    )
                    retry_kwargs["max_tokens"] = retry_limit
                    retry_kwargs.pop("max_completion_tokens", None)
                    try:
                        call_kwargs = retry_kwargs
                        captured_usage.clear()
                        return _capture_litellm_usage(), dict(captured_usage)
                    except Exception as retry_exc:
                        if isinstance(retry_exc, OutputLengthExceededError):
                            raise
                        if _is_context_limit_error(retry_exc):
                            raise ContextLengthExceededError(str(retry_exc)) from retry_exc
                        raise

            def _instrumented_call(_llm: Any, *args: Any, **call_kwargs: Any) -> str:
                call_kwargs = {**llm_call_kwargs, **call_kwargs}
                _trim_message_history(_llm, args, call_kwargs)
                messages = _call_messages(args, call_kwargs)
                prompt_tokens = _approx_context_tokens(_llm, messages) if messages else 0
                available = _bounded_max_tokens(_llm, args, call_kwargs, prompt_tokens)
                _apply_output_bound(call_kwargs, available)

                self._record_event("lm_inference_start", model=litellm_model)
                try:
                    response, usage = _call_with_retry(args, call_kwargs, available)
                except Exception as exc:
                    self._record_event(
                        "lm_inference_end",
                        model=litellm_model,
                        prompt_tokens=0,
                        completion_tokens=0,
                        token_source="missing",
                        error=str(exc),
                    )
                    raise

                if usage:
                    self._usage_calls.append(usage)
                self._record_event(
                    "lm_inference_end",
                    model=litellm_model,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    cost_usd=usage.get("cost_usd"),
                    token_source="litellm_response_usage" if usage else "missing",
                )
                return response

            llm.call = types.MethodType(_instrumented_call, llm)
            setattr(llm, "_ipw_terminus_instrumented", True)

    def set_workspace(self, workspace_path: str) -> None:
        """Mount the per-query workspace into the Terminus container."""
        workspace = Path(workspace_path).resolve()
        if self._workspace == workspace:
            return
        if self._container is not None:
            self.cleanup()
        self._workspace = workspace

    def _get_docker_client(self):
        """Get or create the Docker client."""
        if self._docker_client is None:
            import docker
            self._docker_client = docker.from_env()
        return self._docker_client

    def _get_or_create_container(self):
        """Get an existing container or create a new one with tmux installed."""
        import docker

        if self._container is not None:
            return self._container

        client = self._get_docker_client()

        # Try to get an existing container by name
        try:
            container = client.containers.get(self._container_name)
            if container.status != "running":
                container.start()
            self._container = container
            return container
        except docker.errors.NotFound:
            pass

        container_kwargs: dict[str, Any] = {}
        if self._workspace is not None:
            self._workspace.mkdir(parents=True, exist_ok=True)
            container_kwargs["volumes"] = {
                str(self._workspace): {"bind": "/workspace", "mode": "rw"},
            }
            container_kwargs["working_dir"] = "/workspace"

        # Create a new container with tmux installed
        container = client.containers.run(
            self._docker_image,
            command="/bin/bash -c 'apt-get update && apt-get install -y tmux && tail -f /dev/null'",
            name=self._container_name,
            detach=True,
            tty=True,
            stdin_open=True,
            **container_kwargs,
        )
        self._container = container
        self._owns_container = True

        # Wait for tmux installation to complete
        for _ in range(30):
            exit_code, output = container.exec_run("which tmux")
            if exit_code == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Timeout waiting for tmux installation in container")

        return container

    def get_session(self, tmux_session: Any = None) -> Any:
        """Get or create a TmuxSession.

        Args:
            tmux_session: Either an existing TmuxSession, a session name string,
                or None to create a default session.

        Returns:
            A TmuxSession instance.
        """
        from terminal_bench.terminal.tmux_session import TmuxSession

        if isinstance(tmux_session, TmuxSession):
            return tmux_session

        container = self._get_or_create_container()
        session_name = (
            tmux_session
            if isinstance(tmux_session, str)
            else f"terminus-session-{uuid.uuid4().hex[:8]}"
        )

        session = TmuxSession(
            session_name=session_name,
            container=container,
            disable_recording=True,
        )
        session.start()
        if not session.is_session_alive():
            raise RuntimeError(
                f"Failed to start tmux session '{session_name}' in container "
                f"'{container.name}'."
            )
        return session

    def _cleanup_session(self, session: Any) -> None:
        """Remove the per-query tmux session to avoid leaking sessions."""
        session_name = getattr(session, "_session_name", None)
        container = getattr(session, "container", None)
        if not session_name or container is None:
            return
        try:
            container.exec_run(["tmux", "kill-session", "-t", str(session_name)])
        except Exception:
            pass

    def _instrument_session_tools(self, session: Any) -> Any:
        """Record terminal commands as tool calls on the provided session."""
        if getattr(session, "_ipw_tool_instrumented", False) is True:
            return session

        original_send_keys = session.send_keys

        def _send_keys_with_events(*args: Any, **kwargs: Any) -> Any:
            command = args[0] if args else kwargs.get("keys", "")
            command_text = str(command)
            self._record_event(
                "tool_call_start",
                tool="terminal",
                command=command_text,
            )
            try:
                return original_send_keys(*args, **kwargs)
            finally:
                self._record_event(
                    "tool_call_end",
                    tool="terminal",
                    command=command_text,
                )

        session.send_keys = _send_keys_with_events
        setattr(session, "_ipw_tool_instrumented", True)
        return session

    def run(
        self,
        input: str,
        tmux_session: Any = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Run the Terminus agent.

        Args:
            input: The input message or prompt for the agent.
            tmux_session: Optional TmuxSession or session name.
            **kwargs: Additional keyword arguments passed to agent.perform_task().

        Returns:
            AgentRunResult with the terminal output.
        """
        session = self._instrument_session_tools(self.get_session(tmux_session))
        try:
            self._usage_calls = []
            self.agent.perform_task(input, session=session, **kwargs)

            if not session.is_session_alive():
                raise RuntimeError(
                    "Terminus tmux session ended before terminal output could be captured."
                )
            terminal_output = session.capture_pane(capture_entire=True)
            if "error connecting to /tmp/tmux-" in terminal_output:
                raise RuntimeError(
                    f"Terminus failed to capture tmux pane: {terminal_output.strip()}"
                )
            if self._usage_calls:
                input_tokens = sum(call.get("prompt_tokens", 0) for call in self._usage_calls)
                output_tokens = sum(call.get("completion_tokens", 0) for call in self._usage_calls)
                total_tokens = sum(call.get("total_tokens", 0) for call in self._usage_calls)
                cost_values = [call.get("cost_usd") for call in self._usage_calls]
                cost_usd = (
                    sum(float(value) for value in cost_values)
                    if all(value is not None for value in cost_values)
                    else None
                )
                if total_tokens == 0:
                    total_tokens = input_tokens + output_tokens
                usage_stats = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "num_turns": len(self._usage_calls),
                    "cost": cost_usd,
                    "cost_usd": cost_usd,
                    "token_source": "litellm_response_usage",
                    "missing_usage_responses": 0,
                }
                metadata = {
                    "token_source": "litellm_response_usage",
                    "usage": usage_stats,
                }
            else:
                input_tokens = None
                output_tokens = None
                metadata = {"token_source": "missing"}
            return AgentRunResult(
                content=terminal_output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd if self._usage_calls else None,
                num_turns=len(self._usage_calls),
                metadata=metadata,
            )
        finally:
            self._cleanup_session(session)

    def cleanup(self) -> None:
        """Clean up Docker resources."""
        if self._container is not None and self._owns_container:
            try:
                self._container.stop()
                self._container.remove()
            except Exception:
                pass
            self._container = None

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        self.cleanup()
