"""Terminus-TB agent — calls Terminus2 on a TerminalBench tmux session.

Docker lifecycle and test execution are handled by
:class:`~ipw.execution.terminalbench_env.TerminalBenchTaskEnv` (managed by
the runner).  This agent only needs to call ``Terminus2.perform_task()``
using the session provided in task metadata.
"""

from __future__ import annotations

import logging
import os
import time
import types
from typing import TYPE_CHECKING, Any, MutableMapping, Optional

from ipw.agents.base import BaseAgent
from ipw.agents.openai_usage_proxy import OpenAIUsageProxy
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder

LOGGER = logging.getLogger(__name__)

if AgentRegistry.has("terminus-tb"):
    AgentRegistry._entries().pop("terminus-tb", None)


@AgentRegistry.register("terminus-tb")
class TerminusTB(BaseAgent):
    """TerminalBench agent backed by Terminus2.

    Expects the runner to provide a tmux ``session`` in task metadata
    (populated by :class:`~ipw.execution.terminalbench_env.TerminalBenchTaskEnv`).
    """

    def __init__(
        self,
        model: str,
        event_recorder: Optional["EventRecorder"] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(event_recorder=event_recorder)

        try:
            from terminal_bench.agents.terminus_2 import Terminus2
        except ImportError:
            raise ImportError(
                "terminal-bench package is required for terminus-tb agent. "
                "Install with: pip install terminal-bench"
            )

        # LiteLLM requires a provider prefix.  Only prepend ``openai/`` when the
        # model doesn't already carry a known litellm prefix.
        _LITELLM_PREFIXES = (
            "openai/", "anthropic/", "gemini/", "google/",
            "azure/", "bedrock/", "vertex_ai/", "ollama/",
        )
        if api_base is not None:
            litellm_model = f"openai/{model.removeprefix('openai/')}"
        else:
            litellm_model = model if model.startswith(_LITELLM_PREFIXES) else f"openai/{model}"

        context_limit = kwargs.pop("context_limit", None)
        context_buffer_tokens = int(kwargs.pop("context_buffer_tokens", 512))
        terminal_output_max_bytes = kwargs.pop("terminal_output_max_bytes", None)
        max_turns = kwargs.pop("max_turns", None)

        agent_kwargs: dict[str, Any] = {}
        llm_call_kwargs = dict(kwargs.pop("llm_call_kwargs", None) or {})
        if max_turns is not None:
            kwargs.setdefault("max_episodes", max(1, int(max_turns)))
        self._usage_proxy: OpenAIUsageProxy | None = None
        if api_base is not None:
            os.environ.setdefault("OPENAI_API_KEY", os.getenv("VLLM_API_KEY", "EMPTY"))
            llm_call_kwargs.setdefault(
                "timeout",
                int(os.environ.get("IPW_LLM_TIMEOUT_SECONDS", "120")),
            )
            try:
                self._usage_proxy = OpenAIUsageProxy(
                    api_base,
                    bind_host="127.0.0.1",
                    model=litellm_model,
                    event_callback=self._record_event,
                )
                self._usage_proxy.start()
                agent_kwargs["api_base"] = self._usage_proxy.base_url_for_client(
                    "127.0.0.1"
                )
            except Exception:
                LOGGER.warning("Failed to start Terminus-TB usage proxy", exc_info=True)
                agent_kwargs["api_base"] = api_base
        agent_kwargs.update(kwargs)

        self._terminus = Terminus2(model_name=litellm_model, **agent_kwargs)
        if terminal_output_max_bytes is not None:
            output_limit = int(terminal_output_max_bytes)
            original_limit_output_length = self._terminus._limit_output_length

            def _ipw_limit_output_length(
                _self: Any,
                output: str,
                max_bytes: int = 10000,
            ) -> str:
                return original_limit_output_length(
                    output,
                    max_bytes=min(max_bytes, output_limit),
                )

            self._terminus._limit_output_length = types.MethodType(
                _ipw_limit_output_length,
                self._terminus,
            )

        if context_limit is not None:
            effective_context_limit = max(1, int(context_limit) - context_buffer_tokens)

            def _forced_context_limit(_self: Any) -> int:
                return effective_context_limit

            self._terminus._get_model_context_limit = types.MethodType(
                _forced_context_limit,
                self._terminus,
            )

        self._configure_llm_call_kwargs(llm_call_kwargs)
        self._model_name = model
        self._litellm_model = litellm_model
        self._records_lm_calls = self._usage_proxy is not None
        self._task_metadata: Optional[MutableMapping[str, Any]] = None

    def _configure_llm_call_kwargs(self, llm_call_kwargs: dict[str, Any]) -> None:
        """Apply default LiteLLM call kwargs that Terminus2 does not expose."""
        if not llm_call_kwargs:
            return
        llm = getattr(self._terminus, "_llm", None)
        original_call = getattr(llm, "call", None)
        if not callable(original_call) or getattr(llm, "_ipw_call_kwargs_wrapped", False):
            return

        def _call_with_ipw_kwargs(*args: Any, **call_kwargs: Any) -> Any:
            return original_call(*args, **{**llm_call_kwargs, **call_kwargs})

        setattr(llm, "call", _call_with_ipw_kwargs)
        setattr(llm, "_ipw_call_kwargs_wrapped", True)
        setattr(llm, "_ipw_call_kwargs", dict(llm_call_kwargs))

    # ------------------------------------------------------------------
    # Metadata hook
    # ------------------------------------------------------------------

    def set_task_metadata(self, metadata: MutableMapping[str, Any]) -> None:
        self._task_metadata = metadata

    def _instrument_session_tools(self, session: Any) -> Any:
        """Record TerminalBench tmux interactions as tool-call events."""
        if getattr(session, "_ipw_tool_instrumented", False) is True:
            return session
        original_send_keys = getattr(session, "send_keys", None)
        if not callable(original_send_keys):
            return session

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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        """Run Terminus2 on the tmux session from task metadata."""
        if self._task_metadata is None:
            raise RuntimeError(
                "set_task_metadata() must be called before run(). "
                "Ensure the dataset is 'terminalbench-native'."
            )

        session = self._task_metadata.get("session")
        if session is None:
            raise RuntimeError(
                "No 'session' in task metadata. "
                "Ensure the dataset provides a TerminalBenchTaskEnv."
            )
        session = self._instrument_session_tools(session)

        task = self._task_metadata["task"]
        task_id = self._task_metadata.get("task_id", "unknown")
        timeout = task.max_agent_timeout_sec

        terminal_output = ""
        usage_stats: dict[str, Any] = {}
        if self._usage_proxy is not None:
            self._usage_proxy.reset()

        if not self._records_lm_calls:
            self._record_event("lm_inference_start", model=self._model_name)
        try:
            LOGGER.info("Running Terminus2 on task %s", task_id)
            start = time.time()
            try:
                self._terminus.perform_task(
                    task.instruction,
                    session=session,
                    time_limit_seconds=timeout,
                )
            except Exception:
                LOGGER.exception("Terminus2 agent failed on task %s", task_id)

            elapsed = time.time() - start
            LOGGER.info("Agent finished in %.1fs", elapsed)

            # Capture terminal output
            try:
                terminal_output = session.capture_pane(capture_entire=True)
            except Exception:
                LOGGER.warning("Failed to capture terminal output")

            # Include actual proxy token usage if available. Do not use
            # terminal-bench's local count_tokens fallback for trace metrics.
            if self._usage_proxy is not None:
                usage_stats = self._usage_proxy.snapshot()
            if usage_stats.get("token_source") == "openai_api_usage":
                self._task_metadata.setdefault("test_results", {})
                self._task_metadata["test_results"]["total_input_tokens"] = int(
                    usage_stats.get("input_tokens") or 0
                )
                self._task_metadata["test_results"]["total_output_tokens"] = int(
                    usage_stats.get("output_tokens") or 0
                )
        finally:
            if not self._records_lm_calls:
                self._record_event("lm_inference_end", model=self._model_name)

        if self._usage_proxy is not None and not usage_stats:
            usage_stats = self._usage_proxy.snapshot()

        token_source = "missing"
        if usage_stats.get("token_source") == "openai_api_usage":
            input_tokens = int(usage_stats.get("input_tokens") or 0)
            output_tokens = int(usage_stats.get("output_tokens") or 0)
            token_source = "openai_api_usage"
        else:
            input_tokens = None
            output_tokens = None

        cost = None
        if input_tokens is not None and output_tokens is not None:
            try:
                from ipw.cost.pricing import calculate_cost

                provider, _, model_id = self._model_name.partition("/")
                cost = calculate_cost(provider, model_id, input_tokens, output_tokens)
            except Exception:
                LOGGER.debug("Cost calculation failed for %s", self._model_name, exc_info=True)

        return AgentRunResult(
            content=terminal_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            metadata={
                "task_id": task_id,
                "token_source": token_source,
                "usage_proxy": usage_stats,
            },
        )


__all__ = ["TerminusTB"]
