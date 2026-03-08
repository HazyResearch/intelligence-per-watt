"""Terminus-TB agent — calls Terminus2 on a TerminalBench tmux session.

Docker lifecycle and test execution are handled by
:class:`~ipw.execution.terminalbench_env.TerminalBenchTaskEnv` (managed by
the runner).  This agent only needs to call ``Terminus2.perform_task()``
using the session provided in task metadata.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, MutableMapping, Optional

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder

LOGGER = logging.getLogger(__name__)


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

        agent_kwargs: dict[str, Any] = {}
        if api_base is not None:
            agent_kwargs["api_base"] = api_base
        agent_kwargs.update(kwargs)

        # LiteLLM requires an ``openai/`` prefix for OpenAI-compatible endpoints.
        # Always prepend so LiteLLM strips it back to the original HF model ID.
        litellm_model = f"openai/{model}"
        self._terminus = Terminus2(model_name=litellm_model, **agent_kwargs)
        self._model_name = model
        self._task_metadata: Optional[MutableMapping[str, Any]] = None

    # ------------------------------------------------------------------
    # Metadata hook
    # ------------------------------------------------------------------

    def set_task_metadata(self, metadata: MutableMapping[str, Any]) -> None:
        self._task_metadata = metadata

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

        task = self._task_metadata["task"]
        task_id = self._task_metadata.get("task_id", "unknown")
        timeout = task.max_agent_timeout_sec

        terminal_output = ""

        self._record_event("lm_inference_start", model=self._model_name)
        try:
            LOGGER.info("Running Terminus2 on task %s", task_id)
            start = time.time()
            agent_result = None
            try:
                agent_result = self._terminus.perform_task(
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

            # Include agent token usage if available
            if agent_result is not None:
                self._task_metadata.setdefault("test_results", {})
                self._task_metadata["test_results"]["total_input_tokens"] = getattr(
                    agent_result, "total_input_tokens", 0
                )
                self._task_metadata["test_results"]["total_output_tokens"] = getattr(
                    agent_result, "total_output_tokens", 0
                )
        finally:
            self._record_event("lm_inference_end", model=self._model_name)

        input_tokens = getattr(agent_result, "total_input_tokens", 0) if agent_result else 0
        output_tokens = getattr(agent_result, "total_output_tokens", 0) if agent_result else 0

        return AgentRunResult(
            content=terminal_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "task_id": task_id,
            },
        )


__all__ = ["TerminusTB"]
