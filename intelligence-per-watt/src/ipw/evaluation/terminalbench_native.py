"""Test-script evaluation handler for TerminalBench native integration.

Reads pre-computed test results from dataset_metadata (written by
:class:`~ipw.execution.terminalbench_env.TerminalBenchTaskEnv` during its
Docker-based run).  No LLM judge is needed.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler


@EvaluationRegistry.register("terminalbench-native")
class TerminalBenchNativeHandler(EvaluationHandler):
    """Extract pass/fail from test results stored in metadata."""

    evaluation_method = "terminalbench-native"

    def __init__(self, client=None) -> None:
        # Accept but ignore the client — no LLM judging required.
        # Avoid calling super().__init__ which requires a real client.
        self._client = client  # type: ignore[assignment]

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        is_resolved = metadata.get("is_resolved")
        test_results = metadata.get("test_results")

        details: Dict[str, object] = {
            "match_type": "test_script",
            "test_results": test_results,
        }

        if is_resolved is None:
            return None, {**details, "reason": "no_test_results"}

        return bool(is_resolved), details


__all__ = ["TerminalBenchNativeHandler"]
