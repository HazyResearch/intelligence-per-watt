from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)

_PATCH_MARKERS = ("diff --git", "---", "+++", "@@")


@EvaluationRegistry.register("swebench")
class SWEBenchHandler(EvaluationHandler):
    """Structural validation handler for SWE-bench.

    SWE-bench correctness requires running the repository test suite, which is
    out of scope for inline evaluation.  This handler checks whether the agent
    produced something that looks like a patch and records structural metadata
    so downstream tooling can decide whether to attempt test execution.
    """

    evaluation_method = "swebench"

    def __init__(self, client=None) -> None:  # noqa: D107
        # SWE-bench does not use LLM judging; accept but ignore the client.
        self._client = client

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        if not model_answer or not model_answer.strip():
            return False, {"reason": "empty_response"}

        has_patch = any(marker in model_answer for marker in _PATCH_MARKERS)

        return None, {
            "reason": "requires_test_execution",
            "has_patch": has_patch,
            "patch_length": len(model_answer),
            "instance_id": metadata.get("instance_id", ""),
        }
