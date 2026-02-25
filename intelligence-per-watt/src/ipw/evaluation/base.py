from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from ..clients.base import InferenceClient

LOGGER = logging.getLogger(__name__)


class EvaluationHandler(ABC):
    """Base class for per-dataset evaluation strategies."""

    evaluation_method: str

    def __init__(self, client: InferenceClient) -> None:
        # Handlers require a client for LLM-based judging
        self._client = client

    @abstractmethod
    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        """
        Evaluate a single model answer.

        Returns:
            A tuple of (is_correct, metadata):

            - is_correct: True/False if a decision could be made,
                or None if the example is not scorable.
            - metadata: method-specific payload (e.g., extracted
                answers, judge explanation, or reasons for being unscorable).
        """
