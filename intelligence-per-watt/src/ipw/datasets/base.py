from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Dict, Iterable, Iterator, Optional, Tuple

from ..clients.base import InferenceClient
from ..core.types import DatasetRecord


class DatasetProvider(ABC):
    """Base interface for providing prompts to the profiler."""

    dataset_id: str
    dataset_name: str

    # Preferred evaluation settings (datasets may override)
    eval_client: str | None = "openai"
    eval_base_url: str | None = "https://api.openai.com/v1"
    eval_model: str | None = "gpt-5-nano-2025-08-07"

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.iter_records())

    @abstractmethod
    def iter_records(self) -> Iterable[DatasetRecord]:
        """Yield dataset records in the order they should be executed."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of records."""

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client: Optional[InferenceClient] = None,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        """
        Compute correctness for a single model response.

        Args:
            record: The dataset record containing problem and reference answer
            response: The model's response to evaluate
            eval_client: Optional inference client to use for LLM-based judging

        Returns:
            (is_correct, metadata) tuple where:
            - is_correct: True/False if scored, None if unscorable
            - metadata: method-specific evaluation details
        """
        raise NotImplementedError("score() is not implemented for this dataset")

    def create_task_env(
        self, record: DatasetRecord
    ) -> Optional[AbstractContextManager]:
        """Return a per-task execution environment, or ``None``.

        Override in datasets that need a managed environment (e.g. Docker
        container) for each task.  The runner will use the returned context
        manager to wrap the agent call::

            with dataset.create_task_env(record):
                agent.run(record.problem)
        """
        return None

    def verify_requirements(self) -> list[str]:
        """
        Return a list of unmet requirements for this dataset (e.g., missing env vars).
        An empty list means all required preconditions are satisfied.
        """
        return []


__all__ = ["DatasetProvider"]
