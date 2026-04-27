from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from ..core.types import Response


class InferenceClient(ABC):
    """
    Base class for inference service integrations.

    Subclasses must be registered with ``ClientRegistry`` to become discoverable.
    """

    client_id: str
    client_name: str

    def __init__(self, base_url: str, **config: Any) -> None:
        self.base_url = base_url
        self._config = config

    @abstractmethod
    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Run a streamed chat completion and return the aggregated response.

        Implementations should consume a streaming API so they can measure
        time-to-first-token latency, but must return a fully materialized
        ``Response`` object once the stream finishes. ``prompt`` contains the
        raw text to submit to the inference service.
        """

    def batch_stream_chat_completion(
        self, model: str, prompts: list[str], **params: Any
    ) -> list["Response"]:
        """Run multiple completions concurrently via thread pool."""
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [
                pool.submit(self.stream_chat_completion, model, p, **params)
                for p in prompts
            ]
            return [f.result() for f in futures]

    def configure_batch_size(self, batch_size: int) -> None:
        """Configure engine for batch inference. Override in subclasses."""
        pass

    @abstractmethod
    def list_models(self) -> Sequence[str]:
        """Return the list of models exposed by the client."""

    @abstractmethod
    def health(self) -> bool:
        """Return True when the client is healthy and reachable."""

    def prepare(self, model: str) -> None:
        """Optional hook to perform warmup before serving requests."""
        return None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """
        Synchronous chat completion helper.

        Implementations should return the generated text for the given prompts.
        Subclasses that don't support chat may rely on this default, which
        raises to signal the capability is unavailable.
        """
        raise NotImplementedError


__all__ = ["InferenceClient"]
