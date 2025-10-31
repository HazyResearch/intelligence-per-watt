"""vLLM client integration placeholder."""

from __future__ import annotations

from typing import Any, Sequence

from ..core.registry import ClientRegistry
from ..core.types import Response
from .base import InferenceClient


@ClientRegistry.register("vllm")
class VLLMClient(InferenceClient):
    client_id = "vllm"
    client_name = "vLLM"
    DEFAULT_BASE_URL = "http://localhost:8000"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        super().__init__(base_url or self.DEFAULT_BASE_URL, **config)

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        raise NotImplementedError

    def list_models(self) -> Sequence[str]:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError
