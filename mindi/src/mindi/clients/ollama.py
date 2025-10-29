from __future__ import annotations

import time
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from ollama import Client, ResponseError

from ..core.client import InferenceClient
from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response


def _normalize_base_url(base_url: str) -> str:
    if not base_url.startswith(("http://", "https://")):
        return f"http://{base_url.rstrip('/')}"
    return base_url.rstrip("/")


@ClientRegistry.register("ollama")
class OllamaClient(InferenceClient):
    client_id = "ollama"
    client_name = "Ollama"
    DEFAULT_BASE_URL = "http://127.0.0.1:11434"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 60.0,
        options: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        verify: bool | str = True,
        **config: Any,
    ) -> None:
        resolved = _normalize_base_url(base_url or self.DEFAULT_BASE_URL)
        super().__init__(resolved, **config)
        self._default_options = self._as_plain_dict(options)
        self._client = Client(
            host=resolved,
            timeout=timeout,
            headers=dict(headers) if headers else None,
            verify=verify,
        )

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        payload = self._build_payload(model, prompt, params)

        start_time = time.perf_counter()

        try:
            stream = self._client.chat(**payload)
        except ResponseError as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc

        content_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        ttft_ms: float | None = None

        for chunk in stream:
            message = getattr(chunk, "message", None)
            content_piece = getattr(message, "content", None)
            if content_piece:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start_time) * 1000
                content_parts.append(content_piece)

            if getattr(chunk, "done", False):
                if chunk.prompt_eval_count is not None:
                    prompt_tokens = int(chunk.prompt_eval_count)
                if chunk.eval_count is not None:
                    completion_tokens = int(chunk.eval_count)

        if ttft_ms is None:
            ttft_ms = 0.0

        return Response(
            content="".join(content_parts),
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            time_to_first_token_ms=ttft_ms,
        )

    def list_models(self) -> Sequence[str]:
        try:
            response = self._client.list()
        except ResponseError as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc

        return [
            str(model.model)
            for model in response.models
            if getattr(model, "model", None)
        ]

    def health(self) -> bool:
        try:
            self._client.list()
            return True
        except ResponseError:
            return False

    def _build_payload(
        self,
        model: str,
        prompt: str,
        params: MutableMapping[str, Any],
    ) -> dict[str, Any]:
        payload = dict(params)
        payload.pop("stream", None)
        payload["model"] = model
        payload["stream"] = True

        messages = payload.pop("messages", None)
        if messages is None:
            payload["messages"] = [{"role": "user", "content": prompt}]
        elif isinstance(messages, (str, bytes)):
            payload["messages"] = [{"role": "user", "content": str(messages)}]
        else:
            payload["messages"] = messages

        options = dict(self._default_options)
        override = payload.pop("options", None)
        if override:
            options.update(self._as_plain_dict(override))
        options = {k: v for k, v in options.items() if v is not None}
        if options:
            payload["options"] = options

        return payload

    @staticmethod
    def _as_plain_dict(value: Mapping[str, Any] | Any | None) -> dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "model_dump"):
            return {
                k: v
                for k, v in value.model_dump(exclude_none=True, exclude_unset=True).items()
            }
        if isinstance(value, Mapping):
            return dict(value)
        return dict(value)
