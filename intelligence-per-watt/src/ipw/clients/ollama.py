from __future__ import annotations

import time
from typing import Any, Mapping

from ollama import Client, ResponseError

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient


def _normalize_base_url(base_url: str) -> str:
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


@ClientRegistry.register("ollama")
class OllamaClient(InferenceClient):
    client_id, client_name = "ollama", "Ollama"
    DEFAULT_BASE_URL = "http://127.0.0.1:11434"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host = _normalize_base_url(base_url or self.DEFAULT_BASE_URL)
        super().__init__(host, **config)
        self._client = Client(host=host)

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        payload = self._build_payload(model, prompt, params)
        start = time.perf_counter()
        try:
            stream = self._client.generate(**payload)
        except ResponseError as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc

        content: list[str] = []
        prompt_tokens = completion_tokens = 0
        ttft_ms: float | None = None
        token_times: list[float] = []

        for chunk in stream:
            text = getattr(chunk, "response", None)
            if text:
                token_times.append(time.perf_counter())
                if ttft_ms is None:
                    ttft_ms = (token_times[0] - start) * 1000
                content.append(text)
            if getattr(chunk, "done", False):
                prompt_tokens = int(chunk.prompt_eval_count or prompt_tokens)
                completion_tokens = int(chunk.eval_count or completion_tokens)

        return Response(
            content="".join(content),
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            time_to_first_token_ms=ttft_ms or 0.0,
            token_timestamps=token_times or None,
        )

    def list_models(self) -> list[str]:
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
        except Exception:
            return False

    def _build_payload(
        self, model: str, prompt: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }

        options: dict[str, Any] = {}
        # Normalize/alias max token controls and enforce a default cap.
        if "max_tokens" in params and "num_predict" not in params:
            options["num_predict"] = params.get("max_tokens")
        if "num_predict" in params:
            options["num_predict"] = params["num_predict"]

        # Recognized top-level fields the Ollama client accepts directly.
        passthrough_keys = {
            "system",
            "template",
            "context",
            "format",
            "images",
            "keep_alive",
            "raw",
        }

        for key, value in params.items():
            if key in {"max_tokens", "num_predict"}:
                continue
            if key in passthrough_keys:
                payload[key] = value
            else:
                options[key] = value

        options.setdefault("num_predict", 4096)
        if options:
            payload["options"] = options
        return payload
