from __future__ import annotations

import time
from typing import Any, Dict, MutableMapping, Sequence, Mapping

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

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        options: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        verify: bool | str = True,
        **config: Any,
    ) -> None:
        super().__init__(base_url, **config)
        self._base_url = _normalize_base_url(base_url)
        self._timeout = timeout
        self._default_options: Dict[str, Any] = dict(options or {})
        self._default_headers: Dict[str, str] = dict(headers or {})
        self._client = Client(
            host=self._base_url,
            timeout=timeout,
            headers=self._default_headers or None,
            verify=verify,
        )

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        payload = self._build_payload(model, prompt, params)

        start_time = time.perf_counter()

        try:
            stream = self._client.chat(**payload)
        except (ResponseError, Exception) as exc:
            raise RuntimeError(f"Ollama error: {getattr(exc, 'error', str(exc))}") from exc

        content_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        ttft_ms: float | None = None

        for chunk in stream:
            data = self._coerce_mapping(chunk)

            if error := data.get("error"):
                raise RuntimeError(f"Ollama error: {error}")

            message = data.get("message") or {}
            content_piece = message.get("content")
            if content_piece:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start_time) * 1000
                content_parts.append(content_piece)

            if data.get("done"):
                prompt_tokens = int(data.get("prompt_eval_count") or prompt_tokens)
                completion_tokens = int(data.get("eval_count") or completion_tokens)
                break

        total_tokens = prompt_tokens + completion_tokens
        usage = ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        if ttft_ms is None:
            ttft_ms = 0.0

        return Response(
            content="".join(content_parts),
            usage=usage,
            time_to_first_token_ms=ttft_ms,
        )

    def list_models(self) -> Sequence[str]:
        try:
            data = self._client.list()
        except (ResponseError, Exception) as exc:
            raise RuntimeError(f"Ollama error: {getattr(exc, 'error', str(exc))}") from exc

        models = self._extract_models(data)
        return [model_name for model_name in models if model_name]

    def health(self) -> bool:
        try:
            self._client.list()
            return True
        except (ResponseError, Exception):
            return False

    def _build_payload(
        self,
        model: str,
        prompt: str,
        params: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "stream": True,
        }

        external_messages = params.pop("messages", None)
        if external_messages is not None:
            payload["messages"] = external_messages
        else:
            payload["messages"] = [
                {"role": "user", "content": prompt}
            ]

        merged_options: Dict[str, Any] = dict(self._default_options)
        if "options" in params:
            merged_options.update(params.pop("options") or {})
        if merged_options:
            payload["options"] = merged_options

        if params:
            payload.update(params)

        return payload

    @staticmethod
    def _coerce_mapping(chunk: Any) -> Dict[str, Any]:
        if isinstance(chunk, Mapping):
            return dict(chunk)

        if hasattr(chunk, "model_dump"):
            try:
                return dict(chunk.model_dump())
            except TypeError:
                return chunk.model_dump()

        if hasattr(chunk, "dict"):
            try:
                return dict(chunk.dict())
            except TypeError:
                return chunk.dict()

        try:
            return dict(chunk)
        except TypeError:
            return chunk

    @staticmethod
    def _extract_models(data: Any) -> Sequence[str]:
        if isinstance(data, Mapping):
            models = data.get("models") or []
        else:
            models = getattr(data, "models", [])

        names: list[str] = []
        for model in models:
            if isinstance(model, Mapping):
                name = model.get("model") or model.get("name")
            else:
                name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(name)
        return names

