from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import httpx

from ..core.client import InferenceClient
from ..core.registry import ClientRegistry


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
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            headers=self._default_headers,
            verify=verify,
        )

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Iterable[Mapping[str, Any]]:
        request_id = params.pop("request_id", f"chatcmpl-{uuid.uuid4().hex}")
        payload = self._build_payload(model, prompt, params)
        created = int(time.time())
        url = "/api/chat"

        with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            prompt_tokens = 0
            completion_tokens = 0

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue

                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                if error := data.get("error"):
                    raise RuntimeError(f"Ollama error: {error}")

                if data.get("done"):
                    prompt_tokens = int(data.get("prompt_eval_count") or prompt_tokens)
                    completion_tokens = int(data.get("eval_count") or completion_tokens)
                    total_tokens = prompt_tokens + completion_tokens
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                    yield self._build_chunk(
                        request_id=request_id,
                        created=created,
                        model=model,
                        content=None,
                        finish_reason=data.get("done_reason", "stop"),
                        usage=usage,
                    )
                    break

                message = data.get("message") or {}
                content_piece = message.get("content")
                if content_piece:
                    yield self._build_chunk(
                        request_id=request_id,
                        created=created,
                        model=model,
                        content=content_piece,
                        finish_reason=None,
                    )

    def list_models(self) -> Sequence[str]:
        response = self._client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        models = data.get("models") or []
        return [model_info.get("name", "") for model_info in models if model_info.get("name")]

    def health(self) -> bool:
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
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
    def _build_chunk(
        *,
        request_id: str,
        created: int,
        model: str,
        content: str | None,
        finish_reason: str | None,
        usage: Mapping[str, int] | None = None,
    ) -> Dict[str, Any]:
        delta = {"content": content} if content is not None else {}
        chunk: Dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            chunk["usage"] = usage
        return chunk
