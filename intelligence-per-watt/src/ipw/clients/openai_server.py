"""OpenAI-compatible profiling client for remote inference servers (vLLM, llama.cpp, etc.)."""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

import requests
from requests import exceptions as req_exc

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient


@ClientRegistry.register("openai-server")
class OpenAIServerClient(InferenceClient):
    """Profiling client for any OpenAI-compatible inference server.

    Unlike the eval-only ``OpenAIClient``, this client supports streaming
    chat completions with TTFT measurement and token usage tracking,
    making it suitable for ``ipw profile`` workloads.
    """

    client_id, client_name = "openai-server", "OpenAI Server"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host = (base_url or config.get("base_url") or "http://localhost:8000/v1").rstrip("/")
        # Ensure URL ends with /v1
        if not host.endswith("/v1"):
            host = f"{host}/v1"
        super().__init__(host, **config)
        self.api_key = config.get("api_key") or "EMPTY"
        self.timeout_seconds = float(config.get("timeout_seconds", 120.0))

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = params.pop("messages", None) or [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        max_tokens = params.pop("max_tokens", None) or params.pop("num_predict", None)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        else:
            payload["max_tokens"] = 4096

        for k, v in params.items():
            if k not in ("model", "messages", "prompt", "stream"):
                payload[k] = v

        start = time.perf_counter()
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
        except req_exc.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                try:
                    detail = f" | body: {exc.response.json()}"
                except Exception:
                    detail = f" | body: {exc.response.text[:500]}"
            raise RuntimeError(f"OpenAI server request failed: {exc}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI server request failed: {exc}") from exc

        content_parts: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        ttft_ms: float | None = None
        token_times: list[float] = []

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    token_times.append(time.perf_counter())
                    if ttft_ms is None:
                        ttft_ms = (token_times[0] - start) * 1000
                    content_parts.append(text)

            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )

        return Response(
            content="".join(content_parts),
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            time_to_first_token_ms=ttft_ms or 0.0,
            token_timestamps=token_times or None,
        )

    def list_models(self) -> Sequence[str]:
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as exc:
            raise RuntimeError(f"Failed to list models: {exc}") from exc

    def health(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        params: dict[str, Any] = {"messages": messages}
        if max_output_tokens:
            params["max_tokens"] = max_output_tokens
        resp = self.stream_chat_completion(
            model=self._config.get("model", ""),
            prompt="",
            **params,
        )
        return resp.content


__all__ = ["OpenAIServerClient"]
