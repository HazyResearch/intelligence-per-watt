from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import requests
from dotenv import load_dotenv
from requests import exceptions as req_exc

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"

load_dotenv()


@ClientRegistry.register("openai")
class OpenAIClient(InferenceClient):
    """
    OpenAI-compatible chat client for evaluation-only use (judge models).

    This client is stateless with respect to secrets:
    - ``base_url`` and ``model`` are passed in construction.
    - API key is read on each request (``IPW_EVAL_API_KEY`` or ``OPENAI_API_KEY``) and never stored; if both differ, the IPW key wins.
    """

    client_id, client_name = "openai", "OpenAI"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host_cfg = base_url or config.get("base_url") or "https://api.openai.com/v1"
        model_cfg = (config.get("model") or DEFAULT_MODEL).strip()

        host = str(host_cfg).rstrip("/")
        super().__init__(host, **config)
        # Model defaults to the recommended judge if unspecified
        self.model = model_cfg
        self.timeout_seconds = float(config.get("timeout_seconds", 60.0))

        # Generation controls are intentionally omitted for judge calls (GPT-5 family ignores them)
        self.temperature = None
        self.max_output_tokens = None

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Profiling is not supported with this client."""
        raise RuntimeError(
            "OpenAIClient is evaluation-only; do not use it for profiling. "
            "Use a profiling client like ollama or vllm instead."
        )

    def list_models(self) -> Sequence[str]:
        raise RuntimeError("OpenAIClient is evaluation-only and cannot list models.")

    def health(self) -> bool:
        raise RuntimeError("OpenAIClient is evaluation-only and does not expose health checks.")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Helper method to mimic the old EvaluationClient interface for easy migration.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        params = {"messages": messages}
        # Deliberately avoid forwarding temperature/max tokens; judge models ignore or reject them.

        response = self._chat_completion(self.model, "", **params)
        return response.content

    def _chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Internal helper used by evaluation-only chat."""
        ipw_key = os.getenv("IPW_EVAL_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if ipw_key and openai_key and ipw_key != openai_key:
            api_key = ipw_key
        else:
            api_key = ipw_key or openai_key

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request_model = model or self.model

        if "messages" in params:
            messages = params["messages"]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {
            "model": request_model,
            "messages": messages,
        }
        # Skip temperature/max tokens to avoid unsupported-parameter errors.

        exclude_keys = {"model", "messages", "prompt"}
        for k, v in params.items():
            if k not in exclude_keys:
                payload[k] = v

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except req_exc.HTTPError as exc:
            detail = ""
            resp = exc.response
            if resp is not None:
                try:
                    parsed = resp.json()
                    detail = f" | body: {parsed}"
                except Exception:
                    body_text = resp.text if resp is not None else ""
                    detail = f" | body: {body_text[:500]}"
            raise RuntimeError(f"OpenAIClient request failed: {exc}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAIClient request failed: {exc}") from exc

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError("OpenAIClient received non-JSON response") from exc

        choices = data.get("choices") or []
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        if not choices:
            return Response(
                content="",
                usage=ChatUsage(prompt_tokens, completion_tokens, total_tokens),
                time_to_first_token_ms=0.0,
            )

        first = choices[0]
        message = first.get("message") or {}
        content = message.get("content") or ""

        return Response(
            content=content,
            usage=ChatUsage(prompt_tokens, completion_tokens, total_tokens),
            time_to_first_token_ms=0.0,
        )


__all__ = ["DEFAULT_MODEL", "OpenAIClient"]
