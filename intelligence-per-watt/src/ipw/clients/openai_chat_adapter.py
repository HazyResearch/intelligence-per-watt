"""Thin OpenAI chat-completions adapter for NativeReact's LM contract.

NativeReact expects an async `complete(messages, **kwargs) -> str` callable.
This adapter wraps openai.AsyncOpenAI so a real OpenAI client and the test
_StubLLM are interchangeable.

Strips the IPW-convention "openai/" model-id prefix before calling the SDK
(the prefix is a routing hint for IPW's _is_cloud_model; OpenAI itself
expects the bare id).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class OpenAIChatAdapter:
    """Async OpenAI chat-completions client matching NativeReact's LM contract."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url
        self._client = None  # lazy init

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs: Dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        client = self._ensure_client()
        # Strip openai/ prefix for cloud API calls (Agno-style prefix; OpenAI's
        # native API expects bare model id like 'gpt-4o-mini').
        model = kwargs.pop("model", self.model)
        if model.startswith("openai/"):
            model = model[len("openai/"):]
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""


__all__ = ["OpenAIChatAdapter"]
