"""Tests for clients/openai_chat_adapter.py."""

from __future__ import annotations

import asyncio
import os

import pytest

from ipw.clients.openai_chat_adapter import OpenAIChatAdapter


class TestOpenAIChatAdapter:
    def test_constructor_stores_model(self) -> None:
        adapter = OpenAIChatAdapter(model="gpt-4o-mini")
        assert adapter.model == "gpt-4o-mini"

    def test_constructor_accepts_api_key_and_base_url(self) -> None:
        adapter = OpenAIChatAdapter(
            model="gpt-4o-mini",
            api_key="sk-test-key",
            base_url="http://localhost:8000/v1",
        )
        assert adapter.model == "gpt-4o-mini"
        assert adapter._api_key == "sk-test-key"
        assert adapter._base_url == "http://localhost:8000/v1"

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                        reason="requires OPENAI_API_KEY")
    @pytest.mark.integration
    def test_simple_completion(self) -> None:
        adapter = OpenAIChatAdapter(model="gpt-4o-mini")
        text = asyncio.run(adapter.complete([
            {"role": "user", "content": "Reply with only the word OK."}
        ]))
        assert "OK" in text or "ok" in text.lower()
