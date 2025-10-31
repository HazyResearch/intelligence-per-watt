from __future__ import annotations

import os

import pytest

from trafficbench.clients.ollama import OllamaClient

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def _require_ollama() -> OllamaClient:
    client = OllamaClient(OLLAMA_URL)
    if not client.health():
        pytest.skip(f"Ollama service unavailable at {OLLAMA_URL}")
    return client


def _pick_test_model(client: OllamaClient) -> str:
    configured = os.environ.get("OLLAMA_TEST_MODEL")
    if configured:
        return configured
    models = client.list_models()
    if not models:
        pytest.skip("No models available from Ollama to run live test")
    return models[0]


def test_list_models_live() -> None:
    client = _require_ollama()
    models = client.list_models()
    assert isinstance(models, list)


def test_stream_chat_completion_live() -> None:
    client = _require_ollama()
    model = _pick_test_model(client)
    response = client.stream_chat_completion(model, "Say hello")

    assert response.content
    assert response.usage.total_tokens >= 0
    assert response.time_to_first_token_ms >= 0
