"""Live tests for the offline vLLM client.

These tests exercise the real vLLM dependency—similar to the Ollama tests—so they
only run when both the optional extra is installed and the ``VLLM_TEST_MODEL``
environment variable is set. Without that hint the module is skipped."""

from __future__ import annotations

import os
from typing import cast

import pytest

pytest.importorskip(
    "vllm",
    reason=(
        "vLLM tests require the optional 'vllm' extra. Install via "
        "`uv pip install -e 'intelligence-per-watt[vllm]'`."
    ),
)

VLLM_MODEL = os.environ.get("VLLM_TEST_MODEL")
if not VLLM_MODEL:
    pytest.skip(
        "Set VLLM_TEST_MODEL to run the vLLM live tests (e.g., a HF model name).",
        allow_module_level=True,
    )

MODEL = cast(str, VLLM_MODEL)

from ipw.clients.vllm import VLLMClient  # noqa: E402


@pytest.fixture(scope="module")
def vllm_client():
    client = VLLMClient()
    try:
        client.prepare(MODEL)
        yield client
    finally:  # pragma: no branch - cleanup guard
        client.close()


def test_list_models_live(vllm_client):
    models = vllm_client.list_models()

    assert isinstance(models, list)
    assert MODEL in models


def test_stream_chat_completion_live(vllm_client):
    prompt = os.environ.get(
        "VLLM_TEST_PROMPT",
        "Say hello from the Intelligence Per Watt vLLM integration test.",
    )

    response = vllm_client.stream_chat_completion(MODEL, prompt)

    assert response.content.strip()
    assert response.usage.total_tokens > 0
    assert response.time_to_first_token_ms >= 0
