"""Tests for vLLM client."""

from __future__ import annotations

import pytest

from trafficbench.clients.vllm import VLLMClient
from trafficbench.core.registry import ClientRegistry


class TestVLLMClient:
    """Test VLLMClient implementation."""

    def test_initializes_with_default_url(self) -> None:
        client = VLLMClient()
        assert client.base_url == "http://localhost:8000"

    def test_initializes_with_custom_url(self) -> None:
        client = VLLMClient("http://example.com:8080")
        assert client.base_url == "http://example.com:8080"

    def test_has_correct_client_id(self) -> None:
        client = VLLMClient()
        assert client.client_id == "vllm"

    def test_has_correct_client_name(self) -> None:
        client = VLLMClient()
        assert client.client_name == "vLLM"

    def test_stream_chat_completion_not_implemented(self) -> None:
        client = VLLMClient()
        with pytest.raises(NotImplementedError):
            client.stream_chat_completion("model", "prompt")

    def test_list_models_not_implemented(self) -> None:
        client = VLLMClient()
        with pytest.raises(NotImplementedError):
            client.list_models()

    def test_health_not_implemented(self) -> None:
        client = VLLMClient()
        with pytest.raises(NotImplementedError):
            client.health()

    def test_registered_in_client_registry(self) -> None:
        # VLLMClient should be registered with decorator
        client_cls = ClientRegistry.get("vllm")
        assert client_cls is VLLMClient

    def test_can_be_instantiated_from_registry(self) -> None:
        client = ClientRegistry.create("vllm", "http://test.com")
        assert isinstance(client, VLLMClient)
        assert client.base_url == "http://test.com"

