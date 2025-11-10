"""Tests for the offline vLLM client."""

from __future__ import annotations

import types
from typing import Any

import pytest

from trafficbench.clients.vllm import VLLMClient
from trafficbench.core.registry import ClientRegistry


class DummySamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class DummyAsyncEngineArgs:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class DummyRequestOutputKind:
    DELTA = object()


class DummyAsyncLLM:
    next_outputs: list[list[Any]] = []
    instances: list["DummyAsyncLLM"] = []

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.shutdown_called = False
        self.args = None

    @classmethod
    def from_engine_args(cls, args: Any) -> "DummyAsyncLLM":
        instance = cls()
        instance.args = args
        cls.instances.append(instance)
        return instance

    async def generate(self, **kwargs: Any):
        self.calls.append(kwargs)
        outputs = self.next_outputs.pop(0) if self.next_outputs else []
        for chunk in outputs:
            yield chunk

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture(autouse=True)
def _patch_vllm(monkeypatch: pytest.MonkeyPatch):
    from trafficbench.clients import vllm as module

    DummyAsyncLLM.instances = []
    DummyAsyncLLM.next_outputs = []

    monkeypatch.setattr(module, "SamplingParams", DummySamplingParams, raising=False)
    monkeypatch.setattr(module, "AsyncEngineArgs", DummyAsyncEngineArgs, raising=False)
    monkeypatch.setattr(module, "RequestOutputKind", DummyRequestOutputKind, raising=False)
    monkeypatch.setattr(module, "AsyncLLM", DummyAsyncLLM, raising=False)
    monkeypatch.setattr(module, "_VLLM_IMPORT_ERROR", None, raising=False)


@pytest.fixture(autouse=True)
def _fake_perf_counter(monkeypatch: pytest.MonkeyPatch):
    current = {"value": 0.0}

    def fake_perf_counter() -> float:
        val = current["value"]
        current["value"] += 0.1
        return val

    monkeypatch.setattr("trafficbench.clients.vllm.time.perf_counter", fake_perf_counter)


def _make_chunk(text: str, finished: bool, prompt_tokens: int = 0, new_tokens: int = 0):
    completion = types.SimpleNamespace(text=text, token_ids=[0] * new_tokens)
    prompt_ids = [0] * prompt_tokens if prompt_tokens else []
    return types.SimpleNamespace(
        outputs=[completion],
        prompt_token_ids=prompt_ids,
        finished=finished,
    )


def _queue_warmup_outputs(count: int) -> None:
    for _ in range(count):
        DummyAsyncLLM.next_outputs.append([_make_chunk("", finished=True)])


def test_stream_chat_completion_accumulates_tokens() -> None:
    client = VLLMClient()
    warmups = client._warmup_count  # type: ignore[attr-defined]
    _queue_warmup_outputs(warmups)
    DummyAsyncLLM.next_outputs.append(
        [
            _make_chunk("Hello", finished=False, prompt_tokens=2, new_tokens=1),
            _make_chunk(" world", finished=True, prompt_tokens=0, new_tokens=2),
        ]
    )
    try:
        response = client.stream_chat_completion("meta-llama/Llama", "Prompt text")
    finally:
        client.close()

    assert response.content == "Hello world"
    assert response.usage.prompt_tokens == 2
    assert response.usage.completion_tokens == 3
    assert response.time_to_first_token_ms == pytest.approx(100.0)


def test_sampling_params_and_engine_overrides() -> None:
    client = VLLMClient()
    warmups = client._warmup_count  # type: ignore[attr-defined]
    _queue_warmup_outputs(warmups)
    client.prepare("meta")
    DummyAsyncLLM.next_outputs.append([_make_chunk("Done", finished=True, prompt_tokens=1, new_tokens=1)])
    try:
        response = client.stream_chat_completion(
            "meta",
            "Prompt",
            temperature="0.4",
            sampling_top_p="0.92",
            sampling_max_tokens="32",
            sampling_stop="</s>",
            stop="[END]",
        )
    finally:
        client.close()

    assert response.content == "Done"
    engine_args = client._engine_args.kwargs  # type: ignore[attr-defined]
    assert engine_args.get("enforce_eager") is None

    engine = DummyAsyncLLM.instances[0]
    sampling = engine.calls[-1]["sampling_params"].kwargs
    assert sampling["max_tokens"] == 32
    assert sampling["temperature"] == 0.4
    assert sampling["top_p"] == 0.92
    assert sampling["stop"] == ["[END]"]
    assert sampling["output_kind"] is DummyRequestOutputKind.DELTA


def test_registry_entry_points() -> None:
    client_cls = ClientRegistry.get("vllm")
    assert client_cls is VLLMClient

    client = ClientRegistry.create("vllm", None)
    warmups = client._warmup_count  # type: ignore[attr-defined]
    _queue_warmup_outputs(warmups)
    DummyAsyncLLM.next_outputs.append([_make_chunk("hi", finished=True, new_tokens=1)])
    try:
        assert isinstance(client, VLLMClient)
        assert client.base_url == "offline://vllm"
        client.stream_chat_completion("foo", "Prompt")
    finally:
        client.close()


def test_prepare_runs_warmup_once() -> None:
    client = VLLMClient()
    warmups = client._warmup_count  # type: ignore[attr-defined]
    _queue_warmup_outputs(warmups)

    client.prepare("meta")
    engine = DummyAsyncLLM.instances[0]
    assert len(engine.calls) == warmups

    DummyAsyncLLM.next_outputs.append([_make_chunk("real", finished=True, new_tokens=1)])
    client.stream_chat_completion("meta", "Prompt")
    assert len(engine.calls) == warmups + 1

    client.close()
