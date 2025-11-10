"""Offline vLLM client backed by AsyncLLM."""

from __future__ import annotations

import asyncio
import atexit
import json
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any, Sequence

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient

SamplingParams = None  # type: ignore[assignment]
AsyncEngineArgs = None  # type: ignore[assignment]
RequestOutputKind = None  # type: ignore[assignment]
AsyncLLM = None  # type: ignore[assignment]
_VLLM_IMPORT_ERROR: Exception | None = None


DEFAULT_WARMUP_COUNT = 10
DEFAULT_WARMUP_MAX_TOKENS = 8
_WARMUP_PROMPTS = (
    "This is a warmup prompt.",
    "Hello from the vLLM warmup.",
    "TrafficBench warmup query.",
)


class _AsyncLoopRunner:
    """Run an asyncio event loop in a background thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="trafficbench-vllm", daemon=True)
        self._thread.start()

    def run(self, coro) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def shutdown(self) -> None:
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)
            self._loop.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


def _ensure_vllm_available() -> None:
    global SamplingParams, AsyncEngineArgs, RequestOutputKind, AsyncLLM, _VLLM_IMPORT_ERROR
    if None not in {AsyncLLM, AsyncEngineArgs, SamplingParams, RequestOutputKind}:
        return
    if _VLLM_IMPORT_ERROR is not None:
        raise RuntimeError("Install the 'vllm' package to use the offline vLLM client.") from _VLLM_IMPORT_ERROR
    try:  # pragma: no cover - exercised via monkeypatched fallbacks in tests
        from vllm import SamplingParams as _SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs as _AsyncEngineArgs
        from vllm.sampling_params import RequestOutputKind as _RequestOutputKind
        from vllm.v1.engine.async_llm import AsyncLLM as _AsyncLLM
    except Exception as exc:  # pragma: no cover - import guarded for optional dependency
        _VLLM_IMPORT_ERROR = exc
        raise RuntimeError("Install the 'vllm' package to use the offline vLLM client.") from exc
    SamplingParams = _SamplingParams  # type: ignore[assignment]
    AsyncEngineArgs = _AsyncEngineArgs  # type: ignore[assignment]
    RequestOutputKind = _RequestOutputKind  # type: ignore[assignment]
    AsyncLLM = _AsyncLLM  # type: ignore[assignment]


@ClientRegistry.register("vllm")
class VLLMClient(InferenceClient):
    """Offline AsyncLLM client."""

    client_id = "vllm"
    client_name = "vLLM Offline"
    DEFAULT_BASE_URL = "offline://vllm"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        super().__init__(base_url or self.DEFAULT_BASE_URL, **config)
        _ensure_vllm_available()

        self._engine_kwargs: dict[str, Any] = {}
        self._sampling_defaults: dict[str, Any] = {"max_tokens": 4096}
        self._warmup_count = DEFAULT_WARMUP_COUNT
        self._warmup_max_tokens = DEFAULT_WARMUP_MAX_TOKENS
        self._warmup_done = False
        self._engine = None
        self._engine_args = None
        self._model_name = None
        self._loop_runner: _AsyncLoopRunner | None = _AsyncLoopRunner()
        self._closed = False
        atexit.register(self.close)

    def prepare(self, model: str) -> None:
        if self._closed:
            raise RuntimeError("vLLM client has been closed")
        self._ensure_engine(model)
        self._warmup_if_needed()

    def stream_chat_completion(self, model: str, prompt: str, **params: Any) -> Response:
        if self._closed:
            raise RuntimeError("vLLM client has been closed")
        self._ensure_engine(model)
        self._warmup_if_needed()

        sampling_params = self._build_sampling_params(params)
        request_id = str(params.get("request_id", uuid.uuid4()))
        runner = self._loop_runner
        if runner is None:
            raise RuntimeError("vLLM client is shut down")
        return runner.run(
            self._stream_response(prompt=prompt, request_id=request_id, sampling_params=sampling_params)
        )

    def list_models(self) -> Sequence[str]:
        return [self._model_name] if self._model_name else []

    def health(self) -> bool:
        return not self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._engine is not None:
                self._engine.shutdown()  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - shutdown best-effort
            pass
        finally:
            self._engine = None
            if self._loop_runner is not None:
                self._loop_runner.shutdown()
                self._loop_runner = None

    def _ensure_engine(self, model: str) -> None:
        if not model:
            raise ValueError("model name is required")
        if self._engine is not None:
            if model != self._model_name:
                raise RuntimeError(
                    f"vLLM client already loaded model '{self._model_name}', cannot switch to '{model}'"
                )
            return

        kwargs = dict(self._engine_kwargs)
        kwargs["model"] = model
        try:
            self._engine_args = AsyncEngineArgs(**kwargs)  # type: ignore[arg-type]
            self._engine = AsyncLLM.from_engine_args(self._engine_args)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - forwarded to caller
            raise RuntimeError(f"Failed to initialize vLLM engine: {exc}") from exc
        self._model_name = model

    def _warmup_if_needed(self) -> None:
        if self._warmup_done or self._warmup_count <= 0:
            return
        runner = self._loop_runner
        if runner is None:
            raise RuntimeError("vLLM client is shut down")

        prompts = _WARMUP_PROMPTS or ("Warmup prompt",)
        sampling = SamplingParams(  # type: ignore[call-arg]
            max_tokens=self._warmup_max_tokens,
            temperature=0.0,
            top_p=1.0,
            output_kind=RequestOutputKind.DELTA,  # type: ignore[index]
        )

        for idx in range(self._warmup_count):
            prompt = prompts[idx % len(prompts)]
            request_id = f"warmup-{idx}-{uuid.uuid4()}"
            runner.run(
                self._stream_response(prompt=prompt, request_id=request_id, sampling_params=sampling)
            )

        self._warmup_done = True

    def _build_sampling_params(self, params: Mapping[str, Any]):
        recognized = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "stop",
            "seed",
            "best_of",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "length_penalty",
        }

        def _coerce(value: Any) -> Any:
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return text
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return value

        overrides: dict[str, Any] = {}
        for key, value in params.items():
            if key.startswith("sampling_"):
                overrides[key.split("_", 1)[1]] = _coerce(value)
            elif key in recognized:
                overrides[key] = _coerce(value)

        sampling = {**self._sampling_defaults, **overrides}
        if "stop" in sampling:
            stop_value = sampling["stop"]
            if isinstance(stop_value, str):
                sampling["stop"] = [stop_value]
            elif isinstance(stop_value, (list, tuple)):
                sampling["stop"] = list(stop_value)
        sampling["output_kind"] = RequestOutputKind.DELTA  # type: ignore[index]
        return SamplingParams(**sampling)  # type: ignore[call-arg]

        

    async def _stream_response(self, *, prompt: str, request_id: str, sampling_params: Any) -> Response:
        if self._engine is None:
            raise RuntimeError("vLLM engine is not initialized")

        start_time = time.perf_counter()
        prompt_tokens: int | None = None
        ttft_ms: float | None = None
        completion_states: dict[int, dict[str, Any]] = {}
        primary_index: int | None = None
        primary_completion_tokens = 0

        def _get_state(index: int) -> dict[str, Any]:
            state = completion_states.get(index)
            if state is None:
                state = {"text": "", "token_history": []}
                completion_states[index] = state
            return state

        def _consume_delta_text(completion: Any, index: int, new_tokens: int, incremental: bool) -> str:
            state = _get_state(index)
            prev_text: str = state["text"]
            delta = getattr(completion, "delta_text", None)
            if delta is None:
                delta = getattr(completion, "text_delta", None)
            if delta:
                state["text"] = prev_text + delta
                return delta

            raw_text = getattr(completion, "text", "") or ""
            if not raw_text:
                return ""

            if raw_text.startswith(prev_text):
                new_text = raw_text[len(prev_text) :]
                state["text"] = raw_text
                return new_text

            if not prev_text:
                state["text"] = raw_text
                return raw_text

            if incremental:
                state["text"] = prev_text + raw_text
                return raw_text

            if prev_text.startswith(raw_text):
                # The model may have truncated its hypothesis; keep accumulated text.
                return ""

            # Fallback: treat the payload as a replacement rather than a delta.
            state["text"] = raw_text
            return raw_text

        def _consume_new_token_count(completion: Any, index: int) -> tuple[int, bool]:
            state = _get_state(index)
            history: list[Any] = state.setdefault("token_history", [])

            delta_token_ids = getattr(completion, "delta_token_ids", None)
            if delta_token_ids is None:
                delta_token_ids = getattr(completion, "token_ids_delta", None)
            if delta_token_ids:
                history.extend(delta_token_ids)
                return len(delta_token_ids), True

            token_ids = getattr(completion, "token_ids", None)
            if not token_ids:
                return 0, False

            prefix_len = len(history)
            if prefix_len and len(token_ids) >= prefix_len and token_ids[:prefix_len] == history:
                new_tokens = token_ids[prefix_len:]
                history[:] = token_ids
                return len(new_tokens), bool(new_tokens)

            if not prefix_len:
                history[:] = token_ids
                return len(token_ids), bool(token_ids)

            # Treat as incremental payload containing only the latest tokens.
            history.extend(token_ids)
            return len(token_ids), bool(token_ids)

        try:
            async for chunk in self._engine.generate(  # type: ignore[func-returns-value]
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            ):
                if prompt_tokens is None:
                    prompt_ids = getattr(chunk, "prompt_token_ids", None) or []
                    prompt_tokens = len(prompt_ids)

                outputs = getattr(chunk, "outputs", []) or []
                if outputs and primary_index is None:
                    primary_index = min(getattr(output, "index", idx) for idx, output in enumerate(outputs))

                for idx, completion in enumerate(outputs):
                    completion_index = getattr(completion, "index", idx)
                    new_tokens, is_incremental = _consume_new_token_count(completion, completion_index)
                    new_text = _consume_delta_text(
                        completion, completion_index, new_tokens, is_incremental
                    )

                    if (
                        ttft_ms is None
                        and primary_index is not None
                        and completion_index == primary_index
                        and (new_text or new_tokens)
                    ):
                        ttft_ms = (time.perf_counter() - start_time) * 1000.0

                    if primary_index is not None and completion_index == primary_index:
                        primary_completion_tokens += new_tokens

                if getattr(chunk, "finished", False):
                    break
        except Exception as exc:  # pragma: no cover - actual streaming exercised in integration
            raise RuntimeError(f"vLLM offline generation failed: {exc}") from exc

        if primary_index is None and completion_states:
            primary_index = min(completion_states)

        primary_state = completion_states.get(primary_index, {"text": "", "token_history": []})
        content_text = primary_state["text"]

        usage = ChatUsage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=primary_completion_tokens,
            total_tokens=(prompt_tokens or 0) + primary_completion_tokens,
        )
        return Response(content=content_text, usage=usage, time_to_first_token_ms=ttft_ms or 0.0)
