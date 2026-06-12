"""In-process MLX (Apple Silicon) inference client.

Drives ``mlx_lm.stream_generate`` directly. Energy/power telemetry is captured
by IPW's existing macOS collector (Rust gRPC service sampling ``powermetrics``
at 50ms), which is system-wide and works for in-process inference without any
extra plumbing.

Optional kernel-level optimizations are applied via ``--client-params opts=...``.
``opts`` is a comma-separated list of names from
``ipw.clients._mlx_kernels.available()``; mods are applied left-to-right and
torn down in reverse on close. Example::

    ipw profile --client mlx \\
                --model mlx-community/Qwen3-8B-bf16 \\
                --client-param opts=qkv_fuse

Speculative decoding is enabled by passing ``draft_model=<hf_id>`` (with an
optional ``num_draft_tokens=<N>``, default 4). When set, the same ``opts``
list is applied to the draft so the spec-vs-baseline comparison stays
apples-to-apples; the draft must use the same tokenizer as the target
(:func:`._mlx_kernels.spec_decode.assert_vocab_compat` enforces this).
Example::

    ipw profile --client mlx \\
                --model mlx-community/Qwen3-8B-bf16 \\
                --client-param draft_model=mlx-community/Qwen3-1.7B-bf16 \\
                --client-param num_draft_tokens=4
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Mapping, Tuple

# Top-level imports anchor the optional-extra gate in
# ``ipw.clients.__init__.ensure_registered``: when the ``mlx`` extra isn't
# installed, ``import mlx.core`` raises ``ModuleNotFoundError(name="mlx")``,
# which the registration loop maps to the MISSING_CLIENTS hint instead of
# bubbling. Keep this above any ``mlx_lm`` import so missing-mlx fires first.
import mlx.core  # noqa: F401
from mlx_lm import load, stream_generate

from . import _mlx_kernels
from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient


def _split_opts(opts: Any) -> List[str]:
    if not opts:
        return []
    return [s.strip() for s in str(opts).split(",") if s.strip()]


@ClientRegistry.register("mlx")
class MLXClient(InferenceClient):
    """MLX in-process client. ``base_url`` is unused (kept for ABC compatibility)."""

    client_id, client_name = "mlx", "MLX"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        opts: Any = "",
        draft_model: Any = "",
        num_draft_tokens: Any = 4,
        **config: Any,
    ) -> None:
        super().__init__(base_url or "in-process", **config)
        self._opt_names: List[str] = _split_opts(opts)
        self._draft_model_id: str = str(draft_model or "").strip()
        self._num_draft_tokens: int = int(_coerce_scalar(num_draft_tokens))
        if self._num_draft_tokens < 1:
            raise ValueError(
                f"num_draft_tokens must be >= 1 (got {self._num_draft_tokens})"
            )
        self._model: Any = None
        self._tokenizer: Any = None
        self._draft_model: Any = None
        self._draft_tokenizer: Any = None
        # Stack of (uninstall_fn, handle) — popped LIFO on teardown. Holds
        # opts applied to BOTH target and draft (target first, then draft);
        # LIFO order undoes draft before target so each opt sees the model
        # state it installed against.
        self._undos: List[Tuple[Callable, object]] = []
        self._loaded: str | None = None

    # ---- lifecycle -----------------------------------------------------------

    def prepare(self, model: str) -> None:
        """Load the model + apply opts + warmup so JIT happens outside the
        first energy window."""
        self._ensure_loaded(model)

    def health(self) -> bool:
        return True  # in-process

    def list_models(self) -> List[str]:
        return []  # HF-driven; no enumeration

    def close(self) -> None:
        self._teardown()

    # ---- inference -----------------------------------------------------------

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        self._ensure_loaded(model)

        max_tokens = int(params.get("max_tokens", 512))
        gen_kwargs = self._build_gen_kwargs(params)
        if self._draft_model is not None:
            # mlx_lm.stream_generate dispatches to speculative_generate_step
            # internally when draft_model is non-None; num_draft_tokens flows
            # through **kwargs to that step.
            gen_kwargs["draft_model"] = self._draft_model
            gen_kwargs["num_draft_tokens"] = self._num_draft_tokens

        start = time.perf_counter()
        ttft_ms: float | None = None
        text_parts: List[str] = []
        token_times: List[float] = []

        for r in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            **gen_kwargs,
        ):
            token_times.append(time.perf_counter())
            if ttft_ms is None:
                ttft_ms = (token_times[0] - start) * 1000
            text_parts.append(r.text)
            # GenerationResponse.from_draft is True for draft-accepted tokens
            # in the spec-decode path; the field is exposed here for future
            # callers that want to surface acceptance rate via Response, but
            # we don't materialize it yet to keep the payload compact.

        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = len(token_times)
        return Response(
            content="".join(text_parts),
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            time_to_first_token_ms=ttft_ms or 0.0,
            token_timestamps=token_times or None,
        )

    # ---- internals -----------------------------------------------------------

    def _ensure_loaded(self, model_id: str) -> None:
        if self._loaded == model_id:
            return
        self._teardown()

        self._model, self._tokenizer = load(model_id)
        self._apply_opts(self._model, role="target", role_model_id=model_id)

        # Spec-decode wiring. Reuse the target object as the draft when the
        # caller explicitly self-specs (same HF id) — avoids a 2× memory blow
        # on Apple's unified-memory budget. Otherwise load + apply opts +
        # vocab-compat-check, mirroring apple-silicon-llm's runner gate.
        if self._draft_model_id:
            from ._mlx_kernels.spec_decode import (
                assert_vocab_compat,
                load_draft,
            )

            if self._draft_model_id == model_id:
                self._draft_model = self._model
                self._draft_tokenizer = self._tokenizer
            else:
                self._draft_model, self._draft_tokenizer = load_draft(
                    self._draft_model_id
                )
                self._apply_opts(
                    self._draft_model,
                    role="draft",
                    role_model_id=self._draft_model_id,
                )
                # Vocab-compat AFTER opts because qkv_fuse / future opts can
                # in principle alter tokenizer-adjacent state (none currently
                # do, but the check is cheap and the order is safer).
                assert_vocab_compat(self._tokenizer, self._draft_tokenizer)

        # Warmup so MLX JIT-compiles kernels before IPW opens the first energy
        # window. Drain the generator with a 4-token cap. When spec-decode is
        # active, warm that path too — speculative_generate_step has its own
        # JIT'd verify-step shape that's distinct from the plain decode step.
        warmup_kwargs: dict[str, Any] = {"max_tokens": 4}
        if self._draft_model is not None:
            warmup_kwargs["draft_model"] = self._draft_model
            warmup_kwargs["num_draft_tokens"] = self._num_draft_tokens

        for _ in stream_generate(
            self._model, self._tokenizer, "warmup", **warmup_kwargs
        ):
            pass

        self._loaded = model_id

    def _apply_opts(
        self,
        model: Any,
        *,
        role: str,
        role_model_id: str,
    ) -> None:
        """Apply ``self._opt_names`` to ``model`` in registration order.

        Records each undo on ``self._undos`` so ``_teardown`` reverses them
        LIFO across both target and draft. ``role`` is "target" or "draft"
        and only affects the error message on a no-match install.
        """
        for name in self._opt_names:
            uninstall_fn, handle = _mlx_kernels.install(name, model)
            if not handle:
                self._teardown()
                raise RuntimeError(
                    f"opt {name!r} matched no modules in {role} model "
                    f"{role_model_id!r}. Check that the optimization applies "
                    f"to this model family (e.g. qkv_fuse targets Qwen3-style "
                    f"Attention with q_norm/k_norm — not Llama, not "
                    f"Qwen3.5-27B's Qwen3NextAttention)."
                )
            self._undos.append((uninstall_fn, handle))

    def _build_gen_kwargs(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Translate user params into mlx_lm.stream_generate kwargs.

        Values arrive from ``--client-params`` as strings, so we coerce them.
        mlx_lm 0.31+ takes a ``sampler`` callable rather than raw temp/top_p
        kwargs; we build it via ``mlx_lm.sample_utils.make_sampler`` when any
        sampler param is set. ``temp=0`` (or unset) → greedy argmax via
        ``sampler=None``.
        """
        sampler_fields = ("temp", "temperature", "top_p", "top_k", "min_p")
        sampler_kw: dict[str, Any] = {}
        for k in sampler_fields:
            if k in params:
                sampler_kw[k] = _coerce_scalar(params[k])
        # Normalize "temperature" -> "temp" (mlx_lm's name).
        if "temperature" in sampler_kw and "temp" not in sampler_kw:
            sampler_kw["temp"] = sampler_kw.pop("temperature")

        kw: dict[str, Any] = {}
        # Treat temp == 0 as greedy: leave sampler as None (mlx_lm's default
        # token selection is argmax when sampler is None).
        if sampler_kw and float(sampler_kw.get("temp", 1.0)) > 0.0:
            from mlx_lm.sample_utils import make_sampler

            kw["sampler"] = make_sampler(**sampler_kw)

        return kw

    def _teardown(self) -> None:
        # LIFO so each opt sees the same module state it installed against.
        # Spans both target and draft when spec-decode is active.
        while self._undos:
            uninstall_fn, handle = self._undos.pop()
            try:
                uninstall_fn(handle)
            except Exception:
                # Best-effort cleanup; don't mask the original error path.
                pass
        self._model = None
        self._tokenizer = None
        # Self-spec aliases the draft to the target; clear unconditionally
        # since we already nulled the target above.
        self._draft_model = None
        self._draft_tokenizer = None
        self._loaded = None


def _coerce_scalar(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s
