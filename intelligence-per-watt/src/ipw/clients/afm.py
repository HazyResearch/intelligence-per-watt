"""In-process Apple Foundation Models (AFM 3) client.

Drives Apple's ``apple_fm_sdk`` against the on-device model behind Apple
Intelligence. Energy/power telemetry comes from IPW's existing macOS collector
(the Rust gRPC service sampling ``powermetrics`` at 50ms), which is system-wide
and therefore captures in-process inference with no extra plumbing.

Requires an Apple Silicon Mac on macOS 26+ with Apple Intelligence enabled, and
a full Xcode (not just Command Line Tools) because the SDK compiles Swift
bindings at install time::

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \\
        uv pip install -e 'intelligence-per-watt[afm]'

    ipw profile --client afm --model afm-3-core-advanced \\
                --dataset fixed_length --dataset-param prompt_length=512

Measurement caveats specific to this backend -- read before comparing numbers
against vLLM/MLX:

* ``session.stream_response`` yields **cumulative text snapshots**, and each
  snapshot batches roughly 8-10 tokens. ``time_to_first_token_ms`` is therefore
  time-to-first-*chunk* (an upper bound on true TTFT, measured around 450ms on
  an M1 Pro versus a few tens of ms of real first-token latency), and the
  inter-token-latency percentiles IPW derives from ``token_timestamps`` are
  inter-*chunk* latencies. Neither is comparable to a backend that streams one
  token at a time. Token counts, throughput and per-token energy are unaffected.
* The SDK exposes no way to select AFM 3 Core (dense 3B) versus AFM 3 Core
  Advanced (20B sparse MoE, 1-4B active per request); the framework's dynamic
  profile picks one from the host device's capabilities. ``--model`` is a run
  label only. :meth:`AFMClient.describe` records the SDK version, context size
  and host chip into ``summary.json`` so a run can be attributed after the fact.
* There is no Private Cloud Compute path in the Python SDK, which suits IPW --
  off-device inference would make the on-device energy measurement meaningless.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import Any, Dict, List, Sequence

# Top-level import anchors the optional-extra gate in
# ``ipw.clients.__init__.ensure_registered``: without the ``afm`` extra this
# raises ``ModuleNotFoundError(name="apple_fm_sdk")``, which the registration
# loop maps to a MISSING_CLIENTS hint instead of bubbling up.
import apple_fm_sdk as fm

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from ._afm_support import (
    MODEL_LABELS,
    SnapshotAccumulator,
    build_options_kwargs,
    parse_sampling_spec,
    validate_model_label,
)
from ._async_loop import AsyncLoopRunner
from .base import InferenceClient

LOGGER = logging.getLogger(__name__)

_USE_CASES = {
    "general": fm.SystemLanguageModelUseCase.GENERAL,
    "content_tagging": fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
}
_GUARDRAILS = {
    "default": fm.SystemLanguageModelGuardrails.DEFAULT,
    "permissive_content_transformations": (
        fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
    ),
}


def _resolve_handled_errors() -> tuple[type[BaseException], ...]:
    """Per-query failures that must not abort the whole profiling run.

    ``ProfilerRunner._process_records`` does not wrap ``_invoke_client`` in a
    try/except, so an exception escaping ``stream_chat_completion`` ends the run
    and discards every record since the last flush. A 4096-token context makes
    context-window overflow routine on datasets like ``hle`` or ``gpqa``, so
    these are converted into empty responses instead. Transient/environmental
    errors (rate limiting, concurrency, missing assets) deliberately propagate.

    Looked up by name so a future SDK release renaming one of them degrades to a
    narrower catch rather than breaking import.
    """
    names = (
        "ExceededContextWindowSizeError",
        "GuardrailViolationError",
        "RefusalError",
        "UnsupportedLanguageOrLocaleError",
    )
    found = tuple(
        exc
        for exc in (getattr(fm, name, None) for name in names)
        if isinstance(exc, type) and issubclass(exc, BaseException)
    )
    if not found:  # pragma: no cover - guards against a wholesale SDK rename
        LOGGER.warning(
            "No known apple_fm_sdk per-query error classes found; overflow and "
            "guardrail failures will abort the run instead of skipping a query."
        )
    return found


_HANDLED_ERRORS = _resolve_handled_errors()


def _host_chip() -> str:
    """Identify the Apple Silicon chip, e.g. "Apple M5 Max".

    ``platform.processor()`` returns a bare "arm" on every Apple Silicon Mac,
    which is useless for the one job this field has: the executing AFM variant
    is not observable, so the host chip is how a run gets attributed after the
    fact. ``sysctl`` names the exact part.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - host dependent
        return platform.processor() or platform.machine()
    return result.stdout.strip() or platform.processor() or platform.machine()



@ClientRegistry.register("afm")
class AFMClient(InferenceClient):
    """Apple Foundation Models client. ``base_url`` is unused (ABC compatibility)."""

    client_id, client_name = "afm", "Apple Foundation Models"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        instructions: Any = "",
        use_case: Any = "general",
        guardrails: Any = "default",
        **config: Any,
    ) -> None:
        super().__init__(base_url or "in-process", **config)
        self._instructions: str = str(instructions or "").strip()
        self._use_case_name = str(use_case or "general").strip().lower()
        self._guardrails_name = str(guardrails or "default").strip().lower()
        if self._use_case_name not in _USE_CASES:
            raise ValueError(
                f"Unknown use_case {use_case!r}. Expected one of: {', '.join(_USE_CASES)}."
            )
        if self._guardrails_name not in _GUARDRAILS:
            raise ValueError(
                f"Unknown guardrails {guardrails!r}. Expected one of: {', '.join(_GUARDRAILS)}."
            )
        # Validate the sampling spec eagerly so a typo fails at construction
        # rather than midway through a run.
        parse_sampling_spec(self._config)

        self._model: Any = None
        self._context_size: int | None = None
        self._label: str | None = None
        self._unavailable_reason: Any = None
        self._skipped: Dict[str, int] = {}
        self._loop = AsyncLoopRunner(name="ipw-afm")
        self._closed = False

    # ---- lifecycle -----------------------------------------------------------

    def health(self) -> bool:
        """Report on-device model availability.

        Returns a bool to honour the ABC contract, but logs the SDK's reason for
        unavailability first: the runner's own failure message
        ("... is unavailable") would otherwise hide whether the cause is Apple
        Intelligence being switched off, an ineligible device, or assets that
        have not finished downloading.
        """
        if self._closed:
            return False
        try:
            model = self._ensure_model()
        except Exception as exc:  # pragma: no cover - depends on host state
            LOGGER.warning("Could not initialize Apple Foundation Models: %s", exc)
            return False

        available, reason = model.is_available()
        self._unavailable_reason = reason
        if not available:
            LOGGER.warning(
                "Apple Foundation Models unavailable: %s. Check that Apple "
                "Intelligence is enabled in System Settings, that this device is "
                "eligible, and that model assets have finished downloading.",
                getattr(reason, "name", reason),
            )
        return bool(available)

    def prepare(self, model: str) -> None:
        """Validate the label, then warm up outside the first energy window.

        The first request after boot pays a large one-off cost for model asset
        load and XPC setup (~10s versus ~450ms warm on an M1 Pro). Burning it
        here keeps it out of the telemetry window of the first real query.
        """
        self._label = validate_model_label(model)
        language_model = self._ensure_model()

        available, reason = language_model.is_available()
        if not available:
            raise RuntimeError(
                "Apple Foundation Models is unavailable "
                f"({getattr(reason, 'name', reason)}). Enable Apple Intelligence in "
                "System Settings and wait for model assets to download."
            )

        self._context_size = int(language_model.context_size)
        self._loop.run(self._warmup())

    def list_models(self) -> Sequence[str]:
        return list(MODEL_LABELS)

    def describe(self) -> Dict[str, Any]:
        """Run metadata for ``summary.json``.

        The executing variant (Core vs Core Advanced) is not observable, so the
        host chip and SDK version are the only way to attribute a run later.
        """
        try:
            sdk_version = _dist_version("apple-fm-sdk")
        except PackageNotFoundError:  # pragma: no cover - installed by definition
            sdk_version = None
        return {
            "client_id": self.client_id,
            "model_label": self._label,
            "afm_sdk_version": sdk_version,
            "afm_context_size": self._context_size,
            "afm_use_case": self._use_case_name,
            "afm_guardrails": self._guardrails_name,
            "afm_instructions": self._instructions or None,
            "afm_sampling": parse_sampling_spec(self._config)[0],
            "host_chip": _host_chip(),
            "host_os_version": platform.mac_ver()[0] or None,
            "variant_selection": (
                "device-chosen by the Foundation Models dynamic profile; "
                "--model is a label only"
            ),
            "skipped_queries": dict(self._skipped) or None,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._skipped:
            LOGGER.warning(
                "AFM skipped %d queries: %s",
                sum(self._skipped.values()),
                ", ".join(f"{k}={v}" for k, v in sorted(self._skipped.items())),
            )
        self._loop.shutdown()
        self._model = None

    # ---- inference -----------------------------------------------------------

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        if self._closed:
            raise RuntimeError("AFM client has been closed")
        if self._label is None:
            self.prepare(model)

        # `--client-param` values are passed to the CONSTRUCTOR by the runner and
        # land in self._config; ProfilerRunner does not forward per-call params
        # (its `additional_parameters` is never populated). Without this merge
        # `--client-param max_tokens=N` would be silently ignored -- the same bug
        # commit 6681a30 fixed for the MLX client. Per-call values still win.
        effective = {**self._config, **params}
        return self._loop.run(self._respond(prompt, effective))

    # ---- internals -----------------------------------------------------------

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = fm.SystemLanguageModel(
                use_case=_USE_CASES[self._use_case_name],
                guardrails=_GUARDRAILS[self._guardrails_name],
            )
        return self._model

    def _new_session(self) -> Any:
        """A fresh session per request.

        Sessions accumulate a transcript, so reusing one would grow the prompt
        until it overflows the 4096-token context after a handful of queries and
        would make per-query energy depend on preceding queries.
        """
        return fm.LanguageModelSession(
            instructions=self._instructions or None,
            model=self._ensure_model(),
        )

    def _build_options(self, effective: Dict[str, Any]) -> Any:
        kwargs = build_options_kwargs(effective)
        mode, mode_kwargs = parse_sampling_spec(effective)
        sampling_factory = getattr(fm.SamplingMode, mode, None)
        if callable(sampling_factory):
            kwargs["sampling"] = sampling_factory(**mode_kwargs)
        return fm.GenerationOptions(**kwargs)

    async def _warmup(self) -> None:
        session = self._new_session()
        options = fm.GenerationOptions(maximum_response_tokens=4)
        async for _ in session.stream_response("warmup", options=options):
            pass

    async def _count(self, *args: Any, **kwargs: Any) -> int | None:
        """token_count that degrades to None rather than failing a query."""
        try:
            return int(await self._ensure_model().token_count(*args, **kwargs))
        except Exception:
            LOGGER.debug("AFM token_count failed", exc_info=True)
            return None

    async def _respond(self, prompt: str, effective: Dict[str, Any]) -> Response:
        options = self._build_options(effective)
        session = self._new_session()

        # Token accounting. `token_count` rejects a value and instructions
        # together ("Provide either a value or instructions"), and counting
        # instructions standalone does not match how the transcript encodes
        # them, so both ends are measured against the transcript instead:
        #   prompt_tokens = <transcript before generating> + <prompt alone>
        #   completion_tokens = <transcript after generating> - prompt_tokens
        # Verified on-device: a 4-token-capped generation yields exactly 4.
        pre_tokens = await self._count(session.transcript)
        prompt_only = await self._count(prompt)

        accumulator = SnapshotAccumulator()
        token_times: List[float] = []
        start = time.perf_counter()
        request_start = time.time()
        ttft_ms: float | None = None

        try:
            async for snapshot in session.stream_response(prompt, options=options):
                if not accumulator.add(snapshot):
                    continue
                token_times.append(time.perf_counter())
                if ttft_ms is None:
                    ttft_ms = (token_times[0] - start) * 1000.0
        except _HANDLED_ERRORS as exc:
            reason = type(exc).__name__
            self._skipped[reason] = self._skipped.get(reason, 0) + 1
            LOGGER.warning("AFM query skipped (%s): %s", reason, exc)
            return Response(
                content="",
                usage=ChatUsage(
                    prompt_tokens=prompt_only, completion_tokens=None, total_tokens=None
                ),
                time_to_first_token_ms=0.0,
                request_start_time=request_start,
                request_end_time=time.time(),
            )

        request_end = time.time()
        post_tokens = await self._count(session.transcript)

        prompt_tokens = (
            pre_tokens + prompt_only
            if pre_tokens is not None and prompt_only is not None
            else prompt_only
        )
        completion_tokens = (
            max(post_tokens - prompt_tokens, 0)
            if post_tokens is not None and prompt_tokens is not None
            else None
        )

        return Response(
            content=accumulator.text,
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=(
                    prompt_tokens + completion_tokens
                    if prompt_tokens is not None and completion_tokens is not None
                    else None
                ),
            ),
            time_to_first_token_ms=ttft_ms or 0.0,
            # Wall-clock bounds let ProfilerRunner._compute_phase_metrics split
            # prefill from decode energy accurately instead of falling back to
            # the first telemetry sample.
            request_start_time=request_start,
            request_end_time=request_end,
            token_timestamps=token_times or None,
        )
