"""Speculative-decoding helpers for the MLX client.

Unlike the in-place ``__class__``-swap mods in this package (``qkv_fuse``,
etc.), spec-decode is a *generation-strategy* change rather than a model
mutation: it loads a second, smaller ``draft_model`` and routes generation
through ``mlx_lm.stream_generate(..., draft_model=draft, num_draft_tokens=K)``.
mlx_lm itself implements the speculative-generate-step loop (target verifies
K draft tokens per call); we only own the loading + vocab-compat plumbing
and the IPW-side per-token telemetry.

Wiring lives in ``ipw.clients.mlx.MLXClient``: pass ``draft_model=<hf_id>``
and (optionally) ``num_draft_tokens=<N>`` via ``--client-param``. The opts
list (``opts=qkv_fuse,...``) is applied to BOTH target and draft so the
spec-vs-baseline comparison stays apples-to-apples.

Self-spec edge case: when ``draft_model == model``, the client reuses the
already-loaded target as the draft (same Python object, two distinct caches).
This avoids a 2× memory blow-up on Apple's unified memory and matches the
self-spec gate in ``apple-silicon-llm/src/llm_profiler/runner.py``.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

# Sample IDs we round-trip through both tokenizers to detect reordered
# vocabularies (matching vocab_size with mismatched id->token mapping is a
# silent correctness failure for spec-decode — drafts proposing id 42 mean
# something different to the target).
_DEFAULT_PROBE_IDS: Tuple[int, ...] = (0, 1, 100, 1000, 10000, 50000)


def load_draft(model_id: str) -> Tuple[Any, Any]:
    """Load a draft model + tokenizer via ``mlx_lm.load``.

    Thin wrapper that exists so the MLX client doesn't import ``mlx_lm`` at
    a different layer for drafts than for targets, and so tests can monkey-
    patch this single seam.
    """
    from mlx_lm import load

    return load(model_id)


def assert_vocab_compat(
    target_tokenizer: Any,
    draft_tokenizer: Any,
    *,
    probe_ids: Iterable[int] = _DEFAULT_PROBE_IDS,
) -> None:
    """Raise if the draft's tokenizer is unsafe to pair with the target.

    Speculative decoding is only correct when target and draft sample over
    the same vocabulary in the same order. We check:

    1. ``vocab_size`` match — necessary but not sufficient.
    2. ``decode([id])`` agreement on a small probe set — catches reordered
       IDs that would otherwise corrupt output silently.

    ``probe_ids`` is a small iterable of token IDs. IDs ≥ ``vocab_size`` are
    skipped (so the default works for both small and large vocabs without
    branching at the call site).
    """
    target_vs = int(target_tokenizer.vocab_size)
    draft_vs = int(draft_tokenizer.vocab_size)
    if target_vs != draft_vs:
        raise ValueError(
            f"draft tokenizer vocab ({draft_vs}) != target ({target_vs}); "
            f"speculative decoding requires identical vocabularies."
        )

    for sample_id in probe_ids:
        if sample_id >= target_vs:
            continue
        t_tok = target_tokenizer.decode([sample_id])
        d_tok = draft_tokenizer.decode([sample_id])
        if t_tok != d_tok:
            raise ValueError(
                f"draft tokenizer disagrees on id={sample_id}: "
                f"target={t_tok!r}, draft={d_tok!r}; "
                f"speculative decoding requires identical id->string maps."
            )


__all__ = ["load_draft", "assert_vocab_compat"]
