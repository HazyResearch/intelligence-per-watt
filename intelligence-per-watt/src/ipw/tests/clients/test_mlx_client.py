"""Live tests for the in-process MLX client.

Mirrors ``test_vllm.py``: skipped at import time when the optional ``mlx``
extra isn't installed, and again when ``MLX_TEST_MODEL`` isn't set. Greedy
decoding is requested for the ``qkv_fuse`` correctness check so output text
is deterministic against the baseline run.
"""

from __future__ import annotations

import os
from typing import cast

import pytest

pytest.importorskip(
    "mlx_lm",
    reason=(
        "MLX tests require the optional 'mlx' extra. Install via "
        "`uv pip install -e 'intelligence-per-watt[mlx]'`."
    ),
)

MLX_MODEL = os.environ.get("MLX_TEST_MODEL")
if not MLX_MODEL:
    pytest.skip(
        "Set MLX_TEST_MODEL to a Qwen3-style HF id to run the MLX live tests "
        "(e.g. mlx-community/Qwen3-0.6B-bf16).",
        allow_module_level=True,
    )

MODEL = cast(str, MLX_MODEL)

from ipw.clients import _mlx_kernels  # noqa: E402
from ipw.clients._mlx_kernels.spec_decode import assert_vocab_compat  # noqa: E402
from ipw.clients.mlx import MLXClient, _coerce_scalar, _split_opts  # noqa: E402

# ---- pure-Python helpers (no model load) -------------------------------------


def test_split_opts_handles_empty_and_whitespace() -> None:
    assert _split_opts("") == []
    assert _split_opts(None) == []
    assert _split_opts("qkv_fuse") == ["qkv_fuse"]
    assert _split_opts(" qkv_fuse , gdn_fuse ") == ["qkv_fuse", "gdn_fuse"]


def test_coerce_scalar_typed_strings() -> None:
    assert _coerce_scalar("true") is True
    assert _coerce_scalar("False") is False
    assert _coerce_scalar("0.7") == 0.7
    assert _coerce_scalar("42") == 42
    assert _coerce_scalar("hello") == "hello"


def test_kernels_registry_lists_qkv_fuse() -> None:
    assert "qkv_fuse" in _mlx_kernels.available()


def test_kernels_registry_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown mlx kernel mod"):
        _mlx_kernels.install("not_a_real_opt", model=None)


# ---- spec-decode pure-Python helpers -----------------------------------------


class _StubTokenizer:
    """Minimal tokenizer protocol for vocab-compat checks.

    We only need ``vocab_size`` and ``decode([id]) -> str``; anything more is
    out of scope for ``assert_vocab_compat``.
    """

    def __init__(self, vocab_size: int, table: dict[int, str] | None = None) -> None:
        self.vocab_size = vocab_size
        self._table = table or {}

    def decode(self, ids):
        # Match HF/TokenizerWrapper semantics: decode is a join over per-id
        # strings. Default to f"<{id}>" so two stubs with identical vocab_size
        # but different ``table`` overrides compare unequal at the override IDs.
        return "".join(self._table.get(i, f"<{i}>") for i in ids)


def test_assert_vocab_compat_accepts_identical() -> None:
    target = _StubTokenizer(vocab_size=128)
    draft = _StubTokenizer(vocab_size=128)
    # Should not raise — identical vocab_size and same default decode().
    assert_vocab_compat(target, draft)


def test_assert_vocab_compat_rejects_size_mismatch() -> None:
    target = _StubTokenizer(vocab_size=128)
    draft = _StubTokenizer(vocab_size=256)
    with pytest.raises(ValueError, match="vocab"):
        assert_vocab_compat(target, draft)


def test_assert_vocab_compat_rejects_id_remap() -> None:
    """vocab_size match alone is insufficient — reordered IDs corrupt output."""
    target = _StubTokenizer(vocab_size=128)
    draft = _StubTokenizer(vocab_size=128, table={1: "WRONG"})  # id 1 differs
    with pytest.raises(ValueError, match="id=1"):
        assert_vocab_compat(target, draft, probe_ids=(0, 1, 2))


def test_assert_vocab_compat_skips_out_of_range_probes() -> None:
    """probe_ids ≥ vocab_size are skipped (so the default probe set works for
    both small and large vocabs without callers having to filter)."""
    target = _StubTokenizer(vocab_size=10)
    draft = _StubTokenizer(vocab_size=10)
    # 50000 is well past the vocab; should be ignored, not raise.
    assert_vocab_compat(target, draft, probe_ids=(0, 5, 50000))


def test_num_draft_tokens_validated_at_construction() -> None:
    with pytest.raises(ValueError, match="num_draft_tokens"):
        MLXClient(num_draft_tokens=0)
    with pytest.raises(ValueError, match="num_draft_tokens"):
        MLXClient(num_draft_tokens="-1")


def test_num_draft_tokens_coerces_string() -> None:
    """CLI passes ``--client-param num_draft_tokens=4`` as a string; constructor
    must coerce to int up front so downstream stream_generate sees an int."""
    client = MLXClient(num_draft_tokens="6")
    try:
        assert client._num_draft_tokens == 6
        assert isinstance(client._num_draft_tokens, int)
    finally:
        client.close()


def test_draft_model_default_is_disabled() -> None:
    """Empty/missing ``draft_model`` leaves spec-decode disabled — the
    constructor must not eagerly try to load anything."""
    client = MLXClient()  # no draft_model → no draft load attempted
    try:
        assert client._draft_model_id == ""
        assert client._draft_model is None
    finally:
        client.close()


def test_client_param_max_tokens_stored_at_construction() -> None:
    """``--client-param max_tokens=N`` is forwarded to the constructor by
    the runner (not to stream_chat_completion). It must land in
    ``self._config`` so the per-call merge in stream_chat_completion can
    pick it up — otherwise the value is silently lost."""
    client = MLXClient(max_tokens=42, temp="0.7")
    try:
        assert client._config == {"max_tokens": 42, "temp": "0.7"}
    finally:
        client.close()


def test_build_gen_kwargs_uses_construction_temp_default() -> None:
    """Sampler kwargs from construction (the only path the runner currently
    takes) must reach _build_gen_kwargs. temp>0 → a sampler is built."""
    client = MLXClient(temp="0.7")
    try:
        effective = {**client._config}  # mirrors stream_chat_completion's merge
        kw = client._build_gen_kwargs(effective)
        assert "sampler" in kw
    finally:
        client.close()


def test_build_gen_kwargs_per_call_overrides_construction() -> None:
    """Per-call params win over construction defaults — temp=0.0 at call time
    should force greedy decoding even when construction set temp=0.7."""
    client = MLXClient(temp="0.7")
    try:
        effective = {**client._config, **{"temp": 0.0}}
        kw = client._build_gen_kwargs(effective)
        assert "sampler" not in kw  # temp==0 → greedy via sampler=None
    finally:
        client.close()


# ---- live model fixtures -----------------------------------------------------


@pytest.fixture(scope="module")
def mlx_client():
    """Single shared client for the live tests.

    Holding two ~8B-parameter models in unified memory at once trips OOM on
    M-series Macs (~36 GB ceiling — bf16 weights + KV cache + workspace alone
    can occupy ~16 GB per model). Tests that need to exercise the qkv_fuse
    swap install / uninstall it on this client's loaded model in place.
    """
    client = MLXClient()
    try:
        client.prepare(MODEL)
        yield client
    finally:
        client.close()


# ---- live tests --------------------------------------------------------------


def test_health_and_list(mlx_client) -> None:
    assert mlx_client.health() is True
    assert isinstance(mlx_client.list_models(), list)


def test_baseline_generation(mlx_client) -> None:
    response = mlx_client.stream_chat_completion(
        MODEL, "Hello", max_tokens=8, temp=0.0
    )

    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens >= response.usage.completion_tokens
    assert response.time_to_first_token_ms >= 0


def test_qkv_fuse_layer0_numerics(mlx_client) -> None:
    """Layer-0 q/k/v outputs must match the unfused projections within 5e-3
    in fp32 (the apple-silicon-llm threshold; bf16 dot-product noise floor at
    these shapes is ~1e-3). This is the canonical bit-exactness check —
    full-decode text equality is *not* a property of qkv_fuse: rounding noise
    compounds across 36 layers × N tokens and can flip argmax tie-breaks.

    Installs in-place on the shared client's model and uninstalls when done,
    so subsequent tests still see the baseline class.
    """
    import mlx.core as mx

    from ipw.clients._mlx_kernels.qkv_fuse import (
        install_qkv_fuse,
        uninstall_qkv_fuse,
    )

    model = mlx_client._model
    ref_attn = model.model.layers[0].self_attn
    in_dim = int(ref_attn.q_proj.weight.shape[1])
    dtype = ref_attn.q_proj.weight.dtype

    mx.random.seed(0)
    x = mx.random.normal(shape=(1, 8, in_dim)).astype(dtype)
    mx.eval(x)

    qb = ref_attn.q_proj(x)
    kb = ref_attn.k_proj(x)
    vb = ref_attn.v_proj(x)
    mx.eval(qb, kb, vb)

    handle = install_qkv_fuse(model)
    try:
        sa = model.model.layers[0].self_attn
        assert type(sa).__name__ == "Swapped_QKVFused_Attention"
        qkv = x @ sa._W_qkv.T
        qf = qkv[..., : sa._q_out]
        kf = qkv[..., sa._q_out : sa._q_out + sa._kv_out]
        vf = qkv[..., sa._q_out + sa._kv_out :]
        mx.eval(qf, kf, vf)

        max_q = float(mx.max(mx.abs(qb - qf).astype(mx.float32)))
        max_k = float(mx.max(mx.abs(kb - kf).astype(mx.float32)))
        max_v = float(mx.max(mx.abs(vb - vf).astype(mx.float32)))

        assert max(max_q, max_k, max_v) < 5e-3, (
            f"qkv_fuse layer-0 numerics drifted: "
            f"max|q|={max_q:.3e}, max|k|={max_k:.3e}, max|v|={max_v:.3e}"
        )
    finally:
        uninstall_qkv_fuse(handle)
        sa = model.model.layers[0].self_attn
        assert type(sa).__name__ == "Attention"
        assert not hasattr(sa, "_W_qkv")


def test_qkv_fuse_idempotent_install(mlx_client) -> None:
    """Re-installing on a model that already has the swap is a no-op (an
    empty handle), via the Swapped_QKVFused_ class-name prefix guard."""
    from ipw.clients._mlx_kernels.qkv_fuse import (
        install_qkv_fuse,
        uninstall_qkv_fuse,
    )

    model = mlx_client._model
    h1 = install_qkv_fuse(model)
    try:
        assert len(h1) > 0, "expected at least one Qwen3 Attention to swap"
        h2 = install_qkv_fuse(model)
        assert h2 == [], "second install should be a no-op"
    finally:
        uninstall_qkv_fuse(h1)
        sa = model.model.layers[0].self_attn
        assert type(sa).__name__ == "Attention"
        assert not hasattr(sa, "_W_qkv")


def test_unknown_opt_raises_on_prepare() -> None:
    """Doesn't share the fixture model — unknown opt is raised at install time
    so we never reach the model load."""
    client = MLXClient(opts="nonexistent_opt")
    try:
        with pytest.raises(ValueError, match="unknown mlx kernel mod"):
            client.prepare(MODEL)
    finally:
        client.close()


# ---- live spec-decode tests --------------------------------------------------
#
# Gated on a separate env var so the qkv_fuse fixture above (single-model) and
# the spec-decode fixture below (two-model) don't both have to be paid for in
# every run. Pick a vocab-compatible draft from the same family as MLX_MODEL —
# e.g. for ``mlx-community/Qwen3-8B-bf16`` the natural draft is
# ``mlx-community/Qwen3-1.7B-bf16`` (~10× cheaper, identical tokenizer).

DRAFT_MODEL = os.environ.get("MLX_TEST_DRAFT_MODEL")


@pytest.fixture(scope="module")
def spec_decode_client():
    """Separate fixture for spec-decode — two models in unified memory.

    Skipped when ``MLX_TEST_DRAFT_MODEL`` isn't set; loads target + draft +
    applies warmup over the speculative path so JIT cost is paid here, not
    inside an individual test's timing.
    """
    if not DRAFT_MODEL:
        pytest.skip(
            "Set MLX_TEST_DRAFT_MODEL to a vocab-compatible smaller HF id "
            "(same tokenizer family as MLX_TEST_MODEL) to run the spec-decode "
            "live tests.",
        )
    client = MLXClient(draft_model=DRAFT_MODEL, num_draft_tokens="4")
    try:
        client.prepare(MODEL)
        yield client
    finally:
        client.close()


def test_spec_decode_loads_draft(spec_decode_client) -> None:
    """The fixture having loaded means the vocab-compat gate passed against
    the configured draft. Sanity-check the wired-up state on the client."""
    assert spec_decode_client._draft_model is not None
    assert spec_decode_client._draft_tokenizer is not None
    assert spec_decode_client._num_draft_tokens == 4
    # Draft must NOT alias the target unless self-spec was explicitly asked.
    if DRAFT_MODEL != MODEL:
        assert spec_decode_client._draft_model is not spec_decode_client._model


def test_spec_decode_generation_matches_baseline_greedy(spec_decode_client) -> None:
    """Greedy spec-decode is bit-exact vs greedy baseline — every token the
    target accepts is exactly the token target greedy would have produced.
    Run a short generation under temp=0 against the same model loaded fresh
    without a draft and compare strings.
    """
    spec_resp = spec_decode_client.stream_chat_completion(
        MODEL, "The capital of France is", max_tokens=16, temp=0.0
    )
    assert spec_resp.usage.completion_tokens > 0

    # Fresh client without a draft for the reference run. Reusing the spec
    # fixture's model object isn't safe here — its draft state is part of the
    # generation path.
    baseline = MLXClient()
    try:
        baseline.prepare(MODEL)
        base_resp = baseline.stream_chat_completion(
            MODEL, "The capital of France is", max_tokens=16, temp=0.0
        )
    finally:
        baseline.close()

    assert spec_resp.content == base_resp.content, (
        "spec-decode greedy output diverged from baseline greedy — "
        f"spec={spec_resp.content!r} vs base={base_resp.content!r}"
    )


def test_spec_decode_self_spec_reuses_target_object() -> None:
    """When draft_model == model, the client must alias rather than load
    twice — otherwise a self-spec test on Qwen3-8B-bf16 OOMs at 32 GB."""
    client = MLXClient(draft_model=MODEL, num_draft_tokens=2)
    try:
        client.prepare(MODEL)
        assert client._draft_model is client._model
        assert client._draft_tokenizer is client._tokenizer
    finally:
        client.close()
