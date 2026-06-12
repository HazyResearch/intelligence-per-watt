"""Fuse Q/K/V projections in mlx_lm Qwen3-style Attention via __class__-swap.

For every Qwen3-style ``Attention`` module we replace ``__call__`` by mutating
``mod.__class__`` to a subclass whose forward runs ONE matmul against a
concatenated weight matrix ``[Wq; Wk; Wv]``, then slices the result into Q/K/V
and continues through the original q_norm / k_norm / RoPE / SDPA / o_proj path.

The fusion is bit-exact vs. the unswapped projections (concat-along-out-dim is
mathematically identical to three separate matmuls; only matmul rounding-order
noise differs). It is shape-agnostic in ``L`` (sequence length), so it applies
to both prefill and decode — no L==1 fast path.

Predicate: matches modules whose class name is exactly ``Attention``, lives
under ``mlx_lm.models.``, and has all of ``q_proj`` / ``k_proj`` / ``v_proj`` /
``o_proj`` / ``q_norm`` / ``k_norm``. The q_norm/k_norm gate intentionally
excludes Llama-style attention (where blind fusion would corrupt outputs)
and Qwen3.5-27B's ``Qwen3NextAttention`` (different class name + a fused-Q-
with-gate output shape).
"""

from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx

_PREFIX = "Swapped_QKVFused_"
_FUSED_ATTRS = ("_W_qkv", "_q_out", "_kv_out")


def _is_qwen3_attention(mod) -> bool:
    cls = type(mod)
    if cls.__name__ != "Attention":
        return False
    if not (cls.__module__ or "").startswith("mlx_lm.models."):
        return False
    return all(
        hasattr(mod, n)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    )


def _build_class(orig_cls: type) -> type:
    def call(self, x, mask=None, cache=None):
        # Lazy import — avoids a hard mlx_lm dep at module load time.
        from mlx_lm.models.base import scaled_dot_product_attention as _sdpa

        B, L, _ = x.shape

        qkv = x @ self._W_qkv.T
        q = qkv[..., : self._q_out]
        k = qkv[..., self._q_out : self._q_out + self._kv_out]
        v = qkv[..., self._q_out + self._kv_out :]

        q = self.q_norm(q.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = _sdpa(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    return type(f"{_PREFIX}{orig_cls.__name__}", (orig_cls,), {"__call__": call})


def install_qkv_fuse(model) -> List[Tuple[object, type]]:
    """Replace each Qwen3-style ``Attention`` with a fused-QKV variant.

    Returns a handle (list of ``(module, original_class)`` pairs) suitable for
    :func:`uninstall_qkv_fuse`. Idempotent — re-installing on a model that
    already has the swap is a no-op.
    """
    handle: List[Tuple[object, type]] = []
    for dotted, mod in model.named_modules():
        if not _is_qwen3_attention(mod):
            continue
        orig_cls = type(mod)
        if orig_cls.__name__.startswith(_PREFIX):
            continue  # already swapped — idempotent re-install

        Wq, Wk, Wv = mod.q_proj.weight, mod.k_proj.weight, mod.v_proj.weight
        in_dim = Wq.shape[1]
        assert Wk.shape[1] == in_dim and Wv.shape[1] == in_dim, (
            f"qkv input-dim mismatch at {dotted}: "
            f"q={Wq.shape}, k={Wk.shape}, v={Wv.shape}"
        )
        assert Wv.shape[0] == Wk.shape[0], (
            f"k/v output-dim mismatch at {dotted}: k={Wk.shape}, v={Wv.shape}"
        )
        for n in ("q_proj", "k_proj", "v_proj"):
            assert "bias" not in getattr(mod, n), (
                f"unexpected bias on {dotted}.{n} — fused QKV path assumes bias=False"
            )

        W_qkv = mx.concatenate([Wq, Wk, Wv], axis=0)
        mx.eval(W_qkv)
        mod._W_qkv = W_qkv
        mod._q_out = int(Wq.shape[0])
        mod._kv_out = int(Wk.shape[0])

        mod.__class__ = _build_class(orig_cls)
        handle.append((mod, orig_cls))

    return handle


def uninstall_qkv_fuse(handle: List[Tuple[object, type]]) -> None:
    """Restore each module's original class and drop the fused tensors."""
    for mod, orig_cls in handle:
        mod.__class__ = orig_cls
        for n in _FUSED_ATTRS:
            try:
                delattr(mod, n)
            except (AttributeError, KeyError):
                pass
    handle.clear()
