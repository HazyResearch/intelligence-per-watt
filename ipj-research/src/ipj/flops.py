"""Analytic FLOPs for a TB run — a hardware-independent "intelligence per FLOP" axis.

Forward-pass FLOPs ≈ 2 · N_active · tokens (the 2·N·T rule; the 2 is multiply+add).
For MoE models use ACTIVE params (e.g. 35B-A3B → ~3B active) — that's why sparse models
can be far more FLOP-efficient than their total size suggests. Precision (FP8 vs bf16)
changes energy/throughput but NOT this operation count, so FLOPs is a clean precision- and
hardware-independent axis — immune to the eager-serving confound that inflates J/$/latency.

Cloud/proprietary models have unknown params → FLOPs is None (compare them on $ + accuracy).
"""

from __future__ import annotations

# Active (not total) parameter counts, in billions, for the local sweep models.
MODEL_ACTIVE_PARAMS_B: dict[str, float] = {
    "Qwen/Qwen3.5-4B": 4.0,
    "Qwen/Qwen3.5-9B": 9.0,
    "Qwen/Qwen3-14B": 14.0,
    "Qwen/Qwen3.5-27B-FP8": 27.0,
    "Qwen/Qwen3.6-27B-FP8": 27.0,
    "Qwen/Qwen3.5-35B-A3B-FP8": 3.0,  # MoE: ~3B active of 35B total
    "google/gemma-4-26B-A4B-it": 4.0,  # MoE: ~4B active of 26B total (DGX-Spark edge tier)
}

# Total (not active) parameter counts, in billions — the headline "model size".
# Used for the marker-SIZE encoding in scatter figures (how big is the model), distinct
# from active params above (which drives FLOPs). For dense models the two agree; for the
# MoE (35B-A3B) total = 35 while active = 3.
MODEL_TOTAL_PARAMS_B: dict[str, float] = {
    "Qwen/Qwen3.5-4B": 4.0,
    "Qwen/Qwen3.5-9B": 9.0,
    "Qwen/Qwen3-14B": 14.0,
    "Qwen/Qwen3.5-27B-FP8": 27.0,
    "Qwen/Qwen3.6-27B-FP8": 27.0,
    "Qwen/Qwen3.5-35B-A3B-FP8": 35.0,  # MoE: 35B total (3B active)
    "google/gemma-4-26B-A4B-it": 26.0,  # MoE: 26B total (4B active) — DGX-Spark edge tier
}


def _strip_route_prefix(model: str) -> str:
    """Drop a leading 'openai/' OpenAI-compatible routing prefix if present."""
    return model[len("openai/"):] if model.startswith("openai/") else model


def active_params(model: str) -> float | None:
    """Active parameter count for a model id (tolerates an 'openai/' routing prefix)."""
    b = MODEL_ACTIVE_PARAMS_B.get(_strip_route_prefix(model))
    return b * 1e9 if b is not None else None


def total_params(model: str) -> float | None:
    """Total parameter count for a model id; None when unknown (e.g. cloud/proprietary)."""
    b = MODEL_TOTAL_PARAMS_B.get(_strip_route_prefix(model))
    return b * 1e9 if b is not None else None


def analytic_flops(n_active_params: float, tokens_in: int, tokens_out: int) -> float:
    """Forward-pass FLOPs ≈ 2 · N_active · (tokens_in + tokens_out)."""
    if n_active_params < 0 or tokens_in < 0 or tokens_out < 0:
        raise ValueError("params and tokens must be non-negative")
    return 2.0 * n_active_params * (tokens_in + tokens_out)


def cell_flops(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """Analytic FLOPs for a run; None when the model's params are unknown (e.g. cloud)."""
    n = active_params(model)
    return analytic_flops(n, tokens_in, tokens_out) if n is not None else None
