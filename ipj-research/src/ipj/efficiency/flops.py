from __future__ import annotations

# Active parameters in billions. MoE models map to their *active* params (what determines
# inference FLOPs), not total. Unknown / closed-source models are absent -> params_for returns None.
MODEL_PARAMS: dict[str, float] = {
    "qwen/qwen3.6-27b-fp8": 27.0,
    "cyankiwi/qwen3.6-27b-awq-int4": 27.0,
    "qwen/qwen3.6-35b-a3b-fp8": 3.0,
    "google/gemma-4-31b-it": 31.0,
    "google/gemma-4-26b-a4b-it": 4.0,
}


def params_for(model_id: str) -> float | None:
    """Active params (billions) for a known model, else None (e.g. closed-source)."""
    return MODEL_PARAMS.get(model_id.lower())


def estimate_flops(active_params_b: float, total_tokens: int) -> float:
    """Forward-pass FLOPs estimate: 2 * params * tokens (input + output). Approximate; MoE active params."""
    return 2.0 * active_params_b * 1e9 * total_tokens
