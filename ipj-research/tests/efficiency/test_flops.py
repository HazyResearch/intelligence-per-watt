from __future__ import annotations

from ipj.efficiency.flops import estimate_flops, params_for


def test_estimate_flops_forward_pass():
    # 2 * params * tokens
    assert estimate_flops(27.0, 1000) == 2.0 * 27.0 * 1e9 * 1000

def test_params_for_known_dense():
    assert params_for("Qwen/Qwen3.6-27B-FP8") == 27.0
    assert params_for("cyankiwi/Qwen3.6-27B-AWQ-INT4") == 27.0  # same model, different quant

def test_params_for_moe_uses_active():
    assert params_for("Qwen/Qwen3.6-35B-A3B-FP8") == 3.0  # MoE: active params, not 35B total

def test_params_for_unknown_returns_none():
    assert params_for("gpt-5.5-2026-04-23") is None
