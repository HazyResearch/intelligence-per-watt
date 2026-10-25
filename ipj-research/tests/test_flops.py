from __future__ import annotations

import pytest

from ipj.flops import active_params, analytic_flops, cell_flops


def test_active_params_strips_openai_prefix():
    assert active_params("openai/Qwen/Qwen3.5-27B-FP8") == 27e9
    assert active_params("Qwen/Qwen3.5-27B-FP8") == 27e9


def test_moe_uses_active_not_total():
    # 35B-A3B → ~3B active (the whole point: sparse models are FLOP-cheap)
    assert active_params("Qwen/Qwen3.5-35B-A3B-FP8") == 3e9


def test_unknown_model_is_none():
    assert active_params("openai/gpt-5.5-2026-04-23") is None
    assert cell_flops("openai/gpt-5.5-2026-04-23", 100, 100) is None


def test_analytic_flops_2nt_rule():
    assert analytic_flops(4e9, 100, 100) == 2 * 4e9 * 200


def test_cell_flops_from_tokens():
    assert cell_flops("Qwen/Qwen3.5-4B", 1000, 500) == 2 * 4e9 * 1500


def test_negative_inputs_raise():
    with pytest.raises(ValueError):
        analytic_flops(4e9, -1, 0)
