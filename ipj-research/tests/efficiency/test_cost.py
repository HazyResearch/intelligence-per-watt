from __future__ import annotations

import pytest

from ipj.efficiency.cost import (
    H100_HOURLY_USD,
    PRICING,
    cost_for,
    cost_from_tokens,
    local_dollars,
    local_dollars_per_task,
    multiplier,
    sensitivity_band,
    throughput_dollars_per_task,
)


def test_cost_from_tokens_exact_value():
    # 1M input @ $1.25/M + 1M output @ $10.0/M = $11.25
    result = cost_from_tokens(1_000_000, 1_000_000, 1.25, 10.0)
    assert result == 11.25


def test_cost_from_tokens_zero():
    assert cost_from_tokens(0, 0, 1.25, 10.0) == 0.0


def test_cost_from_tokens_only_input():
    # 2M input @ $2.0/M, no output
    assert cost_from_tokens(2_000_000, 0, 2.0, 5.0) == 4.0


def test_cost_from_tokens_only_output():
    # no input, 500k output @ $10.0/M = $5.0
    assert cost_from_tokens(0, 500_000, 2.0, 10.0) == 5.0


def test_cost_for_known_model():
    # Use whatever the first seeded entry is; just check it returns a float > 0
    # for any reasonable non-zero token count
    for model_id in PRICING:
        result = cost_for(model_id, 1_000_000, 1_000_000)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0
        break


def test_cost_for_unknown_model_returns_none():
    assert cost_for("nonexistent/model-v999", 1_000, 1_000) is None


def test_cost_for_case_insensitive():
    for model_id in PRICING:
        upper_result = cost_for(model_id.upper(), 500_000, 500_000)
        lower_result = cost_for(model_id.lower(), 500_000, 500_000)
        assert upper_result == lower_result
        break


def test_pricing_dict_nonempty():
    assert len(PRICING) >= 1


def test_pricing_dict_values_are_positive():
    for model_id, (price_in, price_out) in PRICING.items():
        assert price_in >= 0.0, f"{model_id} price_in must be non-negative"
        assert price_out >= 0.0, f"{model_id} price_out must be non-negative"


# --- open-source / local cost rule: $ = latency · n_GPUs · shared reference rate ------------

def test_h100_reference_rate_is_shared():
    # The efficiency helpers and live cell/routing paths share one reference.
    from ipj.cost import H100_HOURLY_USD as CANONICAL_H100_HOURLY_USD

    assert H100_HOURLY_USD == CANONICAL_H100_HOURLY_USD


def test_local_dollars_exact_formula():
    assert local_dollars(3600.0, 8) == pytest.approx(8 * H100_HOURLY_USD)
    assert local_dollars(1800.0, 1) == pytest.approx(H100_HOURLY_USD / 2)


def test_local_dollars_custom_rate():
    assert local_dollars(3600.0, 2, hourly_rate=2.0) == pytest.approx(4.0)


def test_local_dollars_per_task_amortizes_window_over_batch():
    assert local_dollars_per_task(3600.0, 8, 89) == pytest.approx(
        8 * H100_HOURLY_USD / 89
    )


def test_local_dollars_per_task_rejects_nonpositive_tasks():
    with pytest.raises(ValueError):
        local_dollars_per_task(3600.0, 8, 0)


# --- saturated-throughput local cost (the honest deployment basis) --------------------------

def test_throughput_dollars_measured_27b_floor():
    # measured 27B on 2xH100: decode 6860 tok/s, prefill 23485 tok/s; per-task 84k in / 7.5k out
    dpt = throughput_dollars_per_task(
        83711, 7530, decode_tok_s=6860, prefill_tok_s=23485, n_gpus=2, cache_frac=0.0
    )
    gpu_s = 83711 / 23485 + 7530 / 6860
    assert dpt == pytest.approx(
        gpu_s / 3600 * 2 * H100_HOURLY_USD,
    )


def test_throughput_dollars_cache_lowers_cost():
    no_cache = throughput_dollars_per_task(80000, 8000, decode_tok_s=7000, prefill_tok_s=23000,
                                           n_gpus=2, cache_frac=0.0)
    cached = throughput_dollars_per_task(80000, 8000, decode_tok_s=7000, prefill_tok_s=23000,
                                         n_gpus=2, cache_frac=0.9)
    assert cached < no_cache  # prefix-cache reuse cuts prefill cost


def test_throughput_dollars_orders_of_magnitude_below_wallclock():
    # saturated GPU-time cost must be far below the single-stream wall-clock upper bound
    sat = throughput_dollars_per_task(83711, 7530, decode_tok_s=6860, prefill_tok_s=23485, n_gpus=2)
    wall = local_dollars_per_task(6.6 * 3600, 2, 89)  # the idle-inflated wall-clock number (~$0.44)
    assert sat < wall / 20


# --- iso-accuracy multiplier + sensitivity band ---------------------------------------------

def test_multiplier_basic():
    # Fable $4.50 vs our $0.30 -> 15x cheaper
    assert multiplier(0.30, 4.50) == pytest.approx(15.0)


def test_multiplier_rejects_nonpositive_our_cost():
    with pytest.raises(ValueError):
        multiplier(0.0, 4.50)


def test_sensitivity_band_brackets_the_point_multiplier():
    lo, hi = sensitivity_band(0.225, 4.50, band=0.5)  # our $0.225 = 20x at nominal
    # +50% cost -> lower multiplier; -50% cost -> higher multiplier; nominal 20x sits between
    assert lo == pytest.approx(4.50 / (0.225 * 1.5))   # 13.33x
    assert hi == pytest.approx(4.50 / (0.225 * 0.5))   # 40x
    assert lo < 20.0 < hi


def test_amortize_run_default_rate_is_reference():
    # the local-cost driver must default to the shared reference, not None
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "experiments" / "fast-rerun" / "amortize_run.py"
    text = src.read_text()
    assert "default=H100_HOURLY_USD" in text
    ast.parse(text)  # still parses
