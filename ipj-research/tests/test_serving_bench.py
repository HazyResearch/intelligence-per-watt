from __future__ import annotations

import pytest

from ipj.serving_bench import (
    avg_power_w,
    build_efficiency_cell,
    decode_tok_s,
    joules_per_token,
    prefill_tok_s,
)


def test_prefill_decode_tok_s():
    assert prefill_tok_s(2048, 0.5) == pytest.approx(4096.0)
    assert decode_tok_s(512, 4.0) == pytest.approx(128.0)
    assert prefill_tok_s(100, 0.0) == 0.0   # guard against divide-by-zero
    assert decode_tok_s(100, 0.0) == 0.0


def test_avg_power_and_joules_per_token():
    # constant 700 W for 10 s = 7000 J; over 1000 tokens = 7 J/tok; mean power 700 W.
    s = [(0.0, 700.0), (5.0, 700.0), (10.0, 700.0)]
    assert avg_power_w(s, 0.0, 10.0) == pytest.approx(700.0)
    assert joules_per_token(s, 0.0, 10.0, 1000) == pytest.approx(7.0)
    assert joules_per_token(s, 0.0, 10.0, 0) == 0.0     # no tokens -> guard
    assert avg_power_w(s, 5.0, 5.0) == 0.0              # zero window -> guard


def test_build_efficiency_cell():
    s = [(0.0, 600.0), (8.0, 600.0)]
    c = build_efficiency_cell(
        "Qwen/Qwen3.5-27B-FP8", "mi350x", input_tokens=2048, ttft_s=0.4, output_tokens=512,
        decode_s=4.0, power_samples=s, win_start=0.0, win_end=8.0, mean_latency_s=4.4,
    )
    assert c.hardware == "mi350x"
    assert c.prefill_tok_s == pytest.approx(2048 / 0.4)
    assert c.decode_tok_s == pytest.approx(128.0)
    assert c.avg_power_w == pytest.approx(600.0)
    assert c.joules_per_tok == pytest.approx((600.0 * 8.0) / (2048 + 512))
    assert c.n_gpus == 1
