from __future__ import annotations

import pytest

from ipj.percentile_amortize import QueryEvent, amortize_all, window_amortize


def _run(n=100, dur_each=1.0, power_w=700.0):
    # n queries completing at t=1..n (start = end - dur_each); constant power over the whole span.
    events = [QueryEvent(start=i - dur_each, end=float(i), tokens_in=100, tokens_out=50) for i in range(1, n + 1)]
    power = [(t, power_w) for t in range(0, n + 2)]  # constant power_w from t=0..n+1
    return events, power


def test_p50_trims_quarters_and_amortizes():
    events, power = _run(n=100, power_w=700.0)
    m = window_amortize(events, power, 50)
    assert m.n_queries == 50                 # central 50 of 100 (drop 25 each end)
    # central tasks complete at t=26..75; window = [25, 75] -> dur 50s
    assert m.window_s == pytest.approx(50.0)
    assert m.avg_power_w == pytest.approx(700.0, rel=1e-3)   # constant power
    assert m.energy_j == pytest.approx(700.0 * 50.0, rel=1e-3)
    assert m.energy_per_query_j == pytest.approx(700.0 * 50.0 / 50.0, rel=1e-3)
    assert m.tokens_out_per_query == pytest.approx(50.0)


def test_total_window_is_all_queries():
    events, power = _run(n=100, power_w=700.0)
    m = window_amortize(events, power, 100)
    assert m.n_queries == 100
    assert m.window_s == pytest.approx(100.0)   # [0, 100]
    assert m.energy_per_query_j == pytest.approx(700.0 * 100.0 / 100.0, rel=1e-3)


def test_trim_widths_per_window():
    events, power = _run(n=100)
    a = amortize_all(events, power)
    assert (a[100].n_queries, a[90].n_queries, a[80].n_queries, a[50].n_queries) == (100, 90, 80, 50)


def test_dollars_local_amortized_over_window():
    events, power = _run(n=100, power_w=700.0)
    m = window_amortize(events, power, 50, gpu_hourly_rate=3.6, n_gpus=1)  # 50s window
    # $ = (50/3600)*3.6 = 0.05 over the window; /50 queries
    assert m.dollars == pytest.approx((50.0 / 3600.0) * 3.6, rel=1e-3)
    assert m.dollars_per_query == pytest.approx(m.dollars / 50, rel=1e-3)


def test_dollars_cloud_token_priced():
    events, power = _run(n=100)
    m = window_amortize(events, power, 100, price_in=1.0, price_out=2.0)
    # 100 queries * (100 in, 50 out): $ = (10000/1e6)*1 + (5000/1e6)*2
    assert m.dollars == pytest.approx((100 * 100 / 1e6) * 1.0 + (100 * 50 / 1e6) * 2.0, rel=1e-6)


def test_empty_is_safe():
    m = window_amortize([], [], 50)
    assert m.n_queries == 0 and m.energy_j == 0.0
