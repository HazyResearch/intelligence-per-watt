from __future__ import annotations

from ipj.energy.integrate import PowerSample, energy_joules


def test_constant_power_full_window():
    s = [PowerSample(0.0, 100.0), PowerSample(10.0, 100.0)]
    assert energy_joules(s, 0.0, 10.0) == 1000.0


def test_constant_power_subwindow():
    s = [PowerSample(0.0, 100.0), PowerSample(10.0, 100.0)]
    assert energy_joules(s, 2.0, 8.0) == 600.0


def test_linear_ramp():
    s = [PowerSample(0.0, 0.0), PowerSample(10.0, 100.0)]
    assert energy_joules(s, 0.0, 10.0) == 500.0


def test_window_between_two_samples_interpolates():
    s = [PowerSample(0.0, 0.0), PowerSample(10.0, 100.0)]
    # avg power over [4,6] = 50W, * 2s = 100J
    assert energy_joules(s, 4.0, 6.0) == 100.0


def test_window_after_all_samples_holds_last_value():
    s = [PowerSample(0.0, 100.0), PowerSample(10.0, 100.0)]
    # window entirely after the last sample → power held flat at the last value
    assert energy_joules(s, 20.0, 30.0) == 1000.0


def test_too_few_points_returns_zero():
    assert energy_joules([PowerSample(1.0, 50.0)], 0.0, 10.0) == 0.0


def test_nonpositive_window_returns_zero():
    s = [PowerSample(0.0, 100.0), PowerSample(10.0, 100.0)]
    assert energy_joules(s, 5.0, 5.0) == 0.0


def test_negative_width_window_returns_zero():
    s = [PowerSample(0.0, 100.0), PowerSample(10.0, 100.0)]
    assert energy_joules(s, 8.0, 2.0) == 0.0
