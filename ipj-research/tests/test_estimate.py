from __future__ import annotations

import pytest

from ipj.energy.estimate import energy_per_token, estimate_from_reference, estimate_joules


def test_energy_per_token_equal_weight():
    # 1000 J over 100 in + 100 out (equal weight) = 200 weighted tok -> 5 J/tok
    assert energy_per_token(1000.0, 100, 100) == pytest.approx(5.0)


def test_energy_per_token_output_weighted():
    # output_weight=3: 100 + 3*100 = 400 weighted -> 1000/400 = 2.5
    assert energy_per_token(1000.0, 100, 100, output_weight=3.0) == pytest.approx(2.5)


def test_energy_per_token_zero_tokens_raises():
    with pytest.raises(ValueError):
        energy_per_token(1000.0, 0, 0)


def test_estimate_joules_band():
    e = estimate_joules(100, 100, 5.0, band=0.3)          # 200 tok * 5 = 1000 J
    assert e.joules == pytest.approx(1000.0)
    assert e.lo == pytest.approx(700.0)
    assert e.hi == pytest.approx(1300.0)


def test_estimate_from_reference_transfer():
    # ref: 1000 J over 200 weighted tok -> 5 J/tok; target 50+50 -> 100 tok -> 500 J
    e = estimate_from_reference(50, 50, ref_joules=1000.0, ref_tokens_in=100, ref_tokens_out=100, band=0.2)
    assert e.j_per_token == pytest.approx(5.0)
    assert e.joules == pytest.approx(500.0)
    assert (e.lo, e.hi) == pytest.approx((400.0, 600.0))
    assert "ESTIMATE not measured" in e.basis


def test_negative_band_raises():
    with pytest.raises(ValueError):
        estimate_joules(100, 100, 5.0, band=-0.1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tokens_in": -1, "tokens_out": 0, "j_per_token": 1.0}, "tokens_in"),
        ({"tokens_in": True, "tokens_out": 0, "j_per_token": 1.0}, "tokens_in"),
        (
            {"tokens_in": 1, "tokens_out": 0, "j_per_token": float("nan")},
            "j_per_token",
        ),
        (
            {
                "tokens_in": 1,
                "tokens_out": 0,
                "j_per_token": 1.0,
                "output_weight": -1.0,
            },
            "output_weight",
        ),
        (
            {"tokens_in": 1, "tokens_out": 0, "j_per_token": 1.0, "band": 1.1},
            "band",
        ),
    ],
)
def test_invalid_estimate_inputs_fail_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        estimate_joules(**kwargs)


def test_reference_energy_must_be_finite_and_nonnegative():
    with pytest.raises(ValueError, match="joules"):
        estimate_from_reference(
            1,
            1,
            ref_joules=float("inf"),
            ref_tokens_in=1,
            ref_tokens_out=1,
        )
