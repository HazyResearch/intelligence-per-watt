from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ipj.efficiency.energy import attribute_energy, load_power_series

# ---------------------------------------------------------------------------
# load_power_series
# ---------------------------------------------------------------------------


def test_load_power_series_parses_and_sorts(tmp_path: Path):
    # Write samples out of order; expect sorted by epoch
    content = textwrap.dedent("""\
        1000.5 200.0
        999.0 150.0
        1001.0 250.0
    """)
    p = tmp_path / "power.log"
    p.write_text(content)

    result = load_power_series(p)
    assert len(result) == 3
    # sorted by epoch ascending
    assert result[0] == (999.0, 150.0)
    assert result[1] == (1000.5, 200.0)
    assert result[2] == (1001.0, 250.0)


def test_load_power_series_skips_malformed_lines(tmp_path: Path):
    content = textwrap.dedent("""\
        1000.0 100.0
        this is junk
        1001.0 110.0
        # comment line
        1002.0 not_a_number
        1003.0 120.0
    """)
    p = tmp_path / "power.log"
    p.write_text(content)

    result = load_power_series(p)
    assert len(result) == 3
    epochs = [t for t, _w in result]
    assert epochs == [1000.0, 1001.0, 1003.0]


def test_load_power_series_empty_file(tmp_path: Path):
    p = tmp_path / "empty.log"
    p.write_text("")
    assert load_power_series(p) == []


def test_load_power_series_accepts_str_path(tmp_path: Path):
    p = tmp_path / "power.log"
    p.write_text("1000.0 100.0\n1001.0 110.0\n")
    result = load_power_series(str(p))
    assert len(result) == 2


def test_load_power_series_all_junk(tmp_path: Path):
    p = tmp_path / "junk.log"
    p.write_text("abc def\n\nhello world\n")
    assert load_power_series(p) == []


# ---------------------------------------------------------------------------
# attribute_energy
# ---------------------------------------------------------------------------


def test_attribute_energy_constant_100w_over_10s():
    # 100 W constant for 10 s → 1000 J
    power = [(float(t), 100.0) for t in range(11)]  # t=0..10, 1s intervals
    j = attribute_energy(power, 0.0, 10.0)
    assert j == pytest.approx(1000.0)


def test_attribute_energy_gap_greater_than_5s_skipped():
    # Two segments: [0,2] at 100W and [10,12] at 100W; gap of 8s is skipped.
    # Each valid segment: trapezoid at 100W over 2s = 200J each → total 400J
    power = [
        (0.0, 100.0),
        (1.0, 100.0),
        (2.0, 100.0),
        # 8-second gap here (> 5 s threshold) — should be skipped
        (10.0, 100.0),
        (11.0, 100.0),
        (12.0, 100.0),
    ]
    j = attribute_energy(power, 0.0, 12.0)
    assert j == pytest.approx(400.0)


def test_attribute_energy_gap_exactly_5s_included():
    # dt = 5.0 exactly is NOT > 5, so it should be included (boundary condition)
    power = [
        (0.0, 100.0),
        (5.0, 100.0),
        (10.0, 100.0),
    ]
    j = attribute_energy(power, 0.0, 10.0)
    # 5s * 100W + 5s * 100W = 1000J (no gap skipped)
    assert j == pytest.approx(1000.0)


def test_attribute_energy_out_of_window_points_excluded():
    # Points far outside [start-2, end+2] are excluded from seg
    power = [
        (0.0, 999.0),   # before start - 2 = 8; excluded
        (9.0, 100.0),   # inside window (start-2=8 ≤ 9)
        (11.0, 100.0),
        (20.0, 999.0),  # after end + 2 = 12; excluded
    ]
    j = attribute_energy(power, 10.0, 11.0)
    # Only the segment (9,100) → (11,100) contributes; 100W * 2s = 200J
    # But within [10,11] via the trapezoid of the single interval (9→11)
    # dt = 2, avg watts = 100, j += 100 * 2 = 200J
    assert j == pytest.approx(200.0)


def test_attribute_energy_empty_power():
    assert attribute_energy([], 0.0, 10.0) == 0.0


def test_attribute_energy_no_points_in_window():
    power = [(100.0, 200.0), (200.0, 300.0)]
    # window [0, 5] is outside all points (start-2 = -2, end+2 = 7, but no points there)
    assert attribute_energy(power, 0.0, 5.0) == 0.0
