"""Unit tests for ipj.rigor.reporting: format_pct, format_ci, truncate_list."""

from __future__ import annotations

from ipj.rigor.reporting import format_ci, format_pct, truncate_list


class TestFormatPct:
    def test_zero(self):
        assert format_pct(0.0) == "0.0%"

    def test_one(self):
        assert format_pct(1.0) == "100.0%"

    def test_typical(self):
        assert format_pct(0.751) == "75.1%"

    def test_rounds_to_one_decimal(self):
        assert format_pct(0.751234) == "75.1%"
        assert format_pct(0.7517) == "75.2%"


class TestFormatCI:
    def test_tuple(self):
        assert format_ci((0.749, 0.788)) == "[74.9, 78.8]"

    def test_list(self):
        assert format_ci([0.726, 0.866]) == "[72.6, 86.6]"


class TestTruncateList:
    def test_short_list_no_truncation(self):
        assert truncate_list(["a", "b", "c"]) == "a, b, c"

    def test_exactly_k_no_truncation(self):
        items = list(range(5))
        assert truncate_list(items, k=5) == "0, 1, 2, 3, 4"

    def test_long_list_truncates_with_more_suffix(self):
        items = list(range(15))
        out = truncate_list(items, k=5)
        assert out == "0, 1, 2, 3, 4 ... (+10 more)"

    def test_default_k_is_10(self):
        items = list(range(12))
        out = truncate_list(items)
        assert "... (+2 more)" in out
