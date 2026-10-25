"""Trapezoidal integration of instantaneous power into energy (Joules)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PowerSample:
    """A single power reading: epoch seconds + instantaneous watts."""

    t: float
    watts: float


def energy_joules(samples: Iterable[PowerSample], start: float, end: float) -> float:
    """Trapezoidal integral of power over [start, end] → Joules.

    Samples are sorted by time and power is linearly interpolated between them.
    For window times outside the captured sample span, power is held flat
    (clamped to the first sample's watts before the first sample, and to the
    last sample's watts after the last sample). Returns 0.0 when end <= start
    or fewer than two samples are available.
    """
    if end <= start:
        return 0.0
    pts = sorted(samples, key=lambda s: s.t)
    if len(pts) < 2:
        return 0.0

    def watts_at(t: float) -> float:
        if t <= pts[0].t:
            return pts[0].watts
        if t >= pts[-1].t:
            return pts[-1].watts
        lo, hi = 0, len(pts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if pts[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = pts[lo], pts[hi]
        frac = (t - a.t) / (b.t - a.t)
        return a.watts + frac * (b.watts - a.watts)

    knots = [start] + [p.t for p in pts if start < p.t < end] + [end]
    total = 0.0
    for t0, t1 in zip(knots, knots[1:]):
        total += 0.5 * (watts_at(t0) + watts_at(t1)) * (t1 - t0)
    return total
