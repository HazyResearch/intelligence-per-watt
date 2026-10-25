"""Per-task energy attribution helpers for the efficiency framework.

Reads a raw power log (epoch seconds + watts per line) and integrates it
trapeziodally over a task window, skipping sampler-restart gaps > 5 s.
This mirrors the logic in analyze_cascade.integrate used for cascade runs.
"""

from __future__ import annotations

from pathlib import Path


def load_power_series(path: Path | str) -> list[tuple[float, float]]:
    """Read a power log file and return samples sorted by epoch.

    Each non-malformed line must contain exactly two whitespace-separated
    fields: an epoch timestamp (float) and instantaneous watts (float).
    Lines that cannot be parsed as two floats are silently skipped.

    Args:
        path: Path to the power log file (string or Path).

    Returns:
        List of (epoch_seconds, watts) tuples sorted ascending by epoch.
    """
    samples: list[tuple[float, float]] = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                t = float(parts[0])
                w = float(parts[1])
            except ValueError:
                continue
            samples.append((t, w))
    samples.sort(key=lambda x: x[0])
    return samples


def attribute_energy(
    power: list[tuple[float, float]],
    start_epoch: float,
    end_epoch: float,
) -> float:
    """Trapezoidal Joules over [start_epoch, end_epoch], skipping gaps > 5 s.

    Only power samples within the 2-second guard band around the task window
    (i.e. start_epoch - 2 ≤ t ≤ end_epoch + 2) are considered. Adjacent
    sample pairs whose time delta exceeds 5 seconds are treated as sampler
    restarts and their interval is excluded from the integral.

    Args:
        power: List of (epoch_seconds, watts) tuples (need not be sorted).
        start_epoch: Task start time in epoch seconds.
        end_epoch: Task end time in epoch seconds.

    Returns:
        Energy in Joules, or 0.0 if no valid intervals exist.
    """
    seg = [(t, w) for t, w in power if start_epoch - 2 <= t <= end_epoch + 2]
    j = 0.0
    for i in range(1, len(seg)):
        dt = seg[i][0] - seg[i - 1][0]
        if 0 < dt <= 5:
            j += 0.5 * (seg[i][1] + seg[i - 1][1]) * dt
    return j
