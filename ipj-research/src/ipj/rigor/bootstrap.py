"""Cluster-bootstrap CI + Codex-style unbiased pass@k estimator."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .trials import N_RESAMPLES, SEED


def cluster_boot_ci(
    stat_of_idx: Callable[[np.ndarray], float],
    n_tasks: int,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of a TASK resample (cluster bootstrap).

    ``stat_of_idx(idx)`` maps an integer index array (a task resample of size n_tasks) to the
    statistic value. Replicates where the statistic is undefined (NaN — e.g. no eligible task in
    the resample) are dropped via ``nanquantile``. Complements ``ipj.efficiency.stats.bootstrap_mean_ci``
    for statistics that are NOT the mean of a per-task vector (e.g. survival's pooled ordered pairs).
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_tasks, size=(n_resamples, n_tasks))
    vals = np.array([stat_of_idx(i) for i in idx])
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def pass_at_k_unbiased(n: int, s: int, k: int = 5) -> float:
    """Codex-style unbiased pass@k estimator ``1 - C(n-s, k) / C(n, k)`` (exact-unbiased for n >= k).

    For n < k this falls back to the any-of-n indicator, which is DOWNWARD-biased for pass@k
    (E = 1 - (1 - p)^n < 1 - (1 - p)^k); callers must disclose when the fallback fires.
    """
    if n >= k:
        return 1.0 - math.comb(n - s, k) / math.comb(n, k)
    return 1.0 if s >= 1 else 0.0
