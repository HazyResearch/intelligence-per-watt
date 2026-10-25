"""k-attempt aggregation + bootstrap confidence intervals for TB V2.1 reporting (Wave A1).

Headline numbers must be reported as **mean ± 95% bootstrap CI over tasks** at **k attempts/task**,
not single-run point estimates. This module turns a list of ``TaskOutcome`` (Harbor emits one per
trial, so k attempts of a task share the same bare name) into:

* per-task solve *rate* (fraction of the task's attempts that the verifier resolved),
* run accuracy = mean solve rate, with a percentile-bootstrap CI (resampling tasks),
* the pass@1 / pass^k / any-of-k family, and
* a paired-difference CI for "does system A beat system B on the same tasks?".

Bootstrap resamples TASKS (the unit of independence), not attempts. Errored trials (timeout, etc.)
count as unsolved for accuracy — the verifier did not resolve them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ipj.harbor.per_task import TaskOutcome


@dataclass(frozen=True)
class AccuracyCI:
    accuracy: float      # mean per-task solve rate
    ci_low: float        # lower 95% bootstrap bound
    ci_high: float       # upper 95% bootstrap bound
    n_tasks: int
    k_mean: float        # mean attempts per task (≈ k)

    def pct(self) -> str:
        return f"{self.accuracy * 100:.1f}% [95% CI {self.ci_low * 100:.1f}–{self.ci_high * 100:.1f}]"


def per_task_solve_rate(outcomes: Iterable[TaskOutcome]) -> dict[str, float]:
    """{task: fraction of its attempts the verifier resolved}. Errored/unresolved count as 0."""
    agg: dict[str, list[float]] = {}
    for o in outcomes:
        agg.setdefault(o.task, []).append(1.0 if o.status == "resolved" else 0.0)
    return {t: sum(v) / len(v) for t, v in agg.items()}


def bootstrap_mean_ci(
    values: Sequence[float], *, n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the MEAN of ``values`` (resample with replacement).

    Deterministic given ``seed``. Returns ``(low, high)``; degenerate for n<2 (returns the point).
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        raise ValueError("need at least one value")
    if n == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def accuracy_ci(
    outcomes: Iterable[TaskOutcome], *, n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> AccuracyCI:
    """Run accuracy (mean per-task solve rate) with a bootstrap CI over tasks."""
    outcomes = list(outcomes)  # materialize: consumed twice below
    rates = per_task_solve_rate(outcomes)
    if not rates:
        raise ValueError("no task outcomes")
    vals = list(rates.values())
    lo, hi = bootstrap_mean_ci(vals, n_resamples=n_resamples, confidence=confidence, seed=seed)
    return AccuracyCI(
        accuracy=float(np.mean(vals)), ci_low=lo, ci_high=hi,
        n_tasks=len(rates), k_mean=len(outcomes) / len(rates),
    )


def _grouped(outcomes: Iterable[TaskOutcome]) -> dict[str, list[TaskOutcome]]:
    g: dict[str, list[TaskOutcome]] = {}
    for o in outcomes:
        g.setdefault(o.task, []).append(o)
    return g


@dataclass(frozen=True)
class PassMetrics:
    pass_at_1: float     # mean over all (task, attempt): expected single-attempt accuracy
    pass_hat_k: float    # fraction of tasks solved on EVERY attempt (worst-case robustness)
    any_of_k: float      # fraction of tasks solved on AT LEAST ONE attempt (best-case / oracle-of-k)


def pass_metrics(outcomes: Iterable[TaskOutcome]) -> PassMetrics:
    """pass@1 / pass^k / any-of-k over the k attempts recorded per task."""
    groups = _grouped(outcomes)
    if not groups:
        raise ValueError("no task outcomes")
    solved = {t: [o.status == "resolved" for o in os] for t, os in groups.items()}
    all_attempts = [s for flags in solved.values() for s in flags]
    return PassMetrics(
        pass_at_1=float(np.mean(all_attempts)),
        pass_hat_k=float(np.mean([all(f) for f in solved.values()])),
        any_of_k=float(np.mean([any(f) for f in solved.values()])),
    )


def paired_diff_ci(
    a_solved: dict[str, float], b_solved: dict[str, float], *,
    n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0,
) -> tuple[float, float, float]:
    """CI on (A − B) accuracy over the SHARED tasks (paired bootstrap).

    ``a_solved``/``b_solved`` map task → solve rate (e.g. from ``per_task_solve_rate``). Returns
    ``(mean_diff, low, high)``; if ``low > 0`` the whole CI is positive → A beats B significantly.
    """
    shared = sorted(set(a_solved) & set(b_solved))
    if not shared:
        raise ValueError("no shared tasks")
    diffs = np.array([a_solved[t] - b_solved[t] for t in shared], dtype=float)
    lo, hi = bootstrap_mean_ci(diffs, n_resamples=n_resamples, confidence=confidence, seed=seed)
    return float(diffs.mean()), lo, hi
