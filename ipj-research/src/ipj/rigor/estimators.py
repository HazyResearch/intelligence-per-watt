"""Primary + survival + k=1-consistency estimators for k>1 serving-run analysis.

All bootstrap/MC seeds and iteration counts default to ``ipj.rigor.trials`` module constants
(SEED, N_RESAMPLES, MC_SEED) so numerical outputs are bit-identical with the pre-extraction
in-experiment definitions in ``EXP-20260711-01-m3-k5-serving/fold_k5.py``.
"""

from __future__ import annotations

import numpy as np

from ipj.efficiency.stats import (
    accuracy_ci,
    bootstrap_mean_ci,
    paired_diff_ci,
    pass_metrics,
)
from ipj.harbor.per_task import TaskOutcome

from .bootstrap import cluster_boot_ci, pass_at_k_unbiased
from .trials import MC_SEED, N_RESAMPLES, SEED, Trial


def primary_stats(groups: dict[str, list[Trial]], *, k: int = 5) -> dict:
    """k-attempt primary stats: per-attempt pooled, per-task pass@1, exact-k pass@k / pass^k / gap.

    Reuses ``ipj.efficiency.stats.accuracy_ci`` on TaskOutcome so pass@1 numerically matches
    the (single-source) significance-table convention. The EXACT-k subset (n_t == k) is what the
    citable pass@k / pass^k / fragility gap use; the mixed-n_t "diagnostics" block quantifies
    the opposite-direction biases from folding n_t != k tasks in.
    """
    tasks = sorted(groups)
    s_arr = np.array([sum(tr.solved for tr in groups[t]) for t in tasks], dtype=float)
    n_arr = np.array([len(groups[t]) for t in tasks], dtype=float)

    outcomes = [
        TaskOutcome(
            t,
            "resolved" if tr.solved else ("errored" if tr.has_exception else "unresolved"),
            1.0 if tr.solved else 0.0,
        )
        for t in tasks
        for tr in groups[t]
    ]
    acc = accuracy_ci(outcomes, n_resamples=N_RESAMPLES, seed=SEED)
    pm = pass_metrics(outcomes)

    pooled = float(s_arr.sum() / n_arr.sum())
    pooled_ci = cluster_boot_ci(lambda i: s_arr[i].sum() / max(n_arr[i].sum(), 1.0), len(tasks))

    exact_tasks = [t for t in tasks if len(groups[t]) == k]
    exact_s = [int(sum(tr.solved for tr in groups[t])) for t in exact_tasks]
    if exact_tasks:
        p_vals = [pass_at_k_unbiased(k, s, k=k) for s in exact_s]
        ph_vals = [1.0 if s == k else 0.0 for s in exact_s]
        ao_vals = [1.0 if s >= 1 else 0.0 for s in exact_s]
        gap_vals = [p - h for p, h in zip(p_vals, ph_vals)]
        exact_k = {
            "n_tasks_nt_eq_k": len(exact_tasks),
            f"pass_at_{k}_unbiased": float(np.mean(p_vals)),
            f"pass_at_{k}_ci": list(bootstrap_mean_ci(p_vals, n_resamples=N_RESAMPLES, seed=SEED)),
            "pass_hat_k": float(np.mean(ph_vals)),
            "pass_hat_k_ci": list(bootstrap_mean_ci(ph_vals, n_resamples=N_RESAMPLES, seed=SEED)),
            "any_of_k": float(np.mean(ao_vals)),
            "fragility_gap_pt": float(np.mean(gap_vals) * 100.0),
            "fragility_gap_ci_pt": [c * 100.0 for c in
                                    bootstrap_mean_ci(gap_vals, n_resamples=N_RESAMPLES, seed=SEED)],
        }
    else:
        exact_k = {"n_tasks_nt_eq_k": 0,
                   "note": f"no tasks with n_t == {k} — pass@{k} / pass^{k} / fragility gap not computable"}

    p_all = [pass_at_k_unbiased(int(n), int(s), k=k) for n, s in zip(n_arr, s_arr)]
    n_fallback = int((n_arr < k).sum())
    diagnostics = {
        f"pass_at_{k}_mixed_nt": float(np.mean(p_all)),
        f"pass_at_{k}_mixed_nt_ci": list(bootstrap_mean_ci(p_all, n_resamples=N_RESAMPLES, seed=SEED)),
        "n_tasks_any_of_n_fallback": n_fallback,
        "any_of_k_indicator_all_tasks": pm.any_of_k,
        "pass_hat_k_all_attempts_all_tasks": pm.pass_hat_k,
        "note": f"mixes n_t != {k}: pass^k counts tasks solved on every attempt regardless of "
                f"attempt count (biased UP for n_t<{k}); pass@{k} uses the any-of-n indicator for "
                f"n_t<{k} (biased DOWN). Diagnostics only — NOT for the significance table.",
    }

    hist = {i: 0 for i in range(k + 1)}
    off_k = {}
    for t, s, n in zip(tasks, s_arr, n_arr):
        if int(n) == k:
            hist[int(s)] += 1
        else:
            off_k[t] = {"s": int(s), "n": int(n)}
    return {
        "n_tasks": len(tasks),
        "k_mean": float(acc.k_mean),
        "n_trials": int(n_arr.sum()),
        "per_attempt_pooled": pooled,
        "per_attempt_pooled_ci": list(pooled_ci),
        "per_attempt_pooled_check_pass_metrics": pm.pass_at_1,
        "pass_at_1_per_task_mean": acc.accuracy,
        "pass_at_1_ci": [acc.ci_low, acc.ci_high],
        "exact_k": exact_k,
        "diagnostics_all_tasks": diagnostics,
        f"s_histogram_nt_eq_{k}": hist,
        f"tasks_nt_ne_{k}": off_k,
        "per_task": {t: {"s": int(s), "n": int(n), "rate": float(s / n)}
                     for t, s, n in zip(tasks, s_arr, n_arr)},
    }


def survival_stats(
    groups: dict[str, list[Trial]],
    *,
    opus_borrowed: dict | None = None,
) -> dict:
    """P(another roll solves | task solved), with the Opus comparison estimator-matched.

    PRIMARY (Opus-comparable): pooled ordered-pairs sum s_t(s_t-1) / sum s_t(n_t-1) —
    solve-event-weighted (estimand E[p^2]/E[p]), the EXACT estimator behind the Opus-borrowed
    80.3% (ceiling_bias.py: 2*both/n_solves at n=2). The pre-extraction fold_k5 loaded the Opus
    number from ceiling_bias.json; here, callers pass ``opus_borrowed`` explicitly to decouple
    from experiment paths (defaults to the hardcoded EXP-20260710-05 fallback).

    SECONDARY (task-weighted, NOT Opus-comparable): fixed-anchor mean. For each roll index j,
    over tasks whose roll j solved, (s_t - 1) / (n_t - 1) is unbiased for p_t; the estimate is
    the mean over those tasks, averaged over anchor indices j. The naive per-task mean of
    (s_t - 1) / (n_t - 1) over tasks with s_t >= 1 is NOT used — conditioning on task
    eligibility biases it DOWN (several pt at n=5).
    """
    if opus_borrowed is None:
        opus_borrowed = {
            "survival": 0.803,
            "ci": [0.726, 0.866],
            "source": "EXP-20260710-05 (hardcoded fallback)",
        }

    tasks = sorted(t for t in groups if len(groups[t]) > 1)
    s = np.array([sum(tr.solved for tr in groups[t]) for t in tasks], dtype=float)
    n = np.array([len(groups[t]) for t in tasks], dtype=float)
    elig = s >= 1
    n_singleton_solved = sum(1 for t in groups if len(groups[t]) == 1 and groups[t][0].solved)

    out: dict = {
        "n_tasks_multi_roll": len(tasks),
        "n_eligible_tasks": int(elig.sum()),
        "n_solved_singletons_excluded": n_singleton_solved,
        "opus_borrowed": opus_borrowed,
    }
    if elig.sum() == 0:
        out["note"] = "no task has s_t>=1 with n_t>1 — survival not estimable on this sample"
        return out

    num, den = float((s * (s - 1.0)).sum()), float((s * (n - 1.0)).sum())
    pooled = num / den if den > 0 else float("nan")

    def _pooled_of(idx):
        ss, nn = s[idx], n[idx]
        d = (ss * (nn - 1.0)).sum()
        return (ss * (ss - 1.0)).sum() / d if d > 0 else float("nan")

    plo, phi = cluster_boot_ci(_pooled_of, len(tasks))

    max_n = int(n.max())
    mat = np.full((len(tasks), max_n), np.nan)
    for i, t in enumerate(tasks):
        flags = [bool(tr.solved) for tr in groups[t]]
        st, nt = sum(flags), len(flags)
        loo = (st - 1.0) / (nt - 1.0)
        for j, f in enumerate(flags):
            if f:
                mat[i, j] = loo

    def _anchor_est(m: np.ndarray) -> float:
        counts = (~np.isnan(m)).sum(axis=0)
        cols = counts > 0
        if not cols.any():
            return float("nan")
        sums = np.nansum(m, axis=0)
        return float(np.mean(sums[cols] / counts[cols]))

    anchor = _anchor_est(mat)
    alo, ahi = cluster_boot_ci(lambda idx: _anchor_est(mat[idx]), len(tasks))

    out.update({
        "pooled_ordered_pairs": pooled,
        "pooled_ci": [plo, phi],
        "pooled_estimand": "solve-event-weighted E[p^2]/E[p] — the exact estimator behind the "
                           "Opus-borrowed 80.3% (2*both/n_solves); PRIMARY, Opus-comparable",
        "delta_vs_opus_pt": float((pooled - opus_borrowed["survival"]) * 100.0),
        "opus_inside_pooled_ci": bool(plo <= opus_borrowed["survival"] <= phi),
        "task_weighted_fixed_anchor_mean": anchor,
        "task_weighted_fixed_anchor_ci": [alo, ahi],
        "task_weighted_estimand": "unweighted-over-tasks mean p via fixed-roll anchoring, averaged "
                                  "over roll indices; DIFFERENT estimand — never compared to the "
                                  "Opus 80.3%",
    })
    return out


def k1_consistency(groups: dict[str, list[Trial]], k1_map: dict[str, bool]) -> dict:
    """Is the committed k=1 M3 run (field-table 49/89) a plausible draw from the k=5 rates?

    Aggregate: posterior-predictive Monte Carlo — per task ``p_t ~ Beta(s_t + 0.5, n_t - s_t + 0.5)``
    (Jeffreys, so the 5-roll estimation noise is propagated), draw a k=1 outcome, compare the
    simulated solve total to the observed k=1 total (two-sided MC p). Plus a paired-bootstrap CI
    (via ``paired_diff_ci``) on (k5 rate - k1 outcome) over the joined tasks.

    Per-task: Jeffreys predictive prob of the observed k=1 outcome, ``(s + 0.5) / (n + 1)`` if
    solved else ``(n - s + 0.5) / (n + 1)``; flags < 0.1 and hard disagreements
    (k1 solved & s_t=0 / k1 failed & s_t=n_t) are listed for audit.
    """
    joined = sorted(set(groups) & set(k1_map))
    out: dict = {
        "n_joined": len(joined),
        "k5_tasks_missing_from_vector": sorted(set(groups) - set(k1_map)),
        "vector_tasks_missing_from_k5": sorted(set(k1_map) - set(groups)),
    }
    if not joined:
        out["note"] = "no joined tasks — consistency check skipped"
        return out

    s = np.array([sum(tr.solved for tr in groups[t]) for t in joined], dtype=float)
    n = np.array([len(groups[t]) for t in joined], dtype=float)
    k1 = np.array([1.0 if k1_map[t] else 0.0 for t in joined])
    obs = int(k1.sum())

    rng = np.random.default_rng(MC_SEED)
    nsim = 20_000
    p = rng.beta(s + 0.5, n - s + 0.5, size=(nsim, len(joined)))
    totals = (rng.random((nsim, len(joined))) < p).sum(axis=1)
    p_le, p_ge = float((totals <= obs).mean()), float((totals >= obs).mean())

    rates = {t: float(si / ni) for t, si, ni in zip(joined, s, n)}
    k1_dict = {t: float(v) for t, v in zip(joined, k1)}
    d, lo, hi = paired_diff_ci(rates, k1_dict, n_resamples=N_RESAMPLES, seed=SEED)

    per_task_pred = []
    for t, si, ni, kk in zip(joined, s, n, k1):
        pred = (si + 0.5) / (ni + 1.0) if kk >= 1.0 else (ni - si + 0.5) / (ni + 1.0)
        per_task_pred.append({
            "task": t, "s": int(si), "n": int(ni), "k1_solved": bool(kk),
            "jeffreys_pred_of_observed": float(pred),
            "flag_low_pred": bool(pred < 0.10),
            "hard_disagreement": bool((kk >= 1.0 and si == 0) or (kk < 1.0 and si == ni)),
        })

    out.update({
        "k5_pass1_per_task_mean": float((s / n).mean()),
        "k1_field_table_pass1": float(obs / len(joined)),
        "k1_field_table_count": [obs, len(joined)],
        "paired_diff_k5_minus_k1": d, "paired_diff_ci": [lo, hi],
        "posterior_predictive": {
            "obs_solves": obs, "nsim": nsim, "mc_seed": MC_SEED,
            "p_le": p_le, "p_ge": p_ge,
            "two_sided_p": float(2.0 * min(p_le, p_ge)),
        },
        "per_task": per_task_pred,
        "n_low_pred": sum(1 for r in per_task_pred if r["flag_low_pred"]),
        "n_hard_disagreements": sum(1 for r in per_task_pred if r["hard_disagreement"]),
    })
    return out
