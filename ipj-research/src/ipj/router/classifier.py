"""Reusable solvability classifier: embeddings → PCA → L2-logistic, leave-one-out.

Ported from ``EXP-20260708-02/trained_router.py`` (and the validated M3 baseline scratch
script). The pipeline is pinned to the numbers the G1 campaign reproduces:
``StandardScaler → PCA(n_components=15, random_state=0) → LogisticRegression(C=0.3, max_iter=1000)``.

The classifier is **local-tier agnostic**: it never mentions GLM or MiniMax-M3. Callers pass
``y`` = whatever local tier's solve label they harvested, so the same code trains the GLM,
MiniMax-M3, or any future local head. Outputs a ``{task: P(local solves)}`` score cache
(§7.3) that :mod:`ipj.router.scorecache` turns into an escalate set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# Pinned hyper-parameters — changing these moves the reproduced AUCs (GLM 0.553 / M3 0.641).
N_COMPONENTS = 15
C = 0.3
MAX_ITER = 1000
RANDOM_STATE = 0


def build_pipeline(
    *,
    n_components: int = N_COMPONENTS,
    C: float = C,
    max_iter: int = MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """The pinned StandardScaler → PCA → L2-logistic pipeline (fresh, unfitted)."""
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=random_state),
        LogisticRegression(C=C, max_iter=max_iter),
    )


def loo_scores(
    X: Sequence[Sequence[float]],
    y: Sequence[bool | int],
    tasks: Sequence[str],
    *,
    n_components: int = N_COMPONENTS,
    C: float = C,
    max_iter: int = MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Leave-one-out ``{task: P(local solves)}``.

    For each task, refit the pipeline on the other ``n-1`` rows and predict the held-out
    task's ``P(local solves)`` — no label leakage. ``X`` rows align with ``tasks`` and ``y``.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray([int(bool(v)) for v in y])
    n = len(tasks)
    if not (Xa.shape[0] == len(ya) == n):
        raise ValueError(f"length mismatch: X={Xa.shape[0]}, y={len(ya)}, tasks={n}")
    if n < 2:
        raise ValueError(f"leave-one-out needs >= 2 tasks, got {n}")

    preds: dict[str, float] = {}
    for i, t in enumerate(tasks):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        clf = build_pipeline(n_components=n_components, C=C, max_iter=max_iter, random_state=random_state)
        clf.fit(Xa[mask], ya[mask])
        preds[t] = float(clf.predict_proba(Xa[i:i + 1])[0, 1])
    return preds


def auc(
    preds: Mapping[str, float],
    solved: Mapping[str, bool],
    tasks: Sequence[str],
) -> float:
    """Mann-Whitney AUC of ``preds`` vs the binary ``solved`` label (ties count 0.5).

    Returns NaN when one class is empty (AUC undefined). Higher preds should coincide with
    ``solved`` = True, so this measures how well the score cache ranks solves above fails.
    """
    pos = [preds[t] for t in tasks if solved[t]]
    neg = [preds[t] for t in tasks if not solved[t]]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
