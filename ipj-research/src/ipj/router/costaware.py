"""Cost-aware router head: escalate by expected accuracy gain per cloud-dollar.

The solvability head (:mod:`ipj.router.classifier`) ranks tasks purely by
``P(local-fail)`` and is **blind to cloud cost** — so at the ≥20× budget it escalates a few
*expensive* tasks and stalls at ~62.9% (EXP-20260709-01 §3). The oracle instead escalates
*cheap* cloud tasks and reaches 82.0%. This module productionizes the finding's
fully-realizable middle ground: predict every quantity leave-one-out from the task embedding
alone — no oracle values at inference — and rank by

    utility(t) = (1 - p_local(t)) * p_cloud(t) / cost_hat(t)         (descending)

where ``p_local`` = P(local solves) (the solvability head), ``p_cloud`` = P(cloud solves)
(a cloud-success head), and ``cost_hat`` = predicted cloud dollars (a cloud-cost head). On
the G1 89-task matrices this reaches **70.8% @ ≈24.5×** with 39 escalations.

**Direction (read this):** :func:`rank_by_utility` is **descending** utility — escalate the
*highest*-utility task first. That is the **opposite** of :func:`ipj.router.scorecache.ranked`,
which is **ascending** ``P(local-solves)`` (escalate the *lowest* first). To reuse the single
score-cache → escalate-set contract, :func:`cost_aware_scores` emits ``-utility`` as a
pseudo-``P(local-solves)``: ascending in ``-utility`` is descending in utility, so
``scorecache.ranked(cost_aware_scores(...))`` reproduces :func:`rank_by_utility` exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Pinned cost-model hyper-parameters — changing these moves the reproduced cost r(log$)=0.464
# and therefore the 70.8% realizable operating point.
N_COMPONENTS = 15
ALPHA = 1.0
RANDOM_STATE = 0
MIN_COST = 1e-6  # dollars; floor so log-cost is finite and division never blows up.


def loo_cost_model(
    X: Sequence[Sequence[float]],
    cloud_cost: Mapping[str, float],
    tasks: Sequence[str],
    *,
    n_components: int = N_COMPONENTS,
    alpha: float = ALPHA,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Leave-one-out predicted cloud cost ``{task: cost_hat}`` (dollars).

    Fits ``StandardScaler → PCA(n_components) → Ridge(alpha)`` on **log(cost)** and
    exponentiates the prediction back to dollars. For each task the model is refit on the
    other ``n-1`` rows, so no task ever sees its own cost label. Costs are clamped to
    ``>= MIN_COST`` on both the training target and the returned prediction. ``X`` rows
    align with ``tasks``.
    """
    Xa = np.asarray(X, dtype=float)
    n = len(tasks)
    if not Xa.shape[0] == n:
        raise ValueError(f"length mismatch: X={Xa.shape[0]}, tasks={n}")
    if n < 2:
        raise ValueError(f"leave-one-out needs >= 2 tasks, got {n}")

    y = np.array([math.log(max(float(cloud_cost[t]), MIN_COST)) for t in tasks])
    out: dict[str, float] = {}
    for i, t in enumerate(tasks):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=random_state),
            Ridge(alpha=alpha),
        )
        model.fit(Xa[mask], y[mask])
        pred = float(model.predict(Xa[i:i + 1])[0])
        out[t] = max(math.exp(pred), MIN_COST)
    return out


def utility(
    p_local: Mapping[str, float],
    p_cloud: Mapping[str, float],
    cost_hat: Mapping[str, float],
    tasks: Sequence[str],
) -> dict[str, float]:
    """Expected accuracy gain per cloud-dollar ``(1 - p_local) * p_cloud / cost_hat``.

    A task is worth escalating when the local model is *likely to fail* (``1 - p_local``
    high), the cloud model is *likely to solve it* (``p_cloud`` high), and it is *cheap*
    (``cost_hat`` low). ``cost_hat`` is floored at ``MIN_COST`` so a zero/negative predicted
    cost cannot produce an infinite or negative utility.
    """
    return {
        t: (1.0 - p_local[t]) * p_cloud[t] / max(cost_hat[t], MIN_COST)
        for t in tasks
    }


def rank_by_utility(util: Mapping[str, float]) -> list[str]:
    """Task names **descending** by utility — escalate the highest-utility task first.

    Ties break by task name (ascending) for determinism. This is the **opposite direction**
    from :func:`ipj.router.scorecache.ranked`, which is ascending ``P(local-solves)``; see
    the module docstring.
    """
    return sorted(util, key=lambda t: (-util[t], t))


def cost_aware_scores(
    p_local: Mapping[str, float],
    p_cloud: Mapping[str, float],
    cost_hat: Mapping[str, float],
    tasks: Sequence[str],
) -> dict[str, float]:
    """Map the cost-aware utility onto the score-cache ``{task: P(local-solves)}`` contract.

    Emits ``-utility`` as a *pseudo*-``P(local-solves)`` so the two heads share one
    interface: because :func:`ipj.router.scorecache.ranked` escalates lowest-first
    (ascending), and ascending ``-utility`` == descending utility, feeding this dict through
    ``scorecache.ranked`` / ``escalate_set`` / ``routed_curve`` yields exactly the
    highest-utility-first order of :func:`rank_by_utility`. The values are *not* calibrated
    probabilities — only their order (and its ties) is meaningful.
    """
    util = utility(p_local, p_cloud, cost_hat, tasks)
    return {t: -util[t] for t in tasks}
