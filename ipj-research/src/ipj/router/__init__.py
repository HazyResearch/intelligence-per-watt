"""G1 solvability + cost-aware router (labeled-data campaign).

Four cooperating pieces, all local-tier agnostic:

* :mod:`ipj.router.classifier` — the reusable embedding→PCA→logistic solvability
  classifier (leave-one-out P(local solves)) + a Mann-Whitney AUC.
* :mod:`ipj.router.costaware` — the primary cost-aware head: leave-one-out cloud-cost
  model plus the ``(1-p_local)*p_cloud/cost_hat`` utility, ranked highest-first, mapped
  back onto the shared score-cache contract.
* :mod:`ipj.router.scorecache` — the score-cache → escalate-set adapter (the pinned
  §7.3 contract): rank ascending by P(local-solves), escalate lowest-first, then trace
  the routed operating curve via ``ipj.efficiency.operating_curve.routed_point``.
* :mod:`ipj.router.dataset` — the router training-data writer: the §7.2 row schema plus
  the classifier-table and SFT-messages emitters.
"""

from __future__ import annotations
