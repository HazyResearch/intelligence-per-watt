"""Statistical machinery for k>1 serving-run analysis.

Extracted from EXP-20260711-01-m3-k5-serving/fold_k5.py, which was previously
imported by EXP-20260714-01-gpt55-k5-serving/fold_gpt55_k5.py via an
`importlib.util.spec_from_file_location` hack. Any downstream `fold_*.py`
should now `from ipj.rigor import ...` and unit tests live at
``ipj/tests/rigor/``.

Distinct from ``ipj.efficiency.stats`` (which operates on ``TaskOutcome``
and provides per-attempt aggregation, pass_metrics, and paired_diff_ci):
this module operates on the richer ``Trial`` dataclass (has_exception,
started/finished timestamps, jobs-dir provenance) that k>1 serving-run
harvests need.
"""

from __future__ import annotations

from .bootstrap import cluster_boot_ci, pass_at_k_unbiased
from .estimators import k1_consistency, primary_stats, survival_stats
from .reporting import format_ci, format_pct, truncate_list
from .trials import (
    AUDIT_LIST_CAP,
    INFRA_MSG,
    MC_SEED,
    N_RESAMPLES,
    PDT,
    SEED,
    Trial,
    capped_append,
    group_by_task,
    is_infra,
    parse_ts,
)

__all__ = [
    "AUDIT_LIST_CAP",
    "INFRA_MSG",
    "MC_SEED",
    "N_RESAMPLES",
    "PDT",
    "SEED",
    "Trial",
    "capped_append",
    "cluster_boot_ci",
    "format_ci",
    "format_pct",
    "group_by_task",
    "is_infra",
    "k1_consistency",
    "parse_ts",
    "pass_at_k_unbiased",
    "primary_stats",
    "survival_stats",
    "truncate_list",
]
