"""Parse Harbor's job-level result.json into the few numbers a Cell needs.

Schema verified against Harbor 0.13.2 (see ipj/tests/fixtures/harbor_summary.json):
the job result nests accuracy under stats.evals["<agent>__<dataset>"].metrics[0].mean,
the task count under n_total_trials, and token totals under stats.n_input_tokens /
stats.n_output_tokens (null when the agent makes no model calls, e.g. the oracle).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HarborSummary:
    n_tasks: int
    accuracy: float
    tokens_in: int
    tokens_out: int
    cost_usd: float | None = None  # Harbor's own billed $ for the run (null for the oracle / no model)


def parse_harbor_summary(path: Path | str) -> HarborSummary:
    data = json.loads(Path(path).read_text())
    n_tasks = int(data.get("n_total_trials") or 0)
    stats = data.get("stats") or {}
    evals = stats.get("evals") or {}
    accuracy = 0.0
    if evals:
        first_eval = next(iter(evals.values()))  # one entry per (agent, dataset)
        metrics = first_eval.get("metrics") or []
        if metrics and metrics[0].get("mean") is not None:
            accuracy = float(metrics[0]["mean"])
    tokens_in = int(stats.get("n_input_tokens") or 0)
    tokens_out = int(stats.get("n_output_tokens") or 0)
    raw_cost = stats.get("cost_usd")
    cost_usd = float(raw_cost) if raw_cost is not None else None
    return HarborSummary(
        n_tasks=n_tasks, accuracy=accuracy, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd
    )
