"""End-to-end: load traces, gate/dedup/select, write the SFT jsonl plus stats."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..harness.trace import is_valid_nonnegative_number
from .filters import DropReason, dedup, diversity_summary, gate_trace, select_cost_optimal
from .loader import load_trajectories
from .sft import to_sft_example, write_jsonl


@dataclass(slots=True)
class Stats:
    n_loaded: int
    dropped: dict[str, int]
    n_kept: int
    strategy_distribution: dict[str, int]
    dollars_total: float | None
    dollars_mean: float | None
    gpu_energy_j_total: float | None
    dollars_known_n: int = 0
    gpu_energy_complete_n: int = 0


def build_sft_dataset(traces_dir: str | Path, out_dir: str | Path) -> Stats:
    out_dir = Path(out_dir)
    trajs = load_trajectories(traces_dir)
    dropped = {r.value: 0 for r in DropReason}

    gated = []
    for t in trajs:
        reason = gate_trace(t)
        if reason is None:
            gated.append(t)
        else:
            dropped[reason.value] += 1

    deduped, dups = dedup(gated)
    dropped[DropReason.DUPLICATE.value] += len(dups)

    kept, not_optimal = select_cost_optimal(deduped)
    dropped[DropReason.NOT_COST_OPTIMAL.value] += len(not_optimal)

    write_jsonl([to_sft_example(t) for t in kept], out_dir / "sft.jsonl")

    dollars = [
        t.outcome.total_dollars
        for t in kept
        if t.outcome is not None
        and is_valid_nonnegative_number(t.outcome.total_dollars)
    ]
    energies = [
        t.outcome.total_gpu_energy_j
        for t in kept
        if t.outcome is not None
        and t.outcome.gpu_energy_complete is True
        and is_valid_nonnegative_number(t.outcome.total_gpu_energy_j)
    ]
    complete_cost_coverage = len(dollars) == len(kept)
    complete_energy_coverage = len(energies) == len(kept)
    stats = Stats(
        n_loaded=len(trajs),
        dropped=dropped,
        n_kept=len(kept),
        strategy_distribution=diversity_summary(kept),
        dollars_total=(sum(dollars) if complete_cost_coverage else None),
        dollars_mean=(
            sum(dollars) / len(dollars)
            if complete_cost_coverage and dollars
            else None
        ),
        dollars_known_n=len(dollars),
        gpu_energy_j_total=(
            sum(energies)
            if complete_energy_coverage and energies
            else None
        ),
        gpu_energy_complete_n=len(energies),
    )
    (out_dir / "stats.json").write_text(json.dumps(asdict(stats), indent=2))
    return stats
