"""v1 per-query escalation cascade: run local, escalate errored tasks to the cloud baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ...cascade import CascadeResult, Tier, cascade_ladder
from ...cell import find_latest_result_json
from ...cost import H100_HOURLY_USD, dollars_local
from ...energy.capture import RunEnergyCapture
from ...harbor.per_task import per_task_outcomes
from ...harbor.results import parse_harbor_summary
from ...harbor.run import harbor_command, run_command
from ...metrics import (
    MEASURED_DEDICATED_JOULES_PROVENANCE,
    UNKNOWN_JOULES_PROVENANCE,
    Cell,
    accelerator_joules_are_complete,
    resolve_joules_provenance,
)
from .signal import escalate_when_oracle, escalate_when_realized


@dataclass
class V1Result:
    realized: CascadeResult
    oracle: CascadeResult
    escalated: list[str]
    realized_cell: Cell
    oracle_cell: Cell


def _outcomes_map(result_json) -> dict[str, str]:
    return {o.task: o.status for o in per_task_outcomes(result_json)}


def _cell(
    model,
    dataset,
    res: CascadeResult,
    joules,
    wall,
    run_dir,
    energy_provenance: str,
    accelerator_device_coverage_complete: bool,
) -> Cell:
    cloud_used = len(res.tier_attempts) > 1 and res.tier_attempts[1] > 0
    measured_local = joules is not None
    joules_provenance = resolve_joules_provenance(joules, energy_provenance)
    if not measured_local:
        energy_source = "unknown local energy (telemetry unavailable or incomplete run-window coverage)"
    elif joules_provenance == MEASURED_DEDICATED_JOULES_PROVENANCE:
        energy_source = "energy-monitor (local accelerator only; dedicated isolation declared)"
    else:
        energy_source = "energy-monitor (local accelerator only; isolation unverified)"
    return Cell(
        model=model, harness="terminus-2-cascade-v1", hardware="h100-local+cloud", dataset=dataset,
        n_tasks=res.n_tasks, accuracy=res.accuracy, tokens_in=0, tokens_out=0,
        dollars=res.dollars,
        dollars_basis="gpu-hours+api" if cloud_used else "gpu-hours",
        joules=joules,
        wall_clock_s=wall,
        energy_source=energy_source,
        run_dir=str(run_dir),
        joules_provenance=joules_provenance,
        joules_scope="local-accelerator",
        joules_complete=(
            accelerator_joules_are_complete(
                joules,
                joules_provenance,
                accelerator_device_coverage_complete,
            )
            and not cloud_used
        ),
        accelerator_device_coverage_complete=(
            accelerator_device_coverage_complete
        ),
    )


def run_v1(
    *,
    local_model: str,
    dataset: str,
    output_dir,
    local_base_url: str,
    cloud_baseline_result,
    cloud_label: str = "gpt-5.5",
    local_api_key: str = "EMPTY",
    k: int = 1,
    n_tasks: int | None = None,
    n_concurrent: int = 1,
    agent_timeout_multiplier: float | None = None,
    gpu_hourly_rate: float = H100_HOURLY_USD,
    n_gpus: int = 1,
    energy_provenance: str = UNKNOWN_JOULES_PROVENANCE,
    run_fn=run_command,
    capture_factory=RunEnergyCapture,
) -> V1Result:
    output_dir = Path(output_dir)
    local_dir = output_dir / "local"

    # --- Local pass, under energy capture ---
    # n_concurrent>1 saturates the local serve (per the inference-config standard); the resulting
    # batch energy is amortized over the central percentile window downstream, not per single task.
    # agent_timeout_multiplier<1 = fail-fast: the local model gives up sooner on hard tasks so it
    # doesn't burn GPU-hours on work it will escalate anyway (the dominant cost in the naive cascade).
    local_cmd = harbor_command(dataset, "terminus-2", f"openai/{local_model}",
                               output_path=local_dir, k=k, n_tasks=n_tasks, n_concurrent=n_concurrent,
                               agent_timeout_multiplier=agent_timeout_multiplier)
    local_env = {**os.environ, "OPENAI_BASE_URL": local_base_url, "OPENAI_API_KEY": local_api_key}
    with capture_factory() as cap:
        win = run_fn(local_cmd, env=local_env)
        local_joules = cap.joules(win.start, win.end)
        device_coverage_complete = bool(
            getattr(cap, "accelerator_device_coverage_complete", lambda *_: False)(
                win.start,
                win.end,
            )
        )
    if win.returncode != 0:
        raise RuntimeError(f"local harbor pass failed (rc={win.returncode})")
    local_map = _outcomes_map(find_latest_result_json(local_dir))

    # --- Cloud outcomes: reuse the existing baseline (cloud-from-scratch on all tasks) ---
    cloud_map = _outcomes_map(cloud_baseline_result)
    cloud_total = parse_harbor_summary(cloud_baseline_result).cost_usd

    n = len(local_map)
    wall = win.end - win.start
    local_cost_pt = dollars_local(wall, gpu_hourly_rate, n_gpus=n_gpus) / max(n, 1)
    # The legacy cascade calculator requires a float per tier. A temporary zero
    # is safe only to discover whether the tier was used; a used unknown cloud
    # bill fails closed before any result Cell can be emitted.
    cloud_cost_pt = (
        cloud_total / max(len(cloud_map), 1)
        if cloud_total is not None
        else 0.0
    )
    local_j_pt = None if local_joules is None else local_joules / max(n, 1)

    local_tier = Tier("local", local_map, cost_per_task=local_cost_pt, joules_per_task=local_j_pt)
    # Cloud joules are not locally measured -> None (cascade joules become None for the cloud-touched
    # ladder; the realized cell still records the measured local joules).
    cloud_tier = Tier(cloud_label, cloud_map, cost_per_task=cloud_cost_pt, joules_per_task=None)

    # Restrict to exactly the tasks the local pass ran; the cloud tier still supplies outcomes
    # for escalated tasks but cloud-only tasks (not in local_map) are excluded from the
    # accuracy denominator and cost totals.
    local_tasks = set(local_map)
    realized = cascade_ladder([local_tier, cloud_tier], tasks=local_tasks, escalate_when=escalate_when_realized)
    oracle = cascade_ladder([local_tier, cloud_tier], tasks=local_tasks, escalate_when=escalate_when_oracle)
    cloud_attempted = realized.tier_attempts[1] > 0 or oracle.tier_attempts[1] > 0
    if cloud_total is None and cloud_attempted:
        raise ValueError(
            "cannot construct v1 Cells: reused cloud baseline has unknown cost_usd"
        )
    escalated = sorted(t for t, s in local_map.items() if escalate_when_realized(s))

    realized_cell = _cell(f"cascade-v1[{local_model}->{cloud_label}]", dataset,
                          realized, local_joules, wall, local_dir, energy_provenance,
                          device_coverage_complete)
    oracle_cell = _cell(f"cascade-v1-oracle[{local_model}->{cloud_label}]", dataset,
                        oracle, local_joules, wall, local_dir, energy_provenance,
                        device_coverage_complete)
    return V1Result(realized=realized, oracle=oracle, escalated=escalated,
                    realized_cell=realized_cell, oracle_cell=oracle_cell)
