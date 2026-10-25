"""Glue: run Harbor under energy capture and assemble one Cell."""

from __future__ import annotations

from pathlib import Path

from .cost import H100_HOURLY_USD, dollars_local, dollars_tokens
from .energy.capture import RunEnergyCapture
from .harbor.results import parse_harbor_summary
from .harbor.run import harbor_command, run_command
from .metrics import (
    MEASURED_DEDICATED_JOULES_PROVENANCE,
    UNKNOWN_JOULES_PROVENANCE,
    Cell,
    accelerator_joules_are_complete,
    resolve_joules_provenance,
)


def find_latest_result_json(output_dir: Path | str) -> Path:
    """Harbor writes its job summary to <jobs-dir>/<job-name-or-timestamp>/result.json.

    Return the most recently modified such file under output_dir.
    """
    output_dir = Path(output_dir)

    def _mtime(p: Path) -> float:
        # Tolerate a sibling result.json vanishing mid-sort (concurrent Harbor cleanup on a shared box).
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    candidates = sorted(output_dir.glob("*/result.json"), key=_mtime)
    if not candidates:
        raise FileNotFoundError(f"No Harbor result.json found under {output_dir}")
    return candidates[-1]


def measure_cell(
    *,
    model: str,
    harness: str,
    hardware: str,
    dataset: str,
    output_dir: Path | str,
    is_local: bool,
    k: int = 1,
    n_tasks: int | None = None,
    env_type: str = "docker",
    gpu_hourly_rate: float = H100_HOURLY_USD,
    n_gpus: int = 1,
    n_concurrent: int = 1,
    energy_provenance: str = UNKNOWN_JOULES_PROVENANCE,
    price_in_per_m: float | None = None,
    price_out_per_m: float | None = None,
    run_fn=run_command,
    capture_factory=RunEnergyCapture,
) -> Cell:
    output_dir = Path(output_dir)
    cmd = harbor_command(
        dataset, harness, model, output_path=output_dir, k=k, n_tasks=n_tasks, env_type=env_type, n_concurrent=n_concurrent
    )

    if is_local:
        with capture_factory() as cap:
            win = run_fn(cmd)
            joules = cap.joules(win.start, win.end)
            device_coverage_complete = bool(
                getattr(cap, "accelerator_device_coverage_complete", lambda *_: False)(
                    win.start,
                    win.end,
                )
            )
        joules_provenance = resolve_joules_provenance(joules, energy_provenance)
        if joules is None:
            energy_source = "unknown (telemetry unavailable or incomplete run-window coverage)"
        elif joules_provenance == MEASURED_DEDICATED_JOULES_PROVENANCE:
            energy_source = "energy-monitor (dedicated accelerator declared)"
        else:
            energy_source = "energy-monitor (accelerator isolation unverified)"
        joules_scope = "accelerator"
        joules_complete = accelerator_joules_are_complete(
            joules,
            joules_provenance,
            device_coverage_complete,
        )
    else:
        win = run_fn(cmd)
        joules = None
        energy_source = "n/a (cloud)"
        joules_provenance = UNKNOWN_JOULES_PROVENANCE
        joules_scope = "provider-infrastructure"
        joules_complete = False
        device_coverage_complete = False

    if win.returncode != 0:
        raise RuntimeError(f"harbor run failed (exit code {win.returncode}); jobs dir: {output_dir}")

    result_path = find_latest_result_json(output_dir)
    summary = parse_harbor_summary(result_path)
    wall = win.end - win.start

    if is_local:
        dollars = dollars_local(wall, gpu_hourly_rate, n_gpus=n_gpus)
        basis = "gpu-hours"
    elif summary.cost_usd is not None:
        # Prefer Harbor's own billed cost (real provider pricing) over our token estimate.
        dollars = summary.cost_usd
        basis = "api-billed"
    else:
        if price_in_per_m is None or price_out_per_m is None:
            raise ValueError(
                "cloud cost is unknown: Harbor did not report cost_usd and both token prices "
                "were not provided"
            )
        dollars = dollars_tokens(summary.tokens_in, summary.tokens_out, price_in_per_m, price_out_per_m)
        basis = "api-tokens"

    return Cell(
        model=model, harness=harness, hardware=hardware, dataset=dataset,
        n_tasks=summary.n_tasks, accuracy=summary.accuracy,
        tokens_in=summary.tokens_in, tokens_out=summary.tokens_out,
        dollars=dollars, dollars_basis=basis, joules=joules,
        wall_clock_s=wall, energy_source=energy_source, run_dir=str(result_path.parent),
        joules_provenance=joules_provenance, joules_scope=joules_scope,
        joules_complete=joules_complete,
        accelerator_device_coverage_complete=device_coverage_complete,
    )
