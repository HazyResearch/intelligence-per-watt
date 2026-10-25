"""The Cell: one (model x harness x hardware) measurement on a dataset."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .energy.estimate import JouleEstimate

UNKNOWN_JOULES_PROVENANCE = "unknown"
MEASURED_DEDICATED_JOULES_PROVENANCE = "measured_dedicated"
TASK_NATIVE_UTILITY_PROVENANCE = "task_native_verifier"
CALIBRATED_JUDGE_UTILITY_PROVENANCE = "calibrated_judge"
DECLARED_COST_ACCOUNTING_VIEWS = frozenset(
    {
        "marginal_electricity",
        "fully_loaded_electricity",
        "marginal_electricity_and_hardware",
        "fully_loaded_electricity_and_hardware",
        "provider_billed",
    }
)
DECLARED_API_CHARGE_KINDS = frozenset({"provider_reported_or_settled", "catalog_quoted", "not_applicable"})


def resolve_joules_provenance(
    joules: float | None,
    declared_provenance: str | None,
) -> str:
    """Return a measurement provenance without inventing an isolation claim."""
    if joules is None or not declared_provenance:
        return UNKNOWN_JOULES_PROVENANCE
    return declared_provenance


def accelerator_joules_are_complete(
    joules: float | None,
    provenance: str | None,
    device_coverage_complete: bool = False,
) -> bool:
    """Return whether accelerator Joules are eligible as a complete measurement.

    Device coverage defaults false. Legacy numeric/provenance fields remain
    readable for two releases, but cannot become rigorous without the additive
    coverage capability.
    """
    return (
        joules is not None
        and math.isfinite(joules)
        and joules >= 0
        and provenance == MEASURED_DEDICATED_JOULES_PROVENANCE
        and device_coverage_complete is True
    )


def utility_is_complete(
    accuracy: float,
    n_tasks: int,
    complete: bool | None,
    provenance: str | None,
    calibration_id: str | None = None,
    verifier_id: str | None = None,
    verifier_version: str | None = None,
    task_manifest_id: str | None = None,
    membership_complete: bool | None = None,
    task_manifest_size: int | None = None,
) -> bool:
    """Return whether aggregate utility has verifier-grade provenance."""
    if (
        complete is not True
        or isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or not 0 <= accuracy <= 1
        or isinstance(n_tasks, bool)
        or not isinstance(n_tasks, int)
        or n_tasks <= 0
        or membership_complete is not True
        or isinstance(task_manifest_size, bool)
        or not isinstance(task_manifest_size, int)
        or task_manifest_size != n_tasks
    ):
        return False
    return utility_evidence_is_complete(
        complete=complete,
        provenance=provenance,
        calibration_id=calibration_id,
        verifier_id=verifier_id,
        verifier_version=verifier_version,
        task_manifest_id=task_manifest_id,
    )


def utility_evidence_is_complete(
    *,
    complete: bool | None,
    provenance: str | None,
    calibration_id: str | None = None,
    verifier_id: str | None = None,
    verifier_version: str | None = None,
    task_manifest_id: str | None = None,
) -> bool:
    """Return whether utility provenance is pinned to reproducible evidence."""
    if complete is not True or provenance not in {
        TASK_NATIVE_UTILITY_PROVENANCE,
        CALIBRATED_JUDGE_UTILITY_PROVENANCE,
    }:
        return False
    bindings = (verifier_id, verifier_version, task_manifest_id)
    if not all(isinstance(value, str) and bool(value.strip()) for value in bindings):
        return False
    if provenance == CALIBRATED_JUDGE_UTILITY_PROVENANCE:
        return isinstance(calibration_id, str) and bool(calibration_id.strip())
    return True


def dollars_are_complete(
    dollars: float,
    complete: bool | None,
    accounting_view: str | None,
    snapshot_id: str | None,
    provenance: str | None,
    api_charge_kind: str | None,
) -> bool:
    """Return whether a dollar value has a complete, pinned accounting basis."""
    return (
        complete is True
        and not isinstance(dollars, bool)
        and isinstance(dollars, (int, float))
        and math.isfinite(float(dollars))
        and dollars >= 0
        and accounting_view in DECLARED_COST_ACCOUNTING_VIEWS
        and api_charge_kind in DECLARED_API_CHARGE_KINDS
        and (accounting_view != "provider_billed" or api_charge_kind == "provider_reported_or_settled")
        and all(isinstance(value, str) and bool(value.strip()) for value in (snapshot_id, provenance))
    )


@dataclass
class Cell:
    model: str
    harness: str
    hardware: str
    dataset: str
    n_tasks: int
    accuracy: float
    tokens_in: int
    tokens_out: int
    dollars: float
    dollars_basis: str  # "gpu-hours" | "api-tokens"
    joules: Optional[float]  # None for cloud (not locally measured)
    wall_clock_s: float
    energy_source: str  # "energy-monitor" | "n/a (cloud)"
    run_dir: str
    n_local: Optional[int] = None
    n_cloud: Optional[int] = None
    classifier_dollars: Optional[float] = None
    # Additive Phase-0 provenance fields.  ``joules`` contains measured energy
    # only; estimates must live in a separate structured field.  ``joules_complete``
    # says whether ``joules`` covers every accelerator component used by this cell,
    # not whether optional host/PUE scopes were also measured.
    joules_provenance: Optional[str] = None
    joules_scope: Optional[str] = None
    joules_complete: Optional[bool] = None
    accelerator_device_coverage_complete: Optional[bool] = None
    cloud_joules_estimate: Optional[JouleEstimate] = None
    utility_complete: Optional[bool] = None
    utility_provenance: Optional[str] = None
    utility_calibration_id: Optional[str] = None
    utility_verifier_id: Optional[str] = None
    utility_verifier_version: Optional[str] = None
    utility_task_manifest_id: Optional[str] = None
    dollars_complete: Optional[bool] = None
    dollars_accounting_view: Optional[str] = None
    dollars_snapshot_id: Optional[str] = None
    dollars_provenance: Optional[str] = None
    dollars_api_charge_kind: Optional[str] = None
    utility_membership_complete: Optional[bool] = None
    utility_task_manifest_size: Optional[int] = None
    # Current-main audit fields are retained as additive keyword-compatible
    # fields. They follow the preserved Phase-0 field sequence so the older
    # provenance schema remains positionally compatible.
    n_terminal: Optional[int] = None
    n_gpus: Optional[int] = None

    def __post_init__(self) -> None:
        if self.joules_complete and not accelerator_joules_are_complete(
            self.joules,
            self.joules_provenance,
            self.accelerator_device_coverage_complete is True,
        ):
            raise ValueError(
                "joules_complete requires finite non-negative Joules and explicitly declared "
                "measured_dedicated provenance plus complete accelerator-device coverage"
            )
        if self.utility_complete and not utility_is_complete(
            self.accuracy,
            self.n_tasks,
            self.utility_complete,
            self.utility_provenance,
            self.utility_calibration_id,
            self.utility_verifier_id,
            self.utility_verifier_version,
            self.utility_task_manifest_id,
            self.utility_membership_complete,
            self.utility_task_manifest_size,
        ):
            raise ValueError(
                "utility_complete requires full task coverage, a pinned verifier ID/version, "
                "a task-manifest binding with validated full membership, and "
                "task_native_verifier or a calibrated_judge with calibration ID"
            )
        if self.dollars_complete and not dollars_are_complete(
            self.dollars,
            self.dollars_complete,
            self.dollars_accounting_view,
            self.dollars_snapshot_id,
            self.dollars_provenance,
            self.dollars_api_charge_kind,
        ):
            raise ValueError(
                "dollars_complete requires finite non-negative dollars, a declared "
                "accounting view, a compatible typed API charge kind, and non-empty "
                "snapshot ID and provenance"
            )


def write_cell(cell: Cell, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cell), indent=2) + "\n")


def append_cell(cell: Cell, jsonl_path: Path) -> None:
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        f.write(json.dumps(asdict(cell)) + "\n")
