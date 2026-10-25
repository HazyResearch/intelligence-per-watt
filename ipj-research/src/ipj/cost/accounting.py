"""Exact dollar-accounting conformance for IPD workload evidence.

``workload_ipd`` is a fail-closed Python reference implementation. It is not a
headline result path until the same schema and fixtures are implemented by the
normative Rust accounting service and wired into runtime exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Iterable

from ..metrics import utility_evidence_is_complete

MoneyValue = Decimal | int | str


class CostAccountingView(str, Enum):
    """A non-overlapping IPD reporting view from ADR 0005."""

    MARGINAL_ELECTRICITY = "marginal_electricity"
    FULLY_LOADED_ELECTRICITY = "fully_loaded_electricity"
    MARGINAL_ELECTRICITY_AND_HARDWARE = "marginal_electricity_and_hardware"
    FULLY_LOADED_ELECTRICITY_AND_HARDWARE = "fully_loaded_electricity_and_hardware"
    PROVIDER_BILLED = "provider_billed"


class ApiChargeKind(str, Enum):
    """Machine-readable origin of the API charge component."""

    PROVIDER_REPORTED_OR_SETTLED = "provider_reported_or_settled"
    CATALOG_QUOTED = "catalog_quoted"
    NOT_APPLICABLE = "not_applicable"


TASK_NATIVE_UTILITY_PROVENANCE = "task_native_verifier"
CALIBRATED_JUDGE_UTILITY_PROVENANCE = "calibrated_judge"


def _exact_nonnegative(value: MoneyValue, *, name: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"{name} must be Decimal, int, or str; binary floats are not exact")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid decimal value") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _token_count(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def dollars_tokens_decimal(
    tokens_in: int,
    tokens_out: int,
    price_in_per_m: MoneyValue,
    price_out_per_m: MoneyValue,
) -> Decimal:
    """Exact USD token cost using per-million-token prices."""
    tokens_in = _token_count(tokens_in, name="tokens_in")
    tokens_out = _token_count(tokens_out, name="tokens_out")
    price_in = _exact_nonnegative(price_in_per_m, name="price_in_per_m")
    price_out = _exact_nonnegative(price_out_per_m, name="price_out_per_m")
    million = Decimal(1_000_000)
    return (Decimal(tokens_in) * price_in + Decimal(tokens_out) * price_out) / million


@dataclass(frozen=True, slots=True)
class CostComponents:
    """Known USD components for one accounting view.

    A zero component is a known zero. An unknown total is represented by a
    missing :class:`TaskCostObservation`, never by zero-valued components.
    ``snapshot_id`` identifies the price/tariff/hardware schedule inputs.
    """

    api_usd: Decimal = Decimal(0)
    electricity_usd: Decimal = Decimal(0)
    hardware_amortization_usd: Decimal = Decimal(0)
    snapshot_id: str | None = None
    provenance: str | None = None
    accounting_view: CostAccountingView | str | None = None
    causal_components_complete: bool | None = None
    api_charge_kind: ApiChargeKind | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_usd", _exact_nonnegative(self.api_usd, name="api_usd"))
        object.__setattr__(
            self,
            "electricity_usd",
            _exact_nonnegative(self.electricity_usd, name="electricity_usd"),
        )
        object.__setattr__(
            self,
            "hardware_amortization_usd",
            _exact_nonnegative(self.hardware_amortization_usd, name="hardware_amortization_usd"),
        )
        if self.snapshot_id is not None and not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty when provided")
        if self.provenance is not None and not self.provenance.strip():
            raise ValueError("provenance must be non-empty when provided")
        if self.accounting_view is not None:
            try:
                view = CostAccountingView(self.accounting_view)
            except ValueError as exc:
                raise ValueError(f"unknown accounting_view: {self.accounting_view!r}") from exc
            object.__setattr__(self, "accounting_view", view)
        if self.causal_components_complete not in (None, True, False):
            raise TypeError("causal_components_complete must be a bool or None")
        if self.api_charge_kind is not None:
            try:
                charge_kind = ApiChargeKind(self.api_charge_kind)
            except ValueError as exc:
                raise ValueError(f"unknown api_charge_kind: {self.api_charge_kind!r}") from exc
            object.__setattr__(self, "api_charge_kind", charge_kind)
            if charge_kind == ApiChargeKind.NOT_APPLICABLE and self.api_usd != 0:
                raise ValueError("api_charge_kind=not_applicable requires api_usd == 0")

    @property
    def headline_eligible(self) -> bool:
        """Whether this is a complete, pinned observation for one view."""
        return (
            self.causal_components_complete is True
            and self.accounting_view is not None
            and self.components_match_view
            and isinstance(self.snapshot_id, str)
            and bool(self.snapshot_id.strip())
            and isinstance(self.provenance, str)
            and bool(self.provenance.strip())
            and self.api_charge_kind is not None
            and (
                self.accounting_view != CostAccountingView.PROVIDER_BILLED
                or self.api_charge_kind == ApiChargeKind.PROVIDER_REPORTED_OR_SETTLED
            )
        )

    @property
    def components_match_view(self) -> bool:
        """Reject components excluded by the declared reporting view."""
        if self.accounting_view in {
            CostAccountingView.MARGINAL_ELECTRICITY,
            CostAccountingView.FULLY_LOADED_ELECTRICITY,
        }:
            return self.hardware_amortization_usd == 0
        if self.accounting_view == CostAccountingView.PROVIDER_BILLED:
            return self.electricity_usd == 0 and self.hardware_amortization_usd == 0
        return self.accounting_view in {
            CostAccountingView.MARGINAL_ELECTRICITY_AND_HARDWARE,
            CostAccountingView.FULLY_LOADED_ELECTRICITY_AND_HARDWARE,
        }

    @property
    def total_usd(self) -> Decimal:
        return self.api_usd + self.electricity_usd + self.hardware_amortization_usd


@dataclass(frozen=True, slots=True)
class TaskCostObservation:
    """Task utility and its complete cost for one declared accounting view."""

    utility: Decimal
    cost: CostComponents | None
    utility_complete: bool | None = None
    utility_provenance: str | None = None
    utility_calibration_id: str | None = None
    utility_verifier_id: str | None = None
    utility_verifier_version: str | None = None
    utility_task_manifest_id: str | None = None
    utility_task_membership_id: str | None = None
    utility_task_manifest_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "utility", _exact_nonnegative(self.utility, name="utility"))
        if self.utility > 1:
            raise ValueError("utility must be in the inclusive range [0, 1]")
        if self.utility_complete not in (None, True, False):
            raise TypeError("utility_complete must be a bool or None")

    @property
    def utility_is_eligible(self) -> bool:
        return (
            utility_evidence_is_complete(
                complete=self.utility_complete,
                provenance=self.utility_provenance,
                calibration_id=self.utility_calibration_id,
                verifier_id=self.utility_verifier_id,
                verifier_version=self.utility_verifier_version,
                task_manifest_id=self.utility_task_manifest_id,
            )
            and isinstance(self.utility_task_membership_id, str)
            and bool(self.utility_task_membership_id.strip())
            and not isinstance(self.utility_task_manifest_size, bool)
            and isinstance(self.utility_task_manifest_size, int)
            and self.utility_task_manifest_size > 0
        )

    @property
    def utility_basis(self) -> tuple[str, str, str, str, str | None, int] | None:
        """Canonical utility evidence basis for workload consistency checks."""
        if not self.utility_is_eligible:
            return None
        calibration = (
            self.utility_calibration_id.strip()
            if self.utility_provenance == CALIBRATED_JUDGE_UTILITY_PROVENANCE
            and self.utility_calibration_id is not None
            else None
        )
        assert self.utility_provenance is not None
        assert self.utility_verifier_id is not None
        assert self.utility_verifier_version is not None
        assert self.utility_task_manifest_id is not None
        assert self.utility_task_manifest_size is not None
        return (
            self.utility_provenance,
            self.utility_verifier_id.strip(),
            self.utility_verifier_version.strip(),
            self.utility_task_manifest_id.strip(),
            calibration,
            self.utility_task_manifest_size,
        )

    @property
    def headline_eligible(self) -> bool:
        return self.utility_is_eligible and self.cost is not None and self.cost.headline_eligible


@dataclass(frozen=True, slots=True)
class IpdResult:
    """Ratio-of-totals IPD and denominator coverage for a workload."""

    utility_total: Decimal
    cost_total_usd: Decimal | None
    intelligence_per_dollar: Decimal | None
    task_count: int
    known_cost_tasks: int
    eligible_tasks: int
    accounting_view: CostAccountingView | None = None
    snapshot_id: str | None = None
    unavailable_reason: str | None = None
    api_charge_kind: ApiChargeKind | None = None

    @property
    def complete(self) -> bool:
        return (
            self.eligible_tasks == self.task_count
            and self.accounting_view is not None
            and self.snapshot_id is not None
            and self.api_charge_kind is not None
            and self.unavailable_reason is None
        )


def workload_ipd(observations: Iterable[TaskCostObservation]) -> IpdResult:
    """Compute ``sum(utility) / sum(cost)`` with strict workload eligibility.

    Every task must have verifier-grade utility, a complete causal cost ledger,
    a pinned snapshot/provenance, and the same accounting view and snapshot.
    A known all-zero denominator is complete but its ratio is unavailable.
    """
    items = tuple(observations)
    if not items:
        raise ValueError("at least one task observation is required")

    utility_total = sum((item.utility for item in items), start=Decimal(0))
    known_costs = tuple(item.cost for item in items if item.cost is not None)
    eligible_items = tuple(item for item in items if item.headline_eligible)
    if len(eligible_items) != len(items):
        return IpdResult(
            utility_total=utility_total,
            cost_total_usd=None,
            intelligence_per_dollar=None,
            task_count=len(items),
            known_cost_tasks=len(known_costs),
            eligible_tasks=len(eligible_items),
            unavailable_reason="incomplete utility, cost provenance, or causal coverage",
        )

    utility_bases = {item.utility_basis for item in eligible_items}
    if len(utility_bases) != 1:
        return IpdResult(
            utility_total=utility_total,
            cost_total_usd=None,
            intelligence_per_dollar=None,
            task_count=len(items),
            known_cost_tasks=len(known_costs),
            eligible_tasks=len(eligible_items),
            unavailable_reason="mixed utility verifier, manifest, or calibration basis",
        )
    utility_basis = next(iter(utility_bases))
    assert utility_basis is not None
    expected_tasks = utility_basis[-1]
    membership_ids = {
        item.utility_task_membership_id.strip()
        for item in eligible_items
        if item.utility_task_membership_id is not None
    }
    if len(items) != expected_tasks or len(membership_ids) != expected_tasks:
        return IpdResult(
            utility_total=utility_total,
            cost_total_usd=None,
            intelligence_per_dollar=None,
            task_count=len(items),
            known_cost_tasks=len(known_costs),
            eligible_tasks=len(eligible_items),
            unavailable_reason="incomplete or duplicate task-manifest population",
        )

    views = {item.cost.accounting_view for item in eligible_items if item.cost is not None}
    snapshots = {item.cost.snapshot_id for item in eligible_items if item.cost is not None}
    charge_kinds = {item.cost.api_charge_kind for item in eligible_items if item.cost is not None}
    if len(views) != 1 or len(snapshots) != 1 or len(charge_kinds) != 1:
        return IpdResult(
            utility_total=utility_total,
            cost_total_usd=None,
            intelligence_per_dollar=None,
            task_count=len(items),
            known_cost_tasks=len(known_costs),
            eligible_tasks=len(eligible_items),
            unavailable_reason="mixed accounting views, snapshots, or API charge kinds",
        )

    complete_costs = tuple(item.cost for item in eligible_items if item.cost is not None)
    cost_total = sum((cost.total_usd for cost in complete_costs), start=Decimal(0))
    ipd = None if cost_total == 0 else utility_total / cost_total
    return IpdResult(
        utility_total=utility_total,
        cost_total_usd=cost_total,
        intelligence_per_dollar=ipd,
        task_count=len(items),
        known_cost_tasks=len(known_costs),
        eligible_tasks=len(eligible_items),
        accounting_view=next(iter(views)),
        snapshot_id=next(iter(snapshots)),
        api_charge_kind=next(iter(charge_kinds)),
    )
