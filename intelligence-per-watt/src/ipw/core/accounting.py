"""Shared eligibility rules for scientific accounting claims."""

from __future__ import annotations

from collections.abc import Iterable

# Phase 0 can prove dedicated-device measurements only. A shared allocation
# needs an allocation method, overlap/lease evidence, and uncertainty metadata
# before its Joules can enter a headline denominator; a provenance label alone
# is insufficient.
RIGOROUS_ENERGY_PROVENANCE = frozenset({"measured_dedicated"})
RIGOROUS_UTILITY_PROVENANCE = frozenset({"task_native_verifier", "calibrated_judge"})


def rigorous_utility_is_eligible(
    *,
    complete: object,
    provenance: object,
    calibration_id: object = None,
    verifier_id: object = None,
    verifier_version: object = None,
    task_manifest_id: object = None,
    task_membership_id: object = None,
    task_manifest_size: object = None,
) -> bool:
    """Return whether one utility observation can back a headline metric.

    Both accepted provenance kinds require a pinned verifier implementation and
    an explicit binding to the evaluated task manifest.  A calibrated judge
    additionally requires a non-empty calibration snapshot identifier.
    Historical or free-form scores default to ineligible instead of being
    inferred from their value.
    """
    return (
        rigorous_utility_basis(
            complete=complete,
            provenance=provenance,
            calibration_id=calibration_id,
            verifier_id=verifier_id,
            verifier_version=verifier_version,
            task_manifest_id=task_manifest_id,
            task_membership_id=task_membership_id,
            task_manifest_size=task_manifest_size,
        )
        is not None
    )


def rigorous_utility_basis(
    *,
    complete: object,
    provenance: object,
    calibration_id: object = None,
    verifier_id: object = None,
    verifier_version: object = None,
    task_manifest_id: object = None,
    task_membership_id: object = None,
    task_manifest_size: object = None,
) -> tuple[str, str, str, str, str | None, int] | None:
    """Return the canonical evidence basis for an eligible utility observation."""
    if complete is not True or provenance not in RIGOROUS_UTILITY_PROVENANCE:
        return None
    required_bindings = (
        verifier_id,
        verifier_version,
        task_manifest_id,
        task_membership_id,
    )
    if not all(isinstance(value, str) and bool(value.strip()) for value in required_bindings):
        return None
    if isinstance(task_manifest_size, bool) or not isinstance(task_manifest_size, int) or task_manifest_size <= 0:
        return None
    normalized_calibration: str | None = None
    if provenance == "calibrated_judge":
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            return None
        normalized_calibration = calibration_id.strip()
    assert isinstance(provenance, str)
    assert isinstance(verifier_id, str)
    assert isinstance(verifier_version, str)
    assert isinstance(task_manifest_id, str)
    return (
        provenance,
        verifier_id.strip(),
        verifier_version.strip(),
        task_manifest_id.strip(),
        normalized_calibration,
        task_manifest_size,
    )


def rigorous_utility_population_is_complete(
    observations: Iterable[tuple[tuple[str, str, str, str, str | None, int] | None, object]],
) -> bool:
    """Require one complete manifest population with no missing or duplicate tasks."""
    items = tuple(observations)
    if not items or any(basis is None for basis, _ in items):
        return False
    bases = {basis for basis, _ in items}
    if len(bases) != 1:
        return False
    basis = next(iter(bases))
    assert basis is not None
    expected_size = basis[-1]
    membership_ids = [value.strip() for _, value in items if isinstance(value, str) and value.strip()]
    return (
        len(items) == expected_size
        and len(membership_ids) == expected_size
        and len(set(membership_ids)) == expected_size
    )


__all__ = [
    "RIGOROUS_ENERGY_PROVENANCE",
    "RIGOROUS_UTILITY_PROVENANCE",
    "rigorous_utility_basis",
    "rigorous_utility_is_eligible",
    "rigorous_utility_population_is_complete",
]
