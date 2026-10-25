"""Cloud-energy estimation (ADR 0003).

We MEASURE Joules only for local GPU runs (NVML/energy-monitor). Cloud inference runs
on hardware we don't control, so its energy is ESTIMATED, never measured: a
transfer-from-local layer that takes J/token from a local measured run and applies it
to the cloud run's token count, with an explicit ±band.

This is always reported as an estimate (with the band + reference), never as a measured
value, and it is only as good as the local reference — use an OPTIMIZED local run, not an
eager-inflated one, for a tight number (eager overstates J/token, so the estimate is a
loose upper bound).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class JouleEstimate:
    joules: float       # point estimate
    lo: float           # joules * (1 - band)
    hi: float           # joules * (1 + band)
    j_per_token: float  # reference energy intensity used
    band: float         # fractional uncertainty applied
    basis: str          # human-readable note on the reference + method

    def __post_init__(self) -> None:
        values = (self.joules, self.lo, self.hi, self.j_per_token, self.band)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("estimate values must be finite real numbers")
        if not 0 <= self.lo <= self.joules <= self.hi:
            raise ValueError("estimate interval must satisfy 0 <= lo <= point <= hi")
        if self.j_per_token < 0 or not 0 <= self.band <= 1:
            raise ValueError("j_per_token must be non-negative and band must be in [0, 1]")
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("basis must be a non-empty string")


def _token_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def energy_per_token(joules: float, tokens_in: int, tokens_out: int, *, output_weight: float = 1.0) -> float:
    """J per (weighted) token from a measured run.

    output_weight scales output tokens relative to input (generation costs more than
    prefill); default 1.0 weights them equally.
    """
    joules = _finite_nonnegative("joules", joules)
    tokens_in = _token_count("tokens_in", tokens_in)
    tokens_out = _token_count("tokens_out", tokens_out)
    output_weight = _finite_nonnegative("output_weight", output_weight)
    weighted = tokens_in + output_weight * tokens_out
    if weighted <= 0:
        raise ValueError("need a positive token count to derive J/token")
    return joules / weighted


def estimate_joules(tokens_in: int, tokens_out: int, j_per_token: float, *,
                    output_weight: float = 1.0, band: float = 0.3, basis: str = "") -> JouleEstimate:
    """Estimate a run's energy from its tokens and a reference J/token, with a ±band."""
    tokens_in = _token_count("tokens_in", tokens_in)
    tokens_out = _token_count("tokens_out", tokens_out)
    j_per_token = _finite_nonnegative("j_per_token", j_per_token)
    output_weight = _finite_nonnegative("output_weight", output_weight)
    band = _finite_nonnegative("band", band)
    if band > 1:
        raise ValueError("band must be at most 1")
    weighted = tokens_in + output_weight * tokens_out
    est = weighted * j_per_token
    return JouleEstimate(est, est * (1 - band), est * (1 + band), j_per_token, band,
                         basis or f"{j_per_token:.4g} J/tok x ({tokens_in}+{output_weight}*{tokens_out}) tok, ±{band:.0%}")


def estimate_from_reference(target_tokens_in: int, target_tokens_out: int, *,
                            ref_joules: float, ref_tokens_in: int, ref_tokens_out: int,
                            output_weight: float = 1.0, band: float = 0.3, ref_name: str = "local ref") -> JouleEstimate:
    """Estimate the target run's Joules using J/token measured on a reference (local) run."""
    jpt = energy_per_token(ref_joules, ref_tokens_in, ref_tokens_out, output_weight=output_weight)
    e = estimate_joules(target_tokens_in, target_tokens_out, jpt, output_weight=output_weight, band=band)
    return JouleEstimate(e.joules, e.lo, e.hi, jpt, band,
                         f"transfer from {ref_name} ({jpt:.4g} J/tok), ±{band:.0%} — ESTIMATE not measured")
