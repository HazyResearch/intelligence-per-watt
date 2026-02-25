"""Compute estimation utilities for LLM inference."""
from __future__ import annotations

from .flops import (
    MODEL_PARAMS,
    estimate_flops,
    estimate_flops_calflops,
    estimate_flops_fallback,
    lookup_params,
    normalize_model_name,
)

__all__ = [
    "MODEL_PARAMS",
    "estimate_flops",
    "estimate_flops_calflops",
    "estimate_flops_fallback",
    "lookup_params",
    "normalize_model_name",
]
