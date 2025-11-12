"""Hardware-related helpers for profiling execution."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Mapping, Optional, Sequence, cast

from ..core.types import GpuInfo, SystemInfo

__all__ = ["derive_hardware_label"]


def derive_hardware_label(
    system_info: Optional[SystemInfo | Mapping[str, object]],
    gpu_info: Optional[GpuInfo | Mapping[str, object]],
) -> str:
    """Return a concise hardware label using GPU or CPU identifiers."""

    def _sanitize(raw: Optional[str]) -> Sequence[str]:
        if not raw:
            return []
        tokens: list[str] = []
        current = []
        for ch in raw:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current.clear()
        if current:
            tokens.append("".join(current))
        return tokens

    def _has_alpha(value: str) -> bool:
        return any(ch.isalpha() for ch in value)

    def _has_digit(value: str) -> bool:
        return any(ch.isdigit() for ch in value)

    def _normalize(label: str) -> str:
        if not label:
            return label
        alpha_chars = [ch for ch in label if ch.isalpha()]
        if alpha_chars and not any(ch.isupper() for ch in alpha_chars):
            return label.upper()
        return label

    def _should_pair(token: str) -> bool:
        """Return True when token should be combined with the next token."""
        if not _has_alpha(token):
            return False
        if len(token) > 4 and not _has_digit(token):
            return False
        if token.isupper() or token.islower() or _has_digit(token):
            return True
        return False

    def _combine(token: str, next_token: Optional[str]) -> Optional[str]:
        if not next_token:
            return None
        if not _has_digit(next_token):
            return None
        if not (_has_alpha(next_token) or _has_alpha(token)):
            return None
        if _has_digit(token) and not _has_alpha(token):
            return None
        if not _should_pair(token) and not _has_digit(token):
            return None
        return token + next_token

    def _derive_label(tokens: Sequence[str]) -> tuple[Optional[str], bool]:
        if not tokens:
            return None, False

        first_digit_candidate: Optional[str] = None
        alpha_fallback: Optional[str] = None

        for index, token in enumerate(tokens):
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            combined = _combine(token, next_token)
            if combined:
                label = _normalize(combined)
                return label, _has_digit(label)

            if first_digit_candidate is None and _has_digit(token):
                first_digit_candidate = token

            if _has_alpha(token):
                alpha_fallback = token

        if first_digit_candidate is not None:
            label = _normalize(first_digit_candidate)
            return label, _has_digit(label)

        if alpha_fallback is not None:
            label = _normalize(alpha_fallback)
            return label, _has_digit(label)

        label = _normalize(tokens[-1])
        return label, _has_digit(label)

    def _extract_field(
        obj: Optional[object], attr: str
    ) -> str:
        if obj is None:
            return ""
        if isinstance(obj, MappingABC):
            mapping = cast(Mapping[str, object], obj)
            value = mapping.get(attr)
            return str(value) if value is not None else ""
        return str(getattr(obj, attr, "") or "")

    gpu_label: Optional[str] = None
    gpu_has_digit = False
    raw_gpu = _extract_field(gpu_info, "name")
    if raw_gpu:
        gpu_label, gpu_has_digit = _derive_label(_sanitize(raw_gpu))
        if gpu_label and gpu_has_digit:
            return gpu_label

    cpu_label: Optional[str] = None
    cpu_has_digit = False
    raw_cpu = _extract_field(system_info, "cpu_brand")
    if raw_cpu:
        cpu_label, cpu_has_digit = _derive_label(_sanitize(raw_cpu))
        if cpu_label and cpu_has_digit and not gpu_has_digit:
            return cpu_label

    if gpu_label:
        return gpu_label
    if cpu_label:
        return cpu_label
    return "UNKNOWN_HW"
