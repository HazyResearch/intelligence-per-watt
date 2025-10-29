"""Hardware-related helpers for profiling execution."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

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
        current = ""
        for ch in raw:
            if ch.isalnum():
                current += ch
            else:
                if current:
                    tokens.append(current)
                current = ""
        if current:
            tokens.append(current)
        return tokens

    def _pick(tokens: Sequence[str]) -> Optional[str]:
        for token in tokens:
            if any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
                return token.upper()
        for token in tokens:
            if token.isalpha():
                return token.upper()
        if tokens:
            return tokens[-1].upper()
        return None

    gpu_candidate: Optional[str] = None
    if gpu_info:
        if isinstance(gpu_info, Mapping):
            raw_name = str(gpu_info.get("name", ""))
        else:
            raw_name = getattr(gpu_info, "name", "")
        gpu_candidate = _pick(_sanitize(raw_name))
        if gpu_candidate and any(ch.isdigit() for ch in gpu_candidate):
            return gpu_candidate

    cpu_candidate: Optional[str] = None
    if system_info:
        if isinstance(system_info, Mapping):
            raw_cpu = str(system_info.get("cpu_brand", ""))
        else:
            raw_cpu = getattr(system_info, "cpu_brand", "")
        cpu_candidate = _pick(_sanitize(raw_cpu))
        if cpu_candidate:
            if any(ch.isdigit() for ch in cpu_candidate):
                return cpu_candidate
            if not gpu_candidate:
                return cpu_candidate

    if gpu_candidate:
        return gpu_candidate
    if cpu_candidate:
        return cpu_candidate
    return "UNKNOWN_HW"

