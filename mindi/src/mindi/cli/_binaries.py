"""Helpers for launching bundled Mindi service binaries."""

from __future__ import annotations

import os
import subprocess
import platform
from pathlib import Path
from typing import Iterable, NoReturn, Sequence

_PLATFORM_ALIASES = {
    ("Darwin", "arm64"): "macos-arm64",
    ("Darwin", "x86_64"): "macos-x86_64",
    ("Linux", "x86_64"): "linux-x86_64",
    ("Linux", "aarch64"): "linux-arm64",
    ("Windows", "amd64"): "windows-x86_64",
}


def _bin_dir() -> Path:
    return Path(__file__).resolve().parent / "bin"


def _platform_id() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    key = (system, machine)
    if key in _PLATFORM_ALIASES:
        return _PLATFORM_ALIASES[key]
    # Normalize for common aliases
    if system == "Darwin" and machine in {"x86_64", "i386"}:
        return "macos-x86_64"
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if system == "Linux" and machine in {"arm64", "aarch64"}:
        return "linux-arm64"
    if system.startswith("Windows") and machine in {"amd64", "x86_64"}:
        return "windows-x86_64"
    raise FileNotFoundError(
        f"Unsupported platform combination ({system}, {machine}). "
        "If you built the binaries yourself, place them under bin/<platform>/ and set _PLATFORM_ALIASES accordingly."
    )


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def binary_path(name: str) -> Path:
    platform_id = _platform_id()
    binary = _bin_dir() / platform_id / _exe(name)
    if not binary.exists():
        raise FileNotFoundError(
            "Bundled binary not found for platform '{platform}'. "
            "Run scripts/build_cli_binaries.py to bundle the latest build.".format(platform=platform_id)
        )
    return binary


def run_attached(name: str, args: Sequence[str] | None = None) -> int:
    binary = binary_path(name)
    cmd: list[str] = [str(binary), *(args or [])]
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def run_detached(name: str, args: Sequence[str] | None = None, *, env: dict[str, str] | None = None) -> int:
    binary = binary_path(name)
    cmd: list[str] = [str(binary), *(args or [])]
    process = subprocess.Popen(cmd, env={**os.environ, **(env or {})})
    return process.pid


def forward_exec(name: str, argv: Iterable[str]) -> NoReturn:
    binary = binary_path(name)
    os.execv(str(binary), [str(binary), *argv])


__all__ = ["binary_path", "run_attached", "run_detached", "forward_exec"]
