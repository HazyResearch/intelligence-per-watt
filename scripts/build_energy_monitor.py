#!/usr/bin/env python3
"""Build and stage the Rust CLI binaries for the Python package."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENERGY_MONITOR_ROOT = PROJECT_ROOT / "energy-monitor"
BIN_ROOT = PROJECT_ROOT / "intelligence-per-watt" / "src" / "ipw" / "telemetry" / "bin"

# Mapping from sys.platform / machine combos to folder names under cli/bin/
PLATFORM_ALIASES = {
    ("Darwin", "arm64"): "macos-arm64",
    ("Darwin", "x86_64"): "macos-x86_64",
    ("Linux", "x86_64"): "linux-x86_64",
    ("Linux", "aarch64"): "linux-arm64",
    ("Windows", "amd64"): "windows-x86_64",
}

TARGET_ALIASES = {
    "aarch64-apple-darwin": "macos-arm64",
    "x86_64-apple-darwin": "macos-x86_64",
    "x86_64-unknown-linux-gnu": "linux-x86_64",
    "aarch64-unknown-linux-gnu": "linux-arm64",
    "x86_64-pc-windows-msvc": "windows-x86_64",
}

CRATES = [
    ("energy-monitor", "energy-monitor"),
]


def check_cargo_installed() -> None:
    """Check if cargo is installed and available."""
    try:
        subprocess.run(
            ["cargo", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise SystemExit(
            "Error: cargo not found. Please install Rust and Cargo from https://rustup.rs/"
        )
    except subprocess.CalledProcessError:
        raise SystemExit(
            "Error: cargo command failed. Please check your Rust installation."
        )


def run(
    cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    subprocess.run(cmd, check=True, cwd=cwd or PROJECT_ROOT, env=env)


def detect_platform_alias(target: str | None) -> tuple[str, Path, bool]:
    if target:
        alias = TARGET_ALIASES.get(target)
        if not alias:
            raise SystemExit(f"No platform alias configured for target '{target}'.")
        base = ENERGY_MONITOR_ROOT / "target" / target
        return alias, base, alias.startswith("windows")

    system = platform.system()
    machine = platform.machine().lower()
    key = (system, machine)
    try:
        alias = PLATFORM_ALIASES[key]
    except KeyError as exc:
        raise SystemExit(
            f"Unsupported platform ({system}, {machine}). Use --target to build for a known target."
        ) from exc
    base = ENERGY_MONITOR_ROOT / "target"
    return alias, base, alias.startswith("windows")


def copy_binary(
    binary_name: str, source_dir: Path, dest_dir: Path, *, is_windows: bool
) -> None:
    src = source_dir / (binary_name + (".exe" if is_windows else ""))
    if not src.exists():
        raise SystemExit(f"Expected binary not found: {src}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (binary_name + (".exe" if is_windows else ""))
    shutil.copy2(src, dest)
    if not is_windows:
        dest.chmod(0o755)
    print(f"Copied {src.relative_to(PROJECT_ROOT)} -> {dest.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and stage CLI binaries for the Python package."
    )
    parser.add_argument(
        "--target",
        help="Optional cargo target triple. If omitted, builds for the host platform.",
    )
    parser.add_argument(
        "--profile",
        default="release",
        choices=["release", "debug"],
        help="Cargo profile to use (default: release).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Copy existing artifacts without running cargo build.",
    )
    args = parser.parse_args()

    alias, target_base, is_windows = detect_platform_alias(args.target)
    profile_dir = "release" if args.profile == "release" else "debug"

    if not args.skip_build:
        check_cargo_installed()
        for crate, _ in CRATES:
            print(f"Building {crate} ({args.profile})")
            cmd = ["cargo", "build"]
            if args.target:
                cmd.extend(["--target", args.target])
            if args.profile == "release":
                cmd.append("--release")

            run(cmd, cwd=ENERGY_MONITOR_ROOT)

    source_dir = target_base
    if args.target:
        source_dir = source_dir / profile_dir
    else:
        source_dir = source_dir / profile_dir

    dest_root = BIN_ROOT / alias
    for _, binary in CRATES:
        copy_binary(binary, source_dir, dest_root, is_windows=is_windows)

    print(
        "\nDone. The CLI binaries are staged under", dest_root.relative_to(PROJECT_ROOT)
    )


if __name__ == "__main__":
    main()
