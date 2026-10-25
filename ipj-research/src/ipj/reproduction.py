"""Offline checks for the pinned ICP paper artifact.

These checks establish reproducibility of supplied inputs, not independent
validation of the experiments from which the inputs originated.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_comments(source: str) -> str:
    parts: list[str] = []
    index = 0
    while index < len(source):
        if source[index] == "\\":
            parts.append(source[index:index + 2])
            index += 2
        elif source[index] == "%":
            end = source.find("\n", index)
            index = len(source) if end < 0 else end
        else:
            parts.append(source[index])
            index += 1
    return "".join(parts)


def tex_macros(source: str) -> dict[str, str]:
    """Read balanced literal definitions with TeX comment/escape handling.

    This parses the zero-argument newcommand/renewcommand/providecommand form
    used by the paper. It does not execute TeX or expand macro references.
    """
    source = _without_comments(source)
    definitions: dict[str, str] = {}
    pattern = r"\\(newcommand|renewcommand|providecommand)\s*\{\\([A-Za-z]+)\}\s*\{"
    for match in re.finditer(pattern, source):
        depth, index = 1, match.end()
        while index < len(source) and depth:
            if source[index] == "\\":
                index += 2
                continue
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
            index += 1
        if depth:
            raise ValueError(f"Unclosed definition: {match[2]}")
        if match[1] == "providecommand" and match[2] in definitions:
            continue
        definitions[match[2]] = re.sub(r"\s+", " ", source[match.end():index - 1]).strip()
    if not definitions:
        raise ValueError("No paper macro definitions found")
    return definitions


def compare_macros(expected: Mapping[str, str], actual: Mapping[str, str]) -> dict:
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in sorted(expected.keys() & actual.keys())
        if expected[key] != actual[key]
    }
    return {
        "status": "passed" if not (missing or extra or changed) else "failed",
        "expected_count": len(expected),
        "matched_count": sum(actual.get(key) == value for key, value in expected.items()),
        "missing": missing,
        "extra": extra,
        "changed": changed,
    }


def verify_inputs(root: Path, files: Mapping[str, str]) -> list[str]:
    """Return integrity failures without reading paths outside the artifact."""
    root = root.resolve()
    failures: list[str] = []
    if not files:
        return ["The input manifest is empty"]
    for relative, expected in sorted(files.items()):
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            failures.append(f"Input escapes artifact directory: {relative}")
        elif not path.is_file():
            failures.append(f"Missing input: {relative}")
        elif sha256(path) != expected:
            failures.append(f"Input hash mismatch: {relative}")
    return failures
