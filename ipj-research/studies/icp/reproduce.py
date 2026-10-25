"""Recompute the author-validated ICP macro bundle in an isolated output directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

from ipj.reproduction import compare_macros, tex_macros, verify_inputs


def compare_figure_text(reference: Path, generated: Path) -> dict:
    """Compare figure labels independently of PDF metadata and text spacing."""
    from pypdf import PdfReader

    if not generated.is_file():
        return {"status": "failed", "reason": "missing generated figure"}

    def labels(path: Path) -> list[str]:
        return ["".join(unicodedata.normalize("NFKC", page.extract_text()).split())
                for page in PdfReader(path).pages]

    return {"status": "passed" if labels(reference) == labels(generated) else "failed",
            "comparison": "PDF text, Unicode-normalized with whitespace removed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare every macro to the frozen paper baseline.")
    parser.add_argument("--figures", action="store_true", help="Also regenerate the six bundled figure generators.")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; existing paths are never overwritten.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    failures = verify_inputs(root, manifest["files"])
    if failures:
        print(json.dumps({"status": "failed", "input_failures": failures}, indent=2))
        return 1
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error("Use a new output directory outside the study source directory.")
    output.mkdir(parents=True)
    pipeline = output / "pipeline"
    shutil.copytree(root / "pipeline", pipeline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (pipeline / "paper/figures_and_tables").mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(output / "matplotlib-cache"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    commands = ["targets/compat.py"]  # compat runs make_numbers and its arithmetic checks itself.
    if args.figures:
        commands += ["targets/sync_figure_inputs.py", *manifest["figure_generators"]]
    completed = []
    failure = None
    for command in commands:
        log = logs / (Path(command).stem + ".log")
        figure_env = {**env, "IPD_FIG_OUT": str(pipeline / "paper/figures_and_tables")}
        try:
            with log.open("w") as stream:
                result = subprocess.run([sys.executable, command], cwd=pipeline, env=figure_env,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=900, check=False)
        except subprocess.TimeoutExpired:
            failure = {"command": command, "reason": "timeout", "log": str(log.relative_to(output))}
            break
        if result.returncode:
            failure = {"command": command, "returncode": result.returncode, "log": str(log.relative_to(output))}
            break
        completed.append(command)
    generated = pipeline / "paper/ipj-numbers.tex"
    comparison = None
    if generated.exists():
        comparison = compare_macros(tex_macros((root / "expected/ipj-numbers.tex").read_text()),
                                    tex_macros(generated.read_text()))
    figures = {}
    if args.figures and not failure:
        figures = {name: compare_figure_text(root / "reference" / name,
                                            pipeline / "paper/figures_and_tables" / name)
                   for name in manifest["regenerated_figures"]}
    figures_pass = all(result["status"] == "passed" for result in figures.values())
    report = {
        "status": "passed" if not failure and comparison and comparison["status"] == "passed" and figures_pass else "failed",
        "scope": "Recomputation from retained study inputs against the author-validated paper; not new benchmark runs.",
        "author_validation": manifest["author_validation"],
        "input_hashes_verified": len(manifest["files"]),
        "completed_commands": completed,
        "failure": failure,
        "macros": comparison,
        "regenerated_figures": sorted(p.name for p in (pipeline / "paper/figures_and_tables").glob("*.pdf")),
        "figure_label_comparison": figures,
        "paper_coverage": manifest["coverage_summary"],
        "retained_reference_only": manifest["retained_reference_only"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
