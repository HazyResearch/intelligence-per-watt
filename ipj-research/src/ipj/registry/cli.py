from __future__ import annotations

from pathlib import Path

import click

from .backfill import backfill_records
from .capture import capture_run
from .check import check_spec
from .record import HardwareSpec
from .store import (
    pull_index,
    pull_traces,
    push_public_record,
    push_record,
    push_traces,
)


def format_query_table(rows: list[dict]) -> str:
    if not rows:
        return "no runs in index"
    header = f"{'model':32} {'benchmark':34} {'hw':8} {'acc':>6} {'sig':10} {'status':10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        acc = r.get("metrics", {}).get("accuracy")
        lines.append(
            f"{r.get('model', {}).get('id', '?')[:32]:32} "
            f"{r.get('benchmark', {}).get('name', '?')[:34]:34} "
            f"{str(r.get('hardware', {}).get('gpu') or '?')[:8]:8} "
            f"{('' if acc is None else f'{acc:.2f}'):>6} "
            f"{str(r.get('energy_sig') or '?')[:8]:10} "
            f"{str(r.get('status') or '?'):10}"
        )
    return "\n".join(lines)


@click.group(help="Model-performance registry (HF-backed).")
def results() -> None:
    pass


@results.command("push")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--gpu", default=None, help="GPU label, e.g. H100 / MI350X")
@click.option("--n-gpus", type=int, default=None)
@click.option("--tp", type=int, default=None)
@click.option(
    "--traces/--no-traces",
    default=False,
    help="upload RUN_DIR/public_artifact after strict sanitizer validation",
)
@click.option(
    "--trace-artifact",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="sanitizer-created public trace artifact to upload",
)
@click.option("--repo-root", default=".", help="git repo to stamp provenance from")
def push_cmd(
    run_dir: str,
    gpu,
    n_gpus,
    tp,
    traces: bool,
    trace_artifact: str | None,
    repo_root: str,
) -> None:
    hw = HardwareSpec(gpu=gpu, count=n_gpus, tp=tp)
    rec = capture_run(run_dir, hardware=hw, repo_root=repo_root)
    if traces and trace_artifact is not None:
        raise click.UsageError("use either --traces or --trace-artifact, not both")
    artifact_path = Path(run_dir) / "public_artifact" if traces else trace_artifact
    if artifact_path is not None:
        rec.trace_uri = push_traces(artifact_path, rec.energy_sig)
        status = push_public_record(rec)
    else:
        # Compatibility/private workflow. This raw index can contain host,
        # run_dir, raw_config, and arbitrary orchestration metadata; it is not
        # eligible for a public artifact or registry trace upload.
        status = push_record(rec)
    click.echo(f"{status}: {rec.model.id} @ {rec.energy_sig} (acc={rec.metrics.accuracy})")


@results.command("query")
@click.option("--model", default=None)
@click.option("--benchmark", default=None)
@click.option("--hardware", "hw", default=None)
def query_cmd(model, benchmark, hw) -> None:
    rows = pull_index()

    def keep(r: dict) -> bool:
        return (
            (model is None or model in (r.get("model", {}).get("id") or ""))
            and (benchmark is None or benchmark in (r.get("benchmark", {}).get("name") or ""))
            and (hw is None or hw in (str(r.get("hardware", {}).get("gpu") or "")))
        )

    click.echo(format_query_table([r for r in rows if keep(r)]))


def spec_matches_block(verdict, force: bool, allow_dup: bool) -> bool:
    """Only an exact match blocks a launch, and only when not explicitly overridden."""
    return verdict.tier == "exact" and not (force or allow_dup)


@results.command("check")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--gpu", default=None)
@click.option("--repo-root", default=".")
def check_cmd(run_dir: str, gpu, repo_root: str) -> None:
    rec = capture_run(run_dir, hardware=HardwareSpec(gpu=gpu), repo_root=repo_root)
    verdict = check_spec(rec, pull_index())
    click.echo(f"[{verdict.tier}] {verdict.message}")


@results.command("run", context_settings={"ignore_unknown_options": True})
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--gpu", default=None)
@click.option("--force", is_flag=True, help="launch even on an exact match")
@click.option("--allow-dup", is_flag=True, help="alias for --force")
def run_cmd(run_dir: str, gpu, force: bool, allow_dup: bool) -> None:
    """Pre-flight dedup gate: capture the spec for RUN_DIR, check it, prompt on an exact match."""
    rec = capture_run(run_dir, hardware=HardwareSpec(gpu=gpu))
    verdict = check_spec(rec, pull_index())
    click.echo(f"[{verdict.tier}] {verdict.message}")
    if spec_matches_block(verdict, force, allow_dup):
        if not click.confirm("Exact duplicate exists. Launch anyway?", default=False):
            click.echo("aborted (duplicate).")
            raise SystemExit(1)
    click.echo("clear to launch.")


@results.command("backfill")
@click.argument("root", type=click.Path(exists=True), default="runs")
@click.option("--push/--dry-run", default=False, help="push records to HF (default: dry-run)")
def backfill_cmd(root: str, push: bool) -> None:
    recs = backfill_records(root)
    click.echo(f"found {len(recs)} run(s) under {root}")
    if not push:
        click.echo("dry-run: not pushing. Re-run with --push to upload.")
        return
    for rec in recs:
        status = push_record(rec)
        click.echo(f"  {status}: {rec.model.id} @ {rec.energy_sig}")


@results.command("pull-traces")
@click.argument("energy_sig")
@click.option("--dest", default="runs/_traces", help="local directory to download traces into")
def pull_traces_cmd(energy_sig: str, dest: str) -> None:
    """Download traces for ENERGY_SIG from the HF dataset repo."""
    path = pull_traces(energy_sig, dest)
    click.echo(f"pulled traces for {energy_sig} -> {path}")
