"""`ipj` command-line entry point."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .metrics import (
    MEASURED_DEDICATED_JOULES_PROVENANCE,
    UNKNOWN_JOULES_PROVENANCE,
)
from .plot import plot_cost_frontier
from .registry.cli import results as _results_group

ENERGY_PROVENANCE_CHOICES = (
    UNKNOWN_JOULES_PROVENANCE,
    MEASURED_DEDICATED_JOULES_PROVENANCE,
)


@click.group(help="Intelligence per Joule CLI")
def cli() -> None:
    """Top-level group."""


@cli.command("plot", help="Scatter accuracy vs cost from a cells JSONL.")
@click.argument("cells_jsonl", type=click.Path(exists=True))
@click.argument("out_png", type=click.Path())
@click.option("--x", type=click.Choice(["dollars", "joules", "flops", "latency", "tokens",
                                        "tokens_in", "tokens_out"]), default="dollars")
@click.option(
    "--rigorous-energy/--allow-incomplete-energy",
    default=False,
    help=(
        "Require complete instrumented Joules for paper claims; the legacy-compatible "
        "default retains incomplete historical points."
    ),
)
def plot_cmd(
    cells_jsonl: str,
    out_png: str,
    x: str,
    rigorous_energy: bool,
) -> None:
    cells = [json.loads(line) for line in Path(cells_jsonl).read_text().splitlines() if line.strip()]
    path = plot_cost_frontier(
        cells,
        out_png,
        x=x,
        allow_incomplete_energy=not rigorous_energy,
    )
    click.echo(f"wrote {path}")


@cli.command("measure", help="Run one measurement cell (Harbor + energy) and write cell.json.")
@click.option("--model", required=True, help="e.g. openai/Qwen/Qwen3.5-27B-FP8 or openai/gpt-5.5-2026-04-23")
@click.option("--harness", default="terminus-2", show_default=True)
@click.option("--dataset", default="terminal-bench/terminal-bench-2-1", show_default=True)
@click.option("--hardware", required=True, help="label for the cell, e.g. H100 / H100-slurm / cloud")
@click.option("--mode", type=click.Choice(["local", "cloud"]), required=True,
              help="local = measure GPU Joules + amortized GPU-hours $; cloud = Joules N/A, billed $")
@click.option("--out", "out_dir", required=True, type=click.Path(), help="Harbor jobs dir for this cell")
@click.option("--n-tasks", type=int, default=None, help="limit to first N TB tasks (omit = full set)")
@click.option("-k", "k", type=int, default=1, show_default=True, help="attempts per task")
@click.option("--env-type", default="docker", show_default=True, help="docker | singularity | ...")
@click.option("--gpu-hourly-rate", type=float, default=None, help="$/GPU-hour for local $ (default: H100 rate)")
@click.option("--n-gpus", type=int, default=1, show_default=True)
@click.option(
    "--energy-provenance",
    type=click.Choice(ENERGY_PROVENANCE_CHOICES),
    default=UNKNOWN_JOULES_PROVENANCE,
    show_default=True,
    help="Declare measured_dedicated only when the metered accelerators were isolated for the full run.",
)
@click.option("--price-in", type=float, default=None, help="$/M input tok (cloud fallback if Harbor has no cost)")
@click.option("--price-out", type=float, default=None, help="$/M output tok (cloud fallback)")
@click.option("--cells-jsonl", type=click.Path(), default=None, help="also append the cell to this JSONL")
@click.option("--n-concurrent", type=int, default=8, show_default=True,
              help="concurrent TB tasks (saturate; energy amortized per percentile method; cloud has no GPU energy to attribute)")
def measure_cmd(model, harness, dataset, hardware, mode, out_dir, n_tasks, k, env_type,
                gpu_hourly_rate, n_gpus, energy_provenance, price_in, price_out,
                cells_jsonl, n_concurrent) -> None:
    from .cell import measure_cell
    from .metrics import append_cell, write_cell

    kwargs = dict(
        model=model, harness=harness, hardware=hardware, dataset=dataset, output_dir=out_dir,
        is_local=(mode == "local"), k=k, n_tasks=n_tasks, env_type=env_type, n_gpus=n_gpus, n_concurrent=n_concurrent,
        energy_provenance=energy_provenance, price_in_per_m=price_in, price_out_per_m=price_out,
    )
    if gpu_hourly_rate is not None:
        kwargs["gpu_hourly_rate"] = gpu_hourly_rate

    cell = measure_cell(**kwargs)
    cell_path = Path(out_dir) / "cell.json"
    write_cell(cell, cell_path)
    if cells_jsonl:
        append_cell(cell, Path(cells_jsonl))
    click.echo(f"cell: acc={cell.accuracy} ${cell.dollars} ({cell.dollars_basis}) "
               f"J={cell.joules} -> {cell_path}")


@cli.command("orchestrate-v1", help="Run the v1 query-routing escalation cascade on TB V2.1.")
@click.option("--local-model", required=True)
@click.option("--local-base-url", required=True)
@click.option("--cloud-baseline-result", type=click.Path(exists=True), required=True,
              help="Path to the existing GPT-5.5 89-task Harbor result.json.")
@click.option("--cloud-label", default="gpt-5.5")
@click.option("--dataset", default="terminal-bench/terminal-bench-2-1")
@click.option("--output-dir", required=True)
@click.option("--cells", required=True)
@click.option("--n-tasks", type=int, default=None)
@click.option("--k", type=int, default=1)
@click.option("--n-concurrent", type=int, default=8, show_default=True,
              help="concurrent TB tasks for the local pass (saturate the serve; energy amortized per percentile method)")
@click.option("--agent-timeout-multiplier", type=float, default=None,
              help="scale each task's local agent timeout; <1 = fail-fast (don't burn GPU-hours on tasks that will escalate)")
@click.option(
    "--energy-provenance",
    type=click.Choice(ENERGY_PROVENANCE_CHOICES),
    default=UNKNOWN_JOULES_PROVENANCE,
    show_default=True,
    help="Declare measured_dedicated only when the metered accelerators were isolated for the full run.",
)
def orchestrate_v1_cmd(local_model, local_base_url, cloud_baseline_result, cloud_label,
                       dataset, output_dir, cells, n_tasks, k, n_concurrent,
                       agent_timeout_multiplier, energy_provenance):
    from .metrics import append_cell
    from .orchestrate.routing.v1_query import run_v1

    res = run_v1(
        local_model=local_model, dataset=dataset, output_dir=output_dir,
        local_base_url=local_base_url, cloud_baseline_result=cloud_baseline_result,
        cloud_label=cloud_label, n_tasks=n_tasks, k=k, n_concurrent=n_concurrent,
        agent_timeout_multiplier=agent_timeout_multiplier, energy_provenance=energy_provenance,
    )
    append_cell(res.realized_cell, cells)
    append_cell(res.oracle_cell, cells)
    click.echo(f"escalated {len(res.escalated)} tasks; wrote realized + oracle cells to {cells}")


@cli.command("orchestrate-v2", help="Run the v2 per-turn routing cascade on TB V2.1.")
@click.option("--local-model", required=True)
@click.option("--cloud-model", default="gpt-5.5-2026-04-23")
@click.option("--local-base-url", required=True)
@click.option("--cloud-base-url", default="https://api.openai.com/v1")
@click.option("--cloud-key-file", type=click.Path(exists=True),
              default=str(Path.home() / ".ipj_openai_key"))
@click.option("--local-turn-budget", type=int, default=15, show_default=True)
@click.option("--deferral-every-k", type=int, default=0, show_default=True,
              help="Self-assessment deferral: every K local turns, ask the model CONTINUE/DEFER and "
                   "escalate on DEFER (0 = off, fall back to the fixed turn budget).")
@click.option("--pre-route/--no-pre-route", default=False, show_default=True,
              help="Enable the SkillOrchestra pre-task router (classify each task -> local/cloud).")
@click.option("--router-model", default="gpt-5-mini", show_default=True)
@click.option("--router-margin", type=float, default=0.8, show_default=True,
              help="Route local if local_score >= margin*cloud_score (lower = more local).")
@click.option("--router-base-url", default=None, help="Classifier endpoint (default: cloud upstream).")
@click.option("--router-key-file", type=click.Path(exists=True), default=None,
              help="Classifier API-key file (default: --cloud-key-file).")
@click.option("--cloud-price-in", type=float, default=None, help="$/M input tok (cloud)")
@click.option("--cloud-price-out", type=float, default=None, help="$/M output tok (cloud)")
@click.option("--dataset", default="terminal-bench/terminal-bench-2-1")
@click.option("--output-dir", required=True)
@click.option("--cells", required=True)
@click.option("--n-tasks", type=int, default=None)
@click.option("--n-concurrent", type=int, default=1, show_default=True,
              help="Concurrent harbor trials (the per-turn proxy is task_key-safe under concurrency).")
@click.option("--k", type=int, default=1)
@click.option(
    "--energy-provenance",
    type=click.Choice(ENERGY_PROVENANCE_CHOICES),
    default=UNKNOWN_JOULES_PROVENANCE,
    show_default=True,
    help="Declare measured_dedicated only when the metered accelerators were isolated for the full run.",
)
def orchestrate_v2_cmd(local_model, cloud_model, local_base_url, cloud_base_url, cloud_key_file,
                       local_turn_budget, deferral_every_k, pre_route, router_model, router_margin,
                       router_base_url, router_key_file, cloud_price_in, cloud_price_out, dataset,
                       output_dir, cells, n_tasks, n_concurrent, k, energy_provenance):
    from .metrics import append_cell
    from .orchestrate.routing.v2_turn import run_v2

    cloud_api_key = Path(cloud_key_file).read_text().strip()
    router_api_key = Path(router_key_file).read_text().strip() if router_key_file else None
    res = run_v2(
        local_model=local_model, cloud_model=cloud_model, dataset=dataset, output_dir=output_dir,
        local_base_url=local_base_url, cloud_base_url=cloud_base_url, cloud_api_key=cloud_api_key,
        local_turn_budget=local_turn_budget, deferral_every_k=deferral_every_k, n_tasks=n_tasks,
        n_concurrent=n_concurrent, k=k,
        energy_provenance=energy_provenance,
        cloud_price_in=cloud_price_in, cloud_price_out=cloud_price_out,
        pre_route=pre_route, router_model=router_model, router_margin=router_margin,
        router_base_url=router_base_url, router_api_key=router_api_key,
    )
    append_cell(res.cell, cells)
    click.echo(f"v2: acc={res.cell.accuracy} escalated={res.n_escalated} deferred={res.n_deferred} "
               f"n_local={res.n_local} n_cloud={res.n_cloud} cloud_tok={res.cloud_tokens} -> {cells}")


cli.add_command(_results_group, name="results")


def main() -> None:
    cli()
