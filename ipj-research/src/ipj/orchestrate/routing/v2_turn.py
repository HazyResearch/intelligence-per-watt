"""v2 per-turn routing orchestrator: run terminus-2 through the routing proxy, emit a Cell."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ...cell import find_latest_result_json
from ...cost import H100_HOURLY_USD, dollars_local, dollars_tokens
from ...energy.capture import RunEnergyCapture
from ...energy.estimate import estimate_from_reference
from ...harbor.results import parse_harbor_summary
from ...harbor.run import harbor_command, run_command
from ...metrics import (
    MEASURED_DEDICATED_JOULES_PROVENANCE,
    UNKNOWN_JOULES_PROVENANCE,
    Cell,
    accelerator_joules_are_complete,
    resolve_joules_provenance,
)
from .v2_proxy import ProxyServer, RoutingState


@dataclass
class V2Result:
    cell: Cell
    n_escalated: int
    local_tokens: dict
    cloud_tokens: dict
    n_local: int = 0
    n_cloud: int = 0
    n_deferred: int = 0
    classifier_dollars: float | None = None


def run_v2(
    *,
    local_model: str,
    cloud_model: str,
    dataset: str,
    output_dir,
    local_base_url: str,
    cloud_base_url: str,
    cloud_api_key: str,
    local_api_key: str = "EMPTY",
    local_turn_budget: int = 15,
    deferral_every_k: int = 0,
    k: int = 1,
    n_tasks: int | None = None,
    n_concurrent: int = 1,
    gpu_hourly_rate: float = H100_HOURLY_USD,
    n_gpus: int = 1,
    energy_provenance: str = UNKNOWN_JOULES_PROVENANCE,
    cloud_price_in: float | None = None,
    cloud_price_out: float | None = None,
    pre_route: bool = False,
    router_model: str = "gpt-5-mini",
    router_margin: float = 0.8,
    router_base_url: str | None = None,
    router_api_key: str | None = None,
    run_fn=run_command,
    capture_factory=RunEnergyCapture,
    proxy_factory=ProxyServer,
) -> V2Result:
    output_dir = Path(output_dir)
    state = RoutingState(local_turn_budget=local_turn_budget, defer_every_k=deferral_every_k)
    upstreams = {
        "local": (local_base_url, local_api_key),
        "cloud": (cloud_base_url, cloud_api_key, cloud_model),
    }
    # Self-assessment deferral: every deferral_every_k local turns, ask the local model itself
    # CONTINUE/DEFER and hand off to cloud on DEFER (the realized "gauge it" router).
    deferral = None
    if deferral_every_k > 0:
        from .deferral import assess_deferral
        from .v2_proxy import _default_forward
        def deferral(messages):  # noqa: E306 — bound to the local upstream
            return assess_deferral(messages, base_url=local_base_url, api_key=local_api_key,
                                   model=local_model, forward_fn=_default_forward)
    router = None
    if pre_route:
        # Phase 0 exposes one shared API price pair for the cloud worker and
        # classifier; distinct classifier price schedules are deferred.
        router = {
            "base_url": router_base_url or cloud_base_url,
            "api_key": router_api_key or cloud_api_key,
            "model": router_model,
            "margin": router_margin,
            "price_in": cloud_price_in,
            "price_out": cloud_price_out,
        }
    proxy = proxy_factory(state, upstreams, router=router, deferral=deferral)
    port = proxy.start()
    try:
        cmd = harbor_command(dataset, "terminus-2", f"openai/{local_model}",
                             output_path=output_dir, k=k, n_tasks=n_tasks, n_concurrent=n_concurrent)
        env = {**os.environ, "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1", "OPENAI_API_KEY": "EMPTY"}
        with capture_factory() as cap:
            win = run_fn(cmd, env=env)
            local_joules = cap.joules(win.start, win.end)
            device_coverage_complete = bool(
                getattr(cap, "accelerator_device_coverage_complete", lambda *_: False)(
                    win.start,
                    win.end,
                )
            )
    finally:
        proxy.stop()
    if win.returncode != 0:
        raise RuntimeError(f"v2 harbor run failed (rc={win.returncode})")

    summary = parse_harbor_summary(find_latest_result_json(output_dir))
    wall = win.end - win.start
    lt, ct = state.tokens["local"], state.tokens["cloud"]

    local_dollars = dollars_local(wall, gpu_hourly_rate, n_gpus=n_gpus)
    cloud_used = state.requests["cloud"] > 0 or state.n_cloud > 0
    # A successful pre-route call is part of online serving. Its accelerator
    # energy is not covered by the local worker meter, even when it routes the
    # task local, so it invalidates rigorous accelerator-IPJ completeness.
    external_classifier_used = pre_route and (state.n_local + state.n_cloud > 0)
    cloud_dollars = None
    if not cloud_used:
        cloud_dollars = 0.0
    elif (
        ct["in"] is not None
        and ct["out"] is not None
        and cloud_price_in is not None
        and cloud_price_out is not None
    ):
        cloud_dollars = dollars_tokens(ct["in"], ct["out"], cloud_price_in, cloud_price_out)

    classifier_dollars = state.classifier_dollars
    unknown_costs = []
    if cloud_dollars is None:
        unknown_costs.append("cloud API")
    if classifier_dollars is None:
        unknown_costs.append("router classifier")
    if unknown_costs:
        joined = ", ".join(unknown_costs)
        raise ValueError(f"cannot construct v2 Cell: unknown cost for used component(s): {joined}")

    unknown_token_tiers = [
        tier
        for tier in ("local", "cloud")
        if state.requests[tier] > 0
        and (state.tokens[tier]["in"] is None or state.tokens[tier]["out"] is None)
    ]
    if unknown_token_tiers:
        joined = ", ".join(unknown_token_tiers)
        raise ValueError(f"cannot construct v2 Cell: unknown token usage for used tier(s): {joined}")

    # Cloud Joules are ESTIMATED by transferring this run's local J/token to the cloud tokens
    # (ADR 0003). Eager local serving overstates J/token, so this is a loose upper bound.
    cloud_estimate = None
    if ct["in"] + ct["out"] > 0 and lt["in"] + lt["out"] > 0 and local_joules is not None:
        cloud_estimate = estimate_from_reference(
            ct["in"], ct["out"], ref_joules=local_joules,
            ref_tokens_in=lt["in"], ref_tokens_out=lt["out"], ref_name="v2 local turns",
        )

    measured_local = local_joules is not None
    joules_provenance = resolve_joules_provenance(local_joules, energy_provenance)
    if not measured_local:
        energy_source = "unknown local energy (telemetry unavailable or incomplete run-window coverage)"
    elif joules_provenance == MEASURED_DEDICATED_JOULES_PROVENANCE:
        energy_source = "energy-monitor (local accelerator only; dedicated isolation declared)"
    else:
        energy_source = "energy-monitor (local accelerator only; isolation unverified)"

    cell = Cell(
        model=f"cascade-v2[{local_model}->{cloud_model}]", harness="terminus-2-cascade-v2",
        hardware="h100-local+cloud", dataset=dataset, n_tasks=summary.n_tasks, accuracy=summary.accuracy,
        tokens_in=lt["in"] + ct["in"], tokens_out=lt["out"] + ct["out"],
        dollars=local_dollars + cloud_dollars + classifier_dollars, dollars_basis="gpu-hours+api",
        # ``joules`` is measured-only.  The cloud estimate remains a separate
        # interval and therefore cannot accidentally enter rigorous IPJ.
        joules=local_joules, wall_clock_s=wall,
        energy_source=energy_source, run_dir=str(output_dir),
        n_local=state.n_local, n_cloud=state.n_cloud, classifier_dollars=classifier_dollars,
        joules_provenance=joules_provenance, joules_scope="local-accelerator",
        joules_complete=(
            accelerator_joules_are_complete(
                local_joules,
                joules_provenance,
                device_coverage_complete,
            )
            and not cloud_used
            and not external_classifier_used
        ),
        accelerator_device_coverage_complete=device_coverage_complete,
        cloud_joules_estimate=cloud_estimate,
    )
    return V2Result(cell=cell, n_escalated=state.n_escalated, local_tokens=lt, cloud_tokens=ct,
                    n_local=state.n_local, n_cloud=state.n_cloud, n_deferred=state.n_deferred,
                    classifier_dollars=classifier_dollars)
