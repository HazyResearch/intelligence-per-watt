"""Aggregate verified Trajectories into per-arm accuracy/$/Joules and a comparison table vs the cascade bar."""

from __future__ import annotations

from dataclasses import dataclass

from ipj.orchestrate.harness.trace import (
    Trajectory,
    is_valid_nonnegative_number,
    is_valid_token_count,
)


@dataclass(slots=True)
class ArmMetrics:
    n: int
    accuracy: float
    dollars_mean: float | None
    gpu_energy_j_mean: float | None
    tokens_out_mean: float | None
    dollars_known_n: int = 0
    gpu_energy_complete_n: int = 0
    tokens_out_known_n: int = 0


def _complete_mean(xs: list[float], expected: int) -> float | None:
    if expected == 0 or len(xs) != expected:
        return None
    return sum(xs) / len(xs)


def arm_metrics(trajectories: list[Trajectory]) -> ArmMetrics:
    n = len(trajectories)
    succ = sum(1 for t in trajectories if t.outcome is not None and t.outcome.success)
    dollars = [
        t.outcome.total_dollars
        for t in trajectories
        if t.outcome and is_valid_nonnegative_number(t.outcome.total_dollars)
    ]
    toks = [
        t.outcome.total_tokens_out
        for t in trajectories
        if t.outcome and is_valid_token_count(t.outcome.total_tokens_out)
    ]
    js = [
        t.outcome.total_gpu_energy_j
        for t in trajectories
        if t.outcome
        and t.outcome.gpu_energy_complete is True
        and is_valid_nonnegative_number(t.outcome.total_gpu_energy_j)
    ]
    return ArmMetrics(
        n=n,
        accuracy=(succ / n if n else 0.0),
        dollars_mean=_complete_mean(dollars, n),
        dollars_known_n=len(dollars),
        gpu_energy_j_mean=_complete_mean(js, n),
        gpu_energy_complete_n=len(js),
        tokens_out_mean=_complete_mean([float(x) for x in toks], n),
        tokens_out_known_n=len(toks),
    )


# The recorded cascade bar to beat (EXP-20260627-02): 76% @ ~$0.38/task
# over 89 tasks. Its historical ~160 kJ/task value remains case-study
# context, but lacks the explicit completeness metadata required here.
BAR_HISTORICAL_GPU_ENERGY_J_MEAN = 160_000.0
BAR = ArmMetrics(
    n=89,
    accuracy=0.76,
    dollars_mean=0.38,
    dollars_known_n=89,
    gpu_energy_j_mean=None,
    gpu_energy_complete_n=0,
    tokens_out_mean=0.0,
    tokens_out_known_n=89,
)


def _row(name: str, m: ArmMetrics) -> str:
    dollars = "—" if m.dollars_mean is None else f"{m.dollars_mean:.4f}"
    j = "—" if m.gpu_energy_j_mean is None else f"{m.gpu_energy_j_mean:.0f}"
    tokens = "—" if m.tokens_out_mean is None else f"{m.tokens_out_mean:.0f}"
    coverage = (
        f"cost {m.dollars_known_n}/{m.n}; energy {m.gpu_energy_complete_n}/{m.n}; "
        f"tokens {m.tokens_out_known_n}/{m.n}"
    )
    return (
        f"| {name} | {m.n} | {m.accuracy:.2f} | {dollars} | {j} | "
        f"{tokens} | {coverage} |"
    )


def compare_table(arms: dict[str, ArmMetrics]) -> str:
    header = (
        "| arm | n | accuracy | $/task | J/task | tokens_out/task | coverage |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = [_row(name, m) for name, m in arms.items()] + [_row("cascade bar", BAR)]
    historical_context = (
        f"\nHistorical cascade energy context: ~{BAR_HISTORICAL_GPU_ENERGY_J_MEAN:.0f} J/task "
        "(not rigorous; completeness metadata unavailable)."
    )
    return "\n".join([header, *rows, historical_context])


def metrics_to_dict(arms: dict[str, ArmMetrics]) -> dict:
    from dataclasses import asdict

    serialized = {
        name: asdict(metrics)
        for name, metrics in {**arms, "cascade_bar": BAR}.items()
    }
    serialized["cascade_bar"]["historical_gpu_energy_j_mean"] = (
        BAR_HISTORICAL_GPU_ENERGY_J_MEAN
    )
    serialized["cascade_bar"]["historical_energy_rigorous"] = False
    return serialized
