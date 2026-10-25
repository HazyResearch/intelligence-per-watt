from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
# --- house-style header (OpenJarvis figure guide): embed editable text, high-res ---
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,  # TrueType, not Type-3 — keeps text editable in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "font.size": 12,
    }
)
import matplotlib.pyplot as plt  # noqa: E402 (must come after backend set)
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .record import PerQueryRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS: list[str] = ["joules", "dollars", "latency_s", "input_tokens", "output_tokens", "flops"]

METRIC_LABELS: dict[str, str] = {
    "joules": "Energy per task (J)",
    "dollars": "Cost per task (USD)",
    "latency_s": "Latency per task (s)",
    "input_tokens": "Input tokens per task",
    "output_tokens": "Output tokens per task",
    "flops": "FLOPs per task",
}

# --- style-guide palette: greens = local/our methods, blues = cloud/frontier ---
GRID = "#AAAAAA"
PARETO = "#4CAF50"
REF_BLUE = "#1976D2"
TEXT_DARK = "#1A1A1A"

# Per-category marker + per-model color family (darker = stronger). Cascade = the deep
# anchor #1B5E20 (the headline result); oracle = amber X (a hypothetical upper bound).
CATEGORY_STYLE: dict[str, dict[str, Any]] = {
    "open": {"marker": "o", "family": ["#2E7D32", "#66BB6A", "#00897B", "#7CB342"], "label": "Open (local)"},
    "closed": {"marker": "^", "family": ["#0D47A1", "#1976D2", "#42A5F5", "#90CAF9"], "label": "Closed (cloud)"},
    "cascade": {"marker": "*", "family": ["#1B5E20"], "label": "Cascade (ours)"},
    "orchestration": {"marker": "D", "family": ["#827717"], "label": "Orchestration"},
}
ORACLE_STYLE: dict[str, Any] = {"marker": "X", "color": "#F9A825", "label": "Oracle (per-query best)"}

BASE_SIZE = 150
STAR_SIZE = 420
ORACLE_SIZE = 230
ANNOT_FS = 8


def style_axis(ax: plt.Axes) -> None:
    """Four black spines at 1.5 lw; dashed grey grid behind the data (zorder 1)."""
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5, color=GRID, zorder=1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def aggregate_by_approach(records: list[PerQueryRecord]) -> dict[str, dict[str, Any]]:
    """Return {approach: {accuracy, category, joules, dollars, ...}} aggregated over tasks.

    accuracy  = mean(solved) across that approach's records.
    Each metric = mean of non-None values; None if all values are None.
    """
    buckets: dict[str, list[PerQueryRecord]] = {}
    for r in records:
        buckets.setdefault(r.approach, []).append(r)

    result: dict[str, dict[str, Any]] = {}
    for approach, recs in buckets.items():
        accuracy = float(np.mean([r.solved for r in recs]))
        category = recs[0].category  # all same approach → same category
        agg: dict[str, Any] = {"accuracy": accuracy, "category": category}
        for metric in METRICS:
            vals = [getattr(r, metric) for r in recs if getattr(r, metric) is not None]
            agg[metric] = float(np.mean(vals)) if vals else None
        result[approach] = agg
    return result


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the Pareto-non-dominated subset of (x_metric, accuracy) pairs.

    A point (x1, y1) is dominated by (x2, y2) when x2 <= x1 AND y2 >= y1 (strictly one).
    Lower x and higher y (accuracy) are both "better".
    """
    if not points:
        return []

    frontier: list[tuple[float, float]] = []
    for candidate in points:
        cx, cy = candidate
        dominated = False
        for other in points:
            if other is candidate:
                continue
            ox, oy = other
            if ox <= cx and oy >= cy and (ox < cx or oy > cy):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def oracle_per_metric(
    records: list[PerQueryRecord],
    metric: str,
) -> tuple[float, float | None]:
    """Compute oracle accuracy and oracle metric value.

    For each task, among non-'oracle'-category approaches that solved it, pick the lowest
    metric value (skipping None). oracle_accuracy = (#tasks with >=1 solver) / (#tasks total).
    oracle_metric = mean of per-task minima (tasks with no valid metric reading are excluded
    from the mean, but still counted toward the denominator for accuracy).

    Returns (accuracy, metric_value).  metric_value is None if there are no valid readings.
    """
    task_ids: set[str] = {r.task_id for r in records}
    total = len(task_ids)
    if total == 0:
        return 0.0, None

    per_task_minima: list[float] = []
    solved_count = 0

    for task_id in task_ids:
        task_recs = [r for r in records if r.task_id == task_id and r.category != "oracle"]
        solved_recs = [r for r in task_recs if r.solved]
        if not solved_recs:
            continue
        solved_count += 1
        metric_vals = [getattr(r, metric) for r in solved_recs if getattr(r, metric) is not None]
        if metric_vals:
            per_task_minima.append(min(metric_vals))

    accuracy = solved_count / total
    metric_value = float(np.mean(per_task_minima)) if per_task_minima else None
    return accuracy, metric_value


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _assign_colors(agg: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Deterministic per-model color drawn from its category's family (sorted by name)."""
    by_cat: dict[str, list[str]] = {}
    for ap, d in agg.items():
        by_cat.setdefault(d["category"], []).append(ap)
    colors: dict[str, str] = {}
    for cat, aps in by_cat.items():
        fam = CATEGORY_STYLE.get(cat, {"family": ["#757575"]})["family"]
        for i, ap in enumerate(sorted(aps)):
            colors[ap] = fam[i % len(fam)]
    return colors


def _annotate(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    """Label a point with its model name in a white-backed box (kept readable on overlap)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=ANNOT_FS,
        color=TEXT_DARK,
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.7, alpha=0.85),
    )


def _draw_pareto_line(ax: plt.Axes, pts: list[tuple[float, float]]) -> None:
    """Dashed green Pareto frontier (style guide §6), behind the markers."""
    if len(pts) < 2:
        return
    sorted_pts = sorted(pts, key=lambda p: p[0])
    ax.plot(
        [p[0] for p in sorted_pts],
        [p[1] for p in sorted_pts],
        linestyle="--",
        color=PARETO,
        linewidth=2.0,
        alpha=0.7,
        zorder=3,
    )


def _plot_single_metric(
    records: list[PerQueryRecord],
    metric: str,
    out_path: Path,
) -> None:
    """Create and save one accuracy-vs-metric figure (PNG + editable PDF)."""
    agg = aggregate_by_approach(records)
    colors = _assign_colors(agg)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    style_axis(ax)
    ax.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(f"Accuracy vs {METRIC_LABELS.get(metric, metric)}", fontsize=15, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))

    cats_present: set[str] = set()
    pareto_points: list[tuple[float, float]] = []
    closed_lines: list[tuple[str, float]] = []  # joules: closed models become reference lines

    for approach, data in agg.items():
        cat = data["category"]
        acc = data["accuracy"]
        x_val = data.get(metric)
        if cat == "closed" and metric == "joules":
            closed_lines.append((approach, acc))
            continue
        if x_val is None:
            continue
        st = CATEGORY_STYLE.get(cat, {"marker": "o"})
        is_cascade = cat == "cascade"
        ax.scatter(
            x_val,
            acc,
            marker=st["marker"],
            color=colors[approach],
            s=STAR_SIZE if is_cascade else BASE_SIZE,
            zorder=6 if cat in ("closed", "cascade") else 5,
            edgecolors="white",
            linewidths=1.0,
        )
        _annotate(ax, x_val, acc, approach, colors[approach])
        pareto_points.append((x_val, acc))
        cats_present.add(cat)

    # Oracle (per-query best solver) — amber X
    oracle_acc, oracle_x = oracle_per_metric(records, metric)
    if oracle_x is not None:
        ax.scatter(
            oracle_x,
            oracle_acc,
            marker=ORACLE_STYLE["marker"],
            color=ORACLE_STYLE["color"],
            s=ORACLE_SIZE,
            zorder=7,
            edgecolors="white",
            linewidths=1.0,
        )
        _annotate(ax, oracle_x, oracle_acc, "Oracle", ORACLE_STYLE["color"])
        pareto_points.append((oracle_x, oracle_acc))
        cats_present.add("oracle")

    _draw_pareto_line(ax, pareto_frontier(pareto_points))

    # On JOULES: cloud models have no local energy → draw as dotted reference lines, each named.
    if metric == "joules" and closed_lines:
        x_left = ax.get_xlim()[0]
        for approach, acc in closed_lines:
            c = colors[approach]
            ax.axhline(y=acc, linestyle=":", color=c, alpha=0.75, linewidth=1.6, zorder=2)
            ax.text(
                x_left,
                acc,
                f" {approach}",
                fontsize=ANNOT_FS,
                color=c,
                va="bottom",
                ha="left",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c, lw=0.7, alpha=0.85),
            )
        cats_present.add("closed")

    # Framed legend: category guide + oracle + Pareto (model names live on the points).
    handles: list[Line2D] = []
    for cat in ("open", "closed", "cascade", "orchestration"):
        if cat in cats_present:
            st = CATEGORY_STYLE[cat]
            handles.append(
                Line2D(
                    [0], [0], marker=st["marker"], color="w",
                    markerfacecolor=st["family"][1 % len(st["family"])],
                    markeredgecolor="white", markersize=12, label=st["label"],
                )
            )
    if "oracle" in cats_present:
        handles.append(
            Line2D(
                [0], [0], marker=ORACLE_STYLE["marker"], color="w",
                markerfacecolor=ORACLE_STYLE["color"], markeredgecolor="white",
                markersize=12, label=ORACLE_STYLE["label"],
            )
        )
    handles.append(Line2D([0], [0], linestyle="--", color=PARETO, lw=2, label="Pareto frontier"))
    leg = ax.legend(handles=handles, fontsize=9, loc="best", frameon=True, framealpha=0.95, edgecolor="black")
    leg.get_frame().set_linewidth(1.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")  # editable vector for the paper
    plt.close(fig)


def plot_all(records: list[PerQueryRecord], out_dir: Path | str) -> None:
    """Write 6 accuracy-vs-efficiency figures (PNG + PDF each) to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        _plot_single_metric(records, metric, out_dir / f"accuracy_vs_{metric}.png")
