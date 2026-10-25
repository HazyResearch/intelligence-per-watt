"""Fig-1: accuracy vs cost ($ or Joules) scatter, in the house figure style.

Conventions follow ipj/references/figure-style-guide.md (OpenJarvis NeurIPS suite):
fonttype-42 embedding, greens for local/ours + blues for cloud, white-edged
markers, dashed-grey grid behind black spines, explicit zorder, %-accuracy axis,
log-scaled cost axis, and a PDF (vector, paper) + PNG (raster, preview) export.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# House-style header (keep text editable in Illustrator; high render quality).
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
    }
)

# Palette (figure-style-guide §2): greens = local/ours, blues = cloud/frontier.
LOCAL_GREEN = "#1B5E20"  # deep anchor — our/local results
CLOUD_BLUE = "#0D47A1"  # strongest cloud
GRID = "#AAAAAA"
PARETO = "#4CAF50"
TEXT_DARK = "#1A1A1A"
BRACKET = "#222222"
MARKER_SCALE = 11.0  # base markersize for in-graph points


def cost_multiplier(cheaper_dollars: float, pricier_dollars: float) -> float:
    """How many times cheaper: pricier / cheaper (e.g. cloud $1.40 vs local $0.07 → ~20)."""
    if cheaper_dollars <= 0:
        return float("inf")
    return pricier_dollars / cheaper_dollars


def draw_improvement_bracket(ax, x_lo, x_hi, y, label, *, color=BRACKET, fontsize=11, tick_h=0.02):
    """Horizontal double-headed 'N× cheaper' bracket between x_lo and x_hi at height y.

    For a log-x (cost) axis the label sits at the geometric midpoint (guide §7).
    """
    ax.annotate(
        "",
        xy=(x_hi, y),
        xytext=(x_lo, y),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.8, mutation_scale=12),
        zorder=11,
    )
    for xv in (x_lo, x_hi):
        ax.plot([xv, xv], [y - tick_h, y + tick_h], color=color, lw=1.5, zorder=11, solid_capstyle="butt")
    x_mid = (x_lo * x_hi) ** 0.5 if (x_lo > 0 and x_hi > 0) else (x_lo + x_hi) / 2
    ax.text(
        x_mid,
        y + tick_h * 1.5,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        zorder=12,
    )


def style_axis(ax, grid: bool = True) -> None:
    """Dashed grey grid behind data; four black spines at 1.5 lw (guide §3)."""
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5, color=GRID, zorder=1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)


def compute_pareto_frontier(x_vals, y_vals):
    """Pareto frontier for minimize-x (cost), maximize-y (accuracy) — guide §6."""
    points = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
    px, py, max_y = [], [], float("-inf")
    for x, y in points:
        if y >= max_y:
            px.append(x)
            py.append(y)
            max_y = y
    return px, py


def _is_local(cell: dict) -> bool:
    """Local (our) cell vs cloud baseline. Prefer dollars_basis; fall back to measured joules."""
    basis = cell.get("dollars_basis")
    if basis is not None:
        return basis == "gpu-hours"
    return cell.get("joules") is not None


# Efficiency axes that accuracy is plotted against — the x-axis menu shared by every
# scatter helper below. `log` chooses a log x-scale (orders-of-magnitude quantities);
# `scale` divides the raw value for axis readability (e.g. Joules -> MJ). Keep raw
# units in the cells; the scaling is presentation-only and applied identically to the
# Pareto frontier so it stays consistent.
METRIC_SPEC: dict[str, dict] = {
    "joules": {"label": "Energy (MJ / run)", "log": False, "scale": 1e6},
    "dollars": {"label": "Cost ($ / run)", "log": True, "scale": 1.0},
    "latency": {"label": "Latency (s / run)", "log": False, "scale": 1.0},
    "flops": {"label": "FLOPs / run", "log": True, "scale": 1.0},
    "tokens": {"label": "Total tokens / run", "log": True, "scale": 1.0},
    "tokens_in": {"label": "Input tokens / run", "log": True, "scale": 1.0},
    "tokens_out": {"label": "Output tokens / run", "log": True, "scale": 1.0},
}

# Default panel order for the trade-off grid: cost-like axes first, token axes second.
DEFAULT_METRICS = ["joules", "dollars", "latency", "flops", "tokens", "tokens_in", "tokens_out"]


def _metric_value(
    cell: dict,
    metric: str,
    *,
    allow_incomplete_energy: bool = True,
    allow_incomplete_cost: bool = True,
):
    """Return a raw metric value with opt-in rigorous evidence gates.

    The compatibility default retains historical plots for two public
    releases. Callers producing paper claims must disable the corresponding
    incomplete-data flag so Joules or dollars require explicit complete,
    pinned evidence and verifier-grade utility.
    """
    from ipj.flops import cell_flops

    ti, to = cell.get("tokens_in", 0) or 0, cell.get("tokens_out", 0) or 0
    if metric == "flops":
        return cell_flops(cell.get("model", ""), ti, to)
    if metric == "latency":
        return cell.get("wall_clock_s")
    if metric == "tokens":
        return (ti + to) or None
    if metric == "tokens_in":
        return ti or None
    if metric == "tokens_out":
        return to or None
    if metric == "joules":
        raw_joules = cell.get("joules")
        if isinstance(raw_joules, bool) or not isinstance(raw_joules, (int, float)):
            return None
        joules = float(raw_joules)
        if not math.isfinite(joules) or joules < 0:
            return None
        if allow_incomplete_energy:
            return joules

        from ipj.metrics import accelerator_joules_are_complete, utility_is_complete

        eligible = (
            cell.get("joules_complete") is True
            and cell.get("joules_scope") in {"accelerator", "local-accelerator"}
            and accelerator_joules_are_complete(
                joules,
                cell.get("joules_provenance"),
                cell.get("accelerator_device_coverage_complete") is True,
            )
            and utility_is_complete(
                cell.get("accuracy"),
                cell.get("n_tasks"),
                cell.get("utility_complete"),
                cell.get("utility_provenance"),
                cell.get("utility_calibration_id"),
                cell.get("utility_verifier_id"),
                cell.get("utility_verifier_version"),
                cell.get("utility_task_manifest_id"),
                cell.get("utility_membership_complete"),
                cell.get("utility_task_manifest_size"),
            )
        )
        return joules if eligible else None
    if metric == "dollars":
        raw_dollars = cell.get("dollars")
        if isinstance(raw_dollars, bool) or not isinstance(raw_dollars, (int, float)):
            return None
        dollars = float(raw_dollars)
        if not math.isfinite(dollars) or dollars < 0:
            return None
        if allow_incomplete_cost:
            return dollars

        from ipj.metrics import dollars_are_complete, utility_is_complete

        eligible = dollars_are_complete(
            dollars,
            cell.get("dollars_complete"),
            cell.get("dollars_accounting_view"),
            cell.get("dollars_snapshot_id"),
            cell.get("dollars_provenance"),
            cell.get("dollars_api_charge_kind"),
        ) and utility_is_complete(
            cell.get("accuracy"),
            cell.get("n_tasks"),
            cell.get("utility_complete"),
            cell.get("utility_provenance"),
            cell.get("utility_calibration_id"),
            cell.get("utility_verifier_id"),
            cell.get("utility_verifier_version"),
            cell.get("utility_task_manifest_id"),
            cell.get("utility_membership_complete"),
            cell.get("utility_task_manifest_size"),
        )
        return dollars if eligible else None
    return cell.get(metric)


def _short_model(model: str) -> str:
    """Compact label for annotations: drop routing/vendor prefixes, prettify special cells."""
    ml = model.lower()
    if "router" in ml or "d6" in ml:  # router-d6[...] -> "D6 router"
        return "D6 router"
    if "cascade" in ml or "orchestr" in ml:  # cascade-v1-realized -> "Cascade realized"
        if "v2" in ml or "perturn" in ml or "per-turn" in ml:
            return "Cascade v2 (per-turn)"
        if "realized" in ml:
            return "Cascade realized"
        if "3tier" in ml or "3-tier" in ml:
            return "Cascade 3-tier"
        if "oracle" in ml:
            return "Cascade oracle"
        return "Cascade"
    m = model.split("/")[-1] if "/" in model else model
    if m.startswith("gpt-5.5"):
        return "GPT-5.5"
    if m.startswith("gemini"):
        return m
    return m


def _validate_strict_comparison_basis(
    cells: list[dict],
    metric: str,
    *,
    allow_incomplete_energy: bool,
    allow_incomplete_cost: bool,
) -> None:
    """Reject a rigorous frontier whose retained points use unlike evidence."""
    strict_metric = (metric == "joules" and not allow_incomplete_energy) or (
        metric == "dollars" and not allow_incomplete_cost
    )
    if not strict_metric:
        return
    retained = [
        cell
        for cell in cells
        if _metric_value(
            cell,
            metric,
            allow_incomplete_energy=allow_incomplete_energy,
            allow_incomplete_cost=allow_incomplete_cost,
        )
        is not None
    ]
    utility_bases = {
        (
            cell.get("utility_provenance"),
            cell.get("utility_verifier_id"),
            cell.get("utility_verifier_version"),
            cell.get("utility_task_manifest_id"),
            cell.get("utility_calibration_id"),
            cell.get("dataset"),
            cell.get("n_tasks"),
        )
        for cell in retained
    }
    if len(utility_bases) > 1:
        raise ValueError(
            "strict comparison requires one utility verifier, version, task manifest, "
            "task count, dataset, provenance kind, and calibration basis"
        )
    if metric == "dollars":
        cost_bases = {
            (
                cell.get("dollars_accounting_view"),
                cell.get("dollars_snapshot_id"),
                cell.get("dollars_api_charge_kind"),
            )
            for cell in retained
        }
        if len(cost_bases) > 1:
            raise ValueError(
                "strict dollar comparison requires one accounting view, snapshot, and typed API charge kind"
            )


# Marker geometry for the rich scatter: SIZE encodes total model params (bigger model =
# bigger dot), COLOR encodes local/ours (green) vs cloud (blue), SHAPE is keyed by harness
# (one harness today -> circle for local, triangle for cloud; new harnesses get new shapes).
# Routing/cascade *strategies* are a third kind: a teal star (a strategy is a system, not a
# single-size model, so it gets a fixed size and its own colour).
_SIZE_BASE, _SIZE_SCALE = 8.0, 2.7  # markersize (pt) = base + scale * sqrt(params_in_B)
_CLOUD_SIZE = 16.0  # cloud params unknown -> fixed reference size
STRATEGY_GREEN = "#004D40"  # teal_deep — routing/cascade (distinct from local #1B5E20)
_STRATEGY_SIZE = 20.0  # fixed; a strategy spans models, so no single param size
# (local_marker, cloud_marker) per harness; extend as harnesses are added.
HARNESS_MARKERS: dict[str, tuple[str, str]] = {"terminus-2": ("o", "^")}
_SIZE_LEGEND_B = [4, 14, 27, 35]  # reference param counts (B) for the size legend


def _size_from_params(params: float | None) -> float:
    """Markersize (pt) from a param count; cloud/unknown -> fixed reference size."""
    if not params:
        return _CLOUD_SIZE
    return _SIZE_BASE + _SIZE_SCALE * (params / 1e9) ** 0.5


def _cell_kind(cell: dict) -> str:
    """'strategy' (routing/cascade), 'local' (our single model), or 'cloud' (baseline)."""
    tag = f"{cell.get('model', '')} {cell.get('harness', '')}".lower()
    if any(w in tag for w in ("cascade", "orchestr", "route")):
        return "strategy"
    return "local" if _is_local(cell) else "cloud"


def _point_style(cell: dict) -> tuple[str, str, float]:
    """(color, marker, markersize) under the color=method / size=params / shape=harness encoding."""
    from ipj.flops import total_params

    kind = _cell_kind(cell)
    if kind == "strategy":
        return STRATEGY_GREEN, "*", _STRATEGY_SIZE
    lm, cm = HARNESS_MARKERS.get(cell.get("harness", ""), ("o", "^"))
    if kind == "cloud":
        return CLOUD_BLUE, cm, _CLOUD_SIZE
    return LOCAL_GREEN, lm, _size_from_params(total_params(cell.get("model", "")))


def plot_cost_frontier(
    cells: list[dict],
    out_path,
    x: str = "dollars",
    *,
    allow_incomplete_energy: bool = True,
    allow_incomplete_cost: bool = True,
) -> Path:
    """Scatter accuracy vs cost; green circles = local/ours, blue triangles = cloud.

    x: "dollars" (log axis) or "joules" (linear; cloud cells have none and are skipped).
    Writes both a vector PDF and a raster PNG (sibling of out_path). Returns out_path.
    """
    _validate_strict_comparison_basis(
        cells,
        x,
        allow_incomplete_energy=allow_incomplete_energy,
        allow_incomplete_cost=allow_incomplete_cost,
    )
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_box_aspect(1)

    xs, ys = [], []
    eligible_costs: list[tuple[dict, float]] = []
    for c in cells:
        xv = _metric_value(
            c,
            x,
            allow_incomplete_energy=allow_incomplete_energy,
            allow_incomplete_cost=allow_incomplete_cost,
        )
        if xv is None:
            continue
        local = _is_local(c)
        ax.plot(
            xv,
            c["accuracy"],
            marker="o" if local else "^",
            color=LOCAL_GREEN if local else CLOUD_BLUE,
            markersize=MARKER_SCALE,
            markeredgecolor="white",
            markeredgewidth=0.8 if local else 1.0,
            linestyle="none",
            zorder=5 if local else 6,
        )
        ax.annotate(
            c.get("model", ""),
            (xv, c["accuracy"]),
            fontsize=9,
            color=TEXT_DARK,
            xytext=(5, 5),
            textcoords="offset points",
            zorder=7,
        )
        xs.append(xv)
        ys.append(c["accuracy"])
        if x == "dollars":
            eligible_costs.append((c, xv))

    # Pareto frontier (only meaningful with >= 2 points), dashed green behind markers.
    if len(xs) >= 2:
        px, py = compute_pareto_frontier(xs, ys)
        if len(px) >= 2:
            ax.plot(px, py, "--", color=PARETO, alpha=0.6, linewidth=2, zorder=3)

    # Headline "N× cheaper" bracket: cheapest local/ours vs the strongest cloud (guide §7).
    if x == "dollars":
        locals_ = [(c, value) for c, value in eligible_costs if _is_local(c) and value > 0]
        clouds_ = [(c, value) for c, value in eligible_costs if not _is_local(c) and value > 0]
        if locals_ and clouds_:
            us, us_dollars = min(locals_, key=lambda item: item[1])
            cloud, cloud_dollars = max(clouds_, key=lambda item: item[0]["accuracy"])
            mult = cost_multiplier(us_dollars, cloud_dollars)
            y = min(us["accuracy"], cloud["accuracy"])
            label = f"{mult:.0f}× cheaper"
            if allow_incomplete_cost:
                label = f"Legacy {label}"
            draw_improvement_bracket(ax, us_dollars, cloud_dollars, y, label)

    _XLABEL = {
        "dollars": "$ per run (log)",
        "joules": "Joules per run",
        "flops": "FLOPs per run (log)",
        "latency": "Latency per run (s)",
        "tokens": "Total tokens per run (log)",
        "tokens_in": "Input tokens per run (log)",
        "tokens_out": "Output tokens per run (log)",
    }
    if x in {"dollars", "flops", "tokens", "tokens_in", "tokens_out"}:
        ax.set_xscale("log")
    ax.set_xlabel(_XLABEL.get(x, x), fontsize=16, fontweight="bold")
    ax.set_ylabel("Accuracy (TB V2.1)", fontsize=16, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v * 100))}%"))
    _TITLE = {
        "dollars": "Accuracy vs Cost — TB V2.1",
        "joules": "Accuracy vs Measured Energy — TB V2.1",
        "flops": "Accuracy vs FLOPs — TB V2.1",
        "latency": "Accuracy vs Latency — TB V2.1",
        "tokens": "Accuracy vs Total Tokens — TB V2.1",
        "tokens_in": "Accuracy vs Input Tokens — TB V2.1",
        "tokens_out": "Accuracy vs Output Tokens — TB V2.1",
    }
    ax.set_title(
        _TITLE.get(x, f"Accuracy vs {x} — TB V2.1"),
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    style_axis(ax)

    # Legend with a visible black frame (guide §11).
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=LOCAL_GREEN,
            markeredgecolor="white",
            markersize=9,
            label="Local / ours",
        ),
        Line2D(
            [],
            [],
            marker="^",
            color="w",
            markerfacecolor=CLOUD_BLUE,
            markeredgecolor="white",
            markersize=9,
            label="Cloud baseline",
        ),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, edgecolor="black")
    leg.get_frame().set_linewidth(1.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Export vector PDF (paper) + raster PNG (preview), tight bbox (guide §13).
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    return out_path


# --- Accuracy-vs-efficiency scatter suite (color=method, size=params, shape=harness) ---------

GREY_MARK = "#9E9E9E"  # neutral grey for the size-legend reference bubbles
_RANGE_GREY = "#999999"  # connector for a dollars range (eager total <-> fast-serving floor)
_Z = {"local": 5, "cloud": 6, "strategy": 8}  # strategies draw on top
# One honest caption baked into every figure: local & cascade numbers are eager lower bounds,
# cascade cost spans an eager total -> fast-serving floor, and the sweep used a single harness.
CAVEAT = (
    "Local & cascade accuracies are eager-serving lower bounds (timeout-capped).  "
    "On the cost axis, the open marker / bar is a $ range: cascade = eager-total -> "
    "cloud-side floor; flash = cache-adjusted -> Harbor-billed.  "
    "Single harness: terminus-2.  TerminalBench V2.1, 89 tasks, k=1."
)


def _legend_handles():
    """Proxy artists for the two legends: method (color+shape) and model size (marker size)."""
    method = [
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=LOCAL_GREEN,
            markeredgecolor="white",
            markersize=11,
            linestyle="none",
            label="Local / ours",
        ),
        Line2D(
            [],
            [],
            marker="^",
            color="w",
            markerfacecolor=CLOUD_BLUE,
            markeredgecolor="white",
            markersize=11,
            linestyle="none",
            label="Cloud baseline",
        ),
        Line2D(
            [],
            [],
            marker="*",
            color="w",
            markerfacecolor=STRATEGY_GREEN,
            markeredgecolor="white",
            markersize=17,
            linestyle="none",
            label="Routing / cascade",
        ),
    ]
    size = [
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=GREY_MARK,
            markeredgecolor="white",
            markersize=_size_from_params(b * 1e9),
            linestyle="none",
            label=f"{b}B params",
        )
        for b in _SIZE_LEGEND_B
    ]
    return method, size


def _draw_panel(
    ax,
    cells: list[dict],
    metric: str,
    *,
    annotate: bool = True,
    allow_incomplete_energy: bool = True,
    allow_incomplete_cost: bool = True,
) -> None:
    """Plot accuracy vs `metric` on `ax`: encoded markers + dashed-green Pareto frontier.

    On the dollars axis a cell may carry `dollars_alt` — a second cost on the same accuracy
    (cascade: cloud-side floor; flash: alternate billing basis). It is drawn as a grey range
    bar to an open marker, and the Pareto frontier still uses the primary `dollars`.
    """
    spec = METRIC_SPEC[metric]
    scale = spec["scale"]
    _validate_strict_comparison_basis(
        cells,
        metric,
        allow_incomplete_energy=allow_incomplete_energy,
        allow_incomplete_cost=allow_incomplete_cost,
    )
    xs, ys = [], []
    for c in cells:
        raw = _metric_value(
            c,
            metric,
            allow_incomplete_energy=allow_incomplete_energy,
            allow_incomplete_cost=allow_incomplete_cost,
        )
        if raw is None:  # cloud has no measured joules/flops -> simply absent from that panel
            continue
        xv = raw / scale
        color, marker, msize = _point_style(c)
        kind = _cell_kind(c)
        z = _Z[kind]
        alt = c.get("dollars_alt") if metric == "dollars" and allow_incomplete_cost else None
        if alt:  # grey range bar from the primary cost to the alternate, open marker at the alt
            xa = alt / scale
            ax.plot(
                [xv, xa],
                [c["accuracy"], c["accuracy"]],
                "-",
                color=_RANGE_GREY,
                lw=1.3,
                alpha=0.8,
                zorder=z - 1,
                solid_capstyle="round",
            )
            ax.plot(
                xa,
                c["accuracy"],
                marker=marker,
                markersize=msize * 0.85,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                linestyle="none",
                zorder=z,
            )
        ax.plot(
            xv,
            c["accuracy"],
            marker=marker,
            color=color,
            markersize=msize,
            markeredgecolor="white",
            markeredgewidth=0.8 if kind == "local" else 1.0,
            linestyle="none",
            zorder=z,
        )
        if annotate:
            ax.annotate(
                _short_model(c.get("model", "")),
                (xv, c["accuracy"]),
                fontsize=8,
                color=TEXT_DARK,
                xytext=(4, 4),
                textcoords="offset points",
                zorder=9,
            )
        xs.append(xv)
        ys.append(c["accuracy"])

    if len(xs) >= 2:  # Pareto frontier: minimize x, maximize accuracy (guide §6)
        px, py = compute_pareto_frontier(xs, ys)
        if len(px) >= 2:
            ax.plot(px, py, "--", color=PARETO, alpha=0.6, linewidth=2, zorder=3)

    if spec["log"]:
        ax.set_xscale("log")
    ax.set_xlabel(spec["label"], fontsize=15, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v * 100))}%"))
    style_axis(ax)


def plot_efficiency_scatter(
    cells: list[dict],
    out_path,
    metric: str = "dollars",
    *,
    size_legend_loc: str = "upper left",
    allow_incomplete_energy: bool = True,
    allow_incomplete_cost: bool = True,
) -> Path:
    """Single accuracy-vs-`metric` scatter with the full encoding + method/size legends.

    Color = local(green)/cloud(blue); marker size = total model params; shape = harness.
    `size_legend_loc` places the model-size legend (move it off a crowded upper-left corner
    when the headline point sits there). Writes a vector PDF + raster PNG sibling of `out_path`.
    Returns `out_path`.
    """
    if metric not in METRIC_SPEC:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRIC_SPEC)}")
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.set_box_aspect(1)

    _draw_panel(
        ax,
        cells,
        metric,
        allow_incomplete_energy=allow_incomplete_energy,
        allow_incomplete_cost=allow_incomplete_cost,
    )
    ax.set_ylabel("Accuracy (TB V2.1)", fontsize=16, fontweight="bold")
    ax.set_ylim(-0.03, 0.86)
    ax.set_title(f"Accuracy vs {METRIC_SPEC[metric]['label']}", fontsize=16, fontweight="bold", pad=12)

    method, size = _legend_handles()
    leg1 = ax.legend(handles=method, loc="lower right", frameon=True, framealpha=0.95, edgecolor="black", fontsize=11)
    leg1.get_frame().set_linewidth(1.5)
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=size,
        loc=size_legend_loc,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=10,
        title="Model size",
        labelspacing=0.7,
    )
    leg2.get_frame().set_linewidth(1.5)
    fig.text(0.5, -0.01, CAVEAT, ha="center", va="top", fontsize=8.5, color="#555555")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    return out_path


def plot_tradeoff_grid(
    cells: list[dict],
    out_path,
    metrics: list[str] | None = None,
    *,
    allow_incomplete_energy: bool = True,
    allow_incomplete_cost: bool = True,
) -> Path:
    """Multi-panel grid: accuracy vs every efficiency axis, shared accuracy y-axis.

    One square panel per metric (4 per row); the trailing cell hosts the method + size
    legends and the caveat. y-tick labels show only on the leftmost column (guide §10).
    Writes a vector PDF + raster PNG sibling of `out_path`. Returns `out_path`.
    """
    metrics = metrics or DEFAULT_METRICS
    bad = [m for m in metrics if m not in METRIC_SPEC]
    if bad:
        raise ValueError(f"unknown metrics {bad}; choose from {sorted(METRIC_SPEC)}")
    plt.rcParams.update({"font.size": 14})

    ncols = 4
    nrows = -(-(len(metrics) + 1) // ncols)  # +1 reserves a cell for the legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 6.5 * nrows))
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for i, metric in enumerate(metrics):
        ax = flat[i]
        ax.set_box_aspect(1)
        _draw_panel(
            ax,
            cells,
            metric,
            allow_incomplete_energy=allow_incomplete_energy,
            allow_incomplete_cost=allow_incomplete_cost,
        )
        ax.set_ylim(-0.03, 0.86)
        if i % ncols == 0:  # leftmost column: keep accuracy label + ticks; others hide them
            ax.set_ylabel("Accuracy (TB V2.1)", fontsize=15, fontweight="bold")
        else:
            ax.set_yticklabels([])

    # Legend / notes panel in the first unused cell; any remaining cells are blanked.
    method, size = _legend_handles()
    legend_ax = flat[len(metrics)]
    legend_ax.axis("off")
    leg1 = legend_ax.legend(
        handles=method,
        loc="upper center",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=13,
        title="Method (color · shape)",
        bbox_to_anchor=(0.5, 0.98),
    )
    leg1.get_frame().set_linewidth(1.5)
    legend_ax.add_artist(leg1)
    leg2 = legend_ax.legend(
        handles=size,
        loc="center",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=12,
        title="Model size (marker area)",
        labelspacing=0.9,
        bbox_to_anchor=(0.5, 0.5),
    )
    leg2.get_frame().set_linewidth(1.5)
    legend_ax.text(
        0.5,
        0.04,
        "Dashed green = Pareto frontier\n(cheaper-and-better dominates)",
        ha="center",
        va="bottom",
        fontsize=11,
        color=TEXT_DARK,
        transform=legend_ax.transAxes,
    )
    for ax in flat[len(metrics) + 1 :]:
        ax.axis("off")

    fig.suptitle("Accuracy vs efficiency — TerminalBench V2.1 (P0 sweep)", fontsize=22, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, CAVEAT, ha="center", va="bottom", fontsize=11, color="#555555")
    fig.subplots_adjust(wspace=0.12, hspace=0.28)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    return out_path
