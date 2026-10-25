"""Shared visual language for IPJ blog scatter figures.

Palette and Pareto treatment are taken from the RECOVERED ICLR generator, now committed at
`ipj/figures/iclr-panels/make_fig1_panelA_only.py` (see the README there for provenance).
This supersedes the earlier hand-matched approximation — the values below are the real ones.

Class-based, not family-based: model tiers deliberately match the Panel B fleet cards, so the
same colour means the same tier across both panels of the paper figure.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import style  # noqa: F401
from style import GRID, TEXT_DARK

# Font sizes copied verbatim from make_fig1_panelA_only.py
plt.rcParams.update({
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "grid.linewidth": 0.65,
})

# ---- canonical palette (ipj/figures/iclr-panels/make_fig1_panelA_only.py) ----
OPEN_CLOUD = "#0891B2"    # cyan   — Open-Weights Cloud
CLOSED_CLOUD = "#7C3AED"  # violet — Closed-Weights Cloud
SAKANA = "#E10600"        # red    — LM Orchestrator baseline
CASCADE = "#16A34A"       # green  — our cascade points
ORACLE = "#F5B301"        # gold   — Oracle Routing ceiling

# aliases kept so existing call sites do not break
OPEN, CLOSED, ORCH = OPEN_CLOUD, CLOSED_CLOUD, SAKANA

SZ = {"frontier": 340, "large": 180, "mid": 115, "small": 75}
INK, INK2, MUTED = "#111318", "#3C4149", "#B5B8BE"

TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 18, 18, 15
MARKER_SIZE = SZ["frontier"]   # 340, as the paper uses for solos
LABEL_SIZE_PT = 13             # annotate fontsize in the paper
EDGE_COLOR, EDGE_LW = "black", 1.0

# Pareto region fill, matching the recovered generator
PARETO_ALPHA = 0.08
PARETO_LINE_LW = 2.2
PARETO_LINE_ALPHA = 0.85


def pareto(points):
    """Up-and-left frontier for (minimise x, maximise y). Returns points sorted by x."""
    out, best = [], float("-inf")
    for x, y in sorted(points, key=lambda p: p[0]):
        if y > best:
            out.append((x, y))
            best = y
    return out


def draw_pareto(ax, points, *, shade: bool = True) -> list:
    """Dotted frontier line plus the shaded 'at least this good' staircase region.

    Reproduces the recovered generator: ax.fill(..., alpha=0.08) under a dotted line at
    lw=2.2, alpha=0.75, both in the cascade green.
    """
    front = pareto(points)
    if len(front) < 2:
        return front

    xs, ys = zip(*front)
    if shade:
        # Verbatim construction from make_fig1_panelA_only.py: the fill follows the
        # frontier line itself (straight segments between points), then closes to the
        # top corners. No staircase — the boundary IS the dotted line.
        x_lo, x_hi = ax.get_xlim()
        y_hi = ax.get_ylim()[1]
        poly_x = [x_lo, x_lo] + list(xs) + [x_hi, x_hi]
        poly_y = [y_hi, ys[0]] + list(ys) + [ys[-1], y_hi]
        ax.fill(poly_x, poly_y, color=CASCADE, alpha=PARETO_ALPHA, zorder=1, linewidth=0)

    ax.plot(xs, ys, ls=":", color=CASCADE, lw=PARETO_LINE_LW,
            alpha=PARETO_LINE_ALPHA, zorder=3)
    return front


def bracket(ax, p_lo, p_hi, text, *, clear=0.055, tick=0.030, fontsize=14.4,
            dx_pts=0.0, connect=False, gap=0.022):
    """Symmetric improvement bracket over two data points, ICLR Panel-A idiom.

    The bar sits a fixed distance above the HIGHER of the two markers and both end ticks drop
    the SAME length. Equal legs are what make it read as balanced — running each leg down to
    its own marker looks lopsided whenever the two points differ in height.

    The label wraps to two lines when a single line would be wider than the bracket itself,
    which happens whenever the two points are close together on a log axis. A caption three
    times the width of the thing it labels does not read as belonging to it.

    clear = bar height above the higher marker, as a fraction of the y-range
    tick  = end-tick length, same units
    """
    (x0, y0), (x1, y1) = p_lo, p_hi
    lo, hi = ax.get_ylim()
    span = hi - lo
    y_bar = max(y0, y1) + clear * span
    y_tick = y_bar - tick * span
    if connect:
        # Legs run down to their OWN markers. Unequal, but unambiguous — use this when other
        # points sit near a leg and an equal-length tick would read as pointing at one of them.
        ax.plot([x0, x0], [y0 + gap * span, y_bar], color=INK, lw=1.8, zorder=8)
        ax.plot([x1, x1], [y1 + gap * span, y_bar], color=INK, lw=1.8, zorder=8)
        ax.plot([x0, x1], [y_bar, y_bar], color=INK, lw=1.8, zorder=8, solid_capstyle="butt")
    else:
        ax.plot([x0, x0, x1, x1], [y_tick, y_bar, y_bar, y_tick],
                color=INK, lw=1.8, zorder=8, solid_capstyle="round")

    xm = (x0 * x1) ** 0.5 if ax.get_xscale() == "log" else (x0 + x1) / 2
    fig = ax.figure
    fig.canvas.draw()
    px0, px1 = (ax.transData.transform((x, y_bar))[0] for x in (x0, x1))
    bracket_px = abs(px1 - px0)
    probe = ax.text(0, 0, text, fontsize=fontsize, fontweight="bold", alpha=0)
    text_px = probe.get_window_extent(fig.canvas.get_renderer()).width
    probe.remove()
    if text_px > bracket_px and ", " in text:
        text = text.replace(", ", ",\n", 1)

    # dx_pts nudges the label off a fixed obstacle (the oracle star) without moving the bracket
    ax.annotate(text, (xm, y_bar), textcoords="offset points", xytext=(dx_pts, 7), ha="center",
                va="bottom", fontsize=fontsize, fontweight="bold", color=INK, zorder=9,
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=INK, lw=1.0, alpha=0.95))
    return y_bar


def callout(ax, text, xy, xytext, fontsize=12.5):
    return ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=fontsize, fontweight="bold", color=INK,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=INK, lw=1.6),
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.6),
    )


def dress(ax, *, xlabel, title, xlim, xticks, xticklabels,
          ylim=(24, 88), yticks=range(30, 86, 10), ylabel=None) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"{v}%" for v in yticks])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, which="major", color=GRID, alpha=0.25, linewidth=0.65)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=14, loc="center")


def plot_points(ax, rows, xkey) -> None:
    for r in rows:
        if r.get(xkey) is None:
            continue
        mk = r["marker"]
        size = r.get("size") or (340 if mk == "*" else (190 if mk == "X" else MARKER_SIZE))
        ax.scatter(r[xkey], r["acc"], s=size, marker=mk, color=r["colour"],
                   edgecolors=EDGE_COLOR, linewidths=EDGE_LW, zorder=6)
