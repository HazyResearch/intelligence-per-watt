"""OpenJarvis-style helpers for the blog-post figure suite.

Header sets font-embedding for editable PDFs, the palette follows the styling
guide's green-for-local / blue-for-cloud convention, and `style_axis` is the
one call every figure makes before plotting.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
})
plt.rcParams.update({"font.size": 15})


# Local / our-method greens
LOCAL_GREENS = {
    "deep_anchor": "#1B5E20",  # headline result
    "teal_deep":   "#004D40",
    "teal":        "#00897B",
    "green":       "#388E3C",
    "green_mid":   "#43A047",
    "green_lt":    "#7CB342",
    "olive_585":   "#558B2F",
    "green_2E":    "#2E7D32",
}

# Cloud / frontier blues
CLOUD_BLUES = {
    "opus_deep": "#0D47A1",
    "gpt_mid":   "#1976D2",
    "gemini_lt": "#64B5F6",
}

# Sequential ramp (small -> large)
RAMP = ["#C5E1A5", "#7CB342", "#43A047", "#1B5E20"]

# Neutrals
GRID = "#AAAAAA"
ERRORBAR = "#555555"
BRACKET = "#222222"
GREY_BAR = "#BDBDBD"
GREY_EDGE = "#757575"
TEXT_DARK = "#1A1A1A"
PARETO = "#4CAF50"
REF_BLUE = "#1976D2"

# Roles used across the suite
COLOR_BY_ROLE = {
    "fable": "#0D47A1",
    "cloud": "#1976D2",
    "open": "#43A047",
    "open_american": "#00897B",   # teal to flag American OSS
    "cascade": "#1B5E20",
    "cascade_american": "#004D40",  # teal-deep for all-American cascade
    "front": "#7CB342",
    "front_american": "#00897B",
    "mid": "#43A047",
}

MARKER_BY_ROLE = {
    "fable": "D",
    "cloud": "^",
    "open": "o",
    "open_american": "o",
    "cascade": "*",
    "cascade_american": "*",
}


def style_axis(ax, grid: bool = True, grid_axis: str = "both"):
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5,
                color=GRID, zorder=1, axis=grid_axis)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)


def compute_pareto_frontier(x_vals, y_vals):
    """Minimize x, maximize y."""
    points = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
    px, py, max_y = [], [], -np.inf
    for x, y in points:
        if y >= max_y:
            px.append(x); py.append(y); max_y = y
    return np.array(px), np.array(py)


def draw_bracket_horizontal(ax, x_left, x_right, y, label,
                            tick_h=0.012, label_offset=-0.045,
                            color=BRACKET, fontsize=11, with_arrows=True,
                            log_x: bool = True):
    """Signature `<->` improvement callout, horizontal, log-x aware label pos."""
    arrow = "<->" if with_arrows else "-"
    ax.annotate("", xy=(x_right, y), xytext=(x_left, y),
                arrowprops=dict(arrowstyle=arrow, color=color,
                                lw=1.8, mutation_scale=14),
                zorder=11)
    for x in (x_left, x_right):
        ax.plot([x, x], [y - tick_h, y + tick_h], color=color,
                lw=1.5, zorder=11, solid_capstyle="butt")
    x_mid = np.sqrt(x_left * x_right) if log_x else 0.5 * (x_left + x_right)
    ax.text(x_mid, y + label_offset, label, ha="center", va="top",
            fontsize=fontsize, fontweight="bold", color=color, zorder=12)


def draw_bracket_vertical(ax, x, y_bottom, y_top, label,
                          tick_half_width=0.15, label_offset=None,
                          color=BRACKET, fontsize=11, with_arrows=True):
    arrow = "<->" if with_arrows else "-"
    ax.annotate("", xy=(x, y_top), xytext=(x, y_bottom),
                arrowprops=dict(arrowstyle=arrow, color=color,
                                lw=1.8, mutation_scale=12),
                zorder=11)
    for y in (y_top, y_bottom):
        ax.plot([x - tick_half_width, x + tick_half_width], [y, y],
                color=color, lw=1.5, zorder=11, solid_capstyle="butt")
    if label_offset is None:
        label_offset = 0.05 * (y_top - y_bottom)
    ax.text(x, y_top + label_offset, label, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=color, zorder=12)


def save(fig, out_stem: str):
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{out_stem}.png", bbox_inches="tight", dpi=180)
    print(f"Saved: {out_stem}.pdf, {out_stem}.png")
