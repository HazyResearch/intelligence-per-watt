"""Figure 3 -- the four orchestration policy families, drawn as execution topologies.

TARGET-PAPER COPY (ICP_TARGET_PAPER, 2026-09-25) of
intelligence-per-joule/ipj/figures/blog-post-ipd/fig3_policy_families.py, which is left untouched.
Only text sizes change (author note: the text inside the diagram printed too small), and the paper now
includes the figure at the full text width like Figures 1, 2 and 4. Printed sizes at \textwidth
(scale 396/861): box labels 5.8 pt, titles 6.4 pt, annotations 4.8 to 5.1 pt, the reads strip 5.5 pt.

Sits at the top of the Methods section. The paper defines a configuration
c=(m,w,h) and a system (C,pi), then distinguishes the families by what pi is
allowed to READ and WHEN: fixed selection reads nothing, routing reads the
prompt t before anything runs, a cascade reads the execution trace tau_1 of a
completed cheap attempt, and plan/execute always runs both stages with a data
dependency between them.

That distinction is the load-bearing one in Section 5.2 ("routing and cascades
differ only in what pi observes"), and it is hard to hold in prose, so the
bottom strip of each panel states it explicitly.

Deliberately NOT a redraw of the models x middleware x machines stack -- that is
already Figure 1.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
})

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures_and_tables")

INK = "#1A1A1A"          # configurations: the things that actually run
DECIDE = "#8C1515"       # the policy element -- scorer or gate
QUIET = "#EFEDE7"        # task in / answer out
EDGE = "#3A3A3A"
GREY = "#6B6B6B"

BOX_W, BOX_H, R = 1.62, 0.50, 0.09


def box(ax, x, y, label, kind="config", w=BOX_W, fs=12.6):
    """One node. `kind` picks the visual register: config, decide, or quiet."""
    face, txt, weight = {
        "config": (INK, "white", "bold"),
        "decide": (DECIDE, "white", "bold"),
        "quiet": (QUIET, INK, "normal"),
    }[kind]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - BOX_H / 2), w, BOX_H,
        boxstyle=f"round,pad=0.012,rounding_size={R}",
        facecolor=face, edgecolor=EDGE, linewidth=1.15, zorder=3))
    ax.text(x, y, label, ha="center", va="center", color=txt,
            fontsize=fs, fontweight=weight, zorder=4)


TITLES = []          # (ax, Text) pairs, underlined after the first draw
TITLE_FS = 14.0
TITLE_Y = 0.978      # axes fraction; the task box tops out at ~0.930


def panel_title(ax, txt):
    """Bold panel title placed just above the diagram. Underlined post-draw."""
    t = ax.text(0.5, TITLE_Y, txt, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=TITLE_FS, fontweight="bold", color=INK)
    TITLES.append((ax, t))


def arrow(ax, x0, y0, x1, y1, label=None, style="-", lx=0.10, fs=10.6,
          ha="left", dy=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.15, color=EDGE, linestyle=style,
        shrinkA=1.0, shrinkB=1.0, zorder=2))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + dy, label, ha=ha, va="center",
                fontsize=fs, color=GREY, style="italic", zorder=4)


fig, axes = plt.subplots(1, 4, figsize=(12.1, 4.05))
TOP, BOT = 3.62, 0.86          # y of the task node and the answer node

for ax in axes:
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(0.02, 4.16)
    ax.axis("off")

# ---------------------------------------------------------------- 1. fixed
ax = axes[0]
panel_title(ax, "Fixed selection")
box(ax, 0, TOP, "task $t$", "quiet")
box(ax, 0, 2.12, "$c_1$", "config")
box(ax, 0, BOT, "answer", "quiet")
arrow(ax, 0, TOP - BOX_H / 2, 0, 2.12 + BOX_H / 2)
arrow(ax, 0, 2.12 - BOX_H / 2, 0, BOT + BOX_H / 2)
ax.text(0.13, 1.49, "same $c$ for\nevery task", ha="left", va="center",
        fontsize=10.6, color=GREY, style="italic")

# ---------------------------------------------------------------- 2. routing
ax = axes[1]
panel_title(ax, "Query routing")
box(ax, 0, TOP, "task $t$", "quiet")
box(ax, 0, 2.62, "scorer $r(t, c_i)$", "decide")
box(ax, 0, 1.62, "$c_j$", "config")
box(ax, 0, BOT, "answer", "quiet")
arrow(ax, 0, TOP - BOX_H / 2, 0, 2.62 + BOX_H / 2)
arrow(ax, 0, 2.62 - BOX_H / 2, 0, 1.62 + BOX_H / 2, r"$\arg\max_i$")
arrow(ax, 0, 1.62 - BOX_H / 2, 0, BOT + BOX_H / 2)
ax.text(0.13, 3.12, "before any\nconfiguration runs", ha="left", va="center",
        fontsize=10.6, color=DECIDE, style="italic")

# ---------------------------------------------------------------- 3. cascade
ax = axes[2]
panel_title(ax, "Local-first escalation")
box(ax, -0.42, TOP, "task $t$", "quiet")
box(ax, -0.42, 2.72, "$c_1$", "config", w=1.30)
box(ax, -0.42, 1.82, "signal $g(\\tau_1)$", "decide", w=1.30)
box(ax, 0.86, 1.82, "$c_2$", "config", w=0.86)
box(ax, -0.42, BOT, "answer", "quiet")
arrow(ax, -0.42, TOP - BOX_H / 2, -0.42, 2.72 + BOX_H / 2)
arrow(ax, -0.42, 2.72 - BOX_H / 2, -0.42, 1.82 + BOX_H / 2, r"trace $\tau_1$",
      lx=-0.10, ha="right")
arrow(ax, -0.42, 1.82 - BOX_H / 2, -0.42, BOT + BOX_H / 2, "pass",
      lx=-0.10, ha="right")
arrow(ax, -0.42 + 1.30 / 2, 1.82, 0.86 - 0.86 / 2, 1.82)
# above the boxes' top edge (2.07): the arrow itself is only 0.20 wide, so the
# label cannot sit on it without running under c_2
ax.text(0.33, 2.13, "escalate", ha="center", va="bottom", fontsize=10.6,
        color=GREY, style="italic")
# headless: the elbow is one path, so only the segment entering "answer" gets a head
ax.add_patch(FancyArrowPatch((0.86, 1.82 - BOX_H / 2), (0.86, BOT),
                            arrowstyle="-", linewidth=1.15, color=EDGE,
                            linestyle=(0, (3, 2)), shrinkA=1.0, shrinkB=0.0, zorder=2))
arrow(ax, 0.86, BOT, -0.42 + 1.62 / 2, BOT)
ax.text(0.78, 1.32, "fresh $t$,\nnot $\\tau_1$", ha="right", va="center",
        fontsize=10.4, color=GREY, style="italic")

# ---------------------------------------------------- 4. cloud plan / local execute
ax = axes[3]
panel_title(ax, "Cloud plan, local execute")
box(ax, 0, TOP, "task $t$", "quiet")
box(ax, 0, 2.62, "$c_1$  (plan)", "config")
box(ax, 0, 1.62, "$c_2\\,(t, p)$", "config")
box(ax, 0, BOT, "answer", "quiet")
arrow(ax, 0, TOP - BOX_H / 2, 0, 2.62 + BOX_H / 2)
arrow(ax, 0, 2.62 - BOX_H / 2, 0, 1.62 + BOX_H / 2, "plan $p$")
arrow(ax, 0, 1.62 - BOX_H / 2, 0, BOT + BOX_H / 2)
ax.add_patch(FancyArrowPatch(
    (-BOX_W / 2, TOP - 0.10), (-1.16, TOP - 0.10), arrowstyle="-",
    linewidth=1.0, color=GREY, linestyle=(0, (2, 2)), zorder=1))
ax.add_patch(FancyArrowPatch(
    (-1.16, TOP - 0.10), (-1.16, 1.62), arrowstyle="-",
    linewidth=1.0, color=GREY, linestyle=(0, (2, 2)), zorder=1))
ax.add_patch(FancyArrowPatch(
    (-1.16, 1.62), (-BOX_W / 2, 1.62), arrowstyle="-|>", mutation_scale=10,
    linewidth=1.0, color=GREY, linestyle=(0, (2, 2)), zorder=1))
ax.text(0.13, 3.12, "both stages\nalways run", ha="left", va="center",
        fontsize=10.6, color=GREY, style="italic")

# --------------------------------------------- the punchline strip: what pi reads
READS = [
    (r"$\pi$ reads: nothing", GREY),
    (r"$\pi$ reads: $t$", DECIDE),
    (r"$\pi$ reads: $\tau_1$", DECIDE),
    (r"$\pi$ reads: nothing", GREY),
]
for ax, (txt, col) in zip(axes, READS):
    ax.add_patch(FancyBboxPatch(
        (-1.16, 0.06), 2.32, 0.34,
        boxstyle="round,pad=0.010,rounding_size=0.07",
        facecolor="#F6F5F2", edgecolor="#C9C6BF", linewidth=0.9, zorder=1))
    ax.text(0, 0.23, txt, ha="center", va="center", fontsize=12.0,
            color=col, fontweight="bold", zorder=4)

fig.subplots_adjust(left=0.010, right=0.990, top=0.965, bottom=0.028, wspace=0.018)

# Underline each title. matplotlib text has no underline attribute, so measure
# the rendered extent and draw a rule at its own width -- this stays correct if
# the titles or the font size change.
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
for _ax, _t in TITLES:
    _bb = _t.get_window_extent(_rend).transformed(_ax.transAxes.inverted())
    # Hang the rule off the text's MEASURED bottom rather than TITLE_Y, so the
    # gap stays constant even though "Query routing" and "Cloud plan, local
    # execute" carry descenders (y, g, p, comma) that "Cascade" does not.
    _ax.plot([_bb.x0, _bb.x1], [_bb.y0 - 0.008] * 2, transform=_ax.transAxes,
             color=INK, linewidth=1.5, solid_capstyle="butt", clip_on=False,
             zorder=5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT if ext == "pdf" else os.environ.get("FIG3_PNG_DIR", OUT),
                             f"fig3_policy_families_target.{ext}"),
                bbox_inches="tight", pad_inches=0.05,
                **({"dpi": 300} if ext == "png" else {}))
print("  wrote fig3_policy_families_target.pdf")
plt.close(fig)
