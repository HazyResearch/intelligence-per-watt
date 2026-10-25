#!/usr/bin/env python3
"""Figure 1 (target paper): the ICP overview, restyled exactly like the original.

The original figure (intelligence-per-joule/ipj/writing/figures_and_tables/fig1_icp_overview.pdf,
504 x 168 pt, Matplotlib 3.11.2) has no generator script. Its geometry, colors, font sizes and
weights were read out of its PDF content stream and are reproduced here in absolute page points.

  * Left panel ("Intelligence System"): contains no data, so it is NOT redrawn. It is clipped out
    of the original PDF as vector and stamped onto the new page (see compose_with_left_panel).
  * Middle panel ("Intelligence Systems Beat Individual Frontier Models") and right panel
    ("Local by Default, Hybrid by Design"): rebuilt from targets/figure_inputs.json["fig1"].

Output: paper/figures_and_tables/fig1_icp_overview_target.pdf (504 x 168 pt).

Round 15 (author, 2026-09-25): the headline's first stage is the laptop (MacBook Pro M5 Max, selected by
the local-by-default rule), so the right panel's bar splits each composition's stage executions three
ways instead of owned / rented: local (the laptop), owned accelerators (the GLM-5.3 stage on 8xH200) and
the closed-source provider. The local share is printed on the bar; the legend names the three segments.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import ArrowStyle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath, text_to_path
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "targets" / "figure_inputs.json"
OUT = ROOT / "paper" / "figures_and_tables" / "fig1_icp_overview_target.pdf"
ORIGINAL = ROOT / "figures" / "fig1_reference_2026-09-24.pdf"   # the author's updated Figure 1 (2026-09-24)

# ---------------------------------------------------------------- style of the original (pt)
PAGE_W, ORIG_H = 504.0, 168.0
# Round 11 (author note, 2026-09-25): the figure is taller. The left panel (the author's vector drawing)
# is scaled uniformly by S to the new height; everything anchored to the top moves up by DY; the middle
# panel shifts right by DX to make room, and its plot and the right panel's rows use the extra height.
PAGE_H = 194.0
S = PAGE_H / ORIG_H
DY = PAGE_H - ORIG_H
INK = "#111318"          # titles, tick labels, table text
BLACK = "#000000"        # axis labels and the annotation
BLUE = "#0072b2"         # composed Intelligence Systems; right panel: local (laptop) segment
PURPLE = "#7c3aed"       # closed, cloud
GREEN = "#16a34a"        # open, local
ORANGE = "#d97706"       # open, cloud
GREY_BAR = "#b5b8be"     # right panel: closed-source provider segment; header rule
ACCEL = "#59a3cd"        # right panel: owned-accelerator segment, BLUE mixed 35% toward white (round 15)
LEGEND_BG = "#f1f3f5"
HILITE_BG = "#f2f7fa"
SEP = "#909398"
LEG_TEXT = "#505660"
HEAD_TEXT = "#61676e"
RENT_TEXT = "#42474d"
ARROW = "#9299a2"
GRID = "#e5e7eb"
KIND_COLOR = {"closed": PURPLE, "open_local": GREEN, "open_cloud": ORANGE}

# Round 5 (author notes, 2026-09-24): the right panel is 30 pt narrower (two-line title and column
# headers), the middle panel takes that width, its legend drops to five rows, and its axes grow down.
SEP1 = 136.8 * S                            # left | middle separator (scaled with the left panel)
DX = SEP1 - 136.8
RIGHT_C0, RIGHT_C1 = 362.2, 499.2           # right panel content, left and right edges (137 pt wide)
SEP2 = RIGHT_C0 - 7.2                       # middle | right separator (was 321.6)
LEG_TOP = 36.5                              # top of the middle legend box (was 47.04)
XLABEL_Y = LEG_TOP + 4.5                    # baseline of the x-axis title (was 51.53)
AX_BOX = (170.4 + DX, XLABEL_Y + 17.59, SEP2 - 8.64, 134.16 + DY)   # middle axes x0, y0, x1, y1
LEFT_CLIP_X = 135.0                         # left panel stamped from the original up to here
MARK_EXT, MARK_MEW = 12.0, 0.6              # plotted marker extent and white ring; author note 2026-09-25: +25% (was 9.6, 0.5)
LEG_MARK = 6.5                              # legend marker extent (was 6.0)
# Display offsets (pt) for markers that would otherwise overlap; the data are unchanged.
# author note 2026-09-25: GPT-5.6-sol and GPT-6 Astra tie in accuracy, so they are separated sideways only (a vertical
# offset made Astra look less accurate); Astra, the costlier, sits to the right
JITTER = {"GPT-5.6-sol": (-4.5, 0.0), "GPT-6 Astra": (4.5, 0.0),      # offsets scaled with the markers (+25%)
          "Qwen3.8-27B → GLM-5.3": (5.5, 2.0), "GLM-5.3": (-5.5, -2.0),   # wider apart than +25% alone: the two nearly coincide
          "Qwen3.8-Flash-Next → GLM-5.3 → GPT-6 Astra": (-1.25, 1.75)}
BUBBLE_EDGE = "#0072b2"
BUBBLE_RAISE = 0.30                         # share of the bubble-to-arrow gap removed (author note)


def _pt(fig):
    """Transform taking page points to display space."""
    return transforms.Affine2D().scale(1 / 72.0) + fig.dpi_scale_trans


def _ms(marker: str, extent: float) -> float:
    """Marker size giving the marker path a max extent of `extent` pt (the original's convention)."""
    ms = MarkerStyle(marker)
    ext = ms.get_path().transformed(ms.get_transform()).get_extents()
    return extent / max(ext.width, ext.height)


def _box(fig, x0, y0, x1, y1, color, r, zorder=0):
    fig.patches.append(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
        transform=_pt(fig), facecolor=color, edgecolor="none", linewidth=0, zorder=zorder))


def _ink(s, size, bold=False):
    """Vertical ink extent (y0, y1) of a one-line string, in pt relative to its baseline."""
    prop = FontProperties(family="DejaVu Sans", size=size, weight="bold" if bold else "normal")
    e = TextPath((0, 0), s, size=size, prop=prop).get_extents()
    return e.y0, e.y1


def _text(fig, x, y, s, size, color=INK, bold=False, ha="left", va="baseline", **kw):
    """Text in page points. va="center" centers the glyph ink on y, as the original's renderer
    (Matplotlib 3.11) did; Matplotlib 3.10 centers the font box instead, 0.3 to 0.6 pt off."""
    if va == "center":
        y0, y1 = _ink(s, size, bold)
        y, va = y - (y0 + y1) / 2.0, "baseline"
    return fig.text(x, y, s, transform=_pt(fig), fontsize=size, color=color, ha=ha, va=va,
                    fontweight="bold" if bold else "normal", **kw)


def _marker(fig, x, y, marker, color, extent, mew, zorder=3, transform=None):
    fig.add_artist(Line2D([x], [y], transform=transform or _pt(fig), linestyle="none",
                          marker=marker, markersize=_ms(marker, extent), markerfacecolor=color,
                          markeredgecolor="white", markeredgewidth=mew, zorder=zorder))


def _frontier(points):
    """Upper-left Pareto frontier (cheapest-first) of (usd, acc) points."""
    out, best = [], -math.inf
    for usd, acc in sorted(points, key=lambda p: (p[0], -p[1])):
        if acc > best:
            out.append((usd, acc))
            best = acc
    return out


def frontier_region(points, xlim, ylim):
    """Polygon shaded in the middle panel, built with the original's rule.

    The original shades the part of the axes above the upper-left cost-accuracy Pareto frontier of
    all plotted points: flat from the left axis edge at the cheapest point's accuracy, straight
    segments (linear in log dollars) through each frontier point, flat to the right axis edge, and
    closed up to the top of the axes.
    """
    f = _frontier(points)
    (xa, xb), (_, ytop) = xlim, ylim
    return [(xa, ytop), (xa, f[0][1])] + f + [(xb, f[-1][1]), (xb, ytop)]


# ---------------------------------------------------------------- middle panel
def draw_middle(fig, d):
    x0, y0, x1, y1 = AX_BOX
    ax = fig.add_axes([x0 / PAGE_W, y0 / PAGE_H, (x1 - x0) / PAGE_W, (y1 - y0) / PAGE_H])
    ax.set_xscale("log")
    ax.set_xlim(0.001, 10)
    ax.set_ylim(*d["ylim"])
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 1, 10]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.yaxis.set_major_locator(FixedLocator(d["yticks"]))
    ax.set_yticklabels([f"{t:g}" for t in d["yticks"]])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.288)
        ax.spines[s].set_color(INK)
    ax.tick_params(axis="both", which="major", length=1.2, width=0.288, color=INK,
                   labelcolor=INK, labelsize=6.96, pad=1.9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.24)
    ax.grid(axis="x", visible=False)

    region = d["region"]
    if region:
        ax.add_patch(Polygon(region, closed=True, facecolor=BLUE, edgecolor="none", alpha=0.05,
                             linewidth=0, zorder=0))

    comp = {p["label"]: p for p in d["composed"]}
    sing = {p["label"]: p for p in d["singles"]}
    ann = d["annotation"]
    head, tail = comp.get(ann["from"]) or sing[ann["from"]], sing.get(ann["to"]) or comp[ann["to"]]

    # arrow: from the reference single model (tail) to the highlighted composed system (head)
    # The drawn tip sits exactly 4.62 pt from the star's centre, as in the original, instead of
    # relying on Matplotlib's shrink bisection (which overshoots on a long sloped arrow). The "->"
    # style pulls its stroked tip back from the requested end point by a fixed amount; request
    # the end point that much closer so the drawn tip lands at 4.62 pt.
    style, mscale, lw = "->", 10.0, 2.0         # author note 2026-09-25: shaft twice as thick, head scaled to match (was 7.2, 1.0)
    segs, _ = ArrowStyle(style)(MplPath([(0.0, 0.0), (100.0, 0.0)]), mscale, lw)
    gap = 5.8 - (100.0 - segs[0].vertices[-1][0])         # tip 5.8 pt from the star's centre (was 4.62; the star is 25% larger)
    sx, sy = ax.transData.transform((head["usd"], head["acc"]))
    tx_, ty_ = ax.transData.transform((tail["usd"], tail["acc"]))
    dist = math.hypot(tx_ - sx, ty_ - sy)
    tip_pt = (sx + gap * (tx_ - sx) / dist, sy + gap * (ty_ - sy) / dist)
    tip = ax.transData.inverted().transform(tip_pt)
    ax.add_patch(FancyArrowPatch((tail["usd"], tail["acc"]), tuple(tip),
                                 arrowstyle=style, mutation_scale=mscale, color=BLACK, linewidth=lw,
                                 shrinkA=5.1, shrinkB=0, zorder=6))   # tail clear of the larger Fable marker (was 4.07)   # above every marker (author note, 2026-09-25)
    ext, mew = MARK_EXT, MARK_MEW
    # draw order of the original: diamond, Fable (o), GPT-5.6-sol (P), GPT-6 Astra (X), triangle,
    # square, GLM (v), Qwen (h); the star is painted last, above the annotation
    sg = d["singles"]
    order = d["composed"][1:2] + [sg[0], sg[2], sg[1]] + d["composed"][2:] + sg[3:]
    for p in order:
        color = BLUE if p["label"] in comp else KIND_COLOR[p["kind"]]
        dx, dy = JITTER.get(p["label"], (0.0, 0.0))
        ax.plot([p["usd"]], [p["acc"]], linestyle="none", color=color,
                marker=MarkerStyle(p["marker"], joinstyle="round"),
                markersize=_ms(p["marker"], ext), markeredgecolor="white", markeredgewidth=mew,
                zorder=3, clip_on=False,
                transform=transforms.offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points"))
    # the takeaway, in a rounded bubble beneath the arrow, with a leader up to it
    bx, by = d["bubble_xy"]
    lines = ann["text"].split(", ")
    lines = [lines[0] + ",", lines[1]] if len(lines) == 2 else lines
    lh = 7.4
    widths = [text_to_path.get_text_width_height_descent(
        l, FontProperties(family="DejaVu Sans", size=6.48, weight="bold"), ismath=False)[0] for l in lines]
    bw, bh = max(widths) + 7.0, lh * len(lines) + 3.6
    # where the leader meets the arrow's shaft, straight above the bubble's centre
    ax_x = (bx - x0) / (x1 - x0)
    lx = 10 ** (math.log10(0.001) + ax_x * 4)                  # x in data at the bubble's centre
    fx, fy, hx, hy = math.log10(tail["usd"]), tail["acc"], math.log10(head["usd"]), head["acc"]
    ay = fy + (math.log10(lx) - fx) * (hy - fy) / (hx - fx)    # arrow's accuracy at that x
    ay_pt = ax.transData.transform((lx, ay))[1] * 72.0 / fig.dpi
    # author note (2026-09-25): bring the bubble 30% closer to the arrow than the base position
    by += BUBBLE_RAISE * ((ay_pt - 1.2) - (by + bh / 2))
    for k, line in enumerate(lines):
        _text(fig, bx, by + lh * ((len(lines) - 1) / 2 - k), line, 6.48, color=BLACK, bold=True,
              ha="center", va="center", zorder=4.2)
    fig.patches.append(FancyBboxPatch(
        (bx - bw / 2, by - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=3.2",
        transform=_pt(fig), facecolor="white", edgecolor=BUBBLE_EDGE, linewidth=0.6, zorder=4.1))
    fig.add_artist(Line2D([bx, bx], [by + bh / 2, ay_pt - 1.2], transform=_pt(fig), color=BUBBLE_EDGE,
                          linewidth=0.6, solid_capstyle="round", zorder=4.0))
    star = d["composed"][0]
    ax.plot([star["usd"]], [star["acc"]], linestyle="none", color=BLUE,
            marker=MarkerStyle(star["marker"], joinstyle="round"),
            markersize=_ms(star["marker"], ext + 3.0),       # +25% with the other markers (was ext + 2.4 at ext 9.6) markeredgecolor="white", markeredgewidth=mew,
            zorder=5, clip_on=True)

    # axis labels
    _text(fig, (x0 + x1) / 2, XLABEL_Y, "Dollars per Task (log scale)", 6.72, color=BLACK, bold=True,
          ha="center")
    yc_ax = (y0 + y1) / 2
    _text(fig, 147.448 + DX, yc_ax, "TerminalBench V2.1", 6.0, color=BLACK, bold=True, ha="center",
          rotation=90, rotation_mode="anchor")
    _text(fig, 153.739 + DX, yc_ax, "Benchmark Accuracy (%)", 6.0, color=BLACK, bold=True,
          ha="center", rotation=90, rotation_mode="anchor")

    # title, centred on the panel as in the original
    mid = (SEP1 + SEP2) / 2
    _text(fig, mid, 154.936 + DY, "Intelligence Systems Beat", 7.92, bold=True, ha="center")
    _text(fig, mid, 145.269 + DY, "Individual Frontier Models", 7.92, bold=True, ha="center")

    # legend box beneath: five rows, "Legend" heads the composed column
    _box(fig, 141.6 + DX, 0.72, SEP2 - 4.8, LEG_TOP, LEGEND_BG, 2.88)
    step = 6.8
    rows = [5.12 + step * k for k in range(4, -1, -1)]
    _text(fig, 146.4 + DX, rows[0], "Legend", 5.52, bold=True, va="center")
    lx_m, lx_t = 146.88 + DX, 153.36 + DX
    rx_t = SEP2 - 4.8 - 5.0 - max(text_to_path.get_text_width_height_descent(
        p["label"], FontProperties(family="DejaVu Sans", size=5.28), ismath=False)[0] for p in d["singles"])
    rx_m = rx_t - 6.48
    def width(s):
        return text_to_path.get_text_width_height_descent(
            s, FontProperties(family="DejaVu Sans", size=5.28), ismath=False)[0]

    entries = []  # (label lines, marker)
    for p in d["composed"]:
        if lx_t + width(p["label"]) + 3.0 > rx_m - 3.0:
            parts = p["label"].split(" → ")
            k = max(1, len(parts) // 2)
            entries.append(([" → ".join(parts[:k]) + " →", " → ".join(parts[k:])],
                            p["marker"]))
        else:
            entries.append(([p["label"]], p["marker"]))
    slots = sum(len(e[0]) for e in entries)
    i = len(rows) - slots
    for lines, marker in entries:
        ys = rows[i:i + len(lines)]
        yc = sum(ys) / len(ys)
        _marker(fig, lx_m, yc, marker, BLUE, LEG_MARK, 0.264)
        for k, line in enumerate(lines):
            _text(fig, lx_t, yc + step * ((len(lines) - 1) / 2 - k), line, 5.28, color=LEG_TEXT,
                  va="center")
        i += len(lines)
    for p, y in zip(d["singles"], rows[len(rows) - len(d["singles"]):]):
        _marker(fig, rx_m, y, p["marker"], KIND_COLOR[p["kind"]], LEG_MARK, 0.264)
        _text(fig, rx_t, y, p["label"], 5.28, color=LEG_TEXT, va="center")
    return ax


# ---------------------------------------------------------------- right panel
def draw_right(fig, d):
    c0, c1 = RIGHT_C0, RIGHT_C1
    mult_x, acc_x = c0 + 90.0, c1 - 17.94
    mid = (SEP2 + PAGE_W) / 2
    _text(fig, mid, 154.936 + DY, "Local by Default,", 7.92, bold=True, ha="center")
    _text(fig, mid, 145.269 + DY, "Hybrid by Design", 7.92, bold=True, ha="center")
    h = d["right_panel_header"]
    # the original renders "Configuration" with the DejaVu "fi" ligature
    _text(fig, c0, 134.557 + DY, h[0].replace("fi", "ﬁ"), 5.52, color=HEAD_TEXT)
    top, _, bottom = h[1].partition(" vs. ")    # "$ vs." over "Fable 5.1", like the next column
    _text(fig, mult_x, 137.124 + DY, top + " vs.", 5.52, color=HEAD_TEXT, ha="center")
    _text(fig, mult_x, 132.012 + DY, bottom, 5.52, color=HEAD_TEXT, ha="center")
    top, _, bottom = h[2].rpartition(" ")       # "TB v2.1" over "accuracy", as in the original
    _text(fig, acc_x, 137.124 + DY, top, 5.52, color=HEAD_TEXT, ha="center")
    _text(fig, acc_x, 132.012 + DY, bottom, 5.52, color=HEAD_TEXT, ha="center")
    fig.add_artist(Line2D([c0, c1], [130.56 + DY, 130.56 + DY], transform=_pt(fig), color=GREY_BAR,
                          linewidth=0.312, solid_capstyle="butt"))
    bar_x0, bar_w, bar_h = c0, 54.72, 6.96
    row_bottom = 20.0     # round 15: was 17.28; lifted 2.7 pt to make room for the two-line legend
    for i, row in enumerate(d["right_panel_rows"]):
        n_rows = len(d["right_panel_rows"])
        by = 113.28 + DY - (113.28 + DY - row_bottom) / (n_rows - 1) * i
        cy = by + bar_h / 2
        bold = row["bold"]
        if bold:
            _box(fig, c0 - 2.88, by - 2.16, c1, by + 16.08, HILITE_BG, 1.68, zorder=-1)
        # three segments, left to right: local (laptop), owned accelerators, closed-source provider
        loc, acc_ = row["local_pct"], row["accel_pct"]
        if loc + acc_ + row["provider_pct"] != 100:
            raise SystemExit(f"{row['label']}: execution shares do not sum to 100")
        _box(fig, bar_x0, by, bar_x0 + bar_w, by + bar_h, GREY_BAR, 0)
        if acc_ > 0:
            _box(fig, bar_x0 + bar_w * loc / 100, by, bar_x0 + bar_w * (loc + acc_) / 100, by + bar_h,
                 ACCEL, 0, zorder=0.5)
        if loc > 0:
            _box(fig, bar_x0, by, bar_x0 + bar_w * loc / 100, by + bar_h, BLUE, 0, zorder=0.5)
            _text(fig, c0 + 2.16, cy, f"{loc}%", 5.28, color="white", bold=bold, va="center")
        else:
            _text(fig, c0 + 52.56, cy, f"{row['provider_pct']}%", 5.28, color=RENT_TEXT, bold=bold,
                  ha="right", va="center")
        _text(fig, c0, cy + 8.04, row["label"], 6.0, bold=bold, va="center")
        _text(fig, mult_x, cy, row["mult"], 6.48, bold=bold, ha="center", va="center")
        _text(fig, acc_x, cy, row["acc"], 6.48, bold=bold, ha="center", va="center")
    # legend (round 15): the three segments on two lines, "Legend" heading the first, the second line
    # aligned under the first entry; 5.28 pt, the middle panel's legend size, so line one fits the panel
    def w(s):
        return text_to_path.get_text_width_height_descent(
            s, FontProperties(family="DejaVu Sans", size=5.28), ismath=False)[0]
    y1, y2, sw, tx, gap = 10.1, 4.3, 4.8, 6.6, 4.0        # line centres, swatch, swatch->text, entry gap
    x1 = c0 + 28.6
    x2 = x1 + tx + w("Local (laptop)") + gap
    if x2 + tx + w("Owned accelerators") > c1 - 2.0:
        raise SystemExit("right-panel legend: the first line overflows the panel")
    _box(fig, c0, 0.72, c1, 14.0, LEGEND_BG, 2.88)
    _text(fig, c0 + 3.6, y1, "Legend", 5.52, bold=True, va="center")
    for x, y, col, lab in ((x1, y1, BLUE, "Local (laptop)"), (x2, y1, ACCEL, "Owned accelerators"),
                           (x1, y2, GREY_BAR, "Closed-source provider")):
        _box(fig, x, y - sw / 2, x + sw, y + sw / 2, col, 0.72, zorder=0.5)
        _text(fig, x + tx, y, lab, 5.28, color=LEG_TEXT, va="center")


def build(d, out_pdf):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.unicode_minus": True})
    fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0), dpi=72)
    fig.patch.set_facecolor("white")
    for x in (SEP1, SEP2):
        fig.add_artist(Line2D([x, x], [161.52 + DY, 2.4], transform=_pt(fig), color=SEP,
                              linewidth=0.36, solid_capstyle="butt"))
    draw_middle(fig, d)
    draw_right(fig, d)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=72)
    plt.close(fig)
    compose_with_left_panel(buf, out_pdf)


def _left_panel_operations(page, reader):
    """The original page's content operations, keeping only what paints left of LEFT_CLIP_X.

    Every path and every text object of the middle and right panels is dropped (not just clipped),
    so the stamped left panel carries no stale numbers or labels. State operators are kept.
    """
    from pypdf.generic import ContentStream

    cs = ContentStream(page.get_contents(), reader)
    paint = {"f", "F", "f*", "S", "s", "B", "B*", "b", "b*"}
    build_ops = {"m", "l", "c", "v", "y", "re", "h"}
    kept, pending, pending_x, stack = [], [], [], []
    ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    text_block, text_keep = None, True

    def tx(x, y):
        return ctm[0] * x + ctm[2] * y + ctm[4]

    for operands, raw in cs.operations:
        op = raw.decode() if isinstance(raw, bytes) else raw
        if text_block is not None:
            text_block.append((operands, raw))
            if op == "Td" and tx(float(operands[0]), float(operands[1])) >= LEFT_CLIP_X:
                text_keep = False
            if op == "ET":
                if text_keep:
                    kept.extend(text_block)
                text_block = None
            continue
        if op == "BT":
            text_block, text_keep = [(operands, raw)], tx(0.0, 0.0) < LEFT_CLIP_X
            continue
        if op in build_ops:
            pending.append((operands, raw))
            nums = [float(v) for v in operands]
            if op == "re":
                pending_x += [tx(nums[0], nums[1]), tx(nums[0] + nums[2], nums[1] + nums[3])]
            elif nums:
                pending_x += [tx(nums[k], nums[k + 1]) for k in range(0, len(nums), 2)]
            continue
        if op in paint:
            if pending_x and max(pending_x) < LEFT_CLIP_X:
                kept.extend(pending + [(operands, raw)])
            pending, pending_x = [], []
            continue
        if op in ("W", "W*", "n"):          # clipping path: keep as is
            kept.extend(pending + [(operands, raw)])
            if op == "n":
                pending, pending_x = [], []
            continue
        if op == "q":
            stack.append(list(ctm))
        elif op == "Q":
            ctm = stack.pop()
        elif op == "cm":
            a = [float(v) for v in operands]
            ctm = [a[0] * ctm[0] + a[1] * ctm[2], a[0] * ctm[1] + a[1] * ctm[3],
                   a[2] * ctm[0] + a[3] * ctm[2], a[2] * ctm[1] + a[3] * ctm[3],
                   a[4] * ctm[0] + a[5] * ctm[2] + ctm[4], a[4] * ctm[1] + a[5] * ctm[3] + ctm[5]]
        kept.append((operands, raw))
    cs.operations = _highlight_apple(kept)
    return cs


def _highlight_apple(ops):
    """Author note (2026-09-25): highlight Apple in the Machines row as NVIDIA is, to show systems straddle NVIDIA and
    Apple machines. Apple's box keeps its fill; its outline takes the highlighted style (1.44 pt, #0072b2), and a short
    connector joins it to the existing machines connector at x = 114.72, above Public Cloud's branch.
    Author note (2026-09-25, later): the middleware line splits at the machines row and enters Apple as well as NVIDIA:
    from its turn at (63.6, 53.76) a branch runs right to x = 114.72 and down into Apple's top edge (y = 51.6)."""
    from pypdf.generic import FloatObject, NameObject, NumberObject

    def nums(o):
        return [float(v) for v in o]

    out, i, box_done, link_done, split_done = [], 0, False, False, False
    blue = [FloatObject(0), FloatObject(0.44705882), FloatObject(0.69803922)]
    while i < len(ops):
        operands, raw = ops[i]
        op = raw.decode() if isinstance(raw, bytes) else raw
        # Apple's box: the path that starts at (101.52, 51.6) and spans x 99.1-128.2, y 42.72-51.6, painted with B
        if (not box_done and op == "m" and len(operands) == 2 and abs(nums(operands)[0] - 101.52) < 0.05
                and abs(nums(operands)[1] - 51.6) < 0.05):
            j = i
            while j < len(ops) and (ops[j][1].decode() if isinstance(ops[j][1], bytes) else ops[j][1]) != "B":
                j += 1
            out.extend(ops[i:j])
            out.append(([FloatObject(1.44)], b"w"))
            out.append((blue, b"RG"))
            out.append(ops[j])
            i, box_done = j + 1, True
            continue
        out.append((operands, raw))
        # the middleware line ends at NVIDIA's top (22.56, 51.6); in the same block (join, width and colour still set),
        # add its branch into Apple from the turn at (63.6, 53.76)
        if (not split_done and op == "S" and len(out) >= 2
                and (out[-2][1].decode() if isinstance(out[-2][1], bytes) else out[-2][1]) == "l"
                and [round(v, 2) for v in nums(out[-2][0])] == [22.56, 51.6]):
            out.extend([([FloatObject(63.6), FloatObject(53.76)], b"m"), ([FloatObject(114.72), FloatObject(53.76)], b"l"),
                        ([FloatObject(114.72), FloatObject(51.6)], b"l"), ([], b"S")])
            split_done = True
        # after the machines connector (it ends at (114.72, 36.96)), add Apple's branch in the same style
        if (not link_done and op == "S" and len(out) >= 2
                and (out[-2][1].decode() if isinstance(out[-2][1], bytes) else out[-2][1]) == "l"
                and [round(v, 2) for v in nums(out[-2][0])] == [114.72, 36.96]):
            out.extend([([], b"Q"), ([], b"q"), ([NumberObject(1)], b"j"), ([FloatObject(1.44)], b"w"), (blue, b"RG"),
                        ([FloatObject(114.72), FloatObject(40.08)], b"m"), ([FloatObject(114.72), FloatObject(42.72)], b"l"),
                        ([], b"S")])
            link_done = True
            # the original Q that closes this block follows next and now closes the added block
        i += 1
    assert box_done and link_done and split_done, (box_done, link_done, split_done)
    return out


def compose_with_left_panel(new_pdf, out_pdf):
    """Stamp the original's left panel (vector, x < LEFT_CLIP_X) onto the new page."""
    from pypdf import PdfReader, PdfWriter

    new_pdf.seek(0)
    base = PdfReader(new_pdf).pages[0]
    orig = PdfWriter(clone_from=str(ORIGINAL))
    left = orig.pages[0]
    left.replace_contents(_left_panel_operations(left, orig))
    from pypdf import Transformation
    base.merge_transformed_page(left, Transformation().scale(S, S), over=True)   # uniform scale, no distortion
    w = PdfWriter()
    w.add_page(base)
    w.pages[0].compress_content_streams()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdf, "wb") as fh:
        w.write(fh)


def target_data():
    f = json.loads(INPUTS.read_text())["fig1"]
    # author ruling (2026-09-25): the Qwen3.8-Flash-Next system is not local-first (its first stage needs datacenter
    # GPUs) yet dominates the headline, so it leaves Figure 1; Table 1 keeps it, and 3.1 explains it
    f = dict(f, composed=[p for p in f["composed"] if not p["label"].startswith("Qwen3.8-Flash-Next")])
    d = dict(f)
    # round 15: Qwen3.8-27B alone on the laptop scores 75.4%, which the original (75, 95) range put on the
    # x axis; the axis now starts 2.5 points lower, ticks unchanged
    d["ylim"] = (72.5, 95) if min(p["acc"] for p in f["composed"] + f["singles"]) < 77.0 else (75, 95)
    d["yticks"] = [75, 80, 85, 90, 95]
    d["region"] = frontier_region([(p["usd"], p["acc"]) for p in f["composed"] + f["singles"]],
                                  (0.001, 10), d["ylim"])
    d["bubble_xy"] = (AX_BOX[2] - 30.0, AX_BOX[1] + 19.0)   # page pt: empty lower right, clear of every marker
    return d


if __name__ == "__main__":
    build(target_data(), OUT)
    print(OUT)
