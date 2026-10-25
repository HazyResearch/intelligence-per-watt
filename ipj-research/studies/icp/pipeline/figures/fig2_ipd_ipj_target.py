#!/usr/bin/env python3
"""Figure 4 (IPD / IPJ) of the target paper, in the canonical figure's style.

The plotting code is copied from fig_ipd_ipj() in
intelligence-per-joule/ipj/figures/blog-post-ipd/build_all.py, with the style helpers it uses
(paper_style.py: pareto, draw_pareto, bracket; style.py: rcParams) inlined here so this
script never imports from, or writes a __pycache__ into, the read-only canonical repository.

Kept verbatim from the canonical figure: figure size, panel titles, shared y label, font sizes,
tick formats, the IPD axis ranges, marker AREA (total parameters for open weights, published
output price for closed weights), the black marker edge, the dotted frontier line and shaded
region (paper_style.draw_pareto, unmodified), the boxed bracket annotations, and the grey legend
boxes stacked in a column to the right of the panels.

Author round of 2026-09-25 (changes from the canonical encodings):
  colour     one colour per company, taken from its logo (FAMILY_COLOR; identical in both panels
             and the legend). This replaces the open/closed colour pair.
  shape      a regular polygon whose number of sides is the model's size rank within its family:
             smallest 3 (triangle), then 4 (square), 5 (pentagon), 6 (hexagon). Every polygon
             rests on a flat edge. Size = total parameters for open weights (a tie is broken by
             release order, older first); closed weights publish no size, so they are ranked by
             published output price, the proxy the canonical figure already uses for their area.
             Each polygon is scaled to the area of the canonical circle at the same s, so marker
             area still encodes parameters / price and never the number of sides.
  dark fills OpenAI dark blue and Z.ai charcoal would lose their black edge, so they get a white
             keyline inside the black outline (plot and legend).
  shape key  a third grey legend box, "Size rank in company", in the slot the canonical
             "Weights" box used.
  removed    Claude Opus 4.8, MiniMax-M2.7, Qwen3.6-27B, GPT-5.5 (legibility); kept in
             targets/fig2_tiers.json with excluded=true.
  added      Claude Fable 5.1 (max effort), IPD panel only, at targets/targets.json
             public.aa_tb21.fable51_max.
  IPJ        bracket Moonshot Kimi K3 -> Qwen3.8-27B; x axis fitted to the plotted energy range.
  IPD        the two bracket endpoints are drawn above their neighbours, so GPT-6 Astra's
             hexagon sits over GPT-5.6-sol's pentagon (same 89.4%, $0.45 vs $0.38).

Author round of 2026-09-25, later (round 15):
  y axis     both panels from 60% to 100%; tiers below 60% (Qwen3.5-397B) are omitted and kept in
             targets/fig2_tiers.json with excluded=true.
  Kimi K3    drawn like every other tier: no dashed edge, no "estimated" note (every plotted value
             is an estimate in the target world, so the distinction is not drawn).
  legend     one "Models" box grouped by company (the "Closed weights" / "Open weights" split is
             removed), above the "Size rank in company" box.
  colour     OpenAI a dark blue (#0A2472) and Qwen an indigo purple (#6123C0); every other company
             keeps its logo colour.

Author round of 2026-09-25 (round 16):
  legend     the legend column (both boxes) is centred vertically on the two panels' plot area (the
             axes' vertical extent), not on the titles or axis labels.

The model set is the canonical figure's set (every tier at or above the 60% floor) minus the four
removed tiers, plus the target paper's own tiers (GPT-6 Astra, GLM-5.3, Qwen3.8-27B,
Qwen3.8-Flash-Next) and Claude Fable 5.1. The "Oracle routing (ceiling)" series stays omitted.

Data rules (target world), recorded per value in targets/fig2_tiers.json:
  accuracy   TerminalBench V2.1, Artificial Analysis leaderboard or the paper's target value
             (sol, Astra, GLM-5.3, Qwen3.8-27B, Flash-Next: derived as solved/graded from
             targets/targets.json tb21.tiers, then checked against figure_inputs.json);
             Sakana Fugu Ultra (no leaderboard value) keeps the canonical value.
  $/task     targets/figure_inputs.json fig2.ipd for the tiers it lists; Claude Fable 5.1 from
             targets/targets.json public.aa_tb21; every other tier at the canonical cost.
  kJ/task    targets/figure_inputs.json fig2.ipj for GLM-5.3, Qwen3.8-27B, Qwen3.8-Flash-Next;
             every other open-weight tier at the energy the canonical figure plots.
  size       canonical parameter counts / output prices, except that a closed tier listed in
             figure_inputs.json fig2.ipd with an out_price uses that price (GPT-5.6-sol $20/M,
             the price its $/task was derived from); Qwen3.8-27B 27B, Qwen3.8-Flash-Next 180B,
             GLM-5.3 753B, GPT-6 Astra $50/M, Claude Fable 5.1 $50/M.

Bracket annotations are hardcoded to the pairs the target paper argues about:
  IPD  GPT-6 Astra (the paper's closed-weight comparison anchor) -> Qwen3.8-Flash-Next
  IPJ  Moonshot Kimi K3 -> Qwen3.8-27B (author, 2026-09-25)
and formatted from the plotted values with the canonical format strings (including the canonical
sign-aware IPJ lead: "Same accuracy" under a point, else "accuracy lost" / "more accurate").

The self-served tiers (GLM-5.3, Qwen3.8-27B, Qwen3.8-Flash-Next) are plotted at the unrounded
values that figure_inputs.json rounds from (targets/targets.json tb21.tiers and
targets/derived.json machines); the script checks each rounds back to the figure_inputs value.

Saved with the canonical call, bbox_inches="tight", pad_inches=0.3.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
FIG = json.loads((ROOT / "targets" / "figure_inputs.json").read_text())["fig2"]
OUT = ROOT / "paper" / "figures_and_tables" / "fig2_ipd_ipj_target.pdf"
TIERS_OUT = ROOT / "targets" / "fig2_tiers.json"

# ---------------------------------------------------------------- style.py (rcParams)
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
})
plt.rcParams.update({"font.size": 15})
GRID = "#AAAAAA"

# ---------------------------------------------------------------- paper_style.py
plt.rcParams.update({
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "grid.linewidth": 0.65,
})
CASCADE = "#16A34A"       # green  - frontier line and region
INK = "#111318"
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
    """Dotted frontier line plus the shaded region above it (paper_style.draw_pareto, verbatim)."""
    front = pareto(points)
    if len(front) < 2:
        return front
    xs, ys = zip(*front)
    if shade:
        x_lo, x_hi = ax.get_xlim()
        y_hi = ax.get_ylim()[1]
        poly_x = [x_lo, x_lo] + list(xs) + [x_hi, x_hi]
        poly_y = [y_hi, ys[0]] + list(ys) + [ys[-1], y_hi]
        ax.fill(poly_x, poly_y, color=CASCADE, alpha=PARETO_ALPHA, zorder=1, linewidth=0)
    ax.plot(xs, ys, ls=":", color=CASCADE, lw=PARETO_LINE_LW,
            alpha=PARETO_LINE_ALPHA, zorder=3)
    return front


def bracket(ax, p_lo, p_hi, text, *, clear=0.055, tick=0.030, fontsize=14.4,
            dx_pts=0.0):
    """Symmetric bracket over two points with a boxed bold label (paper_style.bracket)."""
    (x0, y0), (x1, y1) = p_lo, p_hi
    lo, hi = ax.get_ylim()
    span = hi - lo
    y_bar = max(y0, y1) + clear * span
    y_tick = y_bar - tick * span
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

    ax.annotate(text, (xm, y_bar), textcoords="offset points", xytext=(dx_pts, 7), ha="center",
                va="bottom", fontsize=fontsize, fontweight="bold", color=INK, zorder=9,
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=INK, lw=1.0, alpha=0.95))
    return y_bar


# ---------------------------------------------------------------- encodings
# Colour = company, from its logo (research pass 2026-09-25; the logo colour is shifted only where
# it would collide with another family or with the green frontier):
#   OpenAI    #0A2472  dark blue (author, 2026-09-25: "the characteristic dark blue"; replaces the black
#                      logo symbol). A navy (OKLCH L 0.30, C 0.14, h 265), well below DeepSeek #4D6BFE
#                      (L 0.59) and Kimi #4CA7FA (L 0.71) in lightness, so the three blues stay apart.
#   Anthropic #D97757  Claude logo / Anthropic primary accent
#   Gemini    #D9AE20  the Gemini sparkle's yellow point (#FBBC04), darkened; its blue is Kimi's
#   Sakana    #B71C07  Sakana red fish (#E10600), darkened away from MiniMax pink
#   Moonshot  #4CA7FA  Kimi logo blue (#1783FF), lightened away from DeepSeek
#   Qwen      #6123C0  indigo purple (author, 2026-09-25), from the Qwen3 logo's indigo violet (#632DE4,
#                      logo_qwen3.png; lobe-icons qwen.svg #6336E7) deepened to OKLCH L 0.45, C 0.22,
#                      h 293: the logo value itself sits 12 dE from DeepSeek #4D6BFE, under the 15 floor.
#                      L 0.45 keeps it above the dark-fill threshold, so it carries the black edge.
#   Z-AI      #474747  Z.ai logo tile (#2D2D2D), lifted (it once had to stay apart from OpenAI black)
#   DeepSeek  #4D6BFE  DeepSeek logo
#   MiniMax   #E62176  MiniMax logo gradient, pink end
FAMILY_COLOR = {"OpenAI": "#0A2472", "Anthropic": "#D97757", "Gemini": "#D9AE20",
                "Sakana": "#B71C07", "Moonshot": "#4CA7FA", "Qwen": "#6123C0",
                "Z-AI": "#474747", "DeepSeek": "#4D6BFE", "MiniMax": "#E62176"}
# dataviz validate_palette.js on these nine hexes (--mode light --pairs all, surface #fcfcfb), round 15.
# The FAILs are the two dark colours by design: OpenAI's dark blue (L 0.30, asked for as a dark blue) and
# Z.ai's charcoal (neutral logo colour, also below the chroma floor); both carry the white keyline. The
# contrast WARN is relieved by the 2.0-pt black edge and the legend names.
PALETTE_CHECK = ("validate_palette.js light, all pairs (round 15): FAIL lightness band for #0A2472 "
                 "(L 0.303) and #474747 (L 0.398), FAIL chroma floor for #474747 only (dark blue and "
                 "neutral logo colours, by design; white keyline); PASS CVD, worst #474747/#B71C07 dE 8.3 "
                 "protan (tritan 7.2); PASS normal-vision floor, worst #4D6BFE/#4CA7FA dE 15.3; WARN "
                 "contrast on white #D9AE20 2.04:1, #4CA7FA 2.49:1 (relief: black marker edge, legend "
                 "names). OpenAI #0A2472 vs Qwen #6123C0 dE 18.8 (CVD 15.3), vs Z.ai 16.7 (14.2), vs "
                 "DeepSeek 30.0; Qwen vs DeepSeek #4D6BFE dE 16.8 (CVD 13.4), vs Kimi #4CA7FA 30.2.")
# Fills too dark to carry the black edge (OKLab L < 0.45): white keyline inside a black outline.
DARK_FILL = {"OpenAI", "Z-AI"}      # #0A2472 L 0.30, #474747 L 0.40 (Qwen #6123C0 is at L 0.45)
EDGE_LW = 2.0                    # canonical marker edge
KEY_LW, KEY_HALO = 1.3, 3.6      # dark fills: white keyline, black outline (plot)
LEG_KEY_LW, LEG_KEY_HALO = 0.8, 2.6   # dark fills: legend handles (canonical mew 1.0)
LEG_H = 15.0                     # legend handle height, points (canonical ms 15.0)
# author note 2026-09-25: legend text 25% larger in both right-hand boxes (was 16.0 and 19.0)
LEG_FS, LEG_TFS = 20.0, 23.75

# One "Models" box grouped by company (author, 2026-09-25: no open-/closed-weight sections), family
# order as in the canonical MODEL_LEGEND; within a family, smallest first.
LEGEND = (("Models", ("OpenAI", "Anthropic", "Gemini", "Sakana", "Moonshot", "Z-AI", "Qwen",
                      "DeepSeek", "MiniMax")),)
Y_LO, Y_HI = 60, 100             # both panels (author, 2026-09-25); tiers below Y_LO are omitted


def polygon(n):
    """Regular n-gon marker resting on a flat edge (triangle up, square axis-aligned, ...)."""
    return (n, 0, 0.0 if n % 2 else 180.0 / n)


def area_fix(n):
    """Factor on s that gives an n-gon marker the area of the canonical circle ('o') at s.

    matplotlib draws (n, 0, angle) inscribed in a circle of diameter sqrt(s), so without this a
    triangle would cover 41% of a circle's area and a hexagon 83%: shape would leak into area.
    """
    return (math.pi / 4) / (0.125 * n * math.sin(2 * math.pi / n))


def price_size(usd_per_m):
    """Marker area from output $/M, on the same visual range as the parameter scale."""
    if not usd_per_m:
        return 190.0
    lo, hi = math.log(1.0), math.log(50.0)
    t = (math.log(usd_per_m) - lo) / (hi - lo)
    return 110 + 460 * max(0.0, min(1.0, t))


def msize(params):
    """Marker area from total parameter count; fixed reference when unpublished."""
    if not params:
        return 190.0
    lo, hi = math.log(20), math.log(800)
    t = (math.log(params) - lo) / (hi - lo)
    return 110 + 460 * max(0.0, min(1.0, t))


# ---------------------------------------------------------------- the tier table
# Source tags written to fig2_tiers.json.
AA = "Artificial Analysis TerminalBench V2.1 leaderboard"
TGT = "paper target value (targets/targets.json tb21.tiers, solved/graded)"
TASK = ("task-spec value; leaderboards/public.json has no Qwen3.5-397B TB2.1 row (its AA text "
        "lists 51.3 as Qwen3.6 27B non-reasoning)")
CANON = "canonical figure (build_all.py)"
FI_IPD = "targets/figure_inputs.json fig2.ipd"
FI_IPJ = "targets/figure_inputs.json fig2.ipj"
PUB_FABLE = "targets/targets.json public.aa_tb21.fable51_max"
FABLE_PRICE = ("leaderboards/public.json Terminal-Bench 2.1 row 'Claude Fable 5.1 (Adaptive "
               "Reasoning, Max Effort, Default Fallback) [AA]': AA price $10/$50 per 1M in/out")

# Canonical plotted values, read from build_all.py and its vendored board
# (data/per_task_vectors_v2.json: authoritative_mean_cost_usd, local_dollars_per_task;
# OPEN_MODELS; OUT_PRICE_PER_M). Used only where a rule above says "as in the canonical figure".
#   (legend label, family, weights, canonical acc, canonical $/task, canonical kJ, params_B,
#    output $/M)
CANONICAL = {
    "OpenAI GPT-5.6-sol":        ("OpenAI", "closed", 72.75, 0.324, None, None, 30.0),
    "OpenAI GPT-5.6-Terra":      ("OpenAI", "closed", 62.64, 0.227, None, None, 12.0),
    "OpenAI GPT-5.6-Luna":       ("OpenAI", "closed", 49.16, 0.0373, None, None, 1.20),
    "OpenAI GPT-5.5":            ("OpenAI", "closed", 75.28, 0.511, None, None, None),
    "Anthropic Claude Opus 5":   ("Anthropic", "closed", 76.40, 0.9, None, None, 25.0),
    "Anthropic Claude Opus 4.8": ("Anthropic", "closed", 74.16, 0.757, None, None, 25.0),
    "Google Gemini 3.7 Flash":   ("Gemini", "closed", 79.78, 1.1266, None, None, None),
    "Google Gemini 3.1 Pro":     ("Gemini", "closed", 65.17, 0.7455, None, None, 12.0),
    "Sakana Fugu Ultra":         ("Sakana", "closed", 75.28, 1.52, None, None, None),
    "Moonshot Kimi K3":          ("Moonshot", "open", 64.33, 0.326, 460.0, 2800, None),
    "Z-AI GLM-5.2":              ("Z-AI", "open", 65.2, 0.010977, 453.7, 753, None),
    "Qwen3.6-27B":               ("Qwen", "open", 55.1, 0.002001, 82.7, 27, None),
    "Qwen3.5-397B":              ("Qwen", "open", 41.6, 0.009421, 389.4, 397, None),
    "DeepSeek-V4-Flash":         ("DeepSeek", "open", 66.3, 0.021, 196.9, 284, None),
    "MiniMax-M3":                ("MiniMax", "open", 59.6, 0.007704, 318.4, 428, None),
    "MiniMax-M2.7":              ("MiniMax", "open", 51.7, 0.00571, 236.0, None, None),
}
# Target-world tiers the canonical figure does not carry: (family, weights, params_B, output $/M).
NEW = {
    "OpenAI GPT-6 Astra":          ("OpenAI", "closed", None, 50.0),
    "Anthropic Claude Fable 5.1":  ("Anthropic", "closed", None, 50.0),
    "Z-AI GLM-5.3":                ("Z-AI", "open", 753, None),
    "Qwen3.8-27B":                 ("Qwen", "open", 27, None),
    "Qwen3.8-Flash-Next":          ("Qwen", "open", 180, None),
}
# Removed from both panels and the legend; kept in fig2_tiers.json.
EXCLUDED = {n: "removed for legibility (author, 2026-09-25)" for n in (
    "Anthropic Claude Opus 4.8", "MiniMax-M2.7", "Qwen3.6-27B", "OpenAI GPT-5.5")}

_PUB = json.loads((ROOT / "targets" / "targets.json").read_text())["public"]["aa_tb21"]
PUBLIC_USD = {"Anthropic Claude Fable 5.1": (_PUB["fable51_max"]["usd"], f"{PUB_FABLE}.usd"),
              # author ruling 2026-09-25: Terra at Artificial Analysis's computed cost, like every other single model
              "OpenAI GPT-5.6-Terra": (_PUB["terra_max"]["usd"], "targets/targets.json public.aa_tb21.terra_max.usd")}
_TGT = json.loads((ROOT / "targets" / "targets.json").read_text())["tb21"]["tiers"]


def _tgt_acc(key):
    """Paper target accuracy, derived from targets.json so the figure follows the plan model."""
    t = _TGT[key]
    return (round(100.0 * t["solved"] / t["graded"], 1),
            f"{TGT}: tb21.tiers.{key} {t['solved']}/{t['graded']}")


# TerminalBench V2.1 accuracy (target world). Tiers absent here keep the canonical value.
ACC = {
    "OpenAI GPT-5.6-sol": _tgt_acc("sol"), "OpenAI GPT-6 Astra": _tgt_acc("astra"),
    "OpenAI GPT-5.6-Terra": (88.0, AA), "OpenAI GPT-5.6-Luna": (80.9, AA),
    "OpenAI GPT-5.5": (84.3, AA), "Anthropic Claude Opus 5": (89.1, AA),
    "Anthropic Claude Fable 5.1": (_PUB["fable51_max"]["acc"], f"{AA} ({PUB_FABLE}.acc)"),
    "Anthropic Claude Opus 4.8": (84.6, AA), "Google Gemini 3.7 Flash": (85.8, AA),
    "Google Gemini 3.1 Pro": (73.8, AA), "Moonshot Kimi K3": (85.0, AA),
    "Z-AI GLM-5.3": _tgt_acc("glm53"), "Z-AI GLM-5.2": (77.9, AA),
    "Qwen3.8-27B": _tgt_acc("qwen38"), "Qwen3.8-Flash-Next": _tgt_acc("flashnext"),
    "Qwen3.6-27B": (60.7, AA), "Qwen3.5-397B": (51.3, TASK),
    "DeepSeek-V4-Flash": (78.7, AA), "MiniMax-M3": (65.2, AA),
}
# Self-served tiers: unrounded sources of the figure_inputs.json values (checked to round back).
_MACH = json.loads((ROOT / "targets" / "derived.json").read_text())["machines"]
UNROUNDED = {   # name -> (usd, usd source, kJ, kJ source)
    "Z-AI GLM-5.3": (_TGT["glm53"]["own"], "targets/targets.json tb21.tiers.glm53.own",
                     _TGT["glm53"]["kj_wall"], "targets/targets.json tb21.tiers.glm53.kj_wall"),
    "Qwen3.8-27B": (_MACH["h200_16"]["own"], "targets/derived.json machines.h200_16.own",
                    _MACH["h200_16"]["kj_wall"], "targets/derived.json machines.h200_16.kj_wall"),
    "Qwen3.8-Flash-Next": (_MACH["fn_h200x2"]["own"], "targets/derived.json machines.fn_h200x2.own",
                           _MACH["fn_h200x2"]["kj_wall"],
                           "targets/derived.json machines.fn_h200x2.kj_wall"),
}
# Energy provenance where the canonical figure's value is itself a proxy (recorded in fig2_tiers.json
# only; drawn like every other tier, author 2026-09-25).
KJ_NOTE = {"Moonshot Kimi K3": ("canonical figure (build_all.py OPEN_MODELS / PROXY_ENERGY): "
                                "ESTIMATED per ADR-0003 from GLM-5.2, 460 kJ, band [230, 920]; "
                                "not a measurement")}

# Size rank within a family (polygon sides = 3 + rank). Open weights rank by total parameters,
# ties broken by release order (older first). Closed weights publish no size and rank by
# published output price; a closed tier without a published price needs an explicit rule here.
RELEASE_ORDER = {"Z-AI GLM-5.2": 0, "Z-AI GLM-5.3": 1}
PRICE_RANK_FALLBACK = {
    "Google Gemini 3.7 Flash": (0.0, "no published output price in this figure's sources; ranked "
                                     "below Gemini 3.1 Pro ($12/M) by vendor tier (Flash is "
                                     "Google's lower-priced line)"),
}


def _rounds_to(v, ref):
    """True when v rounded to ref's printed precision equals ref."""
    dec = len(repr(float(ref)).split(".")[1].rstrip("0")) if "." in repr(float(ref)) else 0
    return abs(round(v, dec) - ref) < 1e-12


# Every tier the table knows, canonical family order (plotted or excluded).
ORDER = [
    "OpenAI GPT-5.6-Luna", "OpenAI GPT-5.6-Terra", "OpenAI GPT-5.6-sol", "OpenAI GPT-6 Astra",
    "OpenAI GPT-5.5",
    "Anthropic Claude Opus 5", "Anthropic Claude Fable 5.1", "Anthropic Claude Opus 4.8",
    "Google Gemini 3.7 Flash", "Google Gemini 3.1 Pro",
    "Moonshot Kimi K3",
    "Z-AI GLM-5.2", "Z-AI GLM-5.3",
    "Qwen3.8-27B", "Qwen3.8-Flash-Next", "Qwen3.5-397B", "Qwen3.6-27B",
    "DeepSeek-V4-Flash",
    "MiniMax-M3", "MiniMax-M2.7",
    "Sakana Fugu Ultra",
]


# Bracket endpoints (hardcoded, as in the canonical figure). HI = upper-x end, LO = lower-x end.
IPD_HI, IPD_LO = "OpenAI GPT-6 Astra", "Qwen3.8-Flash-Next"
IPJ_HI, IPJ_LO = "Moonshot Kimi K3", "Qwen3.8-27B"
# IPJ x axis: log, limits and ticks picked from these readable values to fit the plotted energies.
IPJ_TICKS = [20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000]


def build_tiers():
    fi_ipd = {r["label"]: r for r in FIG["ipd"]}
    fi_ipj = {r["label"]: r for r in FIG["ipj"]}
    tiers = []
    for name in ORDER:
        if name in CANONICAL:
            fam, w, c_acc, c_usd, c_kj, par, price = CANONICAL[name]
        else:
            fam, w, par, price = NEW[name]
            c_acc = c_usd = c_kj = None
        acc, acc_src = ACC.get(name, (c_acc, CANON + " (no leaderboard value)"))
        if name in UNROUNDED:
            usd, usd_src = round(UNROUNDED[name][0], 8), f"{FI_IPD} ({fi_ipd[name]['usd']:g}), unrounded from {UNROUNDED[name][1]}"
            if not _rounds_to(usd, fi_ipd[name]["usd"]):
                raise SystemExit(f"{name}: unrounded $ {usd} does not round to {fi_ipd[name]['usd']}")
        elif name in fi_ipd:
            usd, usd_src = fi_ipd[name]["usd"], FI_IPD
        elif name in PUBLIC_USD:
            usd, usd_src = PUBLIC_USD[name]
        else:
            usd, usd_src = c_usd, CANON
        if w == "closed":
            kj, kj_src = None, "not plotted (closed weights)"
        elif name in UNROUNDED:
            kj, kj_src = round(UNROUNDED[name][2], 6), f"{FI_IPJ} ({fi_ipj[name]['kj']:g}), unrounded from {UNROUNDED[name][3]}"
            if not _rounds_to(kj, fi_ipj[name]["kj"]):
                raise SystemExit(f"{name}: unrounded kJ {kj} does not round to {fi_ipj[name]['kj']}")
        elif name in fi_ipj:
            kj, kj_src = fi_ipj[name]["kj"], FI_IPJ
        else:
            kj, kj_src = c_kj, KJ_NOTE.get(name, CANON)
        if acc is None or usd is None:
            raise SystemExit(f"{name}: missing accuracy or cost")
        if name in fi_ipd and abs(fi_ipd[name]["acc"] - acc) > 1e-9:
            raise SystemExit(f"{name}: accuracy {acc} disagrees with figure_inputs")
        if name == "Anthropic Claude Fable 5.1":
            size_src = FABLE_PRICE
        elif w == "closed" and fi_ipd.get(name, {}).get("out_price"):
            # Target world: the output price the plotted $/task was derived from (sol $20/M, not
            # the canonical $30/M), so marker area and cost rest on the same price.
            price, size_src = fi_ipd[name]["out_price"], f"{FI_IPD} out_price"
        else:
            size_src = CANON if name in CANONICAL else "task spec (target tier)"
        tiers.append(dict(
            name=name, family=fam, weights=w, color=FAMILY_COLOR[fam],
            acc=acc, usd=usd, kj=kj,
            params_b=par if w == "open" else None,
            out_price_per_m=price if w == "closed" else None,
            size_basis=(f"{par}B total params" if par else "fixed (params unpublished)")
            if w == "open" else (f"output ${price:g}/M" if price else
                                 "fixed (no published price)"),
            excluded=name in EXCLUDED or acc < Y_LO,
            exclusion_reason=EXCLUDED.get(name) or (
                f"below the {Y_LO}% y-axis floor (author, 2026-09-25: both panels start at {Y_LO}%)"
                if acc < Y_LO else None),
            source=dict(acc=acc_src, usd=usd_src, kj=kj_src, size=size_src),
        ))
    return tiers


def assign_polygons(shown):
    """Sides = 3 + size rank within the family, over the plotted tiers."""
    fams = {}
    for r in shown:
        fams.setdefault(r["family"], []).append(r)
    for fam, rows in fams.items():
        def key(r):
            if r["weights"] == "open":
                if not r["params_b"]:
                    raise SystemExit(f"{r['name']}: open weights without a parameter count to rank")
                return (r["params_b"], RELEASE_ORDER.get(r["name"], 0))
            if r["out_price_per_m"]:
                return (r["out_price_per_m"], 0)
            if r["name"] in PRICE_RANK_FALLBACK:
                return (PRICE_RANK_FALLBACK[r["name"]][0], 0)
            raise SystemExit(f"{r['name']}: closed weights without a price or rank rule")
        if len(rows) > 1:
            rows.sort(key=key)
            for a, b in zip(rows, rows[1:]):
                if key(a) == key(b):
                    raise SystemExit(f"{a['name']} / {b['name']}: size tie with no release order")
        for i, r in enumerate(rows):
            r["size_rank"], r["sides"] = i + 1, 3 + i
            r["marker"] = list(polygon(3 + i))
            if len(rows) == 1:
                basis = "only plotted model of its family"
            elif r["weights"] == "open":
                basis = f"{r['params_b']}B total params"
                if r["name"] in RELEASE_ORDER and any(
                        o is not r and o["params_b"] == r["params_b"] for o in rows):
                    basis += " (tie, broken by release order: older first)"
            elif r["out_price_per_m"]:
                basis = f"output ${r['out_price_per_m']:g}/M (closed: size unpublished)"
            else:
                basis = PRICE_RANK_FALLBACK[r["name"]][1]
            r["rank_basis"] = basis


def scatter_tier(ax, r, x, s, *, z=6.0):
    """One tier's marker: family colour, polygon by size rank, equal-area to the canonical circle."""
    n, col = r["sides"], FAMILY_COLOR[r["family"]]
    kw = dict(s=s * area_fix(n), marker=polygon(n), zorder=z, color=col)
    if r["family"] in DARK_FILL:
        kw.update(edgecolors="white", linewidths=KEY_LW,
                  path_effects=[pe.Stroke(linewidth=KEY_HALO, foreground="black"), pe.Normal()])
    else:
        kw.update(edgecolors="black", linewidths=EDGE_LW)
    ax.scatter(x, r["acc"], **kw)


def legend_marker(n):
    """Legend-only n-gon (same orientation as the plot), centred on its bounding box and sized
    to a common height, so every handle sits in its row like the canonical built-in markers."""
    ang = math.radians(polygon(n)[2]) + math.pi / 2
    pts = [(math.cos(ang + 2 * math.pi * k / n), math.sin(ang + 2 * math.pi * k / n))
           for k in range(n)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    verts = [(x - cx, y - cy) for x, y in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    # matplotlib scales a custom path so its larger half-extent is ms / 2.
    return MPath(verts + verts[:1], closed=True), LEG_H * max(w, h) / h


def legend_handle(r):
    col = FAMILY_COLOR[r["family"]]
    path, ms = legend_marker(r["sides"])
    kw = dict(marker=path, ls="", ms=ms, mfc=col, label=r["name"])
    if r["family"] in DARK_FILL:
        kw.update(mec="white", mew=LEG_KEY_LW,
                  path_effects=[pe.Stroke(linewidth=LEG_KEY_HALO, foreground="black"),
                                pe.Normal()])
    else:
        kw.update(mec="black", mew=1.0)
    return Line2D([], [], **kw)


def _below(fig, leg, gap=0.018):
    """Figure coordinates just under a drawn legend's frame, left-aligned with it."""
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    return (bb.x0, bb.y0 - gap, bb.width, 0.0)


def fig_ipd_ipj():
    tiers = build_tiers()
    FLOOR = float(Y_LO)   # the y axis starts here; build_tiers marks the tiers below it excluded
    shown = [r for r in tiers if not r["excluded"]]
    assert all(r["acc"] >= FLOOR for r in shown)
    assign_polygons(shown)
    closed = [r for r in shown if r["weights"] == "closed"]
    OPEN_SHOWN = [r for r in shown if r["weights"] == "open"]
    ICON = 1.15 ** 2
    area = {r["name"]: (price_size(r["out_price_per_m"]) if r["weights"] == "closed"
                        else msize(r["params_b"])) * ICON for r in shown}
    # Round 5 (author note, 2026-09-24): full text width, legends stacked to the right of the panels.
    fig, (a, b) = plt.subplots(1, 2, figsize=(20.4, 7.55))   # round 7: about 15% shorter

    # ---- IPD: closed tiers + every open-weights model
    a.set_xscale("log"); a.set_xlim(0.0007, 12); a.set_ylim(Y_LO, Y_HI)
    pts = [(r["usd"], r["acc"]) for r in closed] + [(m["usd"], m["acc"]) for m in OPEN_SHOWN]
    front_a = draw_pareto(a, pts)
    # Small markers on top, except the two bracket endpoints, which sit above their neighbours so
    # each tick lands on its own marker (GPT-6 Astra shares 89.4% with GPT-5.6-sol, $0.45 vs $0.38).
    for r in sorted(shown, key=lambda r: -area[r["name"]]):
        scatter_tier(a, r, r["usd"], area[r["name"]],
                     z=6.5 if r["name"] in (IPD_HI, IPD_LO) else 6.0)
    a.set_yticks(range(Y_LO, Y_HI + 1, 10))
    a.set_yticklabels([f"{v}%" for v in range(Y_LO, Y_HI + 1, 10)], fontsize=22.7)
    a.set_xticks([0.001, 0.01, 0.1, 1, 10])
    a.set_xticklabels(["\\$0.001", "\\$0.01", "\\$0.10", "\\$1", "\\$10"], fontsize=22.7)
    a.set_xlabel("Cost per task (USD, log)", fontsize=26.3, fontweight="bold")
    a.set_ylabel("Terminal Bench V2.1 - Accuracy", fontsize=26.3, fontweight="bold")
    a.set_title("Intelligence per Dollar (IPD)", fontsize=31.5, fontweight="bold", pad=14)
    # Canonical: hardcoded Fable 5 -> DeepSeek-V4-Flash. Target paper: hardcoded GPT-6 Astra
    # (the closed-weight anchor every headline multiplier is quoted against) -> Qwen3.8-Flash-Next.
    by_name = {r["name"]: r for r in tiers}
    top, cheap = by_name[IPD_HI], by_name[IPD_LO]
    ipd_label = (f"{top['acc'] - cheap['acc']:.1f} pp accuracy lost, "
                 f"{top['usd'] / cheap['usd']:.0f}× cheaper")
    bracket(a, (cheap["usd"], cheap["acc"]), (top["usd"], top["acc"]), ipd_label,
            dx_pts=62, fontsize=19.3)

    # ---- IPJ: the open-weights models that have an energy figure
    withj = [m for m in OPEN_SHOWN if m["kj"] is not None]
    kjs = [m["kj"] for m in withj]
    x_lo = max(t for t in IPJ_TICKS if t <= min(kjs) / 1.2)
    x_hi = min(t for t in IPJ_TICKS if t >= max(kjs) * 1.4)
    # Pad the left limit below the snapped tick so that tick label does not sit on the axis
    # corner, where it would collide with the lowest y tick label.
    xlim_kj = [round(x_lo / 1.1, 3), x_hi]
    b.set_xscale("log"); b.set_xlim(*xlim_kj); b.set_ylim(Y_LO, Y_HI)
    front_b = draw_pareto(b, [(m["kj"], m["acc"]) for m in withj])
    for m in sorted(withj, key=lambda r: -area[r["name"]]):
        scatter_tier(b, m, m["kj"], area[m["name"]])
    ipj_ticks = [t for t in IPJ_TICKS if x_lo <= t <= x_hi]
    b.set_xticks(ipj_ticks)
    b.set_xticklabels([f"{t:g}" for t in ipj_ticks], fontsize=22.7)
    b.minorticks_off()
    b.set_yticks(range(Y_LO, Y_HI + 1, 10))
    b.set_yticklabels([f"{v}%" for v in range(Y_LO, Y_HI + 1, 10)], fontsize=22.7)
    b.set_xlabel("Energy per task (kJ, log)", fontsize=26.3, fontweight="bold")
    b.set_title("Intelligence per Joule (IPJ)", fontsize=31.5, fontweight="bold", pad=14)
    # Canonical: hardcoded Kimi K3 -> DeepSeek-V4-Flash, with the sign-aware lead below.
    # Target paper (author, 2026-09-25): hardcoded Moonshot Kimi K3 -> Qwen3.8-27B.
    hj, lj = by_name[IPJ_HI], by_name[IPJ_LO]
    _gap = hj["acc"] - lj["acc"]
    if abs(_gap) < 1.0:
        _lead = "Same accuracy"
    elif _gap > 0:
        _lead = f"{_gap:.1f} pp accuracy lost"
    else:
        _lead = f"{-_gap:.1f} pp more accurate"
    ipj_label = f"{_lead}, {hj['kj'] / lj['kj']:.1f}× less energy"
    # clear: the bracket's right tick stops at the apex of Kimi K3's triangle instead of inside it.
    CLEAR_J = 0.085
    bracket(b, (lj["kj"], lj["acc"]), (hj["kj"], hj["acc"]), ipj_label,
                      clear=CLEAR_J, dx_pts=30, fontsize=19.3)
    b.tick_params(labelsize=22.7)

    for ax in (a, b):
        ax.grid(True, color="#AAAAAA", alpha=0.25, linewidth=0.65)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # author note 2026-09-25: the 25%-larger legend takes its room from the panels (was 0.795 / 0.805), so the
    # figure keeps its width and the paper keeps its print scale
    fig.tight_layout(rect=(0, 0, 0.768, 1), w_pad=2.2)
    legends, anchor = [], (0.778, 0.985)
    for title, fams in LEGEND:
        rows = [r for f in fams for r in sorted((r for r in shown if r["family"] == f),
                                                key=lambda r: r["sides"])]
        box = dict(loc="upper left", ncol=1, frameon=True, facecolor="#EFEFEF",
                   edgecolor="black", framealpha=1.0, fontsize=LEG_FS, title=title,
                   title_fontsize=LEG_TFS, labelspacing=0.16, borderpad=0.45, handletextpad=0.5)
        if legends:
            box.update(bbox_to_anchor=_below(fig, legends[-1]), mode="expand", borderaxespad=0.0)
        else:
            box.update(bbox_to_anchor=anchor, columnspacing=1.4)
        leg = fig.legend(handles=[legend_handle(r) for r in rows], **box)
        leg.get_frame().set_linewidth(2.0)
        leg.get_title().set_fontweight("bold")
        legends.append(leg)
    listed = [t.get_text() for leg in legends for t in leg.get_texts()]
    if sorted(listed) != sorted(r["name"] for r in shown):
        raise SystemExit(f"legend lists {listed}, plotted tiers are {[r['name'] for r in shown]}")
    # Shape key, in the slot the canonical "Weights" box used, same box style and the canonical
    # legend-handle grey: sides = size rank within the company. Handles are passed as 3,5,4,6 so
    # the column-major fill reads row by row (smallest, 2nd / 3rd, 4th).
    ordinal = {3: "smallest", 4: "2nd", 5: "3rd", 6: "4th", 7: "5th", 8: "6th"}
    ranks = [(n, ordinal[n]) for n in range(3, max(r["sides"] for r in shown) + 1)]
    ranks = ranks[0::2] + ranks[1::2]
    keys = []
    for n, lab in ranks:
        path, ms = legend_marker(n)
        keys.append(Line2D([], [], marker=path, ls="", ms=ms, mfc="#9AA0A6", mec="black",
                           mew=1.0, label=lab))
    key = fig.legend(handles=keys, loc="upper left", ncol=2, frameon=True, facecolor="#EFEFEF",
                     edgecolor="black", framealpha=1.0, fontsize=LEG_FS,
                     title="Size rank in company", title_fontsize=LEG_TFS, labelspacing=0.16,
                     borderpad=0.45, handletextpad=0.5, columnspacing=1.6,
                     bbox_to_anchor=_below(fig, legends[-1]), mode="expand", borderaxespad=0.0)
    key.get_frame().set_linewidth(2.0)
    key.get_title().set_fontweight("bold")

    # Round 16 (author, 2026-09-25): centre the legend column vertically on the panels' plot area
    # (the axes' vertical extent, not the titles or axis labels). The column was stacked down from
    # a provisional top anchor; measure it, shift that anchor by the offset between the two
    # midpoints, and restack the boxes below it (box sizes and gaps unchanged). The anchors are in
    # fig.transSubfigure, as fig.legend sets them, so they follow the bbox_inches="tight" crop.
    column = legends + [key]
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    col_mid = (column[0].get_window_extent().transformed(inv).y1
               + column[-1].get_window_extent().transformed(inv).y0) / 2
    plot_mid = (min(ax.get_position().y0 for ax in (a, b))
                + max(ax.get_position().y1 for ax in (a, b))) / 2
    column[0].set_bbox_to_anchor((anchor[0], anchor[1] + plot_mid - col_mid), fig.transSubfigure)
    for prev, leg in zip(column, column[1:]):
        leg.set_bbox_to_anchor(_below(fig, prev), fig.transSubfigure)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    TIERS_OUT.write_text(json.dumps(dict(
        note=("Final tier table for Figure 4 (fig2_ipd_ipj_target.pdf), written by "
              "figures/fig2_ipd_ipj_target.py. usd = $/task on TerminalBench V2.1; kj = kJ/task "
              "(open-weight tiers only; closed-weight energy is not observable). Claude Fable 5.1 "
              "is on the IPD panel only (author, 2026-09-25); the oracle-routing series is "
              "omitted per the canonical TODO. Tiers with excluded=true are in the table but not "
              "plotted."),
        floor_pct=FLOOR,
        encoding=dict(
            color="company, from its logo (family_color)",
            shape=("regular polygon, sides = 3 + size rank within the family among plotted "
                   "tiers (3 triangle = smallest, 4 square, 5 pentagon, 6 hexagon), resting on a "
                   "flat edge; size = total params (open; ties by release order, older first) or "
                   "published output price (closed; size unpublished)"),
            area=("unchanged: total params (open) or output price (closed); each polygon drawn "
                  "with the area of the canonical circle at the same s"),
            dark_fill_keyline=sorted(DARK_FILL),
            y_axis=f"{Y_LO}% to {Y_HI}% on both panels; tiers below {Y_LO}% excluded",
            legend="one 'Models' box grouped by company, then the size-rank key",
            shape_key="'Size rank in company' legend box (grey handles), in the old Weights slot",
            palette_check=PALETTE_CHECK),
        family_color=FAMILY_COLOR,
        ipd=dict(frontier=front_a, annotation=ipd_label,
                 bracket=[top["name"], cheap["name"]]),
        ipj=dict(frontier=front_b, annotation=ipj_label, bracket=[hj["name"], lj["name"]],
                 xlim_kj=xlim_kj, xticks_kj=ipj_ticks,
                 kj_plotted_range=[min(kjs), max(kjs)]),
        tiers=tiers,
    ), indent=1, ensure_ascii=False) + "\n")

    print(f"  wrote {OUT}  (IPD {len(closed)} closed + {len(OPEN_SHOWN)} open, IPJ {len(withj)})")
    print(f"  IPD frontier: {front_a}")
    print(f"  IPJ frontier: {front_b}")
    print(f"  IPJ x axis: {xlim_kj[0]}-{xlim_kj[1]} kJ (ticks {ipj_ticks[0]}-{ipj_ticks[-1]}), "
          f"plotted {min(kjs):.1f}-{max(kjs):.1f} kJ")
    print(f"  IPD bracket {top['name']} -> {cheap['name']}: {ipd_label!r}")
    print(f"  IPJ bracket {hj['name']} -> {lj['name']}: {ipj_label!r}")
    print(f"  wrote {TIERS_OUT}")


if __name__ == "__main__":
    fig_ipd_ipj()
