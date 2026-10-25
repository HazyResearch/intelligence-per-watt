#!/usr/bin/env python3
"""Intelligence System composition: where the headline system's tasks, and its dollars, go.

Qwen3.8-27B -> GLM-5.3 -> GPT-6 Astra on TerminalBench V2.1 (plan "headline" in targets.json
tb21.plans). Every task starts on the first stage; each stage keeps what it solved, keeps what it got
wrong without noticing, and escalates the rest. Band widths are shares of all tasks, coloured by the
stage they leave (Figure 1's palette); hatched bands are work escalated (the stage was paid for and its
answer discarded). Each stage's two kept outcomes (solved, failed) get their own node, so every outcome
node receives exactly one band and no two bands cross; the outcomes are grouped by whether a
closed-source model ever sees the task (the stages before GPT-6 Astra, against GPT-6 Astra itself, in a
shaded box). Under the flow, a cost row gives each stage's contribution to the owned-hardware cost per
task, averaged over all tasks (plan()'s own cost terms: escalation premiums included, and each stage
charged for the judge call that reads its attempt), and the system's total and its multiple below
Claude Fable 5.1.

Every flow is recomputed here from the same variables plan() uses (targets/make_numbers.py, kind
"three"), and the script asserts that they reproduce PLANS["headline"] (accuracy, escalation rates,
per-stage shares, the share reaching GPT-6 Astra, and the owned cost) exactly, and checks the
GLM-5.3 / GPT-6 Astra split of the solved tasks against plan() itself. The first stage's machine is
read from the plan, never fixed here. Printed shares are rounded so that every printed total (and every
paper macro a reader can add them up to: accuracy, the local, middle and closed shares, the escalation
rate) is the sum of its printed parts and also its own rounding, shifting a part by 0.1 where needed;
likewise GPT-6 Astra's share of the cost is its own rounding (the paper's macro) and the three stage
shares sum to 100. The script prints every shifted part, and says so if a model state ever makes a
total impossible to match. It also asserts that no two bands overlap. The model runs on a throwaway
copy of targets/, so the figure PDF is the only file this script writes.
Output: paper/figures_and_tables/fig_task_flow.pdf (a 396 x 228 pt page, printed at 0.95\textwidth, 376.2 x 216.6 pt)
"""
from __future__ import annotations

import contextlib
import io
import itertools
import math
import re
import runpy
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.textpath import TextPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures_and_tables" / "fig_task_flow.pdf"

# ---- Style: Figure 4 (fig2_ipd_ipj_target.py) at one shared scale, as in fig_escalation_curves.py: every size
# below is Figure 4's native value x K on this figure's 396 pt (5.5 in) page. Both figures print at
# 0.95\textwidth = 376.2 pt (round 16), this one at 0.95 of native and Figure 4 at 376.2/1486.61 of native
# (K x 0.95 to within 0.05%), so in-paper sizes are 0.95 x the values below: headings 6.66, labels 5.75,
# annotations 4.88 pt (TITLE is unused here).
K = 396 / 1485.88
TITLE, LABEL, TICK = 31.5 * K, 26.3 * K, 22.7 * K          # 8.40, 7.01, 6.05 pt
NOTE = 19.3 * K                                             # 5.14 pt: annotations
FRAME_LW, THIN = 2.0 * K, 1.0 * K                           # 0.533, 0.267 pt
HALO = [pe.withStroke(linewidth=1.4, foreground="white")]   # keeps hatching off escalated-band labels
SEP_LW = 0.45                                               # white rule between flows that share an edge

# ---- Palette: Figure 1's colours with Figure 1's meanings; Figure 4's legend fill and handle grey.
INK = "#111318"
GREEN = "#16a34a"      # open weights, local (Qwen3.8-27B)
ORANGE = "#d97706"     # open weights on owned accelerators (GLM-5.3)
PURPLE = "#7c3aed"     # closed-source provider (GPT-6 Astra)
BLUE = "#0072b2"       # a composed Intelligence System
G_MID = "#61676e"      # Figure 1 column-header grey
G_LIGHT = "#9aa0a6"    # Figure 4 legend-handle grey
LEG_FACE = "#EFEFEF"   # Figure 4 legend fill
BUBBLE_FACE = "#F1F3F5"  # the cost bubble: Figure 1's light legend grey
BAND_ALPHA = 0.36
HATCH = "//////"

W = 396.0                                                   # page width, pt (printed at 0.95 of it)


def model():
    """The plan model (targets/make_numbers.py), run on a throwaway copy of targets/.

    make_numbers.py rewrites targets/derived.json and paper/ipj-numbers-target.tex relative to its
    own location, so running it in place would touch generated files; the copy keeps this script's
    only output the figure PDF. If its own checks fail (it exits), say so with its last output."""
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(ROOT / "targets", Path(tmp) / "targets", ignore=shutil.ignore_patterns("workflow_raw"))
        (Path(tmp) / "paper").mkdir()
        try:
            with contextlib.redirect_stdout(buf):
                ns = runpy.run_path(str(Path(tmp) / "targets" / "make_numbers.py"), run_name="mn")
        except (SystemExit, Exception) as e:
            raise RuntimeError(f"targets/make_numbers.py failed ({e!r}); the end of its output:\n"
                               + buf.getvalue()[-3000:]) from e
    return ns


def flows(ns) -> dict:
    """plan()'s kind "three" branch, written out with the model's own variables."""
    sig, tb = ns["sig"], ns["tb"]
    p = tb["plans"]["headline"]
    assert p["kind"] == "three" and p["closed"] == "astra" and p["local"] == "qwen38", p
    loc = ns["MACH"][p["machine"]]             # the first stage's machine, as the plan names it
    L = loc["acc"]
    Dt = loc.get("to", 0.0)                    # solves lost to the time budget (always flagged)
    Lq = L + Dt
    r1, f1, r2, f2 = sig["s1_recall"], sig["s1_fpr"], sig["s2_recall"], sig["s2_fpr"]
    fe, caught = L * f1, (1 - Lq) * r1         # solved but flagged; failed and flagged
    E1 = fe + caught + Dt                      # escalated to GLM-5.3
    kept = L - fe                              # solved by Qwen3.8-27B and kept
    silent = (1 - L) - caught - Dt             # wrong, not escalated
    g_c = sig["glm_solve_caught"]
    c3 = sig["astra3_solve_doubly_failed"]
    gc = fe * sig["glm_solve_false_esc"] + caught * g_c + Dt * sig.get("timeout_glm_solve", g_c)
    gf = E1 - gc
    E2 = gf * r2 + gc * f2                     # escalated to GPT-6 Astra
    glm_kept = gc * (1 - f2)                   # solved by GLM-5.3 and kept
    glm_silent = gf * (1 - r2)                 # wrong at GLM-5.3, not escalated
    astra_solved = gc * f2 * sig["closed3_solve_flagged_correct"] + gf * r2 * c3
    astra_wrong = E2 - astra_solved
    acc = kept + glm_kept + astra_solved

    # ---- dollars per task on the owned basis, plan()'s cost terms split by the stage that incurs them:
    # the first stage and the judge reading its attempt on every task; GLM-5.3 (at its escalation
    # premium) and the judge reading its attempt on the E1 tasks; GPT-6 Astra at its premium on E2
    judge = sig["judge_own"]
    g = tb["tiers"]["glm53"]
    cost = dict(qwen=loc["own"] + judge,
                glm=E1 * (g["own"] * ns["PG"] + judge),
                astra=E2 * ns["closed_usd"]("astra") * ns["PC3"]("astra"))   # round 23: the doubly escalated set's own premium
    cost["total"] = cost["qwen"] + cost["glm"] + cost["astra"]

    # ---- the flows must be plan()'s headline, exactly
    h = ns["PLANS"]["headline"]
    tol = 1e-12
    assert abs(acc - h["acc"]) < tol, (acc, h["acc"])
    assert abs(E1 - h["E1"]) < tol and abs(kept - h["kept"]) < tol and abs(silent - h["silent"]) < tol
    shares = (kept + silent, glm_kept + glm_silent, E2)
    assert all(abs(a - b) < tol for a, b in zip(shares, h["shares"])), (shares, h["shares"])
    assert abs(E2 - h["closed"]) < tol and abs(1 + E1 - h["calls"]) < tol
    assert abs(cost["total"] - h["cost"]["own"]) < tol, (cost, h["cost"]["own"])
    # every task ends in exactly one outcome
    assert abs(kept + silent + E1 - 1) < tol and abs(glm_kept + glm_silent + E2 - E1) < tol
    assert abs(acc + silent + glm_silent + astra_wrong - 1) < tol
    # the GLM-5.3 / GPT-6 Astra split of the solved tasks, from plan() itself: with GPT-6 Astra's
    # two solve rates zeroed, plan() must credit exactly the tasks kept by the first two stages
    saved = {k: sig[k] for k in ("closed3_solve_flagged_correct", "astra3_solve_doubly_failed")}
    try:
        sig.update({k: 0.0 for k in saved})
        assert abs(ns["plan"](p)["acc"] - (kept + glm_kept)) < tol
    finally:
        sig.update(saved)
    assert abs(ns["plan"](p)["acc"] - h["acc"]) < tol
    anchor = ns["ANCHOR"]
    mach = tb["machines"][p["machine"]]
    return dict(kept=kept, silent=silent, E1=E1, glm_kept=glm_kept, glm_silent=glm_silent, E2=E2,
                astra_solved=astra_solved, astra_wrong=astra_wrong, acc=acc, shares=shares, cost=cost,
                mult=anchor["usd"] / cost["total"], anchor=anchor, machine=p["machine"],
                machine_label=mach["label"], machine_kind=mach.get("kind"),
                glm_gpus=tb["machine_model"]["glm53"].get("ngpu"))


# ---------------------------------------------------------------- printed numbers
def apportion(vals, step, totals):
    """Round each of vals to a multiple of step (down or up), choosing the combination that makes the
    most printed totals equal their own rounding, then the smallest total rounding error.
    totals: [(indices, target)] with target already a multiple of step. Returns (printed, misses)."""
    best = None
    for pick in itertools.product((0, 1), repeat=len(vals)):
        shown = [(math.floor(v / step + 1e-9) + k) * step for v, k in zip(vals, pick)]
        miss = [t for idx, t in totals if abs(sum(shown[i] for i in idx) - t) > step / 2]
        key = (len(miss), sum(abs(s - v) for s, v in zip(shown, vals)))
        if best is None or key < best[0]:
            best = (key, shown, miss)
    return best[1], best[2]


def _shifted(names, vals, shown, step, fmt):
    """The printed parts that are not their own rounding, for the run log."""
    return [f"{n} {v:{fmt}} printed {s:{fmt}}" for n, v, s in zip(names, vals, shown)
            if abs(s - round(v / step) * step) > step / 2]


def shown_shares(F):
    """The six outcome flows in percent at one decimal, such that every total printed in the figure or
    readable off it as a paper macro (accuracy, the share never reaching a closed-source model, the
    local, middle and closed shares, the escalation rate) is both the sum of its printed parts and its
    own rounding."""
    keys = ("kept", "silent", "glm_kept", "glm_silent", "astra_solved", "astra_wrong")
    v = [100 * F[k] for k in keys]
    r = lambda x: float(f"{x:.1f}")
    ix = {k: i for i, k in enumerate(keys)}
    groups = {
        "all": keys, "acc": ("kept", "glm_kept", "astra_solved"), "local": ("kept", "silent"),
        "E1": ("glm_kept", "glm_silent", "astra_solved", "astra_wrong"), "mid": ("glm_kept", "glm_silent"),
        "E2": ("astra_solved", "astra_wrong"), "never": ("kept", "silent", "glm_kept", "glm_silent"),
    }
    totals = [([ix[k] for k in g], r(sum(v[ix[k]] for k in g))) for g in groups.values()]
    shown, miss = apportion(v, 0.1, totals)
    out = {k: shown[ix[k]] for k in keys}
    for name, g in groups.items():
        out[name] = sum(out[k] for k in g)
    out["_shifted"] = _shifted(keys, v, shown, 0.1, ".3f")
    if miss:
        print(f"  note: {len(miss)} printed total(s) differ from their own rounding by 0.1")
    return out


def pct(x):
    return f"{x:.1f}%"


def shown_costs(cost, usd):
    """Stage costs printed with the total's decimals (one more if a stage would print as zero), summing
    to the printed total; and whole-percent shares of the total summing to 100, GPT-6 Astra's being its
    own rounding (the paper's macro for the closed stage's share of the owned cost)."""
    tot_s = usd(cost["total"])
    nd = len(tot_s.split(".")[1]) if "." in tot_s else 0
    parts = [cost[k] for k in ("qwen", "glm", "astra")]
    while True:
        step = 10.0 ** -nd
        shown, _ = apportion(parts, step, [([0, 1, 2], round(cost["total"], nd))])
        if min(shown) > 0 or nd >= 5:
            break
        nd += 1
    raw = [100 * c / cost["total"] for c in parts]
    share, miss = apportion(raw, 1.0, [([2], float(round(raw[2]))), ([0, 1, 2], 100.0)])
    if miss:
        print("  note: the cost shares cannot both sum to 100 and keep GPT-6 Astra's own rounding")
    shifted = _shifted(("qwen", "glm", "astra"), raw, share, 1.0, ".2f")
    return [f"${s:.{nd}f}" for s in shown], [f"{s:.0f}%" for s in share], f"${tot_s}", shifted


def plain(label: str) -> str:
    """A targets.json label (LaTeX) as plain figure text."""
    s = label.replace("$\\times$", "\u00d7").replace("\\times", "\u00d7").replace("\\,", " ").replace("~", " ")
    s = s.replace("\\_", "_").replace("\\&", "&").replace("\\%", "%")
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return s.replace("$", "").replace("{", "").replace("}", "")


# ---------------------------------------------------------------- drawing helpers (page pt)
def _smooth(t):
    return t * t * (3 - 2 * t)


BANDS = []                                     # (name, xs, top, h) of every band drawn, for the overlap check


def band(ax, xa, xb, ya, yb, h, color, *, tr=None, hatch=None, z=2, sep=True, alpha=BAND_ALPHA, name=""):
    """A flow of constant vertical thickness h whose top edge moves from ya (at xa) to yb (at xb),
    easing over the x-range tr (default: the whole band). A thin white rule on the top edge keeps
    flows that share an edge apart."""
    t0, t1 = tr if tr else (xa, xb)
    xs = np.linspace(xa, xb, 400)
    top = ya + (yb - ya) * _smooth(np.clip((xs - t0) / (t1 - t0), 0, 1))
    pts = np.vstack([np.c_[xs, top], np.c_[xs[::-1], (top - h)[::-1]]])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=alpha, edgecolor="none", lw=0, zorder=z))
    if hatch:
        ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor=color, lw=0, hatch=hatch,
                             alpha=0.9, zorder=z + 0.1))
    if sep:
        ax.plot(xs, top, color="white", lw=SEP_LW, solid_capstyle="butt", zorder=z + 0.2)
    BANDS.append((name, xs, top, h))
    return lambda x: float(np.interp(x, xs, top))       # top edge at x


def assert_no_overlap(tol=0.02):
    """No two bands share any area (they may touch): the figure has no crossings."""
    for (n1, x1, t1, h1), (n2, x2, t2, h2) in itertools.combinations(BANDS, 2):
        lo, hi = max(x1[0], x2[0]), min(x1[-1], x2[-1])
        if hi - lo <= 1e-6:
            continue
        xs = np.linspace(lo + 1e-6, hi - 1e-6, 600)
        a_top, b_top = np.interp(xs, x1, t1), np.interp(xs, x2, t2)
        ov = np.minimum(a_top, b_top) - np.maximum(a_top - h1, b_top - h2)
        assert ov.max() <= tol, f"bands {n1} and {n2} overlap by {ov.max():.2f} pt"


def node(ax, x, ytop, h, color, w):
    r = min(1.1, h / 2)
    ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=color, edgecolor="black", linewidth=THIN, zorder=5))


def _prop(size, bold):
    return FontProperties(family="DejaVu Sans", size=size, weight="bold" if bold else "normal")


def _ink(s, size, bold=False):
    """Vertical ink extent (y0, y1) of a one-line string, in pt relative to its baseline
    (fig1_overview_target.py's helper: centre the glyph ink, not the font box)."""
    e = TextPath((0, 0), s, size=size, prop=_prop(size, bold)).get_extents()
    return e.y0, e.y1


def _adv(s, size, bold=False):
    """Advance width of a one-line string, in pt (trailing spaces included)."""
    tp = TextPath((0, 0), s + "|", size=size, prop=_prop(size, bold))
    bar = TextPath((0, 0), "|", size=size, prop=_prop(size, bold))
    return tp.get_extents().x1 - bar.get_extents().x1


def main():
    ns = model()
    F = flows(ns)
    P = shown_shares(F)
    cost_s, cost_share, total_s, cost_shifted = shown_costs(F["cost"], ns["usd"])
    mult_s = ns["mult"](F["mult"]).replace("{,}", ",") + "\u00d7"

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.unicode_minus": True, "hatch.linewidth": 0.35})

    # ---------------------------------------------------------------- vertical layout (pt, from the top)
    S = 104.0                                  # pt per 100% of tasks
    g1 = 5.5                                   # between outcomes within a group
    G = 20.0                                   # between the two groups (room for the second header)
    HEAD_NAME, HEAD_SUB = 7.4, 14.6            # stage-name and hardware baselines, below the top
    FLOW_TOP = 20.5                            # top of the flow
    span = S + 4 * g1 + G                      # outcome column: 6 nodes, 4 inner gaps, 1 group gap
    ZONE_PAD = 4.0                             # the closed-source box below its last outcome
    # author note (2026-09-25): a full-width rule, then the cost breakdown in its own rounded light-grey bubble
    SEP_Y = FLOW_TOP + span + ZONE_PAD + 5.5   # rule between the diagram and the cost section
    BUB_TOP = SEP_Y + 4.5                      # cost bubble, top
    COST_RULE = BUB_TOP + 7.0                  # cost-row heading (centre)
    COST_VAL = COST_RULE + 10.5                # cost-row baseline
    COST_BAR = COST_VAL + 2.4                  # cost bars, top
    BAR_H = 3.6
    BUB_BOT = COST_BAR + BAR_H + 4.5           # cost bubble, bottom
    LEG_TOP = BUB_BOT + 5.5
    LEG_H = 12.5
    H = LEG_TOP + LEG_H + 1.5

    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    yt = lambda d: H - d                       # distance from the top -> page y

    def text(x, y, s, size=NOTE, bold=False, color=INK, ha="center", va="center", halo=False, **kw):
        if va == "center":                     # centre the ink, so a label sits mid-band
            y0, y1 = _ink(s, size, bold)
            y, va = y - (y0 + y1) / 2, "baseline"
        kw = dict(fontsize=size, fontweight="bold" if bold else "normal", ha=ha, va=va,
                  zorder=kw.pop("zorder", 8), **kw)
        if halo:                               # a white copy under the label (the label stays text)
            ax.text(x, y, s, color="white", path_effects=HALO, **{**kw, "zorder": kw["zorder"] - 0.1})
        return ax.text(x, y, s, color=color, **kw)

    def runs(x, y, parts, size, va="center"):
        """Left-aligned text made of (string, bold) runs, centred as one line."""
        if va == "center":
            y0 = min(_ink(s, size, b)[0] for s, b in parts)
            y1 = max(_ink(s, size, b)[1] for s, b in parts)
            y = y - (y0 + y1) / 2
        for s, b in parts:
            ax.text(x, y, s, fontsize=size, fontweight="bold" if b else "normal", color=INK, ha="left",
                    va="baseline", zorder=8)
            x += _adv(s, size, b)
        return x

    # ---------------------------------------------------------------- horizontal layout
    PL, PR = 8.0, 388.0
    NW = 4.4                                   # node width
    OUT_GAP = 3.2                              # outcome node to its label
    leaf = [  # (share, text, node colour): each stage's kept outcomes, then GPT-6 Astra's
        (P["kept"], "solved by Qwen3.8-27B", BLUE),
        (P["silent"], "failed by Qwen3.8-27B", G_LIGHT),
        (P["glm_kept"], "solved by GLM-5.3", BLUE),
        (P["glm_silent"], "failed by GLM-5.3", G_LIGHT),
        (P["astra_solved"], "solved by GPT-6 Astra", BLUE),
        (P["astra_wrong"], "failed by GPT-6 Astra", G_LIGHT),
    ]
    head_a = [(pct(P["never"]), True), (" never reach a closed-source model", False)]
    head_b = [(pct(P["E2"]), True), (" reach GPT-6 Astra", False)]
    anchor = re.sub(r"\s*\(.*\)\s*$", "", F["anchor"]["label"])      # "Claude Fable 5.1"
    total_run = [("= " + total_s, True), (" per task, ", False), (mult_s, True), (f" below {anchor}", False)]
    lab_w = max([_adv(pct(v) + " ", TICK, True) + _adv(t, TICK) for v, t, _ in leaf] +
                [sum(_adv(s, TICK, b) for s, b in hd) for hd in (head_a, head_b)])
    X1 = PL
    # outcome labels end at PR, and so does the cost row's total, which starts under the outcome nodes
    X4 = min(PR - lab_w - OUT_GAP - NW, PR - sum(_adv(s, TICK, b) for s, b in total_run) + _adv("= ", TICK, True))
    X2 = X1 + 0.36 * (X4 - X1)
    X3 = X1 + 0.70 * (X4 - X1)

    a, sl, k2, s2 = F["kept"], F["silent"], F["glm_kept"], F["glm_silent"]
    b, e2, k3, w3 = F["E1"], F["E2"], F["astra_solved"], F["astra_wrong"]

    # outcome column (page y of each node's top): the first stage's solved and failed, GLM-5.3's solved
    # and failed, then (after the group gap) GPT-6 Astra's solved and failed
    Y0 = yt(FLOW_TOP)
    tops, y = [], Y0
    for i, h in enumerate((a, sl, k2, s2, k3, w3)):
        tops.append(y)
        y -= h * S + (G if i == 3 else g1)
    R1, R2, R3, R4, R5, R6 = tops
    Y_BOT = R6 - w3 * S
    # stage nodes: each one's outputs, top to bottom, are its solved, failed and escalated bands
    Q_TOP = Y0
    G_TOP = R3                                 # GLM-5.3 level with its solved outcome: that band runs flat
    A_TOP = (R5 + Y_BOT) / 2 + e2 * S / 2      # GPT-6 Astra, centred on its two outcomes

    # ---- bands, back to front; each eases early so it has reached its outcome's height before the next
    # stage's node (the first stage's failures pass above GLM-5.3, GLM-5.3's above GPT-6 Astra)
    TR = 30.0
    band(ax, X1 + NW, X4, Q_TOP, R1, a * S, GREEN, sep=False, name="qwen kept")
    band(ax, X1 + NW, X4, Q_TOP - a * S, R2, sl * S, GREEN, tr=(X1 + NW, X2), name="qwen failed")
    top_e1 = band(ax, X1 + NW, X2, Q_TOP - (a + sl) * S, G_TOP, b * S, GREEN, hatch=HATCH, name="E1")
    band(ax, X2 + NW, X4, G_TOP, R3, k2 * S, ORANGE, name="glm kept")
    band(ax, X2 + NW, X4, G_TOP - k2 * S, R4, s2 * S, ORANGE, tr=(X2 + NW, X2 + NW + TR), name="glm failed")
    # the hatched band to GPT-6 Astra eases over most of its length, so it never runs steep enough to
    # look thinner than its share
    E2_TR = (X2 + NW, X2 + NW + 0.72 * (X3 - X2 - NW))
    band(ax, X2 + NW, X3, G_TOP - (k2 + s2) * S, A_TOP, e2 * S, ORANGE, hatch=HATCH, tr=E2_TR, name="E2")
    band(ax, X3 + NW, X4, A_TOP, R5, k3 * S, PURPLE, name="astra solved")
    band(ax, X3 + NW, X4, A_TOP - k3 * S, R6, w3 * S, PURPLE, name="astra failed")
    assert_no_overlap()

    # ---- the tasks a closed-source model sees: a faint purple box behind GPT-6 Astra and its outcomes
    zb_top = R5 + 13.5
    ax.add_patch(FancyBboxPatch((X3 - 2.0, Y_BOT - ZONE_PAD), PR + 2.0 - (X3 - 2.0), zb_top - (Y_BOT - ZONE_PAD),
                                boxstyle="round,pad=0,rounding_size=2.5", facecolor=PURPLE, alpha=0.07,
                                edgecolor="none", zorder=0.5))
    # ---- and the tasks none does: a neutral box behind the first group of outcomes
    za_bot = R4 - s2 * S - 3.0
    ax.add_patch(FancyBboxPatch((X4 - 2.0, za_bot), PR + 2.0 - (X4 - 2.0), Y0 + 1.8 - za_bot,
                                boxstyle="round,pad=0,rounding_size=2.5", facecolor=G_LIGHT, alpha=0.10,
                                edgecolor="none", zorder=0.5))
    assert za_bot > zb_top + 1.0, "the two outcome groups' boxes touch"

    # ---- nodes
    node(ax, X1, Q_TOP, S, GREEN, NW)
    node(ax, X2, G_TOP, b * S, ORANGE, NW)
    node(ax, X3, A_TOP, e2 * S, PURPLE, NW)
    sizes = (a, sl, k2, s2, k3, w3)
    for (v, t, c), top, h in zip(leaf, tops, sizes):
        node(ax, X4, top, h * S, c, NW)

    # ---- stage headers: name, and where the stage runs (the first stage's machine from the plan)
    gpus = F["glm_gpus"]
    first = plain(F["machine_label"]) + ("" if F["machine_kind"] == "desk" else ", owned")
    subs = (first, f"{gpus}\u00d7H200, owned" if gpus else "owned accelerators", "provider API")
    for x, name, sub in ((X1, "Qwen3.8-27B", subs[0]), (X2, "GLM-5.3", subs[1]), (X3, "GPT-6 Astra", subs[2])):
        text(x, yt(HEAD_NAME), name, LABEL, bold=True, color="black", ha="left", va="baseline")
        text(x, yt(HEAD_SUB), sub, NOTE, color=G_MID, ha="left", va="baseline")

    # ---- escalation labels: on the wide band from the first stage; under the narrow one to GPT-6 Astra,
    # right-aligned where it has levelled off before the node
    xe = (X1 + NW + X2) / 2
    text(xe, top_e1(xe) - b * S / 2, f"{pct(P['E1'])} escalated", zorder=9, halo=True)
    lab_e2 = f"{pct(P['E2'])} escalated"
    y0, y1 = _ink(lab_e2, NOTE)
    text(X3 - 3.0, A_TOP - e2 * S - 1.6 - y1, lab_e2, ha="right", va="baseline", zorder=9)

    # ---- outcomes: group headers and one label per node
    xo = X4 + NW + OUT_GAP
    text(X4, yt(HEAD_NAME), "Outcome", LABEL, bold=True, color="black", ha="left", va="baseline")
    text(X4, yt(HEAD_SUB), f"{pct(P['acc'])} solved overall", NOTE, color=G_MID, ha="left", va="baseline")
    for hd, yb in ((head_a, R1 - 5.2), (head_b, R5 + 6.0)):
        runs(xo, yb, hd, TICK, va="baseline")
        ax.plot([xo, PR], [yb - 2.0, yb - 2.0], color=G_LIGHT, lw=THIN * 1.5, zorder=1)
    for (v, t, c), top, h in zip(leaf, tops, sizes):
        runs(xo, top - h * S / 2, [(pct(v) + " ", True), (t, False)], TICK)

    # ---- cost row: what each stage adds to the owned cost per task, averaged over all tasks
    rule_y = yt(COST_RULE)
    c_title, c_tail = "Cost per task", "on owned hardware, averaged over all tasks (share of the total)"
    ty0, ty1 = _ink(c_title, LABEL, True)
    base = rule_y - (ty0 + ty1) / 2
    text(PL, base, c_title, LABEL, bold=True, color="black", ha="left", va="baseline")
    x_tail = PL + _adv(c_title + " ", LABEL, True)
    text(x_tail, base, c_tail, NOTE, color=G_MID, ha="left", va="baseline")
    ax.plot([PL - 4.5, PR + 4.5], [yt(SEP_Y), yt(SEP_Y)], color=G_MID, lw=THIN * 1.7, solid_capstyle="butt",
            zorder=1)                            # separates the diagram above from the cost section
    ax.add_patch(FancyBboxPatch((PL - 4.5, yt(BUB_BOT)), PR - PL + 9.0, BUB_BOT - BUB_TOP,
                                boxstyle="round,pad=0,rounding_size=3.0", facecolor=BUBBLE_FACE,
                                edgecolor="none", linewidth=0, zorder=0.5))   # the cost bubble
    bar_L = X4 - X3 - 8.0                      # bar length for 100% of the cost
    for i, (x, c) in enumerate(((X1, GREEN), (X2, ORANGE), (X3, PURPLE))):
        pre = "" if i == 0 else "+ "
        runs(x - _adv(pre, TICK, True), yt(COST_VAL), [(pre + cost_s[i], True), (f" ({cost_share[i]})", False)],
             TICK, va="baseline")
        share = F["cost"][("qwen", "glm", "astra")[i]] / F["cost"]["total"]
        ax.add_patch(Rectangle((x, yt(COST_BAR) - BAR_H), bar_L, BAR_H, facecolor="white", edgecolor="none",
                               zorder=4))                    # the track: 100% of the cost (white on the bubble)
        ax.add_patch(Rectangle((x, yt(COST_BAR) - BAR_H), max(bar_L * share, 0.6), BAR_H, facecolor=c,
                               edgecolor="black", lw=THIN, zorder=5))
    runs(X4 - _adv("= ", TICK, True), yt(COST_VAL), total_run, TICK, va="baseline")
    ax.add_patch(Rectangle((X4, yt(COST_BAR) - BAR_H), bar_L, BAR_H, facecolor=BLUE, edgecolor="black",
                           lw=THIN, zorder=5))

    # ---- legend: one row in a Figure 4 box, under the cost row, every label on one baseline
    local = "Local machine" if F["machine_kind"] == "desk" else "Local model"
    items = [("sq", GREEN, local), ("sq", ORANGE, "Owned accelerators"),
             ("sq", PURPLE, "Closed-source provider"), ("hatch", G_MID, "Escalated: paid for, answer discarded")]
    LS, SQ, PADX, ITEM_GAP, SW_GAP = TICK, 5.6, 6.0, 10.0, 3.0
    wid = sum(SQ + SW_GAP + _adv(t, LS) for _, _, t in items) + ITEM_GAP * (len(items) - 1)
    bx0 = (PL + PR) / 2 - wid / 2 - PADX
    by1 = yt(LEG_TOP)
    by0 = by1 - LEG_H
    ax.add_patch(FancyBboxPatch((bx0, by0), wid + 2 * PADX, LEG_H, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=LEG_FACE, edgecolor="black", linewidth=FRAME_LW, zorder=6))
    ym = (by0 + by1) / 2
    hy0, hy1 = _ink("Hx", LS)
    ybl = ym - (hy0 + hy1) / 2                 # one baseline for the row: centre the capital ink on the box
    x = bx0 + PADX
    for kind, c, t in items:
        if kind == "sq":
            ax.add_patch(Rectangle((x, ym - SQ / 2), SQ, SQ, facecolor=c, edgecolor="black", lw=THIN, zorder=7))
        else:
            ax.add_patch(Rectangle((x, ym - SQ / 2), SQ, SQ, facecolor="white", edgecolor="black", lw=THIN,
                                   zorder=7))
            ax.add_patch(Rectangle((x, ym - SQ / 2), SQ, SQ, facecolor="none", edgecolor=c, lw=0, hatch=HATCH,
                                   zorder=7.1))
        x += SQ + SW_GAP
        text(x, ybl, t, LS, color="black", ha="left", va="baseline", zorder=7)
        x += _adv(t, LS) + ITEM_GAP

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)          # fixed page, no tight bbox; printed at 0.95\textwidth: in-paper pt = 0.95 x native pt
    plt.close(fig)
    print(OUT)
    print(f"  page {W:.1f} x {H:.1f} pt; first stage on {F['machine']} ({first}; legend '{local}')")
    for k in ("kept", "silent", "E1", "glm_kept", "glm_silent", "E2", "astra_solved", "astra_wrong", "acc"):
        print(f"  {k:13s} {100 * F[k]:7.3f}%  printed {P.get(k, float('nan')):5.1f}")
    print(f"  printed groups: never {P['never']:.1f}, reach {P['E2']:.1f}, acc {P['acc']:.1f}, "
          f"local {P['local']:.1f}, mid {P['mid']:.1f}, escalated {P['E1']:.1f}")
    print(f"  shares printed off their own rounding: {P['_shifted'] or 'none'}")
    print(f"  cost {cost_s} {cost_share} = {total_s}, {mult_s} below the anchor; "
          f"shares off their own rounding: {cost_shifted or 'none'}")
    print(f"  flow {Y_BOT:.1f} to {Y0:.1f} pt, {S:.1f} pt per 100% of tasks; X {X1:.1f} {X2:.1f} {X3:.1f} {X4:.1f}")


if __name__ == "__main__":
    main()
