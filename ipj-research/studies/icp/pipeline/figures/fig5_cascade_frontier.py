"""F5 — the escalate-only cascade frontier: accuracy vs $/task.

TARGET-PAPER COPY (ICP_TARGET_PAPER, 2026-09-25) of
intelligence-per-joule/ipj/figures/blog-post-ipd/fig5_cascade_frontier.py, which is left untouched.
Run unmodified, that script reproduces the paper's fig5_cascade_frontier.pdf byte for byte except
its CreationDate, so this copy starts from the true generator. It changes only what the paper's
TODO for fig:cascade asks:
  * no Claude Fable 5 accuracy point (marker and "Claude / Fable 5 / 80.4%" label removed, and the
    anchor dropped from the Pareto input; it was dominated, so the frontier and shading are unchanged);
  * no "matches Fable 5 within noise, 23× cheaper" bracket (label box and its connector lines);
  * no "N× cheaper" multiples on the frontier labels. They were computed against the S1 anchor in
    per_task_vectors_v2.json (80.4% at $4.50), so they disagreed with the paper's $3.36 anchor and
    its cascade table; each label now reads "<local> → <cloud>" over "<acc>%".
Data, axes, figure size, fonts, colours and every other mark are unchanged. Data and paper_style
are read in place from the canonical directory (read-only; no bytecode is written there). The PDF is
written to paper/figures_and_tables/; a PNG preview is written only if FIG5_PNG_DIR is set.

The canonical docstring follows. Where it mentions the Fable anchor, the bracket or "× cheaper"
labels, it describes the canonical figure, not this copy.

Replaces the §4.2 table + price-board pair with one chart. Same visual language as the IPD/IPJ
panels: log-x, dotted Pareto frontier over the shaded better-than region, and an improvement
bracket on the frontier's top cascade.

SCOPE (Jon, 2026-08-14): escalate-on-failure ONLY. The parallel_any_of ("run both, take either")
diamonds were dropped — this figure now argues one thing, that running a cheap local model first
and escalating its observed failures buys most of the frontier model's accuracy for a fraction of
the price. Run-both numbers remain in cascade_grid.json for anyone who wants them.

BASE MODEL = GLM-5.2 (changed 2026-08-29). All four local tiers are now priceable: GLM-5.2's energy
was measured 2026-08-15 (453.7 kJ/task, 8 boards, 99.3% active, all five gates), which unblocked the
ADR-0004 dollar cost that had been cost_blocked_on since the board's first release. With that in
hand GLM-5.2 dominates the cascade frontier outright: every Pareto point above 66.3% is a GLM
cascade, and the Qwen3.6-27B family this figure previously plotted is dominated at every price.
GLM -> GPT-5.6-sol reaches 84.3% at $0.1965 (22.9x cheaper than Fable 5), which matches the Fable
anchor (80.4%, S1 per ADR-0005) to within sampling noise. Say "matches within noise", never "beats".

CAVEAT ON THE BASE TIER. The board's local_energy_verification note for GLM_local records that its
energy run resolved 57/89 while the spine's GLM accuracy is 58/89 from a different run, so this
row pairs energy and accuracy measured one task apart. Every headline number in this figure now
rests on that cost, so the pairing should be closed before submission.

UNCAPPED (Jon, 2026-08-14). cascade_grid.json's deploy_honest cells apply a $0.225/task budget
ceiling — 20x cheaper than Fable 5 — and 7 of the 10 hit it (budget_trimmed: true), escalating only
17-32 of their 35 candidates. That made every expensive target cost the same $0.213-$0.224 and read
5-11 pp below its true accuracy: an artifact of the ceiling, not a property of the model. Worse, the
blog spec (decision D) drops budget framing entirely, so a capped axis smuggles back a frame that was
deliberately cut. This figure therefore recomputes deploy_honest with NO ceiling, straight from the
per-task vectors. Uncapped, the ten span $0.010-$0.772 (76x) instead of piling up at one price.

The reconstruction is validated against the grid: for every pair the grid did NOT trim, recomputed
(acc, cost) must match its published cell to 1e-4 or the build aborts. See _validate_against_grid.

COST BASIS IS MIXED, and that is disclosed rather than papered over. Escalated tasks are the ones
the local tier timed out on — the hard ones — and clouds cost 1.3-1.8x their all-task mean there.
Seven tiers are priced from real per-task costs. Three cannot be, per the board README:
    deepseek_v4f  0/89 per-task costs               -> flat mean, UNDERSTATED
    opus5         0/89 (Bedrock cache under-report) -> flat mean, UNDERSTATED
    gpt56_terra   86/89 but litellm-mispriced       -> flat mean, UNDERSTATED
Two of those (deepseek, terra) sit ON the frontier, so the frontier's cheap end is priced by the
weaker method. They are drawn with a dashed edge and carry a dagger. Inventing a premium multiplier
for them from other models' data would be worse than disclosing the gap.

DATA — every row read from the committed artifacts, not retyped:
    per_task_vectors_v2.json  per-task solved/errored/cost_usd; fable_reference; the cost basis
    cascade_grid.json         local_dollars_per_task (ADR-0004); validation target
Both are byte-identical to ScalingIntelligence/ipj-tb21-board as of 2026-08-18.
"""
from __future__ import annotations

import json
import os
import sys

# Canonical generator directory: paper_style (+ style) and data/ are read from here, never written.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # ICP_TARGET_PAPER
CANON = HERE  # Reviewed generator dependencies are bundled beside this script.
sys.dont_write_bytecode = True     # do not drop __pycache__ into the canonical repo
sys.path.insert(0, CANON)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paper_style as ps
from paper_style import CASCADE, INK, INK2, OPEN_CLOUD

OUT = os.path.join(ROOT, "paper", "figures_and_tables")
PNG_DIR = os.environ.get("FIG5_PNG_DIR")    # optional preview PNG, off by default
DATA = os.path.join(CANON, "data")

LOCAL = "GLM_local"
LOCAL_PRETTY = "GLM-5.2"

PRETTY = {"gpt56_sol": "GPT-5.6-sol", "gpt56_terra": "GPT-5.6-Terra",
          "deepseek_v4f": "DeepSeek-V4-Flash", "kimi_k3": "Kimi K3", "gpt55": "GPT-5.5",
          "opus5": "Opus 5", "opus48": "Opus 4.8", "gemini-31pro": "Gemini 3.1 Pro",
          "gemini-flash": "Gemini Flash", "fugu_ultra": "Sakana Fugu Ultra"}

# Tiers with no usable per-task cost — priced from the all-task mean, so their cascade cost is
# understated. Board README: deepseek_v4f 0/89, opus5 0/89 (Bedrock), gpt56_terra 86/89 mispriced.
# These three carry no usable per-task invoice. Rather than pricing their escalated tasks at
# the fleet mean -- which understates them, because the escalated set is exactly the tasks
# the local tier failed -- we use per_task_cost_reconstruction.json, which allocates each
# tier's invoiced total across tasks by measured per-task tokens (approved 2026-09-01).
# Escalation premiums over the fleet mean: DeepSeek 1.46x, Terra 1.76x, Opus 5 0.58x.
MEAN_PRICED = {"deepseek_v4f", "opus5", "gpt56_terra"}
with open(os.path.join(DATA, "per_task_cost_reconstruction.json")) as _f:
    RECON = {k: v["per_task"] for k, v in json.load(_f).items() if not k.startswith("_")}

# Values from a re-run (2026-08-22) that the vendored 2026-08-18 board does not yet contain.
# These are NOT derived from data/, so they are applied after the reconstruction and logged on
# every build. Delete this dict the moment the board catches up; do not add to it casually.
# EMPTY as of 2026-08-29. The previous entry ({"gpt56_sol": acc 78.4, cost 0.17}) was a
# QWEN3.6-27B cascade value from the 2026-08-22 re-run. The local tier is now GLM-5.2, whose
# cascade cells come straight from the board (GLM -> gpt56_sol = 84.3% @ $0.1965, untrimmed),
# so applying that override here would silently corrupt a different model's numbers.
OVERRIDE = {}
# STALE (2026-08-22): DeepSeek-V4-Flash's standalone accuracy was corrected to 76.1% in
# build_all.ACC_OVERRIDE, but the "-> DeepSeek-V4-Flash" row below is a CASCADE accuracy
# reconstructed from deepseek_v4f's per-task solve vector, which is still the old 59/89
# run. A better DeepSeek would solve more of the escalated tasks, so 61.8% is a LOWER
# BOUND. It is deliberately NOT overridden: OVERRIDE sets cascade values, and pasting a
# standalone accuracy in here would silently misreport the cascade. Recompute when the
# board ships per-task vectors for the re-run.

# Withheld from the plot (2026-08-29). With GLM-5.2 as the local tier these are all dominated,
# and five of them are budget_trimmed against the $0.225 ceiling, so they pile up at one price and
# say nothing about the model. They are still computed and still validated against the grid.
EXCLUDE = {"gpt55", "gemini-31pro", "opus5", "opus48", "gemini-flash", "fugu_ultra"}

# Frontier cascades get a full two-line label; offsets in points from the marker.
# Terra's label runs LEFT: a right-running label from $0.041 would collide with sol/kimi at $0.18.
# Four frontier points inside 1.2 decades, so labels alternate side and vertical band.
OFF_FRONTIER = {
    "deepseek_v4f": (12, -8),
    "gpt56_terra": (-12, 8),
    "kimi_k3": (12, -8),
    "gpt56_sol": (-16, 9),
}
# Dominated cascades: cloud name only, muted. Uncapped they scatter across $0.19-$0.77 in a narrow
# accuracy band (69.7-75.3%), so collisions are horizontal now — labels alternate above/below.
OFF_DOMINATED = {
    "gemini-31pro": (0, -12),
    "opus5": (0, 12),
    "opus48": (0, 12),
    "gemini-flash": (0, -12),
    "fugu_ultra": (0, 12),
}
# Dominated points carry the model name only. Their accuracies are in the printed table and in
# the post's prose; repeating them here costs ~40% more label width than the 0.6-decade window
# these seven share can absorb.


def load(n):
    with open(os.path.join(DATA, n)) as fh:
        return json.load(fh)


def uncapped(vec, grid, cloud):
    """Recompute deploy_honest with no budget ceiling, from the per-task vectors.

    The gate is 'all_timeouts': escalate every task the local tier errored on (AgentTimeout or
    crash), and take the cloud's verdict on those. Three of the 35 candidates are tasks the local
    tier both solved AND errored, so escalating them can LOSE a solve — which is exactly what
    distinguishes deploy_honest from deploy_keep_solves. Hence the kept set is (solved - errored),
    not solved.
    """
    tasks, local = vec["tasks"], vec["vectors"][LOCAL]
    n = len(tasks)
    solved = {t for t in tasks if local[t]["solved"]}
    errored = {t for t in tasks if local[t]["errored"]}
    cvec = vec["vectors"][cloud]

    kept = solved - errored
    recovered = {t for t in errored if cvec[t]["solved"]}
    acc = 100.0 * len(kept | recovered) / n

    local_cost = grid["local_dollars_per_task"][LOCAL]
    if cloud in MEAN_PRICED:
        recon = RECON.get(cloud, {})
        if recon and all(t in recon for t in errored):
            cloud_spend = sum(recon[t] for t in errored) / n
        else:
            cloud_spend = len(errored) / n * vec["authoritative_mean_cost_usd"][cloud]
    else:
        missing = [t for t in errored if cvec[t].get("cost_usd") is None]
        if missing:
            # Partial coverage. Which tasks lack a per-task cost depends on the LOCAL tier, since
            # the escalated set is whatever that tier failed -- so a cloud can be fully priceable
            # for one local model and not another. Fall back to the all-task mean for the whole
            # row rather than mixing two bases within one number, and record it so the caller
            # can mark the point as a lower bound exactly like the MEAN_PRICED tiers.
            print(f"  NOTE: {cloud} has {len(missing)}/{len(errored)} escalated tasks with no "
                  f"per-task cost under {LOCAL}; falling back to the all-task mean (lower bound)")
            return acc, local_cost + len(errored) / n * vec["authoritative_mean_cost_usd"][cloud], \
                len(errored), True
        cloud_spend = sum(cvec[t]["cost_usd"] for t in errored) / n
    return acc, local_cost + cloud_spend, len(errored), cloud in MEAN_PRICED


def _validate_against_grid(grid, rows):
    """Every pair the grid did NOT budget-trim must reproduce exactly. That is the proof that the
    uncapped recomputation is the same policy as the published one, minus the ceiling."""
    published = {p["cloud"]: p["deploy_honest"] for p in grid["pairs"] if p["local"] == LOCAL}
    checked = 0
    for r in rows:
        d = published[r["cloud"]]
        if d.get("budget_trimmed"):
            continue
        # The grid publishes accuracy to 1 dp and cost to 4 dp, so compare at ITS precision —
        # a tighter tolerance would fail on rounding alone (62.9213% is published as 62.9%).
        if round(r["acc"], 1) != round(d["acc_pct"], 1) or \
                round(r["cost"], 4) != round(d["cost_usd_per_task"], 4):
            raise SystemExit(
                f"reconstruction disagrees with cascade_grid.json on untrimmed pair {r['cloud']}: "
                f"grid={d['acc_pct']:.4f}% ${d['cost_usd_per_task']:.6f}  "
                f"recomputed={r['acc']:.4f}% ${r['cost']:.6f}")
        checked += 1
    # 5 as of the 2026-08-18 board (Gemini Flash stopped hitting the ceiling when its accuracy rose).
    # How many pairs the grid leaves untrimmed is a property of the LOCAL tier, not a constant:
    # GLM-5.2 has 4, Qwen3.6-27B had 5. This is therefore a floor, not an exact count.
    if checked < 3:
        raise SystemExit(f"validated only {checked} untrimmed pairs against the grid; "
                         f"expected at least 3 for the reconstruction to be trustworthy")
    print(f"  validated {checked} untrimmed pairs against cascade_grid.json (exact)")


def main() -> None:
    grid, vec = load("cascade_grid.json"), load("per_task_vectors_v2.json")
    # TARGET-PAPER COPY: no Fable 5 anchor. The canonical script loads it here (S1, 80.4% @ $4.50)
    # only to plot it and to print "N× cheaper" multiples; both are removed in this copy.

    local_acc = grid["base_accuracy_pct"][LOCAL]
    local_cost = grid["local_dollars_per_task"][LOCAL]
    if local_cost is None:
        raise SystemExit(f"{LOCAL} has no ADR-0004 dollar cost; it cannot anchor this figure.")

    rows = []
    for p in grid["pairs"]:
        if p["local"] != LOCAL:
            continue
        if p["deploy_honest"].get("cost_usd_per_task") is None:
            continue  # cost_blocked_on — never draw a point we cannot price
        acc, cost, n_esc, mean_priced = uncapped(vec, grid, p["cloud"])
        rows.append(dict(cloud=p["cloud"], name=PRETTY[p["cloud"]], acc=acc, cost=cost,
                         escalated=n_esc,
                         mean_priced=mean_priced,
                         was_trimmed=bool(p["deploy_honest"].get("budget_trimmed"))))
    if len(rows) < 8:
        raise SystemExit(f"expected ~10 priceable escalate plans for {LOCAL}, got {len(rows)}")
    _validate_against_grid(grid, rows)          # all 10, before any editorial exclusion
    for r in rows:
        if r["cloud"] in OVERRIDE:
            o = OVERRIDE[r["cloud"]]
            print(f"  OVERRIDE {r['cloud']}: {r['acc']:.1f}% ${r['cost']:.4f} -> "
                  f"{o['acc']}% ${o['cost']} (2026-08-22 re-run, not in the vendored board)")
            r["acc"], r["cost"] = o["acc"], o["cost"]
    rows = [r for r in rows if r["cloud"] not in EXCLUDE]

    # The frontier is computed over the cascades plus the local model alone (where you start).
    # TARGET-PAPER COPY: the canonical script also passed the Fable 5 anchor here. It was dominated
    # (GLM -> GPT-5.6-sol is both cheaper and more accurate), so dropping it leaves the dotted
    # frontier and its shading unchanged.
    anchors = [(local_cost, local_acc)]
    pts = [(r["cost"], r["acc"]) for r in rows] + anchors
    front = set(ps.pareto(pts))
    for r in rows:
        r["on_front"] = (r["cost"], r["acc"]) in front

    fig, ax = plt.subplots(figsize=(18.5, 11.0))
    ax.set_xscale("log")
    ax.set_xlim(0.0011, 11.0)
    ax.set_ylim(50, 95)

    ps.draw_pareto(ax, pts)

    # A dashed edge means the point is priced from the all-task mean, not per-task actuals, so its
    # x is a lower bound (escalated tasks run 1.3-1.8x the mean on every tier we CAN measure).
    def edge(r):
        # Uniform solid 2pt outline (Jon, 2026-08-22). The dashed edge used to mark mean-priced
        # points; that distinction now rests on the dagger in the label alone, so the caption must
        # not say "dashed edge".
        return dict(linestyle="-", linewidths=2.0)

    dag = ""   # daggers removed 2026-08-22; the caveat now lives only in the caption

    # --- frontier cascades --------------------------------------------------------------------
    for r in (r for r in rows if r["on_front"]):
        ax.scatter(r["cost"], r["acc"], s=601, marker="o", color=CASCADE,
                   edgecolors="black", zorder=6, **edge(r))
        dx, dy = OFF_FRONTIER[r["cloud"]]
        # TARGET-PAPER COPY: the canonical second line ends " · {mult}× cheaper" (vs the $4.50 S1
        # anchor); removed so the figure carries no multiple that disagrees with the paper's table.
        ax.annotate(f"{LOCAL_PRETTY} → {r['name']}{dag if r['mean_priced'] else ''}\n"
                    f"{r['acc']:.1f}%",
                    (r["cost"], r["acc"]), textcoords="offset points", xytext=(dx, dy),
                    ha="right" if dx < 0 else "left", va="bottom" if dy > 0 else "top",
                    fontsize=20.8, fontweight="bold", color=INK, zorder=9, linespacing=1.3)

    # --- endpoints ----------------------------------------------------------------------------
    ax.scatter(local_cost, local_acc, s=531, marker="^", color=OPEN_CLOUD,
               edgecolors="black", linewidths=1.3, zorder=6)
    ax.annotate(f"{LOCAL_PRETTY} alone\n{local_acc:.1f}% · no escalation",
                (local_cost, local_acc), textcoords="offset points", xytext=(26, -4),
                ha="left", va="top", fontsize=20.8, fontweight="bold", color=INK, zorder=9,
                linespacing=1.3)

    # TARGET-PAPER COPY: the canonical script drew the Claude Fable 5 anchor here (orange square,
    # "Claude / Fable 5 / 80.4%") and a bracket from the top cascade to it labelled "matches Fable 5
    # within noise, 23× cheaper". Both are removed; nothing else is drawn in their place.
    head = max((r for r in rows if r["on_front"]), key=lambda r: r["acc"])

    # --- note on the dominated stack ------------------------------------------------------------
    ax.set_xticks([0.002, 0.01, 0.03, 0.1, 0.3, 1, 3])
    ax.set_xticklabels(["\\$0.002", "\\$0.01", "\\$0.03", "\\$0.10", "\\$0.30", "\\$1", "\\$3"],
                       fontsize=25.8)
    ax.set_yticks(range(50, 91, 10))
    ax.set_yticklabels([f"{v}%" for v in range(50, 91, 10)], fontsize=25.8)
    ax.set_xlabel("Cost per task (USD, log scale)", fontsize=30.0, fontweight="bold", labelpad=14)
    ax.set_ylabel("Accuracy", fontsize=30.0, fontweight="bold", labelpad=14)
    ax.set_title("TerminalBench v2.1", fontsize=34.8, fontweight="bold", color=INK, pad=22)
    ax.grid(True, color="#AAAAAA", alpha=0.25, linewidth=0.65)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.0)      # 2x matplotlib's 0.8 default
    leg = ax.legend(handles=[
        Line2D([], [], marker="^", ls="", ms=17.3, mfc=OPEN_CLOUD, mec="black", mew=2.0,
               label="First stage alone"),
        Line2D([], [], marker="o", ls="", ms=17.3, mfc=CASCADE, mec="black", mew=2.0,
               label="Escalate on observed failure (→)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3, frameon=True,
       fontsize=21.6, handletextpad=0.7, columnspacing=2.6,
       facecolor="#EFEFEF", edgecolor="black", framealpha=1.0, borderpad=0.9)
    leg.get_frame().set_linewidth(2.0)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.215)
    # The mean-priced disclosure is NOT drawn on the figure (Jon, 2026-08-14) — it lives in the
    # post's figcaption instead. The dashed edge + dagger stay, so the caveat is still visible on
    # the image itself; any venue that reuses this PNG must carry the caption with it.
    # TARGET-PAPER COPY: the PDF goes to paper/figures_and_tables/; the PNG only if FIG5_PNG_DIR is set.
    targets = [os.path.join(OUT, "fig5_cascade_frontier.pdf")]
    if PNG_DIR:
        targets.insert(0, os.path.join(PNG_DIR, "fig5_cascade_frontier.png"))
    for path in targets:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
        print(f"  wrote {path}")
    print(f"  {'PLAN':<32}{'ACC':>7}{'$/TASK':>11}  FRONT  PRICING  WAS-CAPPED")
    print(f"  {LOCAL_PRETTY + ' alone':<32}{local_acc:>6.1f}%${local_cost:>10.4f}"
          f"  yes    measured  —")
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(f"  {'→ ' + r['name']:<32}{r['acc']:>6.1f}%${r['cost']:>10.4f}"
              f"  {'yes' if r['on_front'] else '— ':<6} {'MEAN†' if r['mean_priced'] else 'per-task':<9}"
              f"{'yes' if r['was_trimmed'] else 'no'}")
    print(f"  {len(rows)} plans shown, all on the frontier; top = {head['acc']:.1f}% "
          f"at ${head['cost']:.4f}/task")
    plt.close(fig)


if __name__ == "__main__":
    main()
