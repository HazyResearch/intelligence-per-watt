"""Figure 7 — generational energy on consumer Apple silicon.

The TB v2.1 energy panel (fig2, right) varies model AND accelerator together, so it cannot
separate the two. This figure holds the model fixed and varies only the machine: seven models,
each measured on M3 Ultra, M4 Max and M5 Max, on the ipw benchmark.

DATA  data/apple_silicon_board.json  (HF metrics/, 2026-09-07; 26 ipw rows)

WHAT THIS FIGURE MAY AND MAY NOT SAY
  MAY  energy. Every model is monotone M3U > M4Max > M5Max; ratios 2.35x-3.43x, mean ~2.7x.
  NOT  accuracy. Per the board's own caveat, ipw accuracy swings up to 15.9 pt for an IDENTICAL
       model+quant across silicon at N~15-30, which is inside binomial noise. Accuracy is not
       plotted here and must not be read off this panel.
  NOT  merged with the TB board. These are J/QUERY on ipw at Q4_K_M; the TB tiers are J/TASK at
       bf16/FP8 on datacenter parts. Different workload, different quantisation.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps
from paper_style import INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("IPD_FIG_OUT", HERE)

with open(os.path.join(HERE, "data", "apple_silicon_board.json")) as f:
    BOARD = json.load(f)

# M5 Pro is omitted: it covers only 4 of the 7 models, so including it would make the
# generational series ragged and invite a comparison the data does not support.
PLATFORMS = ["Mac Studio M3 Ultra", "Mac Studio M4 Max", "MacBook Pro M5 Max"]
SHORT = {"Mac Studio M3 Ultra": "M3 Ultra", "Mac Studio M4 Max": "M4 Max",
         "MacBook Pro M5 Max": "M5 Max"}

rows = BOARD["data"]["ipw"]
by = {(r["platform"], r["model"]): r["energy_j_per_query"] for r in rows}
models = sorted({r["model"] for r in rows}
                if all((p, r["model"]) in by for p in PLATFORMS for r in rows)
                else {m for m in {r["model"] for r in rows}
                      if all((p, m) in by for p in PLATFORMS)},
                key=lambda m: -by[(PLATFORMS[0], m)])

RAMP = ["#7C3AED", "#0891B2", "#16A34A"]   # one colour per generation


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    x = np.arange(len(models))
    w = 0.26

    for i, p in enumerate(PLATFORMS):
        vals = [by[(p, m)] for m in models]
        ax.bar(x + (i - 1) * w, vals, w, label=SHORT[p], color=RAMP[i],
               edgecolor="black", linewidth=1.0, zorder=3)

    # Per-model generational ratio, printed over the M5 Max bar it describes.
    for j, m in enumerate(models):
        ratio = by[(PLATFORMS[0], m)] / by[(PLATFORMS[-1], m)]
        ax.annotate(f"{ratio:.1f}$\\times$", (x[j] + w, by[(PLATFORMS[-1], m)]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=13.5, fontweight="bold", color=INK, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (Q4_K_M)", "") for m in models],
                       rotation=18, ha="right", fontsize=13.5)
    ax.set_ylabel("Energy per query (J)", fontsize=17, fontweight="bold")
    ax.set_xlabel("Model (Q4_K_M quantisation, held fixed across silicon)", fontsize=17,
                  fontweight="bold", labelpad=10)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, axis="y", color="#AAAAAA", alpha=0.25, linewidth=0.65)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ratios = [by[(PLATFORMS[0], m)] / by[(PLATFORMS[-1], m)] for m in models]
    ax.set_title(f"Two hardware generations cut energy per query "
                 f"{min(ratios):.1f}$\\times$ to {max(ratios):.1f}$\\times$, "
                 f"with the model held fixed",
                 fontsize=18, fontweight="bold", pad=16)
    ax.legend(fontsize=14.5, frameon=True, facecolor="#EFEFEF", edgecolor="black",
              framealpha=1.0, title="Apple silicon", title_fontsize=14.5, loc="upper right")
    ax.set_ylim(0, max(by[(PLATFORMS[0], m)] for m in models) * 1.16)

    fig.text(0.012, -0.02,
             "Energy only. Per-platform accuracy on this benchmark is not discriminative at "
             "N of roughly 15 to 30 and is not plotted.",
             fontsize=11.5, color=MUTED, ha="left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig7_apple_silicon.{ext}"),
                    bbox_inches="tight", pad_inches=0.3)
    print(f"  wrote fig7_apple_silicon.png/.pdf  ({len(models)} models x {len(PLATFORMS)} platforms)")
    print(f"  generational ratio M3 Ultra -> M5 Max: {min(ratios):.2f}x-{max(ratios):.2f}x, "
          f"mean {sum(ratios)/len(ratios):.2f}x")
    for m in models:
        r = by[(PLATFORMS[0], m)] / by[(PLATFORMS[-1], m)]
        print(f"    {m:26} {by[(PLATFORMS[0], m)]:7.0f} -> {by[(PLATFORMS[-1], m)]:6.0f} J  {r:.2f}x")
    plt.close(fig)


if __name__ == "__main__":
    main()
