#!/usr/bin/env python3
"""Refresh the numbers in targets/figure_inputs.json from the plan model, so Figures 1 and 4 follow targets.json.

Labels, markers, kinds, order and styling fields are left as they are; only accuracies, dollars, Joules,
execution shares, multipliers and Figure 1's annotation text are rewritten (round 9, 2026-09-25: the H200
moved to 4-bit and the figure inputs had been maintained by hand).

Round 15 (2026-09-25): the headline's first stage is the laptop the local-by-default rule selects
(make_numbers HEAD), so Figure 1's right panel splits each composition's stage executions three ways:
local (the laptop), owned accelerators (GLM-5.3 on 8xH200) and the closed-source provider.
"""
from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()):
    NS = runpy.run_path(str(ROOT / "targets" / "make_numbers.py"), run_name="make_numbers")
PLANS, MACH, tiers, base_acc, ANCHOR, mult = NS["PLANS"], NS["MACH"], NS["tiers"], NS["base_acc"], NS["ANCHOR"], NS["mult"]
P = ROOT / "targets" / "figure_inputs.json"
d = json.loads(P.read_text())
f1 = d["fig1"]

PLAN_OF = {"Qwen3.8-27B → GLM-5.3 → GPT-6 Astra": "headline", "Qwen3.8-27B → GLM-5.3": "openonly",
           "Qwen3.8-27B → GPT-6 Astra": "two_astra", "Qwen3.8-Flash-Next → GLM-5.3 → GPT-6 Astra": "flashnext"}
SINGLE = {"Fable 5.1": (ANCHOR["acc"] / 100, ANCHOR["usd"]),
          "GPT-6 Astra": (base_acc["astra"], tiers["astra"]["usd"]),
          "GPT-5.6-sol": (base_acc["sol"], tiers["sol"]["usd"]),
          "GLM-5.3": (base_acc["glm53"], tiers["glm53"]["own"]),
          "Qwen3.8-27B": (MACH[NS["HEAD"]]["acc"], MACH[NS["HEAD"]]["own"])}   # round 15: the headline's desk machine (Table 1's row)


def exec_pcts(r: dict, kind: str) -> tuple[int, int, int]:
    """Figure 1, right panel: the percent of a composition's stage executions that run locally (every task
    starts on the laptop), on owned accelerators (the GLM-5.3 stage) and at the closed-source provider,
    rounded by largest remainder so the three segments of a bar sum to 100."""
    if kind not in ("three", "open", "two"):
        raise SystemExit(f"figure 1 right panel: no execution split for plan kind {kind!r}")
    # author ruling 2026-09-25: the bars count TASKS by where each one finishes (laptop / owned accelerators /
    # closed-source provider), matching Table 1 and the text; they used to count stage executions (a two-stage
    # plan showed 80% local while keeping 75.1% of tasks on the laptop)
    raw = [100 * x for x in r["shares"]]
    out = [int(v) for v in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - out[i], reverse=True)[:100 - sum(out)]:
        out[i] += 1
    return tuple(out)


for c in f1["composed"]:
    r = PLANS[PLAN_OF[c["label"]]]
    c["acc"], c["usd"] = round(100 * r["acc"], 1), round(r["cost"]["own"], 4)
for s in f1["singles"]:
    a, u = SINGLE[s["label"]]
    s["acc"], s["usd"] = round(100 * a, 1), round(u, 5 if u < 0.01 else 3)
h = PLANS["headline"]
f1["annotation"]["text"] = f"−{ANCHOR['acc'] - 100 * h['acc']:.1f} points, {mult(ANCHOR['usd'] / h['cost']['own'])}× cheaper"
for row in f1["right_panel_rows"]:
    lab = row["label"]
    if lab in PLAN_OF:
        k = PLAN_OF[lab]
        r = PLANS[k]
        row.pop("owned_pct", None)   # round 15: replaced by the three-way split
        row["local_pct"], row["accel_pct"], row["provider_pct"] = exec_pcts(r, NS["tb"]["plans"][k]["kind"])
        row["mult"] = f"{mult(ANCHOR['usd'] / r['cost']['own'])}×"
        row["acc"] = f"{100 * r['acc']:.1f}%"
    else:
        name = lab.replace(" alone", "").replace("Claude ", "")
        a, u = SINGLE[name]
        m = "1×" if name == "Fable 5.1" else f"{mult(ANCHOR['usd'] / u)}×"   # the anchor itself
        row.pop("owned_pct", None)
        row["local_pct"], row["accel_pct"], row["provider_pct"] = 0, 0, 100   # a closed model served alone
        row["mult"], row["acc"] = m, f"{100 * a:.1f}%"

f2 = d["fig2"]
# Figure 2 plots each tier at its own TerminalBench accuracy (tb21.tiers solved/graded; the figure script
# figures/fig2_ipd_ipj_target.py holds the same values and aborts on disagreement), with the cost and energy of the
# configuration that serves it. Round 10: machine accuracies now differ from the tier's (format and time-budget
# losses, targets/make_numbers.py), so the tier accuracy is written here rather than the machine's.
SELF = {"Z-AI GLM-5.3": (base_acc["glm53"], tiers["glm53"]["own"], tiers["glm53"]["kj_wall"]),
        "Qwen3.8-27B": (base_acc["qwen38"], MACH["h200_16"]["own"], MACH["h200_16"]["kj_wall"]),
        "Qwen3.8-Flash-Next": (base_acc["flashnext"], MACH["fn_h200x2"]["own"], MACH["fn_h200x2"]["kj_wall"])}
for key, field in (("ipd", "usd"), ("ipj", "kj")):
    for e in f2[key]:
        if e["label"] in SELF:
            a, u, kj = SELF[e["label"]]
            e["acc"] = round(100 * a, 1)
            e[field] = round(u, 5) if field == "usd" else round(kj)
# Figure 2 bracket labels, formatted exactly as figures/fig2_ipd_ipj_target.py formats them from the plotted values
# (its pairs are fixed there: GPT-6 Astra -> Qwen3.8-Flash-Next for IPD, GLM-5.3 -> Qwen3.8-Flash-Next for IPJ).
_acc = lambda k: round(100 * base_acc[k], 1)
f2["ipd_annotation"] = (f"{_acc('astra') - _acc('flashnext'):.1f} pp accuracy lost, "
                        f"{tiers['astra']['usd'] / round(MACH['fn_h200x2']['own'], 8):.0f}× cheaper")
# Round 12: the IPJ bracket pair is chosen in the figure script (now Moonshot Kimi K3 -> Qwen3.8-27B, whose Kimi energy is
# the script's own canonical estimate), which formats its label from the plotted values; the stale GLM-5.3 ->
# Qwen3.8-Flash-Next annotation that round 10 wrote here is dropped rather than left to disagree with the figure.
f2.pop("ipj_annotation", None)
P.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("figure_inputs.json synced:", f1["annotation"]["text"],
      [(r["label"][:24], r["local_pct"], r["accel_pct"], r["provider_pct"], r["mult"], r["acc"]) for r in f1["right_panel_rows"]])
