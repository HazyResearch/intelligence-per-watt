#!/usr/bin/env python3
"""Map the canonical paper's macro names onto target-world values.

Runs make_numbers.py (for PLANS, MACH, the plan model), computes the extra quantities the canonical
prose uses (capability-target frontier, flags-only variant, signal-lift range, attempt counts), and
writes paper/ipj-numbers.tex = canonical macro file + \\renewcommand overrides + target macros.
Macros that are not overridden keep their canonical (measured) value; STATUS.md lists which.
"""
from __future__ import annotations

import json
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = runpy.run_path(str(ROOT / "targets" / "make_numbers.py"), run_name="make_numbers")
PLANS, MACH, plan, T = NS["PLANS"], NS["MACH"], NS["plan"], NS["T"]
tiers, sig, PREM, ANCHOR, PUB = NS["tiers"], NS["sig"], NS["PREM"], NS["ANCHOR"], NS["PUB"]
PC, ICC, IV, acc_hw = NS["PC"], NS["ICC"], NS["IV"], NS["acc_hw"]
PC3, PCG, PD, PLAN_ICC, plan_icc = NS["PC3"], NS["PCG"], NS["PD"], NS["PLAN_ICC"], NS["plan_icc"]   # round 23     # round 10: per-target premiums, intervals
base_acc, usd, mult = NS["base_acc"], NS["usd"], NS["mult"]
HEAD = NS["HEAD"]      # round 15: the headline's first stage, the M5 Max configuration the local-by-default rule selects
TX = T["compat"]
FAILS: list[str] = []
OV: dict[str, str] = {}


def ov(name: str, value: str) -> None:
    assert name not in OV, name
    OV[name] = value


def x(v: float) -> str:
    return mult(v) + "$\\times$"


h = PLANS["headline"]
N = T["tb21"]["n_tasks"] * T["constants"]["k"]

# ------------------------------------------------------------ extra plans
# flags-only variant (errors + time budget, no trace judge)
flag_sig = dict(sig)
sig_saved = dict(sig)
sig.update(s1_recall=TX["flags_s1_recall"], s1_fpr=TX["flags_s1_fpr"])
flags = plan(T["tb21"]["plans"]["headline"])
sig.clear(); sig.update(sig_saved)

# lift over random escalation at equal rate, per plan
lifts = {}
for k in ("openonly", "two_sol", "two_astra", "headline"):
    r = PLANS[k]
    rnd = plan(T["tb21"]["plans"][k], random_stage1=True)
    lifts[k] = 100 * (r["acc"] - rnd["acc"])


# GLM-first two-stage plans (no local stage) and Qwen3.6-local plans for the ladder
def glmfirst(closed: str) -> dict:
    g = tiers["glm53"]
    L = base_acc["glm53"]
    r2, f2 = sig["s2_recall"], sig["s2_fpr"]
    fe, caught = L * f2, (1 - L) * r2
    E = fe + caught
    acc = L - fe + fe * sig["closed3_solve_flagged_correct"] + caught * TX["closed_solve_glm_failure"][closed]
    c = tiers[closed]["usd"] * PCG(closed)          # round 23: its own escalated subset
    cost = {b: g[b] + sig[f"judge_{b}"] + E * c for b in ("own", "amort", "rent")}
    return dict(acc=acc, cost=cost)


def qwen36(kind: str, closed: str | None = None) -> dict:
    # same machine and signal as the headline local stage, with Qwen3.6-27B's leaderboard accuracy
    loc = dict(MACH[HEAD])
    saved = base_acc["qwen38"]
    base_acc["qwen38"] = TX["qwen36_acc"] / 100
    MACH["_q36"] = dict(loc, acc=TX["qwen36_acc"] / 100, to=0.0)
    try:
        return plan({"kind": kind, "local": "qwen38", "machine": "_q36", "closed": closed})
    finally:
        base_acc["qwen38"] = saved
        del MACH["_q36"]


def single(acc, own, amort, rent, local):
    return dict(acc=acc, cost=dict(own=own, amort=amort, rent=rent), local=local)


def space(qwen_new: bool) -> dict:
    S = {
        "GLM-5.3 alone": single(base_acc["glm53"], tiers["glm53"]["own"], tiers["glm53"]["amort"], tiers["glm53"]["rent"], False),
        "GPT-5.6-sol alone": single(base_acc["sol"], tiers["sol"]["usd"], tiers["sol"]["usd"], tiers["sol"]["usd"], False),
        "GPT-6 Astra alone": single(base_acc["astra"], tiers["astra"]["usd"], tiers["astra"]["usd"], tiers["astra"]["usd"], False),
        "GLM-5.3 $\\rightarrow$ GPT-5.6-sol": dict(glmfirst("sol"), local=False),
        "GLM-5.3 $\\rightarrow$ GPT-6 Astra": dict(glmfirst("astra"), local=False),
    }
    if qwen_new:
        # round 15: every Qwen3.8-27B-first system runs on the headline's desk machine (the M5 Max configuration the rule
        # selects); the H200 stays a datacenter comparison in Table 2 and is no longer a first-stage option in the plan space
        m = MACH[HEAD]
        S["Qwen3.8-27B alone"] = single(m["acc"], m["own"], m["amort"], m["rent"], True)
        mp = MACH["m4pro"]
        S["Qwen3.8-27B alone (M4 Pro)"] = single(mp["acc"], mp["own"], mp["amort"], mp["rent"], True)
        fn = MACH["fn_h200x2"]
        # 180B total: too large for any desk machine, served on two owned H200s, so an on-prem
        # open-weight tier like GLM-5.3 rather than a local stage (author note, 2026-09-24)
        S["Qwen3.8-Flash-Next alone"] = single(fn["acc"], fn["own"], fn["amort"], fn["rent"], False)
        names = {"headline": "Qwen3.8-27B $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-6 Astra",
                 "solvar": "Qwen3.8-27B $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-5.6-sol",
                 "openonly": "Qwen3.8-27B $\\rightarrow$ GLM-5.3",
                 "two_astra": "Qwen3.8-27B $\\rightarrow$ GPT-6 Astra", "two_sol": "Qwen3.8-27B $\\rightarrow$ GPT-5.6-sol",
                 "retry": "Qwen3.8-27B, retried once, $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-6 Astra",
                 "flashnext": "Qwen3.8-Flash-Next $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-6 Astra"}
        for k, n in names.items():
            S[n] = dict(PLANS[k], local=(k != "flashnext"))
    else:
        m = MACH[HEAD]
        S["Qwen3.6-27B alone"] = single(TX["qwen36_acc"] / 100, m["own"], m["amort"], m["rent"], True)
        S["Qwen3.6-27B $\\rightarrow$ GLM-5.3"] = dict(qwen36("open"), local=True)
        S["Qwen3.6-27B $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-6 Astra"] = dict(qwen36("three", "astra"), local=True)
        S["Qwen3.6-27B $\\rightarrow$ GPT-5.6-sol"] = dict(qwen36("two", "sol"), local=True)
        S["Qwen3.6-27B $\\rightarrow$ GPT-6 Astra"] = dict(qwen36("two", "astra"), local=True)
    for n, v in S.items():      # self-served first stage (local, or open-weight on owned accelerators)
        v["owned"] = v["local"] or n.startswith(("GLM-5.3", "Qwen3.8-Flash-Next"))
    return S


KAPPAS = TX["kappas"]


def ladder(S: dict, basis: str) -> list[tuple[float, str, bool]]:
    out = []
    for k in KAPPAS:
        feas = [(v["cost"][basis], n, v["local"]) for n, v in S.items() if v["acc"] >= k - 1e-9]
        c, n, loc = min(feas)
        out.append((k, n, loc))
    return out


new_own, new_rent = ladder(space(True), "own"), ladder(space(True), "rent")
old_own = ladder(space(False), "own")
for name, L in (("owned, Qwen3.8 family", new_own), ("rented, Qwen3.8 family", new_rent), ("owned, Qwen3.6 as newest", old_own)):
    print(name)
    for k, n, loc in L:
        print(f"   kappa {k:.2f}: {n}  local={loc}")


def kmax(S: dict, basis: str) -> float:
    best = 0.0
    for v in S.values():
        if not v["local"]:
            continue
        a = v["acc"]
        feas = [(w["cost"][basis], w["local"]) for w in S.values() if w["acc"] >= a - 1e-9]
        if min(feas)[1]:
            best = max(best, a)
    return best


S_new = space(True)
S_no_flash = {k: v for k, v in S_new.items() if "Flash-Next" not in k}
WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
n_new = sum(1 for _, _, loc in new_own if loc)
n_old = sum(1 for _, _, loc in old_own if loc)
n_rent = sum(1 for _, _, loc in new_rent if loc)

# ------------------------------------------------------------ overrides of canonical macros (main text)
fable5 = PUB["aa_tb21"]["fable5_max"]["acc"]
ov("ANCHORACC", f"{ANCHOR['acc']:.1f}\\%")
ov("ANCHORUSD", usd(ANCHOR["usd"]))
ov("ASTRAESCPREM", f"{PC('astra'):.2f}")
ov("ASTRAESCPREMHEAD", f"{PC3('astra'):.2f}")     # round 23: the headline's doubly escalated set
ov("ASTRADEPTH", f"{PD['astra']['after_glm']:.2f}")
# round 28: 3.2 title, the rented pick's closed-source share over the headline's, from the printed shares
ov("RentClosedX", x(round(100 * PLANS["two_astra"]["shares"][2], 1) / round(100 * h["shares"][2], 1)))
ov("ASTRAMEANUSD", usd(tiers["astra"]["usd"]))
ov("TBTerraAAAcc", f"{PUB['aa_tb21']['terra_max']['acc']:.1f}")   # round 47: the leaderboard entry behind GPT-5.6-Terra in RENTDOMPROSE
ov("TBTerraAAUSD", usd(PUB["aa_tb21"]["terra_max"]["usd"]))
# round 12: P(Astra only) - P(plan only) = pA - pH is an identity, so the Astra-only share follows from the plan-only target
TX["cover_only_astra"] = TX["cover_only_plan"] + 100 * (base_acc["astra"] - h["acc"])
ov("COVERONLYPLAN", f"{TX['cover_only_plan']:.1f}\\%")
ov("COVERONLYASTRA", f"{TX['cover_only_astra']:.1f}\\%")
check_cov = abs((TX["cover_only_astra"] - TX["cover_only_plan"]) - (100 * base_acc["astra"] - 100 * h["acc"])) < 0.3
if not check_cov:
    FAILS.append("coverage asymmetry must equal the Astra minus plan accuracy gap")

rows = {"Base": ("single", HEAD), "Open": "openonly", "Sol": "two_sol", "Astra": "two_astra",
        "SolThree": "solvar", "Head": "headline", "Retry": "retry", "Desk": "desk", "Flash": "flashnext"}
for tag, key in rows.items():
    if tag == "Base":
        m = MACH[HEAD]
        acc, own = m["acc"], m["own"]
    else:
        acc, own = PLANS[key]["acc"], PLANS[key]["cost"]["own"]
    ov_name = "Casc" + tag
    OV["CascAcc" + tag] = f"{100 * acc:.1f}\\%"
    OV["CascUSD" + tag] = usd(own)
    OV["CascMult" + tag] = x(ANCHOR["usd"] / own)
    if tag != "Base":
        r = PLANS[key]
        OV["CascRent" + tag] = usd(r["cost"]["rent"])
        OV["CascRentMult" + tag] = x(ANCHOR["usd"] / r["cost"]["rent"])
        OV["CascClosed" + tag] = f"{100 * r['shares'][2]:.1f}\\%"
g1 = glmfirst("sol"); g2 = glmfirst("astra")
for tag, r in (("GLMSol", g1), ("GLMAstra", g2)):
    OV["CascAcc" + tag] = f"{100 * r['acc']:.1f}\\%"
    OV["CascUSD" + tag] = usd(r["cost"]["own"])
    OV["CascMult" + tag] = x(ANCHOR["usd"] / r["cost"]["own"])
    OV["CascRent" + tag] = usd(r["cost"]["rent"])
    OV["CascRentMult" + tag] = x(ANCHOR["usd"] / r["cost"]["rent"])
esc_rows = ["Open", "Sol", "Astra"]
accs = [PLANS[rows[t]]["acc"] for t in esc_rows]
ov("CascSpanPP", f"{100 * (max(accs) - min(accs)):.1f}")
mults = [ANCHOR["usd"] / PLANS[rows[t]]["cost"]["own"] for t in esc_rows]
ov("CascMultHi", x(max(mults)))
ov("CascMultLo", x(min(mults)))
ov("CascESCFIRE", f"{100 * h['E1']:.1f}\\%")

ov("DESKPLAN", "Qwen3.8-27B (Q4\\_K\\_M) on a 48\\,GB MacBook Pro M4~Pro")
d, gb, g1a = MACH["m4pro"], MACH["h200_16"], MACH["h200_1"]
ov("DESKACC", f"{100 * d['acc']:.1f}\\%")
ov("DESKKJ", f"{d['kj_wall']:.0f}")
ov("DESKKJSOLVED", f"{d['kj_solved']:.0f}")
ov("GPUACCMATCH", f"{100 * gb['acc']:.1f}")
ov("GPUKJ", f"{gb['kj_wall']:.0f}")
ov("GPUKJSOLVED", f"{gb['kj_solved']:.0f}")
ov("MacVsGPU", f"{gb['kj_wall'] / d['kj_wall']:.1f}$\\times$")
ov("MacVsGPUSolved", f"{gb['kj_solved'] / d['kj_solved']:.1f}$\\times$")
ov("MacWatts", f"{1000 * d['kj_wall'] / (d['wall_min'] * 60):.0f}")

# signal counts over the five rolls of the local stage (target: every attempt graded)
L = MACH[HEAD]["acc"]
solved = round(L * N)
fails = N - solved
fires = round(h["E1"] * N)
fe = round(L * sig["s1_fpr"] * N)
caught = fires - fe
ov("ESCATTEMPTS", str(N))
ov("LOCALFAILS", str(fails))
ov("ESCFIRES", str(fires))
ov("DISCARDEDSOLVES", str(fe))
ov("SYMDIFF", str((fails - caught) + fe))
ov("ESCCAUGHT", str(caught))   # round 47: failed attempts the signal escalates, over all failures (RuleRecall excludes the solves lost to the time budget)
ov("SIGRECALL", {0.5: "half", 0.6: "three fifths", 0.7: "seven tenths", 0.8: "four fifths"}[round(sig["s1_recall"], 1)])
ov("GATEAUCTO", f"{sig['auc_combined_all']:.2f}")
ov("ESCLIFTLO", f"{min(lifts.values()):.1f}")
ov("ESCLIFTHI", f"{max(lifts.values()):.1f}")
ov("GATEAUCTB", f"{TX['judge_auc_completed_tb']:.2f}")
ov("GATEAUCSWE", f"{TX['judge_auc_completed_swe']:.3f}")
ov("SWEGATEAUC", f"{TX['judge_auc_completed_swe']:.3f}")
ov("JUDGEAUCALL", f"{sig['auc_judge_all']:.2f}")
ov("JUDGEAUCCOMP", f"{TX['judge_auc_completed_tb']:.2f}")
ov("JUDGEGAIN", f"{100 * (h['acc'] - flags['acc']):.1f}")
ov("JUDGEACC", f"{100 * flags['acc']:.1f}\\%")      # now: the flags-only headline (what the system reaches without the judge)
ov("JUDGEMULT", x(ANCHOR["usd"] / flags["cost"]["own"]))

# GAIA
ga = T["gaia"]
D = json.loads((ROOT / "targets" / "derived.json").read_text())
ov("GAIAESCFIRE", f"{ga['signal']['errored_fire_pct_measured']:.1f}\\%")
ov("GAIAFRONT", f"{ga['tiers']['astra']['acc']:.1f}")
ov("GAIALFACC", f"{D['gaia']['localfirst']['acc']:.1f}\\%")
ov("GAIALOCAL", f"{ga['tiers']['qwen38']['acc']:.1f}")
ov("GAIALOCALSOLO", f"{ga['tiers']['qwen38']['acc']:.1f}\\%")

# energy paragraph
ov("QKJSOLVED", f"{MACH[HEAD]['kj_solved']:.0f}")        # round 15: the headline's first stage (the laptop)
GLMT = NS["GLM"]                                     # round 12: GLM-5.3 from its own measured campaign
ov("GLMINFLIGHT", f"{GLMT['inflight']:.1f}")
ov("GLMCAMPAIGNS", f"{GLMT['campaign_s']:,.0f}".replace(",", "{,}"))
# the energy span of the metered-tier table (tab:provenance) now includes GLM-5.3 at its measured occupancy (round 12)
_prov = [82.7, 141.4, 318.4, 453.7, gb["kj_wall"], GLMT["kj_wall"]]    # Qwen3.6, Gemma-4-31B, MiniMax-M3, GLM-5.2 measured
ov("JSPANMEASURED", f"{max(_prov) / min(_prov):.1f}")
ov("PROVKJMIN", f"{min(_prov):.1f}")
ov("PROVKJMAX", f"{max(_prov):.1f}")
glm_solved = tiers["glm53"]["kj_wall"] / base_acc["glm53"]
ov("GLMKJSOLVED", f"{glm_solved:.0f}")
ov("LOCALJOULEMULT", f"{glm_solved / MACH[HEAD]['kj_solved']:.1f}$\\times$")

# GLM generation step (target: both at the leaderboard protocol, five rolls each)
ov("GLMSTEPPP", f"{TX['glm_step_pp']:.2f}")
ov("GLMSTEPP", f"{TX['glm_step_p']}")
ov("GLMSTEPCI", TX["glm_step_ci"])

# ladder
ov("KAPPALOW", f"{KAPPAS[0]:.2f}")
ov("KAPPALOWMID", f"{KAPPAS[1]:.2f}")
ov("KAPPAMID", f"{KAPPAS[2]:.2f}")
ov("KAPPAHIGH", f"{KAPPAS[-1]:.2f}")
ov("KAPPATOP", f"{TX['kappa_top']:.2f}")
ov("KAPPAOWNMAX", f"{kmax(S_new, 'own'):.3f}")
ov("KAPPAOWNMAXQEIGHT", f"{kmax(S_no_flash, 'own'):.3f}")
ov("NLMTNEW", WORD[n_new])
ov("NLMTOLD", WORD[n_old])
ov("NLMTALL", WORD[len(KAPPAS)])
_miss_own = [k for k, _, loc in new_own if not loc]
_miss_rent = [k for k, _, loc in new_rent if not loc]
if _miss_own != _miss_rent:     # round 10: a report, not a failure (the prose now states each basis separately)
    print(f"note: owned and rented ladders lose different targets: {_miss_own} vs {_miss_rent}")
ov("KAPPANOTLOCAL", ", ".join(f"{k:.2f}" for k in _miss_own) or "none")

# headline
two_sol = PLANS["two_sol"]
ov("LFPLAN", "Qwen3.8-27B $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-6 Astra")
ov("LFACC", f"{100 * h['acc']:.1f}\\%")
ov("LFASTRAGAP", f"{100 * (base_acc['astra'] - h['acc']):.1f}")
ov("LFCLOUDGAP", f"{100 * (base_acc['sol'] - h['acc']):.1f}")        # now: points BELOW sol alone
ov("LFCLOUDMULT", x(tiers["astra"]["usd"] / h["cost"]["own"]))
ov("LFSOLMULT", x(tiers["sol"]["usd"] / h["cost"]["own"]))
ov("LFFABLEGAP", f"{ANCHOR['acc'] - 100 * h['acc']:.1f}")
ov("LFFABLEFIVEGAP", f"{100 * h['acc'] - fable5:.1f}")
ov("LFMULT", x(ANCHOR["usd"] / h["cost"]["own"]))
ov("LFMULTRENT", x(ANCHOR["usd"] / h["cost"]["rent"]))
ov("LFMULTAMORT", x(ANCHOR["usd"] / h["cost"]["amort"]))
ov("LFUSD", usd(h["cost"]["own"]))
ov("LFUSDRENT", usd(h["cost"]["rent"]))
ov("LFUSDAMORT", usd(h["cost"]["amort"]))
ov("LFLOCALSHARE", f"{100 * h['shares'][0]:.1f}\\%")
ov("LFCLOSEDSHARE", f"{100 * h['shares'][2]:.1f}\\%")
ov("LFNOCLOSED", f"{100 * (1 - h['shares'][2]):.1f}\\%")
ov("LFASTRARENTMULT", x(tiers["astra"]["usd"] / h["cost"]["rent"]))
ov("LFGLMFIRSTCOSTX", f"{g2['cost']['own'] / h['cost']['own']:.1f}$\\times$")
ov("RETRYACC", f"{100 * PLANS['retry']['acc']:.1f}\\%")
ov("RETRYMULT", x(ANCHOR["usd"] / PLANS["retry"]["cost"]["own"]))
ov("RETRYMULTAMORT", x(ANCHOR["usd"] / PLANS["retry"]["cost"]["amort"]))
ov("RETRYLOCAL", f"{100 * PLANS['retry']['shares'][0]:.1f}\\%")

# owned vs rented ratio over the tiers served on owned accelerators
ratios = [MACH["h200_16"]["rent"] / MACH["h200_16"]["own"], tiers["glm53"]["rent"] / tiers["glm53"]["own"]]

# SWE / TB4 / tier counts / table 2
sw = T["swe"]
ov("SWECEIL", f"{D['swe']['headline']['acc']:.1f}")
ov("SWEFRONT", f"{sw['tiers']['opus5']['acc']:.1f}")
ov("SWELOCAL", f"{sw['tiers']['qwen38']['acc']:.1f}")
ov("SWETASKS", str(sw["n_tasks"]))
t4 = T["tb4"]
ov("TBFOURTASKS", str(t4["n_tasks"]))
ov("TBFOURLOCAL", f"{t4['tiers']['qwen38']['acc']:.1f}")
ov("TBFOURKIMI", f"{t4['tiers']['astra']['acc']:.1f}")
ov("TBFOURCOMP", f"{D['tb4']['localfirst']['acc']:.1f}")
ov("TBROWBEST", f"{100 * base_acc['astra']:.1f}")
ov("TBROWLOCAL", f"{100 * MACH[HEAD]['acc']:.1f}")      # round 15: TerminalBench V2.1's local stage is the laptop
ov("NTIERSSCORED", str(TX["n_tiers_scored"]))
ov("NTIERSENUM", str(TX["n_tiers_enum"]))
ov("NTIERSENUMWORD", TX["n_tiers_enum_word"])

# held-out selection (appendix)
ov("HOIN", f"{sig['heldout_in']:.1f}\\%")
ov("HOOUT", f"{sig['heldout_out']:.1f}\\%")
ov("HOGAP", f"{sig['heldout_in'] - sig['heldout_out']:.1f}")
ov("HOPICK", f"{sig['heldout_pick_pct']:.0f}\\%")
ov("HOHEAD", f"{100 * h['acc']:.1f}\\%")
# round 12: the held-out half's 5th-95th percentile follows the sampling model of the intervals (half of the tasks,
# each with five rolls at the composed-plan within-task correlation, finite-population correction); round 10 carried
# fixed offsets from an earlier round, narrower than that model allows
_nh = T["tb21"]["n_tasks"] / 2
_fpc = (T["tb21"]["n_tasks"] - _nh) / (T["tb21"]["n_tasks"] - 1)
_seh = (h["acc"] * (1 - h["acc"]) * (1 + (T["constants"]["k"] - 1) * PLAN_ICC["headline"]) / (T["constants"]["k"] * _nh) * _fpc) ** 0.5
ov("HOLO", f"{100 * (h['acc'] - 1.645 * _seh):.1f}\\%")
ov("HOHI", f"{100 * (h['acc'] + 1.645 * _seh):.1f}\\%")
ov("MULTAMORT", x(ANCHOR["usd"] / h["cost"]["rent"]))

# ladder narration for the objectives subsection (target plan names per kappa)
for i, (k, n, loc) in enumerate(new_own):
    OV["LADDEROWN" + "ABCDEFG"[i]] = n
for i, (k, n, loc) in enumerate(new_rent):
    OV["LADDERRENT" + "ABCDEFG"[i]] = n
ov("NLMTRENT", WORD[n_rent])


# ------------------------------------------------------------ facets (steps of the frontier) over the target plan space
def facets(S: dict, basis: str):
    accs = sorted(set(round(v["acc"], 4) for v in S.values()))
    steps, prev = [], None
    for a in accs:
        feas = [(v["cost"][basis], n, v["local"]) for n, v in S.items() if v["acc"] >= a - 5e-5]
        c, n, loc = min(feas)
        if n != prev:
            steps.append((a, n, S[n]["owned"])); prev = n
    return steps
fo, fr = facets(S_new, "own"), facets(S_new, "rent")
print("owned facets:", [(round(100*a,1), n[:40], l) for a, n, l in fo])
print("rented facets:", [(round(100*a,1), n[:40], l) for a, n, l in fr])
WORDS = {i: w for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen".split())}
ov("NFACETSOWNNEW", WORDS[len(fo)])
ov("NFACETSRENTNEW", WORDS[len(fr)])
ov("NLOCALFACETSOWN", WORDS[sum(1 for *_, l in fo if l)])
ov("NLOCALFACETSRENT", WORDS[sum(1 for *_, l in fr if l)])


def _facets_have(steps, word: str) -> str:   # round 47: never "five facets, of which five", never "of which one have"
    n = sum(1 for *_, l in steps if l)
    return ("all of which have" if n == len(steps) else "none of which has" if n == 0 else
            f"of which {word} has" if n == 1 else f"of which {word} have")


ov("OWNFACETSHAVE", _facets_have(fo, OV["NLOCALFACETSOWN"]))
ov("RENTFACETSHAVE", _facets_have(fr, OV["NLOCALFACETSRENT"]))
ov("EINFLIGHT", f"{MACH['h200_16']['gpu_share']:.1f}")   # realized tasks in flight (sixteen nominal), round 10
m1 = MACH["h200_1"]
one_plan = m1["rent"] + (h["cost"]["rent"] - MACH[HEAD]["rent"])     # the headline with an H200 serving one agent as its first stage
ov("EONEUSD", usd(m1["rent"]))
ov("HEADONEUSD", usd(one_plan))
ov("HEADONEMULT", x(ANCHOR["usd"] / one_plan))
ov("QWENRATIO", f"{MACH['h200_16']['rent'] / MACH['h200_16']['own']:.0f}")
ov("GLMRATIO", f"{tiers['glm53']['rent'] / tiers['glm53']['own']:.0f}")
ov("HEADRATIO", f"{h['cost']['rent'] / h['cost']['own']:.1f}")
ov("QWENIPJ", f"{MACH['h200_16']['acc'] / (MACH['h200_16']['kj_wall'] / 1000):.2f}")
ov("QWENIPD", f"{MACH['h200_16']['acc'] / MACH['h200_16']['own']:.1f}")


# ------------------------------------------------------------ macros requested by the verification pass
def glm_esc(closed):
    L = base_acc["glm53"]; return (L * sig["s2_fpr"] + (1 - L) * sig["s2_recall"])
ov("CascClosedGLMSol", f"{100 * glm_esc('sol'):.1f}\\%")
ov("CascClosedGLMAstra", f"{100 * glm_esc('astra'):.1f}\\%")
ov("CascRentBase", usd(MACH[HEAD]["rent"]))
ov("CascRentMultBase", x(ANCHOR["usd"] / MACH[HEAD]["rent"]))
def _kap(L, key, last=True):
    ks = [k for k, n, _ in L if key in n]
    if not ks:
        return "none"
    return f"{(ks[-1] if last else ks[0]):.2f}"
ov("KAPPADESKSOLO", _kap(new_own, "alone (M4 Pro)"))
ov("KAPPAFLASHSOLO", _kap(new_own, "Qwen3.8-Flash-Next alone"))
ov("KAPPARENTSOLO", _kap(new_rent, "Qwen3.8-27B alone"))
ov("KAPPARENTFLASHSOLO", _kap(new_rent, "Qwen3.8-Flash-Next alone"))
ov("AmortYears", "three")
ov("SPARKMEM", "128")   # round 47: Table 7 gives the nominal 128 GB that Table 3, targets.json and the dgxspark entry give; the canonical 121 disagreed
ov("TBGLMIPJ", f"{base_acc['glm53'] / (tiers['glm53']['kj_wall'] / 1000):.2f}")
ov("TBGLMIPD", f"{base_acc['glm53'] / tiers['glm53']['own']:.1f}")
ov("NMETERED", "six")
ov("HOCLEAR", "100\\%")
ov("OWNMULTLO", f"{MACH['h200_16']['rent'] / MACH['h200_16']['own']:.0f}")
ov("OWNMULTHI", "115")          # max of the Ratio column of tab:bases (GLM-5.2 campaign row)
ov("ACCSPANPP", f"{100 * base_acc['astra'] - 28.1:.1f}")   # strongest tier minus Gemma-4-31B (28.1, lowest measured)
ov("NLOCALFIRST", "five")
ov("QWENSIXLBACC", f"{TX['qwen36_acc']:.1f}")   # round 47: Qwen3.6-27B's leaderboard accuracy, the value the Qwen3.6 ladder (qwen36()) and the 60% figure floor use
ov("TBFOURRUNNABLEOLD", "63")
ov("TBFOURWRONG", "34")         # per-roll mean of Qwen3.8-27B's failed TB4 tasks that completed wrong
ov("TBFOURTIMEOUT", "28")       # per-roll mean that timed out (62 of 66 fail per roll at 5.6%)
ov("TBFOURLOCALN", "four")      # per-roll mean solved tasks (5.6% of 66)


# ------------------------------------------------------------ cross-benchmark table (tab:benchmarks; round 15, xbench spec 1 and 4)
# Per benchmark: (local-first acc, owned $), (open-weight-first acc, owned $), owned kJ of each where both are stored, all as
# printed (one decimal / usd()), so every gap and ratio quoted in the caption matches the cells.
def _p(a, c):
    return round(a, 1), float(usd(c))
_sw, _ga, _t4 = T["swe"]["plans"], T["gaia"]["plans"], T["tb4"]["plans"]
XB = {"tb21": (_p(100 * h["acc"], h["cost"]["own"]), _p(100 * g2["acc"], g2["cost"]["own"]), h["kj"], None),
      "swe": (_p(D["swe"]["headline"]["acc"], _sw["headline"]["own"]), _p(D["swe"]["glmfirst"]["acc"], _sw["glmfirst"]["own"]), None, None),
      "gaia": (_p(D["gaia"]["localfirst"]["acc"], _ga["localfirst"]["own"]), _p(D["gaia"]["glmfirst"]["acc"], _ga["glmfirst"]["own"]), None, None),
      "tb4": (_p(D["tb4"]["localfirst"]["acc"], _t4["localfirst"]["own"]), _p(D["tb4"]["glmfirst"]["acc"], _t4["glmfirst"]["own"]),
              _t4["localfirst"]["kj"], _t4["glmfirst"]["kj"])}
SHADE_RULE_PTS = 5.0     # shade the cheaper composition unless it gives up more than this many points to the other
EXPECT = {"tb21": "lf", "swe": "lf", "gaia": "of", "tb4": "of"}
for _b, ((la, lc), (oa, oc), lkj, okj) in XB.items():
    if lc < oc:
        win = "lf" if oa - la <= SHADE_RULE_PTS else "of"
    elif oc < lc:
        win = "of" if la - oa <= SHADE_RULE_PTS else "lf"
    else:                                   # equal printed cost: the higher accuracy, then the fewer owned-stage Joules
        win = "lf" if la > oa else "of" if oa > la else ("lf" if (lkj or 0) < (okj or 0) else "of")
    if win != EXPECT[_b]:
        FAILS.append(f"tab:benchmarks shading: {_b} winner is {win}, the table shades {EXPECT[_b]}")
_obs = [XB["tb21"], XB["swe"]]
_gaps = [oa - la for (la, _), (oa, _), *_ in _obs]
_costs = [oc / lc for (_, lc), (_, oc), *_ in _obs]
ov("XBObsGapLo", f"{min(_gaps):.1f}")
ov("XBObsGapHi", f"{max(_gaps):.1f}")
ov("XBObsCostLo", f"{min(_costs):.1f}")                 # printed as "2.5 to 2.7$\\times$"
ov("XBObsCostHi", x(max(_costs)))
ov("XBGAIAGap", f"{XB['gaia'][1][0] - XB['gaia'][0][0]:.1f}")


# ------------------------------------------------------------ Figure 1 caption macros (canonical names)
ov("CompAccGap", f"{ANCHOR['acc'] - 100 * h['acc']:.1f}")
ov("CompMult", x(ANCHOR["usd"] / h["cost"]["own"]))
ov("PanelCAcc", f"{100 * h['acc']:.1f}\\%")
ov("PanelCUSD", usd(h["cost"]["own"]))
ov("PanelCOwn", f"{100 * (1 + h['E1']) / (1 + h['E1'] + h['shares'][2]):.0f}\\%")
ov("PanelCMult", x(ANCHOR["usd"] / h["cost"]["own"]))
ov("PanelCNoClosed", f"{100 * (1 - h['shares'][2]):.1f}\\%")   # round 35: Figure 1 caption, tasks never reaching a closed-source provider
ov("PanelCGap", f"{ANCHOR['acc'] - 100 * h['acc']:.1f}")


# ------------------------------------------------------------ round 10 (2026-09-25): machines from evidence, intervals
MN = NS["MACROS"]
# (a) Table 1 accuracy intervals: planned 89 x 5 design, within-task correlation per tier (constants.icc)
for tag, key in rows.items():
    if tag == "Base":
        a, rho = MACH[HEAD]["acc"], ICC["qwen38"]
    else:
        a, rho = PLANS[key]["acc"], PLAN_ICC[key]
    OV["CascAccPM" + tag] = f"{100 * acc_hw(a, rho):.1f}"
for tag, r, cl in (("GLMSol", g1, "sol"), ("GLMAstra", g2, "astra")):
    _E = glm_esc(cl)
    OV["CascAccPM" + tag] = f"{100 * acc_hw(r['acc'], plan_icc(('glm53', cl), (1 - _E, _E))):.1f}"
ov("CascAccPMAstraT", f"{100 * acc_hw(base_acc['astra'], ICC['astra']):.1f}")
hw_head, hw_astra = acc_hw(h["acc"], PLAN_ICC["headline"]), acc_hw(base_acc["astra"], ICC["astra"])
_p1, _h1 = round(100 * h["acc"], 1), round(100 * hw_head, 1)      # round 12: the printed point +/- printed half-width
LFCI = (_p1 - _h1, _p1 + _h1)
ov("LFACCCI", f"[{LFCI[0]:.1f}, {LFCI[1]:.1f}]")
# paired headline vs GPT-6 Astra (round 12): one joint per-task model. Rolls are independent across systems given the
# task, so the covariance of the two systems' per-task solve rates is P(both) - pH pA = pH - P(plan only) - pH pA, with
# P(plan only) the printed discordance target (compat.cover_only_plan); each system's task-mean variance carries its own
# within-task correlation (constants.icc). Round 10 borrowed the correlation of task means (0.845) from a measured
# analog, which is inconsistent with the discordance the paper prints.
_n, _k = T["tb21"]["n_tasks"], T["constants"]["k"]
_pH, _pA = h["acc"], base_acc["astra"]
_cov = _pH - TX["cover_only_plan"] / 100 - _pH * _pA
_vH = _pH * (1 - _pH) * (1 + (_k - 1) * PLAN_ICC["headline"]) / _k
_vA = _pA * (1 - _pA) * (1 + (_k - 1) * ICC["astra"]) / _k
if not 0 <= _cov <= (PLAN_ICC["headline"] * _pH * (1 - _pH) * ICC["astra"] * _pA * (1 - _pA)) ** 0.5:
    FAILS.append("the discordance targets imply a covariance outside what the within-task correlations allow")
R_JOINT = _cov / (_vH * _vA) ** 0.5
_gap = 100 * (_pA - _pH)
_dh = 100 * 1.96 * ((_vH + _vA - 2 * _cov) / _n) ** 0.5
GAPCI = (_gap - _dh, _gap + _dh)
ov("LFASTRAGAPRESOLVED", "" if GAPCI[0] > 0 else ", so the gap is not resolved at 95\\%")   # round 43: "on the planned design" removed (projection wording)
_sg = lambda v: (f"$-${-v:.1f}" if round(v, 1) < 0 else f"{v:.1f}")    # typeset minus (round 12)
ov("LFASTRAGAPCI", f"[{_sg(GAPCI[0])}, {_sg(GAPCI[1])}]")
ov("LFMULTCI", MN["TBHeadOwnMultCI"])
if not (LFCI[0] <= 100 * h["acc"] <= LFCI[1] and GAPCI[0] <= _gap <= GAPCI[1]):
    FAILS.append("an accuracy interval excludes its point")
if not LFCI[0] <= ANCHOR["acc"] <= LFCI[1] or not LFCI[0] <= fable5 <= LFCI[1]:
    FAILS.append("text: the Fable 5.1 and Fable 5 anchors are said to lie inside the headline's 95% interval")

ov("SOLESCPREM", f"{PC('sol'):.2f}")                 # per-target premiums (constants.premium_by_target)
ov("GLMESCPREM", f"{NS['PG']:.2f}")
# (b) the intro's token premium range, per receiving model (compat.esc_tokens_premium)
_tp = [v for k, v in TX["esc_tokens_premium"].items() if k != "src"]
ov("ESCPREMLO", f"{min(_tp):.1f}")
ov("ESCPREMHI", f"{max(_tp):.1f}")

# (c) machine text: the laptop row, the campaign wall at the realized occupancy
_d, _g = MACH["m4pro"], MACH["h200_16"]
ov("DESKCUT", f"{100 * _d['cut']:.0f}\\%")
ov("DESKGPUGAP", f"{100 * (_g['acc'] - _d['acc']):.1f}")
ov("QWENCAMPAIGNS", f"{N * _g['wall_s'] / _g['gpu_share']:,.0f}".replace(",", "{,}"))
if not all(MACH[k]["kj_wall"] > _g["kj_wall"] for k in ("m5max", "m5max_q40", "m4max", "m3ultra", "spark", "strix")):
    FAILS.append("text: every other desk machine is said to draw more per task than the batched H200")
# (c2) round 12: the laptop and rented-basis comparisons are generated from the numbers they state (which machine
# draws fewer Joules, and whether a single model beats the plan on rented hardware), not asserted by the prose
_LV = NS["DERIVED"]["laptop_vs_batched_h200"]
_ra, _rs = _g["kj_wall"] / _d["kj_wall"], _g["kj_solved"] / _d["kj_solved"]      # batched H200 over laptop
_one = MACH["h200_1"]["kj_solved"] / _d["kj_solved"]
_fold = lambda r: r if r >= 1 else 1 / r
_pt, _lo, _hi = _LV["sol"]
_ci = (_lo, _hi) if _rs >= 1 else (1 / _hi, 1 / _lo)
OV["MacVsGPU"] = f"{_fold(_ra):.1f}$\\times$"
OV["MacVsGPUSolved"] = f"{_fold(_rs):.1f}$\\times$"
OV["MacVsGPUSolvedCI"] = f"[{_ci[0]:.2f}, {_ci[1]:.2f}]"
_unres = _ci[0] <= 1.0
_lap_ahead = _rs >= 1
if _lap_ahead:
    ov("ABSLAPTOP", "A 48\\,GB laptop runs its local stage at \\MacVsGPUSolved\\ fewer joules per solved task than a datacenter "
                    "accelerator serving sixteen agents" + (", a margin within sampling," if _unres else "") +
                    " and \\MachMFourProVsOne\\ fewer than one serving a single agent.")
    ov("MACTABTITLE", "On the same tasks, the same model completes a solved task for \\MacVsGPUSolved\\ fewer joules on the "
                      "48\\,GB laptop than on a datacenter accelerator serving sixteen agents, and \\MachMFourProVsOne\\ fewer "
                      "than on one serving a single agent.")
    ov("DESKGPUSENT", "that desk-class machine draws fewer joules per task it solves than the batched H200: \\DESKKJSOLVED\\ "
                      "against \\GPUKJSOLVED\\,kJ at the wall, \\MacVsGPUSolved\\ fewer (paired 95\\% interval "
                      "\\MacVsGPUSolvedCI)" + (", a margin within sampling" if _unres else "") + "; per attempted task the "
                      "laptop draws \\DESKKJ\\,kJ and the batched H200 \\GPUKJ\\,kJ.")
else:
    ov("ABSLAPTOP", "A 48\\,GB laptop runs its local stage at \\MachMFourProVsOne\\ fewer joules per solved task than a "
                    "datacenter accelerator serving a single agent, but at \\MacVsGPUSolved\\ more than one serving sixteen "
                    "agents" + (", a margin within sampling." if _unres else "."))
    ov("MACTABTITLE", "On the same tasks, a datacenter accelerator serving sixteen agents completes a solved task for "
                      "\\MacVsGPUSolved\\ fewer joules than the 48\\,GB laptop, and one serving a single agent for "
                      "\\MachMFourProVsOne\\ more.")
    ov("DESKGPUSENT", "the batched H200 draws fewer joules per task it solves than that desk-class machine: \\GPUKJSOLVED\\ "
                      "against \\DESKKJSOLVED\\,kJ at the wall, \\MacVsGPUSolved\\ fewer (paired 95\\% interval "
                      "\\MacVsGPUSolvedCI)" + (", a margin within sampling" if _unres else "") + ", because the 48\\,GB laptop's slower "
                      "attempts lose solves to the time budget; per attempted task the laptop draws \\DESKKJ\\,kJ and the "
                      "batched H200 \\GPUKJ\\,kJ.")
ov("DESKALLPHRASE", "Every other desk machine" if _d["kj_wall"] < _g["kj_wall"] else "Every desk machine, the laptop included,")
_rd_names = None
_rd = sorted(NS["DERIVED"]["rented_dominators"], key=lambda d: d[1])
_names = []
for _a, _u, _lab in _rd:
    _nm = _lab.split(" (")[0].replace("GPT-6 Astra", "GPT-6~Astra")
    if _nm not in _names:
        _names.append(_nm)
_nlist = (_names[0] if len(_names) == 1 else ", ".join(_names[:-1]) + (", and " if len(_names) > 2 else " and ") + _names[-1])
ov("RENTDOMPROSE", (_nlist + " served alone " + ("is" if len(_names) == 1 else "are each") + " both") if _names else
   "no single model is both")
_hr, _ar = h["cost"]["rent"], tiers["astra"]["usd"]
if _hr <= _ar:
    ov("LFASTRARENTCMP", "\\LFASTRARENTMULT\\ below GPT-6~Astra served alone" +
       ("; on this basis " + OV["RENTDOMPROSE"] + " more accurate and cheaper than the plan" if _names else ""))
else:
    _desk_first = T["tb21"]["machines"][HEAD]["kind"] == "desk"
    ov("LFASTRARENTCMP", f"{_hr / _ar:.1f}$\\times$ the cost of GPT-6~Astra served alone, because "
                         + ("the laptop's amortized list price adds \\$\\CascRentBase\\ to every task and " if _desk_first else "")
                         + "GLM-5.3 pays for its share "
                         "of eight H200s at the \\GLMINFLIGHT\\ tasks its campaign kept in flight, \\$\\TBGLMRent\\ per "
                         "attempt before the escalation premium; on this basis " + OV["RENTDOMPROSE"] + " more accurate and "
                         "cheaper than the plan" + (" (GPT-5.6-Terra at its leaderboard \\TBTerraAAAcc\\% and \\$\\TBTerraAAUSD, Figure~\\ref{fig:ipd-ipj})" if "GPT-5.6-Terra" in _names else "") + ", and Qwen3.8-27B escalating straight to GPT-6~Astra is the cheapest composition "
                         "that reaches the headline's accuracy (\\S\\ref{sec:middleware})")   # round 25
    if not (PLANS["two_astra"]["acc"] >= h["acc"] and PLANS["two_astra"]["cost"]["rent"] < _hr):
        FAILS.append("text (4.1): Qwen3.8-27B -> GPT-6 Astra is said to be cheaper at the headline's accuracy on rented hardware")
_open_gap = 100 * (base_acc["glm53"] - PLANS["openonly"]["acc"])
ov("TBOpenBelowGLM", f"{_open_gap:.1f}")
if not 0 < _open_gap < 1:
    FAILS.append("text: Qwen3.8-27B -> GLM-5.3 is said to land just below GLM-5.3 served alone")
_ts, _hd = PLANS["two_astra"], PLANS["headline"]          # round 25: the rented pick is Qwen3.8-27B -> GPT-6 Astra
if not (_ts["acc"] >= _hd["acc"] and _ts["cost"]["rent"] < _hd["cost"]["rent"] and _hd["cost"]["own"] < _ts["cost"]["own"]):
    FAILS.append("text (4.2): the bases no longer pick different Qwen3.8-27B systems at the headline's accuracy")
_q_at = {k: o for k, o in PLANS.items() if T["tb21"]["plans"][k]["local"] == "qwen38" and T["tb21"]["plans"][k]["machine"] == HEAD
         and o["acc"] >= _hd["acc"] - 1e-12}          # the laptop systems of Table 1
if min(_q_at, key=lambda k: _q_at[k]["cost"]["rent"]) != "two_astra" or min(_q_at, key=lambda k: _q_at[k]["cost"]["own"]) != "headline":
    FAILS.append("text (4.2): at the headline's accuracy the owned pick must be the headline and the rented pick Qwen3.8-27B -> GPT-6 Astra")

# (d) ladder prose, generated from the ladders it narrates
PROSE = {"GLM-5.3 alone": "GLM-5.3 served alone", "GPT-5.6-sol alone": "GPT-5.6-sol served alone",
         "GPT-6 Astra alone": "GPT-6~Astra served alone",
         "GLM-5.3 $\\rightarrow$ GPT-5.6-sol": "GLM-5.3 escalating to GPT-5.6-sol",
         "GLM-5.3 $\\rightarrow$ GPT-6 Astra": "GLM-5.3 escalating to GPT-6~Astra",
         "Qwen3.8-27B alone (M5 Max)": "Qwen3.8-27B alone on a MacBook Pro M5~Max",
         "Qwen3.8-27B alone (M4 Pro)": "Qwen3.8-27B alone on a 48\\,GB MacBook Pro M4~Pro",
         "Qwen3.8-Flash-Next alone": "Qwen3.8-Flash-Next alone on two owned H200s",
         "Qwen3.8-27B $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-6 Astra": "Qwen3.8-27B through GLM-5.3 to GPT-6~Astra",
         "the same, local stage on an M5 Max": "the headline system with its local stage on an M5~Max",
         "Qwen3.8-27B $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-5.6-sol": "Qwen3.8-27B through GLM-5.3 to GPT-5.6-sol",
         "Qwen3.8-27B $\\rightarrow$ GLM-5.3": "Qwen3.8-27B escalating to GLM-5.3",
         "Qwen3.8-27B (M5 Max) $\\rightarrow$ GLM-5.3": "Qwen3.8-27B on an M5~Max escalating to GLM-5.3",
         "Qwen3.8-27B $\\rightarrow$ GPT-6 Astra": "Qwen3.8-27B escalating to GPT-6~Astra",
         "Qwen3.8-27B $\\rightarrow$ GPT-5.6-sol": "Qwen3.8-27B escalating to GPT-5.6-sol",
         "Qwen3.8-27B, retried once, $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-6 Astra": "Qwen3.8-27B retried once through GLM-5.3 to GPT-6~Astra",
         "Qwen3.8-Flash-Next $\\rightarrow$ GLM-5.3 $\\rightarrow$ GPT-6 Astra": "Qwen3.8-Flash-Next through GLM-5.3 to GPT-6~Astra"}


def _prose(n: str, basis: str) -> str:
    if n == "Qwen3.8-27B alone":        # round 15: the headline's desk machine on both bases
        return ("Qwen3.8-27B alone on a MacBook Pro M5~Max" if T["tb21"]["machines"][HEAD]["hw"] == "m5max" else
                "Qwen3.8-27B alone on an owned H200" if basis == "own" else "Qwen3.8-27B alone on a shared H200")
    if n == "Qwen3.8-Flash-Next alone" and basis == "rent":   # round 47: the rented ladder names no owned H200s
        return "Qwen3.8-Flash-Next alone on two H200s"
    return PROSE[n]


def _narrate(L, basis: str) -> str:
    runs = []
    for k, n, _ in L:
        if runs and runs[-1][2] == n:
            runs[-1][1] = k
        else:
            runs.append([k, k, n])
    parts = [(_prose(n, basis) + (" is cheapest" if i == 0 else "") + (f" from {a:.2f} to {b:.2f}" if b > a else f" at {a:.2f}"))
             for i, (a, b, n) in enumerate(runs)]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]


ov("LADDERNARROWN", _narrate(new_own, "own"))
ov("LADDERNARRENT", _narrate(new_rent, "rent"))
_loc = [k for k, _, l in new_own if l]
if _loc != KAPPAS[:len(_loc)]:
    FAILS.append("text: the owned ladder is said to start from a local stage at every target up to the last local one")
ov("KAPPALOCALTOP", f"{_loc[-1]:.2f}" if _loc else "none")
_nl = [(k, n) for k, n, l in new_own if not l]
if not all(S_new[n]["owned"] for _, n in _nl):
    FAILS.append("text: every owned target without a local stage is said to start from an open-weight model on owned accelerators")
_nlp = [f"at {k:.2f} {_prose(n, 'own')}" for k, n in _nl]
ov("LADDERNOTLOCALOWN", (_nlp[0] + " is cheaper" if len(_nlp) == 1 else
                         _nlp[0] + " is cheaper" + "".join(", " + p for p in _nlp[1:-1]) + ", and " + _nlp[-1]) if _nlp else "")
ov("KAPPANOTLOCALFIRST", f"{_nl[0][0]:.2f}" if _nl else "none")
ov("NLMTOTHER", WORD[len(KAPPAS) - n_new])
ov("NLMTRENTOF", ("all " + WORD[n_rent]) if n_rent == len(KAPPAS) else WORD[n_rent])
# round 15: with every Qwen3.8-27B-first system on the laptop, the rented basis charges the laptop's amortized list price and
# the top target can fall to a single closed model there; the implication's clause is generated from the ladders it states
# (was a fixed claim, "on both bases", checked here)
_top_comp = {"own": "rightarrow" in new_own[-1][1], "rent": "rightarrow" in new_rent[-1][1]}
if not any(_top_comp.values()):
    FAILS.append("text (Appendix, implication): the top target is said to be reached by a composed system on at least one basis")
ov("IMPLTOPCLAUSE", ("on both bases" if all(_top_comp.values()) else "on the owned basis" if _top_comp["own"] else "on the rented basis")
   + " the \\KAPPAHIGH\\ target is reached by a composed one rather than a single configuration")
if not all(_top_comp.values()):
    print("NOTE (round 15): the implication paragraph's fixed 'on both bases the \\KAPPAHIGH\\ target is reached by a composed one' "
          "is now false; use \\IMPLTOPCLAUSE:", OV["IMPLTOPCLAUSE"])
_fown = [f for f in fo if f[2]]
_last = max(i for i, f in enumerate(fo) if f[2])
if any(not f[2] for f in fo[:_last]) or not all(f[1] in ("GPT-5.6-sol alone", "GPT-6 Astra alone") for f in fo[_last + 1:]):
    FAILS.append("text (Appendix, facets): owned facets are said to be contiguous and followed only by single closed models")
ov("FACETNARROWN", ", then ".join(_prose(n, "own") + (" at the lowest capabilities" if i == 0 else "") for i, (a, n, _) in enumerate(_fown)))
ov("FACETTOPOWNACC", f"{100 * S_new[_fown[-1][1]]['acc']:.1f}")
# round 15: what reaches past the top owned facet, generated (the fixed "above it only GPT-5.6-sol and GPT-6~Astra served alone
# reach further" is false once the one deployed threshold lifts Qwen3.8-Flash-Next -> GLM-5.3 -> GPT-6 Astra above both)
_above = [n for n in ("GPT-5.6-sol alone", "GPT-6 Astra alone") if S_new[n]["acc"] > S_new[_fown[-1][1]]["acc"] + 5e-5]
_nm = {"GPT-5.6-sol alone": "GPT-5.6-sol", "GPT-6 Astra alone": "GPT-6~Astra"}
ov("FACETABOVEPHRASE", ("above it only " + " and ".join(_nm[n] for n in _above) + " served alone " + ("reaches" if len(_above) == 1 else "reach")
                        + " further") if _above else "no plan we cost reaches further")
if not all(n in ("GPT-5.6-sol alone", "GPT-6 Astra alone") for n in [n for n, v in S_new.items() if v["acc"] > S_new[_fown[-1][1]]["acc"] + 5e-5]):
    FAILS.append("text (Appendix, facets): only single closed models are said to reach past the top owned facet")
_fl, _so = round(100 * PLANS["flashnext"]["acc"], 1), round(100 * base_acc["sol"], 1)
ov("TBFlashVsSolPhrase", f"within \\TBFlashVsSol\\ points of GPT-5.6-sol" if _fl < _so else
   ("level with GPT-5.6-sol and GPT-6~Astra served alone" if _fl == _so else
    f"{_fl - _so:.1f} points above GPT-5.6-sol and GPT-6~Astra served alone"))
_w_own = min((v["cost"]["own"], n) for n, v in S_new.items() if v["acc"] >= h["acc"] - 5e-5)[1]
_w_rent = min((v["cost"]["rent"], n) for n, v in S_new.items() if v["acc"] >= h["acc"] - 5e-5)[1]
if _w_own == _w_rent:
    FAILS.append("text (Appendix, facets): at the headline's accuracy the composition that wins is said to differ between bases")
if not any("Terra" in d[2] for d in NS["DERIVED"]["rented_dominators"]):
    FAILS.append("text (conclusion): GPT-5.6-Terra is said to be more accurate and cheaper than the headline on rented hardware")


# (e) projection Monte Carlo: every machine parameter, format delta and GLM-5.3 node parameter (round 12) drawn over its
# evidence range; the signal, conditional solve rates, premiums and within-task correlations stay at their central values
def projection_mc() -> dict:
    import numpy as np
    from collections import Counter, defaultdict
    rng = np.random.default_rng(IV["mc_seed"])
    saved_m, saved_p, saved_g = dict(MACH), dict(PLANS), dict(tiers["glm53"])
    rec, n_own, n_rent = defaultdict(list), Counter(), Counter()
    try:
        for _ in range(IV["mc_draws"]):
            _prm = NS["mm_params"](rng)
            _, M = NS["machines_from"](_prm)
            MACH.update(M)
            tiers["glm53"].update({k: v for k, v in NS["glm_tier"](_prm).items() if k in ("own", "amort", "rent", "kj_wall")})
            PLANS.update({k: plan(p) for k, p in T["tb21"]["plans"].items()})
            for k, m in M.items():
                rec[k + ":kj"].append(m["kj_wall"])
                rec[k + ":acc"].append(100 * m["acc"])
            rec["mac_vs_gpu_solved"].append(M["h200_16"]["kj_solved"] / M["m4pro"]["kj_solved"])
            rec["other_desk_above"].append([M[k]["kj_wall"] > M["h200_16"]["kj_wall"] for k in ("m5max", "m5max_q40", "m4max", "m3ultra", "spark", "strix")])
            hh = PLANS["headline"]
            rec["head_acc"].append(100 * hh["acc"])
            rec["head_own_mult"].append(ANCHOR["usd"] / hh["cost"]["own"])
            rec["head_rent_mult"].append(ANCHOR["usd"] / hh["cost"]["rent"])
            S = space(True)
            n_own[sum(1 for *_, l in ladder(S, "own") if l)] += 1
            n_rent[sum(1 for *_, l in ladder(S, "rent") if l)] += 1
    finally:
        MACH.clear(); MACH.update(saved_m)
        PLANS.clear(); PLANS.update(saved_p)
        tiers["glm53"].clear(); tiers["glm53"].update(saved_g)
    q = lambda k, a: float(np.quantile(rec[k], a))
    return dict(rec=rec, q=q, n_own=n_own, n_rent=n_rent, n=IV["mc_draws"],
                other=np.array(rec["other_desk_above"]).mean(0))


MC = projection_mc()
_q = MC["q"]
_pc = lambda v: f"{100 * v / MC['n']:.0f}\\%" if 100 * v / MC["n"] >= 1 else "under 1\\%"
_share = lambda c: ", ".join(f"{WORD[k]} in {_pc(v)}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], -kv[0])))
_andlast = lambda s: (s.rsplit(", ", 1)[0] + ", and " + s.rsplit(", ", 1)[1]) if s.count(", ") >= 2 else s.replace(", ", " and ")
ov("MCNDRAWS", f"{MC['n']:,}".replace(",", "{,}"))
ov("MCDeskKJRange", f"{_q('m4pro:kj', 0.05):.0f} to {_q('m4pro:kj', 0.95):.0f}")
ov("MCGPUKJRange", f"{_q('h200_16:kj', 0.05):.0f} to {_q('h200_16:kj', 0.95):.0f}")
ov("MCMacVsGPULo", f"{_q('mac_vs_gpu_solved', 0.05):.2f}")
ov("MCMacVsGPUHi", f"{_q('mac_vs_gpu_solved', 0.95):.2f}")
ov("MCMacAhead", f"{100 * sum(1 for v in MC['rec']['mac_vs_gpu_solved'] if v > 1) / MC['n']:.0f}\\%")
ov("MCOtherDeskAbove", f"{100 * min(MC['other']):.0f}\\%")
ov("MCOtherDeskAbovePhrase", "in every draw" if min(MC["other"]) >= 1.0 else "in at least \\MCOtherDeskAbove\\ of draws")
ov("MCHeadAccLo", f"{_q('head_acc', 0.05):.1f}")
ov("MCHeadAccHi", f"{_q('head_acc', 0.95):.1f}")
ov("MCHeadMultLo", x(_q("head_own_mult", 0.05)))
ov("MCHeadMultHi", x(_q("head_own_mult", 0.95)))
ov("MCHeadRentMultLo", x(_q("head_rent_mult", 0.05)))
ov("MCHeadRentMultHi", x(_q("head_rent_mult", 0.95)))
ov("MCLadderOwn", _andlast(_share(MC["n_own"])))
ov("MCLadderRent", _andlast(_share(MC["n_rent"])))
ov("MCAccLo", f"{min(_q(k + ':acc', 0.05) for k in NS['TAB2']):.1f}")
ov("MCAccHi", f"{max(_q(k + ':acc', 0.95) for k in NS['TAB2']):.1f}")
for k in NS["TAB2"]:
    c = MACH[k]["kj_wall"]
    if not _q(k + ":kj", 0.05) <= c <= _q(k + ":kj", 0.95):
        print(f"note: {k} central {c:.0f} kJ lies outside its projection range")
print("projection MC: share of draws each other desk machine draws more per task than the batched H200:",
      dict(zip(("m5max", "m5max_q40", "m4max", "m3ultra", "spark", "strix"), (round(float(v), 3) for v in MC["other"]))))
print("projection MC:", {k: OV[k] for k in OV if k.startswith("MC")})

# ------------------------------------------------------------ emit
canon = (ROOT / "inputs" / "canonical-numbers-fork-20260924-1240.tex").read_text()
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", canon))
target = (ROOT / "paper" / "ipj-numbers-target.tex").read_text()
out = ["% ipj-numbers.tex for the target draft: canonical macro file, then target overrides.",
       "% GENERATED by targets/compat.py. Do not hand-edit.", "", canon, "",
       "% ======================= target overrides (targets/compat.py) =======================", ""]
for k in sorted(OV):
    cmd = "\\renewcommand" if k in defined else "\\newcommand"
    out.append(f"{cmd}{{\\{k}}}{{{OV[k]}}}")
out += ["", "% ======================= target macros (targets/make_numbers.py) =======================", "", target.replace("\\newcommand{", "\\providecommand{")]
(ROOT / "paper" / "ipj-numbers.tex").write_text("\n".join(out) + "\n")
print(f"{len(OV)} overrides; flags-only headline {100 * flags['acc']:.2f}% at ${flags['cost']['own']:.4f}; lifts {lifts}")
print(f"GLM-first: sol {100 * g1['acc']:.2f}% ${g1['cost']['own']:.4f}/{g1['cost']['rent']:.3f}; astra {100 * g2['acc']:.2f}% ${g2['cost']['own']:.4f}/{g2['cost']['rent']:.3f}")
print(f"kappa_max owned {kmax(S_new, 'own'):.3f}; without Flash-Next {kmax(S_no_flash, 'own'):.3f}; rented {kmax(S_new, 'rent'):.3f}")
if FAILS:
    print("CHECKS FAILED:", *FAILS, sep="\n  - ")
    sys.exit(1)
print("compat checks passed")
