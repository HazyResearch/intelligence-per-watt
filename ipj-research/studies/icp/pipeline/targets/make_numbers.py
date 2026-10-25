#!/usr/bin/env python3
"""Emit paper/ipj-numbers-target.tex and targets/derived.json from targets/targets.json.

Every composed quantity is derived here from its primitives; nothing derived is stored in
targets.json. The script fails (non-zero exit) if any arithmetic check fails.

    python3 targets/make_numbers.py            # write macros + derived.json, run checks
    python3 targets/make_numbers.py --measured # also report how many values are still targets
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = json.loads((ROOT / "targets" / "targets.json").read_text())
C = T["constants"]
PUB = T["public"]
PREM = C["premium"]
# round 10: each escalated leg carries the premium measured for its own target (constants.premium_by_target)
PB = C.get("premium_by_target", {})
PG = PB.get("glm53", PREM)       # GLM-5.3 middle stage (energy, hours, owned dollars scale with agent time)
PR = PB.get("retry", PREM)       # Qwen3.8-27B retried on its own fired attempts
PC = lambda name: PB.get(name, PREM)   # closed leg: GPT-6 Astra or GPT-5.6-sol
# round 23: a closed leg that receives only tasks both earlier stages escalated, or GLM-5.3's escalations, carries its
# own measured cost ratio over the singly escalated set (constants.premium_depth)
PD = C.get("premium_depth", {})
PC3 = lambda name: PC(name) * PD.get(name, {}).get("after_glm", 1.0)
PCG = lambda name: PC(name) * PD.get(name, {}).get("glm_first", 1.0)
ANCHOR = PUB["aa_tb21"]["fable51_max"]  # Fable 5.1 max, the TB2.1 cost anchor
FAILS: list[str] = []
MACROS: dict[str, str] = {}
DERIVED: dict = {}


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def mac(name: str, value: str, comment: str = "") -> None:
    assert name.isalpha(), name
    assert name not in MACROS, f"duplicate macro {name}"
    MACROS[name] = value + (f"   % {comment}" if comment else "")


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def usd(x: float) -> str:
    if x >= 1:
        return f"{x:.2f}"
    if x >= 0.1:
        return f"{x:.3f}"
    if x >= 0.01:
        return f"{x:.3f}"
    return f"{x:.4f}"


def mult(x: float) -> str:
    if x >= 100:
        return f"{x:,.0f}".replace(",", "{,}")
    if x >= 10:
        return f"{x:.0f}"
    return f"{x:.1f}"


def kwh_usd(kj: float) -> float:
    return kj / 3600.0 * C["elec_usd_per_kwh"]


# ---------------------------------------------------------------- machines (TB2.1 local stage)
tb = T["tb21"]
tiers = tb["tiers"]
sig = tb["signal"]
hw = PUB["hardware"]
base_acc = {k: v["solved"] / v["graded"] for k, v in tiers.items()}


# ---------------------------------------------------------------- machine model (round 10, 2026-09-25)
# Each machine's time and energy follow from its own decode speed, prefill speed and power (tb21.machine_model,
# every parameter with its source) applied to one measured workload: the 445 E1 Qwen3.8-27B attempts
# (measured.e1_qwen38_attempts), scaled to AA's output tokens per attempt. Wall energy uses each platform's own
# conversion, phase by phase. Accuracy = leaderboard base + format delta - solves lost to attempts cut at the budget.
MM = tb["machine_model"]
MMC, MMW = MM["common"], MM["workload"]
IV = C["intervals"]
ICC = C["icc"]
NB = IV["n_boot"]
K_ROLLS = C["k"]
PARTIAL = MMW["partial_solve_after_cut"]["c"]


def _c(p):
    return p["c"] if isinstance(p, dict) else p


def _e1_profile() -> dict:
    e = T["measured"]["e1_qwen38_attempts"]
    cols = e["cols"].split(",")
    rows = sorted((dict(zip(cols, r.split(","))) for r in e["rows"]), key=lambda r: r["task"])   # stable: rolls keep order
    f = lambda k: np.array([float(r[k]) for r in rows])
    tout, tin, nreq, agent, llm, total, to1 = (f(k) for k in ("tout", "tin", "nreq", "agent", "llm", "total", "to1"))
    solved = np.array([r["rew"] != "" and float(r["rew"]) >= 1 for r in rows])
    s = C["tokens_out_per_task_qwen38_aa"] / tout.mean()          # AA output tokens per attempt / E1 mean
    R = np.maximum(1.0, nreq)
    c0 = _c(MMW["c0_first_prompt"])
    d_pre = tin / R
    U = np.where(R == 1, tin, np.maximum(c0, 2 * tin / R - c0))     # uncached prefill (see the c0 source)
    tasks = sorted(set(r["task"] for r in rows))
    ti = np.array([tasks.index(r["task"]) for r in rows])
    return dict(O=tout * s, U=U * s, Dp=d_pre * s, Dd=(d_pre + tout / (2 * R)) * s,
                tool=np.maximum(0.0, agent - llm) * s, over=np.maximum(0.0, total - agent), solved=solved,
                B=np.maximum(MMW["budget_floor_s"]["c"], to1), scale=s, tasks=tasks, ti=ti)


PROF = _e1_profile()
N_TASKS, N_ATT = len(PROF["tasks"]), len(PROF["ti"])
TI = np.arange(N_ATT).reshape(N_TASKS, K_ROLLS)                     # attempt index by (task, roll)
_FD = np.array(MMW["depth_factor"]["pts"], dtype=float)


def _tri(rng, p):
    lo, hi = p["range"]
    return p["c"] if hi <= lo else float(rng.triangular(lo, p["c"], hi))


def mm_params(rng=None):
    """Scalar parameters: central values, or one joint projection draw (shared parameters drawn once per draw,
    one draw per quantization format so identical formats keep identical deltas). The GLM-5.3 node (round 12)
    is drawn with the machines."""
    val = (lambda p: _c(p)) if rng is None else (lambda p: _tri(rng, p) if "range" in p else _c(p))
    com = {k: val(v) for k, v in MMC.items() if isinstance(v, dict) and "c" in v}
    mach = {k: {kk: val(vv) for kk, vv in m.items() if isinstance(vv, dict) and "c" in vv}
            for k, m in MM["machines"].items()}
    mach["_glm53"] = {kk: val(vv) for kk, vv in MM["glm53"].items() if isinstance(vv, dict) and "c" in vv}
    q = {k: val(v) for k, v in MM["acc_q"].items() if isinstance(v, dict)}
    return com, mach, q


def _mtp(m: dict, com: dict) -> tuple[float, float, float]:
    """(decode speedup, decode power factor, prefill factor) for the machine's engine and format (round 12):
    Metal K-quants gain little from MTP, Metal Q8_0 gains and slows prefill, CUDA/ROCm engines use common.mtp."""
    cls = m.get("mtp_class", "")
    if cls == "metal_kquant":
        return com["mtp_metal_kquant"], com["mtp_power_metal_kquant"], 1.0
    if cls == "metal_q8":
        return com["mtp_metal_q8"], com["mtp_power"], com["mtp_prefill_metal_q8"]
    if cls == "metal_q40":      # round 15: #29110's multi-column kernels cover Q4_0 too; MTP prefill penalty as measured on Q8_0
        return com["mtp_metal_q40"], com["mtp_power"], com["mtp_prefill_metal_q8"]
    return com["mtp"], com["mtp_power"], 1.0


def _host(p_board, scale):
    """Host share of one GPU slot of an 8-GPU node at the wall (Latif et al. Table 1, less the H200 idle floor)."""
    x = np.clip((p_board - MMC["gpu_idle_w"]) / (MMC["gpu_burn_w"] - MMC["gpu_idle_w"]), 0.0, 1.0)
    return scale * (MMC["host_w_idle"] + (MMC["host_w_burn"] - MMC["host_w_idle"]) * x)


def phase_watts(key: str, p: dict, com: dict) -> dict:
    """Boundary and wall power per phase (decode, prefill, idle = tool + trial overhead), W."""
    m = MM["machines"][key]
    kind = m["kind"]
    if kind == "h200_batch":                 # whole board over wall-clock, shared by the tasks in flight
        ng = m.get("ngpu", 1)
        b = ng * p["p_board"] / p["inflight"]
        w = ng * (p["p_board"] + _host(p["p_board"], com["host_scale"])) / p["inflight"]
        return dict(b=(b, b, b), w=(w, w, w))
    pd, pp, pt = p["p_dec"] * _mtp(m, com)[1], p["p_pre"], p["p_tool"]
    if kind == "h200":
        hs = com["host_scale"]
        return dict(b=(pd, pp, pt), w=(pd + _host(pd, hs), pp + _host(pp, hs), pt + _host(pt, hs)))
    if kind == "spark":
        return dict(b=(pd, pp, pt), w=((pd + p["rest"]) / p["eta"], (pp + p["rest"]) / p["eta"], p["wall_tool"]))
    # Apple and Strix Halo: SoC/package power plus DRAM outside that boundary, platform, fan, supply. Round 12: DRAM is
    # an idle floor per installed GB (Apple; inside the platform term on Strix) plus an increment per GB/s of traffic.
    inc = com["kdram_inc"] * p["r0"] * com["eff"] * MMC["weights_gb"][m["weights"]]   # MTP does not raise weight reads
    dram_i = com["dram_idle_w_per_gb"] * m["mem_gb"] if kind == "apple" else 0.0
    dram_d, dram_p = dram_i + inc, dram_i + com["dram_prefill_frac"] * inc
    vrm = com["vrm"] if kind == "apple" else 0.0
    fan = p.get("fan", 0.0)
    eta = p.get("eta", com["eta_desk"])
    plat = p["plat"]
    return dict(b=(pd, pp, pt), w=((pd * (1 + vrm) + dram_d + plat + fan) / eta, (pp * (1 + vrm) + dram_p + plat + fan) / eta,
                                   (pt * (1 + vrm) + dram_i + plat) / eta))


def mm_eval(key: str, p: dict, com: dict) -> dict:
    """Per-attempt phase times (s), budget cuts and energies (J) on one machine."""
    m = MM["machines"][key]
    kind = m["kind"]
    mtp, _, pre_f = _mtp(m, com)
    if kind == "h200_batch":
        r = p["rfn"] if "rfn" in p else p["r16"] * p["w4"] * p["mtp16"]
        fd = 1.0            # round 12: in-campaign per-agent rates already include depth (no second depth factor)
    else:
        r = (p["fp8_b1"] / p["w4_time_ratio"] if kind == "h200" else p["r0"]) * mtp * com["eff"]
        fd = np.interp(PROF["Dd"], _FD[:, 0], _FD[:, 1]) ** com["depth_gamma"]
    t_dec = PROF["O"] / (r * fd)
    t_pre = PROF["U"] / (p["pp0"] * pre_f / (1 + PROF["Dp"] / p["dh"]))
    t_tool = PROF["tool"]
    agent = t_dec + t_pre + t_tool
    cut = agent > PROF["B"]
    k = np.where(cut, PROF["B"] / agent, 1.0)          # a cut attempt is charged up to the budget, pro rata by phase
    t_dec, t_pre, t_idle = t_dec * k, t_pre * k, t_tool * k + PROF["over"]
    pw = phase_watts(key, p, com)
    eb = pw["b"][0] * t_dec + pw["b"][1] * t_pre + pw["b"][2] * t_idle
    ew = pw["w"][0] * t_dec + pw["w"][1] * t_pre + pw["w"][2] * t_idle
    return dict(T=t_dec + t_pre + t_idle, agent=np.minimum(agent, PROF["B"]), t_dec=t_dec, t_pre=t_pre, t_idle=t_idle,
                cut=cut, eb=eb, ew=ew, r=r, pw=pw)


def mm_summary(key: str, ev: dict, q: dict, p: dict, base: float) -> dict:
    fmt = tb["machines"][key]["format"]
    solved = PROF["solved"]
    lam = float((solved & ev["cut"]).sum() / solved.sum())   # share of E1 solves the budget cuts on this machine
    acc0 = base + q[fmt] / 100.0
    dt = acc0 * lam * (1 - PARTIAL)
    m = MM["machines"][key]
    share = (p["inflight"] / m.get("ngpu", 1)) if m["kind"] == "h200_batch" else 1.0
    return dict(kj_boundary=float(ev["eb"].mean() / 1e3), kj_wall_model=float(ev["ew"].mean() / 1e3),
                wall_s=float(ev["T"].mean()), wall_s_median=float(np.median(ev["T"])), acc_q=q[fmt],
                acc_timeout=-100 * dt, acc=acc0 - dt, lam=lam, cut=float(ev["cut"].mean()), gpu_share=share,
                t_dec=float(ev["t_dec"].mean()), t_pre=float(ev["t_pre"].mean()), t_idle=float(ev["t_idle"].mean()),
                format=fmt)


def machine(mkey: str, model: str, s: dict) -> dict:
    m = tb["machines"][mkey]
    wall_factor = s["kj_wall_model"] / s["kj_boundary"]     # energy-weighted result of the per-phase wall model
    kj_wall = s["kj_boundary"] * wall_factor
    acc = s["acc"]
    own = kwh_usd(kj_wall)
    if m["kind"] == "dc":
        gpu_s = s["wall_s"] / s["gpu_share"]
        rent = gpu_s * C["h200_rent_usd_per_gpu_h"] / 3600
        amort = gpu_s * C["h200_own_amort_usd_per_gpu_h"] / 3600
    else:
        price = hw[m["hw"]]["usd"]
        per_h = price / (C["amort_years"] * 8760 * C["desk_duty_cycle"])
        amort = per_h * s["wall_s"] / 3600
        rent = amort  # no rental market for a desk machine: rented basis = amortized
    return dict(acc=acc, to=-s["acc_timeout"] / 100.0, kj_wall=kj_wall, kj_solved=kj_wall / acc, own=own, amort=amort,
                rent=rent, wall_min=s["wall_s"] / 60, label=m["label"], boundary=m["boundary"], kind=m["kind"],
                kj_boundary=s["kj_boundary"], wall_factor=wall_factor, wall_s=s["wall_s"],
                wall_min_median=s["wall_s_median"] / 60, acc_q=s["acc_q"], acc_timeout=s["acc_timeout"],
                cut=s["cut"], lam=s["lam"], gpu_share=s["gpu_share"], format=s["format"],
                watts_wall=1000 * kj_wall / s["wall_s"], t_dec=s["t_dec"], t_pre=s["t_pre"], t_idle=s["t_idle"])


MODEL_OF = lambda k: "flashnext" if k == "fn_h200x2" else "qwen38"


def machines_from(params) -> tuple[dict, dict]:
    com, mach, q = params
    EV = {k: mm_eval(k, mach[k], com) for k in tb["machines"]}
    return EV, {k: machine(k, MODEL_OF(k), mm_summary(k, EV[k], q, mach[k], base_acc[MODEL_OF(k)])) for k in tb["machines"]}


def glm_tier(params) -> dict:
    """GLM-5.3 on eight owned H200s from its own measured campaign (round 12; tb21.machine_model.glm53): node seconds
    per attempt = mean agent time x token scale / tasks in flight; the node's board power plus the host share per GPU
    slot at the wall; rented and amortized dollars from GPU-seconds."""
    com, mach, _ = params
    g, n = mach["_glm53"], MM["glm53"]["ngpu"]
    node_s = g["agent_s"] * g["token_scale"] / g["inflight"]
    wall_w = g["p_node_board"] + n * float(_host(g["p_node_board"] / n, com["host_scale"]))
    kj_b, kj_w = g["p_node_board"] * node_s / 1e3, wall_w * node_s / 1e3
    return dict(kj_boundary=kj_b, kj_wall=kj_w, wall_factor=kj_w / kj_b, own=kwh_usd(kj_w),
                rent=n * node_s * C["h200_rent_usd_per_gpu_h"] / 3600, amort=n * node_s * C["h200_own_amort_usd_per_gpu_h"] / 3600,
                node_s=node_s, campaign_s=node_s * N_ATT, inflight=g["inflight"], watts_wall=wall_w)


PARAMS = mm_params()
EV, MACH = machines_from(PARAMS)
DERIVED["machines"] = MACH
GLM = glm_tier(PARAMS)
_gs = tiers["glm53"]
for _k in ("own", "amort", "rent", "kj_wall"):     # stored copies exist only for figures/fig2_ipd_ipj_target.py
    check(abs(_gs[_k] - GLM[_k]) <= 0.005 * GLM[_k],
          f"tb21.tiers.glm53.{_k} = {_gs[_k]} is stored for the figure script but the derivation gives {GLM[_k]:.6g}")
    _gs[_k] = GLM[_k]
DERIVED["glm53_tier"] = GLM

# model checks: energy = power x time per phase, phases sum to the attempt, budget respected, wall >= boundary
for k, ev in EV.items():
    pw, m = ev["pw"], MACH[k]
    e_b = sum(w * float(t.mean()) for w, t in zip(pw["b"], (ev["t_dec"], ev["t_pre"], ev["t_idle"]))) / 1e3
    e_w = sum(w * float(t.mean()) for w, t in zip(pw["w"], (ev["t_dec"], ev["t_pre"], ev["t_idle"]))) / 1e3
    check(abs(e_b - m["kj_boundary"]) < 1e-6 * e_b and abs(e_w - m["kj_wall"]) < 1e-6 * e_w,
          f"{k}: energy is not phase power x phase time")
    check(abs(m["t_dec"] + m["t_pre"] + m["t_idle"] - m["wall_s"]) < 1e-6 * m["wall_s"], f"{k}: phase times do not sum")
    check(bool(np.all(ev["agent"] <= PROF["B"] + 1e-6)), f"{k}: an attempt runs past its budget")
    check(m["wall_factor"] >= 1.0, f"{k}: wall energy below boundary energy")
    check(0 <= m["lam"] <= 1 and 0 <= m["cut"] <= 1 and m["acc_timeout"] <= 0, f"{k}: timeout loss out of range")
_fmts: dict = {}
for k, m in MACH.items():
    _fmts.setdefault(m["format"], set()).add(m["acc_q"])
check(all(len(v) == 1 for v in _fmts.values()), "identical formats must give identical format deltas")
check(MACH["h200_1"]["lam"] < 0.01, "the H200 serving one agent should lose no solves to the budget")
# cross-checks against published wall readings (spec, section 1): decode-phase wall power at central values
_dw = {k: EV[k]["pw"]["w"][0] for k in ("m3ultra", "spark", "strix", "m4pro")}
check(_dw["m3ultra"] < 227, f"M3 Ultra decode wall {_dw['m3ultra']:.0f} W exceeds Geerling's 227 W peak (80-core)")
check(60 <= _dw["spark"] <= 90, f"DGX Spark decode wall {_dw['spark']:.0f} W outside ServeTheHome's 60-90 W")
check(abs(_dw["strix"] - 112) / 112 < 0.15, f"Strix Halo decode wall {_dw['strix']:.0f} W vs Tom's ~112 W")
# round 12 cross-check, reported: E-C at one task in flight (Qwen3.6-27B FP8, 5 tasks) averaged 471 W on the board over
# its whole window, tool phases included; the model's mean board power over an H200 single-agent attempt
DERIVED["h200_1_mean_board_w"] = 1000 * MACH["h200_1"]["kj_boundary"] / MACH["h200_1"]["wall_s"]


# ---------------------------------------------------------------- sampling intervals (round 10)
# Repo convention (ipj/src/ipj/rigor): 95% percentile bootstrap over task clusters, 10,000 resamples, seed 0.
def acc_hw(p: float, rho: float) -> float:
    """95% half-width of a pooled accuracy under the planned n_tasks x k design with within-task correlation rho."""
    n = tb["n_tasks"] * K_ROLLS
    return 1.96 * math.sqrt(p * (1 - p) * (1 + (K_ROLLS - 1) * rho) / n)


def _q(x, a):
    return float(np.nanquantile(x, a))


def solve_vec(k: str) -> np.ndarray:
    """Per-attempt solve on machine k on the E1 profile: solved and not cut, or a cut attempt that still passes."""
    s, cut = PROF["solved"], EV[k]["cut"]
    return np.where(s & ~cut, 1.0, np.where(s & cut, PARTIAL, 0.0))


_rng = np.random.default_rng(IV["seed"])
_idx = _rng.integers(0, N_TASKS, size=(NB, N_TASKS))
PM: dict = {}
for k, m in MACH.items():
    E = EV[k]["ew"][TI] / 1e3
    S = solve_vec(k)[TI]
    bE = E.sum(1)[_idx].sum(1) / N_ATT
    bS = S.sum(1)[_idx].sum(1) / N_ATT
    lo, hi = _q(bE, 0.025), _q(bE, 0.975)
    e_pm = max(m["kj_wall"] - lo, hi - m["kj_wall"])
    corr = float(np.corrcoef(np.log(bE), np.log(np.maximum(bS, 1e-9)))[0, 1])
    rel_e = (hi - lo) / 2 / 1.96 / m["kj_wall"]
    icc = ICC["flashnext"] if k == "fn_h200x2" else ICC["qwen38"]
    rel_a = acc_hw(m["acc"], icc) / 1.96 / m["acc"]
    rel_s = math.sqrt(rel_e ** 2 + rel_a ** 2 - 2 * corr * rel_e * rel_a)
    PM[k] = dict(kj=(lo, hi), kj_pm=e_pm, kj_solved_pm=1.96 * rel_s * m["kj_solved"], corr=corr,
                 acc_pm=100 * acc_hw(m["acc"], icc))
    check(lo <= m["kj_wall"] <= hi, f"{k}: kJ interval excludes its point")
DERIVED["machine_intervals"] = PM


def paired_ratio(a: str, b: str, rng) -> dict:
    """b over a, same tasks, rolls redrawn independently per machine; percentile interval relative to the
    bootstrap point, applied to the target-level point (per attempt and per solved task)."""
    out = {}
    Ea, Sa, Eb, Sb = (EV[a]["ew"][TI], solve_vec(a)[TI], EV[b]["ew"][TI], solve_vec(b)[TI])
    ra, rb = rng.integers(0, K_ROLLS, size=(NB, N_TASKS, K_ROLLS)), rng.integers(0, K_ROLLS, size=(NB, N_TASKS, K_ROLLS))
    ii = _idx[:, :, None]
    mEa, mSa = Ea[ii, ra].mean((1, 2)), Sa[ii, ra].mean((1, 2))
    mEb, mSb = Eb[ii, rb].mean((1, 2)), Sb[ii, rb].mean((1, 2))
    for tag, boot, pt_b, pt in (("att", mEb / mEa, Eb.mean() / Ea.mean(), MACH[b]["kj_wall"] / MACH[a]["kj_wall"]),
                                ("sol", (mEb / mSb) / (mEa / mSa), (Eb.mean() / Sb.mean()) / (Ea.mean() / Sa.mean()),
                                 MACH[b]["kj_solved"] / MACH[a]["kj_solved"])):
        lo, hi = _q(boot, 0.025) / pt_b * pt, _q(boot, 0.975) / pt_b * pt
        out[tag] = (pt, lo, hi)
        check(lo <= pt <= hi, f"paired {b}/{a} {tag}: interval excludes its point")
    return out


_rngp = np.random.default_rng(IV["seed"] + 1)
PAIRS = {("m4pro", "h200_16"): paired_ratio("m4pro", "h200_16", _rngp),
         ("m4pro", "h200_1"): paired_ratio("m4pro", "h200_1", _rngp)}
DERIVED["paired_ratios"] = {f"{b}/{a}": v for (a, b), v in PAIRS.items()}
# machine minus batched H200 accuracy, same tasks, independent rolls: the task effect cancels (E1 ICC)
ACC_PAIR_PM = max(100 * 1.96 * math.sqrt(2 * p * (1 - p) * (1 - ICC["e1"]) / N_ATT)
                  for p in ((MACH[k]["acc"] + MACH["h200_16"]["acc"]) / 2 for k in MACH if k != "h200_16"))

MNAME = {"h200_1": "HOne", "h200_16": "HBatch", "m5max": "MFiveMax", "m5max_q8": "MFiveMaxQ", "m5max_q40": "MFiveMaxQZero",
         "m4max": "MFourMax", "m4pro": "MFourPro", "m3ultra": "MThreeUltra", "spark": "Spark",
         "strix": "Strix", "fn_h200x2": "HTwoFlash"}
for k, m in MACH.items():
    n = "Mach" + MNAME[k]
    mac(n + "KJPM", f"{PM[k]['kj_pm']:.0f}", "95% sampling half-width, kJ/attempt")
    mac(n + "KJSolvedPM", f"{PM[k]['kj_solved_pm']:.0f}")
    mac(n + "AccPM", f"{PM[k]['acc_pm']:.1f}")
    mac(n + "Cut", f"{100 * m['cut']:.0f}", "percent of attempts cut at the time budget")
    mac(n + "AccTimeout", f"{-m['acc_timeout']:.1f}", "points lost to the time budget")
    mac(n + "Watts", f"{m['watts_wall']:.0f}", "mean wall power over an attempt")
HEAD = tb["signal"]["s1_rule"]["selected_machine"] or "m5max"     # round 15: checked against the rule below
TAB2 = ("m4pro", HEAD, "m4max", "m3ultra", "spark", "strix", "h200_16", "h200_1")
mac("MachAccMin", pct(min(MACH[k]["acc"] for k in TAB2)))
mac("MachAccMax", pct(max(MACH[k]["acc"] for k in TAB2)))
mac("MachAccPairPM", f"{ACC_PAIR_PM:.1f}", "machine minus batched H200 accuracy, paired, 95%")
_r = PAIRS[("m4pro", "h200_16")]
mac("MacVsGPUCI", f"[{_r['att'][1]:.2f}, {_r['att'][2]:.2f}]")
mac("MacVsGPUSolvedCI", f"[{_r['sol'][1]:.2f}, {_r['sol'][2]:.2f}]")
_r = PAIRS[("m4pro", "h200_1")]
mac("MachMFourProVsOneCI", f"[{_r['sol'][1]:.1f}, {_r['sol'][2]:.1f}]")
# round 12: which of the two draws fewer Joules is reported, not asserted (the prose is generated from it in compat.py)
DERIVED["laptop_vs_batched_h200"] = dict(att=PAIRS[("m4pro", "h200_16")]["att"], sol=PAIRS[("m4pro", "h200_16")]["sol"],
                                         resolved=not (PAIRS[("m4pro", "h200_16")]["sol"][1] <= 1.0 <= PAIRS[("m4pro", "h200_16")]["sol"][2]))

for k, m in MACH.items():
    n = "Mach" + MNAME[k]
    mac(n + "Acc", pct(m["acc"]))
    mac(n + "KJ", f"{m['kj_wall']:.0f}", "kJ/attempt at the wall")
    mac(n + "KJSolved", f"{m['kj_solved']:.0f}")
    mac(n + "Min", f"{m['wall_min']:.0f}")
    mac(n + "Own", usd(m["own"]))
    mac(n + "Amort", usd(m["amort"]))
    mac(n + "VsBatch", f"{MACH['h200_16']['kj_solved'] / m['kj_solved']:.1f}$\\times$")
    mac(n + "VsOne", f"{MACH['h200_1']['kj_solved'] / m['kj_solved']:.1f}$\\times$")
desks = [k for k, m in MACH.items() if m["kind"] == "desk" and k != "fn_h200x2"]
gap = max(abs(MACH["h200_16"]["acc"] - MACH[k]["acc"]) for k in desks)
mac("DeskMaxGap", pct(gap), "largest desk accuracy gap vs H200, Qwen3.8-27B")
mac("DeskMinGap", pct(min(abs(MACH["h200_16"]["acc"] - MACH[k]["acc"]) for k in desks)))
mac("NDeskMachines", {5: "five", 6: "six", 7: "seven"}[len({tb["machines"][k]["hw"] for k in desks})])   # machines, not formats
mac("MFiveMaxGap", pct(MACH["h200_16"]["acc"] - MACH["m5max"]["acc"]))


# ---------------------------------------------------------------- TB2.1 plans
def closed_usd(name: str) -> float:
    return tiers[name]["usd"]


def plan(p: dict, random_stage1: bool = False) -> dict:
    loc = MACH[p["machine"]]
    L = loc["acc"]
    Dt = loc.get("to", 0.0)                 # round 10: solves lost to the time budget; a cut attempt is always flagged
    Lq = L + Dt
    r1, f1, r2, f2 = sig["s1_recall"], sig["s1_fpr"], sig["s2_recall"], sig["s2_fpr"]
    fe, caught = L * f1, (1 - Lq) * r1
    E1 = fe + caught + Dt
    if random_stage1:
        fe, caught = L * E1, (1 - L) * E1
        Dt = 0.0
    kept = L - fe
    silent = (1 - L) - caught - Dt
    g = tiers["glm53"]
    g_c = sig["flashnext_glm_solve_caught"] if p["local"] == "flashnext" else sig["glm_solve_caught"]
    c3 = sig["flashnext_astra3_doubly_failed"] if p["local"] == "flashnext" else (
        sig["astra3_solve_doubly_failed"] if p.get("closed") == "astra" else sig["sol3_solve_doubly_failed"])
    out = dict(L=L, E1=E1, kept=kept, silent=silent)
    local_cost = {b: loc[b] for b in ("own", "amort", "rent")}
    judge = {"own": sig["judge_own"], "amort": sig["judge_amort"], "rent": sig["judge_rent"]}
    glm_cost = {"own": g["own"], "amort": g["amort"], "rent": g["rent"]}
    kind = p["kind"]
    if kind == "three":
        gc = fe * sig["glm_solve_false_esc"] + caught * g_c + Dt * sig.get("timeout_glm_solve", g_c)
        gf = E1 - gc
        E2 = gf * r2 + gc * f2
        acc = kept + gc * (1 - f2) + gc * f2 * sig["closed3_solve_flagged_correct"] + gf * r2 * c3
        calls = 1 + E1
        cost = {b: local_cost[b] + calls * judge[b] + E1 * glm_cost[b] * PG + E2 * closed_usd(p["closed"]) * PC3(p["closed"])
                for b in local_cost}
        kj = loc["kj_wall"] + calls * sig["judge_kj_wall"] + E1 * g["kj_wall"] * PG
        shares = (1 - E1, E1 - E2, E2)
    elif kind == "open":
        gc = fe * sig["glm_solve_false_esc"] + caught * g_c + Dt * sig.get("timeout_glm_solve", g_c)
        acc = kept + gc
        calls = 1
        cost = {b: local_cost[b] + calls * judge[b] + E1 * glm_cost[b] * PG for b in local_cost}
        kj = loc["kj_wall"] + calls * sig["judge_kj_wall"] + E1 * g["kj_wall"] * PG
        shares = (1 - E1, E1, 0.0)
    elif kind == "two":
        cfe = sig[f"{p['closed']}2_solve_false_esc"]
        cc = sig[f"{p['closed']}2_solve_caught"]
        acc = kept + fe * cfe + caught * cc + Dt * sig.get(f"timeout_{p['closed']}2_solve", cc)
        calls = 1
        cost = {b: local_cost[b] + calls * judge[b] + E1 * closed_usd(p["closed"]) * PC(p["closed"]) for b in local_cost}
        kj = loc["kj_wall"] + calls * sig["judge_kj_wall"]
        shares = (1 - E1, 0.0, E1)
    elif kind == "retry":
        rc = fe * sig["retry_solve_false_esc"] + (caught + Dt) * sig["retry_solve_caught"]
        rf = E1 - rc
        G = rc * f1 + rf * r1                 # the retry is read by the same stage-1 signal
        retry_kept = rc * (1 - f1)
        gc = G * sig["glm_solve_after_retry"]
        gf = G - gc
        E2 = gf * r2 + gc * f2
        acc = kept + retry_kept + gc * (1 - f2) + gc * f2 * sig["closed3_solve_flagged_correct"] + gf * r2 * sig["astra_solve_after_retry"]
        calls = 1 + E1 + G
        cost = {b: local_cost[b] * (1 + E1 * PR) + calls * judge[b] + G * glm_cost[b] * PG + E2 * closed_usd(p["closed"]) * PC3(p["closed"])
                for b in local_cost}
        kj = loc["kj_wall"] * (1 + E1 * PR) + calls * sig["judge_kj_wall"] + G * g["kj_wall"] * PG
        shares = (1 - G, G - E2, E2)  # tasks finishing on owned local hardware (first attempt or retry)
        out["retry_share"] = E1
    else:
        raise ValueError(kind)
    out.update(acc=acc, cost=cost, kj=kj, shares=shares, closed=shares[2], calls=calls)
    check(abs(sum(shares) - 1) < 1e-9, f"shares do not sum to 1 for {p}")
    check(0 < acc < 1, f"accuracy out of range for {p}")
    return out


# ---------------------------------------------------------------- stage-1 threshold: local by default (round 15)
# Author ruling 2026-09-25: the deployed stage-1 threshold is the most accurate operating point of the headline plan whose owned
# cost is at least RULE["min_mult"] below the Fable 5.1 anchor and that keeps at least RULE["min_local_share"] of tasks finishing
# on the laptop. The false-escalation rate is swept along the deployed signal's equal-variance binormal ROC (the construction
# of figures/fig_escalation_curves.py); stage 2 is held at its operating point; one threshold serves every plan. Of the two M5
# Max configurations the one whose rule-selected headline is more accurate runs every Qwen3.8-27B-first plan.
_ND = NormalDist()
RULE = sig["s1_rule"]


def roc_recall(auc: float, fpr: float) -> float:
    """Recall at a false-escalation rate on an equal-variance binormal ROC of the given AUC."""
    return _ND.cdf(math.sqrt(2) * _ND.inv_cdf(auc) + _ND.inv_cdf(fpr))


def rule_select(mkey: str) -> dict:
    """Operating points of the headline plan with its first stage on machine mkey: 'best' satisfies the rule, 'free' drops the
    local-share requirement (what the requirement gives up)."""
    saved = sig["s1_recall"], sig["s1_fpr"]
    auc, step = sig[RULE["auc_key"]], RULE["fpr_step"]
    best = free = None
    try:
        for i in range(1, round(1 / step)):
            f = round(i * step, 9)
            r = roc_recall(auc, f)
            sig["s1_recall"], sig["s1_fpr"] = r, f
            o = plan(dict(tb["plans"]["headline"], machine=mkey))
            pt = dict(fpr=f, recall=r, acc=o["acc"], own=o["cost"]["own"], rent=o["cost"]["rent"], shares=o["shares"], E1=o["E1"])
            if o["cost"]["own"] <= ANCHOR["usd"] / RULE["min_mult"]:
                if free is None or o["acc"] > free["acc"]:
                    free = pt
                if (o["shares"][0] >= RULE["min_local_share"] and o["shares"][2] <= RULE.get("max_closed_share", 1.0)
                        and (best is None or o["acc"] > best["acc"])):
                    best = pt
    finally:
        sig["s1_recall"], sig["s1_fpr"] = saved
    return dict(best=best, free=free)


RULE_PICK = {k: rule_select(k) for k in RULE["candidates"]}
check(all(v["best"] is not None for v in RULE_PICK.values()), "s1_rule: no operating point satisfies the rule on a candidate machine")
_pick = max(RULE["candidates"], key=lambda k: RULE_PICK[k]["best"]["acc"])
_sel = RULE_PICK[_pick]["best"]
check(RULE["selected_machine"] == _pick, f"s1_rule: the rule selects {_pick}, targets.json records {RULE['selected_machine']}")
check(RULE["selected_fpr"] is not None and abs(RULE["selected_fpr"] - _sel["fpr"]) < 1e-9 and abs(sig["s1_fpr"] - _sel["fpr"]) < 1e-9
      and abs(sig["s1_recall"] - _sel["recall"]) < 5e-6,
      f"s1_rule: the rule selects recall {_sel['recall']:.6f} at false escalation {_sel['fpr']}, targets.json records "
      f"{sig['s1_recall']} / {sig['s1_fpr']}")
check(abs(roc_recall(sig[RULE["auc_key"]], sig["s1_fpr"]) - sig["s1_recall"]) < 5e-6, "s1_rule: the deployed point is off the signal's ROC")
_QFIRST = [k for k, p in tb["plans"].items() if p["local"] == "qwen38" and k not in ("desk", "desk_open", "head_dc", "open_dc")]
check(all(tb["plans"][k]["machine"] == HEAD for k in _QFIRST) and HEAD == _pick,
      f"s1_rule: every Qwen3.8-27B-first plan must run on the selected machine {_pick}")
DERIVED["s1_rule"] = dict(selected_machine=_pick, **{k: v for k, v in RULE_PICK.items()})

PLANS = {k: plan(p) for k, p in tb["plans"].items()}
# round 23: each composed plan's within-task correlation, scaled from icc.plan by its share-weighted mix of measured stage
# correlations over the mix of the measured plan that implies icc.plan (constants.icc_plan_ref)
ICC_REF = (1 - C["icc_plan_ref"]["fire"]) * ICC["e1"] + C["icc_plan_ref"]["fire"] * ICC["astra"]
def plan_icc(stages, shares):
    return ICC["plan"] * sum(s * ICC[k] for k, s in zip(stages, shares)) / ICC_REF
PLAN_ICC = {k: plan_icc((tb["plans"][k]["local"], "glm53", tb["plans"][k].get("closed", "astra")), o["shares"]) for k, o in PLANS.items()}
DERIVED["plan_icc"] = PLAN_ICC
DERIVED["tb21_plans"] = PLANS
# the deployed point satisfies the rule it was selected by
check(PLANS["headline"]["cost"]["own"] <= ANCHOR["usd"] / RULE["min_mult"] and PLANS["headline"]["shares"][0] >= RULE["min_local_share"]
      and PLANS["headline"]["shares"][2] <= RULE.get("max_closed_share", 1.0),
      "s1_rule: the deployed headline breaks the local-by-default rule (owned cost, share on the laptop, or closed-source share)")
check(tb["machines"][HEAD]["kind"] == "desk", "s1_rule: the headline's first stage must run on a desk machine")
PNAME = {"headline": "Head", "desk": "Desk", "solvar": "SolVar", "openonly": "Open", "desk_open": "DeskOpen",
         "two_astra": "TwoAstra", "two_sol": "TwoSol", "retry": "Retry", "flashnext": "Flash", "head_dc": "HeadDC",
         "open_dc": "OpenDC"}
for k, r in PLANS.items():
    n = "TB" + PNAME[k]
    mac(n + "Acc", pct(r["acc"]))
    mac(n + "Local", pct(r["shares"][0]), "share of tasks finishing on owned local hardware")
    mac(n + "Mid", pct(r["shares"][1]))
    mac(n + "Closed", pct(r["shares"][2]))
    mac(n + "NoClosed", pct(1 - r["shares"][2]))
    mac(n + "Fire", pct(r["E1"]))
    mac(n + "KJ", f"{r['kj']:.0f}")
    mac(n + "KJSolved", f"{r['kj'] / r['acc']:.0f}")
    for b, B in (("own", "Own"), ("amort", "Amort"), ("rent", "Rent")):
        mac(n + B, usd(r["cost"][b]))
        mac(n + B + "Mult", mult(ANCHOR["usd"] / r["cost"][b]) + "$\\times$")

h = PLANS["headline"]
# held-out selection (appendix): targets are stated relative to the headline accuracy (round 10)
sig["heldout_in"] = 100 * h["acc"] + sig["heldout_in_offset"]
sig["heldout_out"] = 100 * h["acc"] + sig["heldout_out_offset"]
rnd = plan(tb["plans"]["headline"], random_stage1=True)
mac("TBHeadRandomAcc", pct(rnd["acc"]))
mac("TBHeadLift", f"{100 * (h['acc'] - rnd['acc']):.1f}")
mac("TBHeadVsFableFive", f"{100 * h['acc'] - PUB['aa_tb21']['fable5_max']['acc']:.1f}")
mac("TBHeadVsFableOne", f"{PUB['aa_tb21']['fable51_max']['acc'] - 100 * h['acc']:.1f}")
mac("TBHeadVsAstra", f"{100 * base_acc['astra'] - 100 * h['acc']:.1f}")
mac("TBHeadVsSolCost", f"{tiers['sol']['usd'] / h['cost']['rent']:.1f}$\\times$", "sol alone $ / plan rented $")
mac("TBHeadVsAstraCost", f"{tiers['astra']['usd'] / h['cost']['rent']:.1f}$\\times$")
mac("TBHeadVsAstraOwnCost", f"{tiers['astra']['usd'] / h['cost']['own']:.1f}$\\times$")
mac("TBRetryVsAstraCost", f"{tiers['astra']['usd'] / PLANS['retry']['cost']['rent']:.1f}$\\times$")
# headline owned multiplier, 95% sampling interval: the clustered closed share (beta-binomial at the ICC of composed
# plans, planned 89 x 5 design) and the measured GPT-6 Astra premium interval; the anchor is fixed (D-ANCHOR)
_rngm = np.random.default_rng(IV["seed"] + 2)
_p2, _sb = h["shares"][2], 1 / PLAN_ICC["headline"] - 1
_e2 = _rngm.binomial(K_ROLLS, _rngm.beta(_p2 * _sb, (1 - _p2) * _sb, size=(NB, N_TASKS))).sum(1) / N_ATT
_plo, _phi = C["premium_by_target_ci"]["astra"]
_prem = _rngm.normal(PC3("astra"), (_phi - _plo) * PD["astra"]["after_glm"] / (2 * 1.96), NB)
_fixed = h["cost"]["own"] - _p2 * tiers["astra"]["usd"] * PC3("astra")
_mb = ANCHOR["usd"] / (_fixed + _e2 * tiers["astra"]["usd"] * _prem)
HEAD_MULT_CI = (_q(_mb, 0.025), _q(_mb, 0.975))
check(_fixed > 0 and HEAD_MULT_CI[0] <= ANCHOR["usd"] / h["cost"]["own"] <= HEAD_MULT_CI[1],
      "headline owned multiplier interval excludes its point")
mac("TBHeadOwnMultCI", f"[{mult(HEAD_MULT_CI[0])}$\\times$, {mult(HEAD_MULT_CI[1])}$\\times$]")
DERIVED["headline_own_mult_ci"] = HEAD_MULT_CI
mac("TBHeadClosedShareOfOwn", pct(h["shares"][2] * tiers["astra"]["usd"] * PC3("astra") / h["cost"]["own"], 0),
    "percent of owned-marginal cost spent on the closed stage")
mac("TBOpenVsGLM", f"{100 * PLANS['openonly']['acc'] - 100 * base_acc['glm53']:.1f}")
mac("TBOpenVsFableFive", f"{100 * PLANS['openonly']['acc'] - PUB['aa_tb21']['fable5_max']['acc']:.1f}")
mac("TBFlashVsSol", f"{100 * base_acc['sol'] - 100 * PLANS['flashnext']['acc']:.1f}")
mac("TBRetryVsHead", f"{100 * h['acc'] - 100 * PLANS['retry']['acc']:.1f}")
mac("TBMidCost", f"{100 * (PLANS['two_astra']['acc'] - h['acc']):.1f}", "accuracy the owned middle stage costs vs going straight to Astra")
mac("TBDeskVsHead", f"{round(100 * h['acc'] - 100 * PLANS['desk']['acc'], 1) + 0.0:.1f}")   # no "-0.0"

# single tiers (ours, k=5, same protocol)
TIER = {"qwen38": "Qwen", "glm53": "GLM", "sol": "SolT", "astra": "AstraT", "flashnext": "FlashT"}
for k, n in TIER.items():
    t = tiers[k]
    mac("TB" + n + "Acc", pct(base_acc[k]))
    mac("TB" + n + "Solved", str(t["solved"]))
for k in ("sol", "astra"):
    mac("TB" + TIER[k] + "USD", usd(tiers[k]["usd"]))
    mac("TB" + TIER[k] + "Mult", mult(ANCHOR["usd"] / tiers[k]["usd"]) + "$\\times$")
g = tiers["glm53"]
q = MACH["h200_16"]
for b, B in (("own", "Own"), ("amort", "Amort"), ("rent", "Rent")):
    mac("TBGLM" + B, usd(g[b]))
    mac("TBQwen" + B, usd(q[b]))
mac("TBGLMKJ", f"{g['kj_wall']:.0f}")
mac("TBQwenKJ", f"{q['kj_wall']:.0f}")
mac("TBAstraAAAcc", f"{PUB['aa_tb21']['astra_medium']['acc']:.1f}")
mac("TBSolAAAcc", f"{PUB['aa_tb21']['sol_xhigh']['acc']:.1f}")
mac("TBQwenAAAcc", f"{PUB['aa_tb21']['qwen38_27b_xhigh']['acc']:.1f}")
mac("TBGLMAAAcc", f"{PUB['aa_tb21']['glm53_max']['acc']:.1f}")
mac("TBFlashAAAcc", f"{PUB['aa_tb21']['flashnext']['acc']:.1f}")
_REP = (('qwen38', 'qwen38_27b_xhigh'), ('glm53', 'glm53_max'), ('sol', 'sol_xhigh'), ('astra', 'astra_medium'))
mac("TBMaxRepGap", f"{max(abs(100 * base_acc[k] - PUB['aa_tb21'][a]['acc']) for k, a in _REP):.1f}")
check(max(abs(100 * base_acc[k] - PUB['aa_tb21'][a]['acc']) for k, a in _REP) <= 0.1 + 1e-9,
      "text (3.3): every tier but Qwen3.8-Flash-Next is said to reproduce its leaderboard score within 0.1 points")
mac("TBFlashRepGap", f"{PUB['aa_tb21']['flashnext']['acc'] - 100 * base_acc['flashnext']:.1f}", "Flash-Next target below its leaderboard score")

# anchor + frontier constants
mac("AnchorAcc", f"{ANCHOR['acc']:.1f}")
mac("AnchorUSD", usd(ANCHOR["usd"]))
mac("AnchorTwentyX", usd(ANCHOR["usd"] / 20))
mac("FableFiveAcc", f"{PUB['aa_tb21']['fable5_max']['acc']:.1f}")
mac("FableOneXhighAcc", f"{PUB['aa_tb21']['fable51_xhigh']['acc']:.1f}")
mac("FableOneXhighUSD", usd(PUB["aa_tb21"]["fable51_xhigh"]["usd"]))
mac("Premium", f"{PREM:.2f}")
mac("ElecRate", f"{C['elec_usd_per_kwh']:.4f}")
mac("RentRate", f"{C['h200_rent_usd_per_gpu_h']:.2f}")
mac("OwnAmortRate", f"{C['h200_own_amort_usd_per_gpu_h']:.2f}")
mac("DutyPct", f"{100 * C['desk_duty_cycle']:.0f}")
mac("KDEPTH", "five")
mac("NTASKS", str(tb["n_tasks"]))
mac("NTrials", str(tb["n_tasks"] * C["k"]))

# signal
for k, n in (("s1_recall", "SigOneRecall"), ("s1_fpr", "SigOneFPR"), ("s2_recall", "SigTwoRecall"), ("s2_fpr", "SigTwoFPR")):
    mac(n, f"{sig[k]:.2f}")
for k, n in (("auc_flags_all", "AUCFlags"), ("auc_judge_all", "AUCJudge"), ("auc_combined_all", "AUCCombined"),
             ("auc_judge_completed", "AUCJudgeDone"), ("auc_automix_completed", "AUCAutoMix"),
             ("auc_frugal_completed", "AUCFrugal"), ("auc_solvhead_completed", "AUCSolvHead"),
             ("auc_routellm_completed", "AUCRouteLLM")):
    mac(n, f"{sig[k]:.2f}")
mac("HeldIn", f"{sig['heldout_in']:.1f}")
mac("HeldOut", f"{sig['heldout_out']:.1f}")
mac("HeldGap", f"{sig['heldout_in'] - sig['heldout_out']:.1f}")
mac("HeldPick", f"{sig['heldout_pick_pct']:.0f}")
mac("JudgeOwn", f"{sig['judge_own']:.5f}")
mac("JudgeCalls", f"{h['calls']:.2f}")
# local-by-default rule (round 15): the requirement, the selected point, what it gives up, and the other M5 Max configuration
_alt = [k for k in RULE["candidates"] if k != HEAD][0]
_ab, _hf = RULE_PICK[_alt]["best"], RULE_PICK[HEAD]["free"]
mac("RuleMinMult", f"{RULE['min_mult']:.0f}$\\times$", "owned cost at least this far below the Fable 5.1 anchor")
mac("RuleMinLocal", f"{100 * RULE['min_local_share']:.0f}\\%", "share of tasks that must finish on the laptop")
mac("RuleMaxClosed", f"{100 * RULE['max_closed_share']:.0f}\\%", "largest share of tasks sent to a closed-source provider")
mac("RuleRecall", f"{sig['s1_recall']:.2f}", "selected stage-1 recall (= SigOneRecall)")
mac("RuleFPR", f"{sig['s1_fpr']:.3f}", "selected stage-1 false-escalation rate, three decimals")
mac("RuleFreeAcc", pct(_hf["acc"]), "headline at the most accurate point under the cost requirement alone")
mac("RuleFreeLocal", pct(_hf["shares"][0]), "share finishing on the laptop at that point")
mac("RuleGiveUp", f"{100 * (_hf['acc'] - h['acc']):.1f}", "points the local-share requirement gives up")
mac("RuleAltAcc", pct(_ab["acc"]), "rule-selected headline with the other M5 Max configuration")
mac("RuleAltMult", mult(ANCHOR["usd"] / _ab["own"]) + "$\\times$")
mac("RuleAltLocal", pct(_ab["shares"][0]))
_FMT = {"q40": "Q4\\_0", "q4km": "Q4\\_K\\_M", "q8": "Q8\\_0"}
mac("HeadFormat", _FMT[tb["machines"][HEAD]["format"]], "weight format of the headline's first stage")
mac("AltFormat", _FMT[tb["machines"][_alt]["format"]])


# ---------------------------------------------------------------- secondary benchmarks
def secondary(bkey: str, prefix: str, anchor_usd: float | None = None) -> None:
    b = T[bkey]
    out = {}
    for pk, p in b["plans"].items():
        sh = p["shares"]
        check(abs(sum(sh) - 100) < 0.05, f"{bkey}/{pk} shares sum to {sum(sh)}")
        if "cond" in p:
            acc = sum(s * c for s, c in zip(sh, p["cond"])) / 100
        elif "correct" in p:
            acc = sum(p["correct"])
        else:
            acc = p["acc"]
        if "acc" in p and "cond" in p:
            check(abs(acc - p["acc"]) < 0.15, f"{bkey}/{pk} acc {acc} vs stated {p['acc']}")
        n = prefix + {"headline": "Head", "openonly": "Open", "solvar": "SolVar", "two_astra": "TwoAstra",
                      "localfirst": "LF", "glmfirst": "GLMFirst", "glmfirst90": "GLMFirstNinety", "glmfirst80": "GLMFirstEighty"}[pk]
        mac(n + "Acc", f"{acc:.1f}")
        mac(n + "Local", f"{sh[0]:.1f}")
        mac(n + "Closed", f"{sh[-1]:.1f}" if len(sh) == 3 or pk == "two_astra" or pk.startswith("glmfirst") else "0.0")
        mac(n + "NoClosed", f"{100 - (sh[-1] if (len(sh) == 3 or pk == 'two_astra' or pk.startswith('glmfirst')) else 0):.1f}")
        for bb, B in (("own", "Own"), ("amort", "Amort"), ("rent", "Rent")):
            if bb in p:
                mac(n + B, usd(p[bb]))
        mac(n + "KJ", f"{p['kj']:.0f}")
        out[pk] = dict(acc=acc, **{k: p[k] for k in p if k in ("own", "amort", "rent", "kj", "shares", "label")})
    for tk, t in b["tiers"].items():
        n = prefix + {"qwen38": "Qwen", "glm53": "GLM", "sol": "SolT", "astra": "AstraT", "opus5": "OpusT"}[tk]
        mac(n + "Acc", f"{t['acc']:.1f}")
        if "usd" in t:
            mac(n + "USD", usd(t["usd"]))
        for bb, B in (("own", "Own"), ("amort", "Amort"), ("rent", "Rent")):
            if bb in t:
                mac(n + B, usd(t[bb]))
    DERIVED[bkey] = out


# ---------------------------------------------------------------- GAIA plan model (plan()'s structure, GAIA primitives; round 15, xbench spec 2-3)
ga = T["gaia"]
gm, gsig, gt = ga["model"], ga["signal"], ga["tiers"]
GAIA_L, GAIA_G = gt["qwen38"]["acc"] / 100, gt["glm53"]["acc"] / 100
GAIA_GS = gm["glm_solve_false_esc"]
GAIA_GF = (GAIA_G - GAIA_L * GAIA_GS) / (1 - GAIA_L)   # P(GLM-5.3 | local failed), from the tier accuracies
check(0 < GAIA_GF < GAIA_GS < 1, "GAIA GLM-5.3 conditionals out of order")


def gaia_plan(r1: float, f1: float, kind: str = "three", random_stage1: bool = False) -> dict:
    """GAIA local-first plan at a stage-1 operating point; same structure as plan(), kind 'three' or 'open'."""
    L = GAIA_L
    fe, caught = L * f1, (1 - L) * r1
    E1 = fe + caught
    if random_stage1:
        fe, caught = L * E1, (1 - L) * E1
    kept = L - fe
    gc = fe * GAIA_GS + caught * GAIA_GF
    gf = E1 - gc
    r2, f2 = gsig["s2_recall"], gsig["s2_fpr"]
    if kind == "three":
        E2 = gf * r2 + gc * f2
        acc = kept + gc * (1 - f2) + gc * f2 * gm["closed3_solve_flagged_correct"] + gf * r2 * gm["astra3_solve_glm_failed"]
        calls = 1 + E1
    else:
        E2, acc, calls = 0.0, kept + gc, 1
    cost = {b: gt["qwen38"][b] + calls * gm["judge_" + b] + E1 * gt["glm53"][b] * gm["glm_premium"]
            + E2 * gt["astra"]["usd"] * PREM for b in ("own", "rent")}
    kj = gt["qwen38"]["kj"] + calls * gm["judge_kj_wall"] + E1 * gt["glm53"]["kj"] * gm["glm_premium"]  # owned stages
    shares = (1 - E1, E1 - E2, E2)
    check(abs(sum(shares) - 1) < 1e-9 and 0 < acc < 1, "GAIA plan out of range")
    return dict(acc=acc, E1=E1, E2=E2, cost=cost, kj=kj, shares=shares)


def gaia_threshold(auc: float, E1: float) -> tuple[float, float]:
    """(recall, false escalation) at which a stage-1 signal of this AUC escalates a share E1 of GAIA tasks."""
    lo, hi = 1e-9, 1 - 1e-9
    for _ in range(100):
        f = (lo + hi) / 2
        lo, hi = (f, hi) if (1 - GAIA_L) * roc_recall(auc, f) + GAIA_L * f < E1 else (lo, f)
    f = (lo + hi) / 2
    return roc_recall(auc, f), f


GAIA_OP = gaia_threshold(gsig["auc_deployed"], gm["e1_cap"])
GAIA_LF = gaia_plan(*GAIA_OP)
GAIA_LF_RND = gaia_plan(*GAIA_OP, random_stage1=True)
GAIA_OPEN = gaia_plan(*GAIA_OP, kind="open")
for _k, _r, _n in (("localfirst", GAIA_LF, 3), ("openonly", GAIA_OPEN, 2)):
    ga["plans"][_k].update(acc=100 * _r["acc"], own=_r["cost"]["own"], rent=_r["cost"]["rent"], kj=_r["kj"],
                           shares=[100 * x for x in _r["shares"][:_n]])
_gf = ga["plans"]["glmfirst"]
_r2, _f2 = gsig["glmfirst_s2_recall"], gsig["glmfirst_s2_fpr"]
check(abs(100 * (GAIA_G * (1 - _f2) + GAIA_G * _f2 * gm["closed3_solve_flagged_correct"]
                 + (1 - GAIA_G) * _r2 * gm["astra3_solve_glm_failed"]) - _gf["acc"]) < 0.15,
      "GAIA GLM-first accuracy does not follow from the model's Astra conditionals")
check(abs(100 * ((1 - GAIA_G) * _r2 + GAIA_G * _f2) - _gf["shares"][1]) < 0.15, "GAIA GLM-first provider share")
DERIVED["gaia_model"] = dict(glm_solve_caught=GAIA_GF, op=GAIA_OP, localfirst=GAIA_LF, random=GAIA_LF_RND)

# ---------------------------------------------------------------- SWE GLM-first (the target sheet's joint structure; round 15)
sw = T["swe"]
sj, st = sw["joint"], sw["tiers"]
_Q, _G, _A = st["qwen38"]["acc"] / 100, st["glm53"]["acc"] / 100, st["astra"]["acc"] / 100
_gq0 = (_G - _Q * sj["glm_given_qwen_solved"]) / (1 - _Q)
_W = {(1, 1): _Q * sj["glm_given_qwen_solved"], (1, 0): _Q * (1 - sj["glm_given_qwen_solved"]),
      (0, 1): (1 - _Q) * _gq0, (0, 0): (1 - _Q) * (1 - _gq0)}
_Aq = {(1, 1): sj["astra_given_both_solved"], (1, 0): sj["astra_given_qwen_only"], (0, 1): sj["astra_given_glm_only"]}
_Aq[(0, 0)] = (_A - sum(_W[k] * _Aq[k] for k in _Aq)) / _W[(0, 0)]
check(0 < _gq0 < 1 and 0 < _Aq[(0, 0)] < 1, "SWE joint structure out of range")
_hacc = 0.0                                                  # the joint must reproduce the stored headline chain
for (_q, _g), _w in _W.items():
    _e = _w * (sw["signal"]["recall"] if _q == 0 else sw["signal"]["fpr"])
    _p2 = sw["signal"]["recall"] if _g == 0 else sw["signal"]["fpr"]
    _hacc += _w * _q - (_e * _q) + _e * (1 - _p2) * _g + _e * _p2 * _Aq[(_q, _g)]
check(abs(100 * _hacc - sum(s_ * c_ for s_, c_ in zip(sw["plans"]["headline"]["shares"], sw["plans"]["headline"]["cond"])) / 100) < 0.15,
      "SWE joint does not reproduce the stored headline plan")
_r2, _f2 = sw["signal"]["recall"], sw["signal"]["fpr"]
_pf = {k: (_r2 if k[1] == 0 else _f2) for k in _W}          # the owned gate reads GLM-5.3's patch
_kept = sum(w * (1 - _pf[k]) for k, w in _W.items())
_kept_c = sum(w * (1 - _pf[k]) for k, w in _W.items() if k[1] == 1)
_astra_c = sum(w * _pf[k] * _Aq[k] for k, w in _W.items())
sw["plans"]["glmfirst"].update(
    shares=[100 * _kept, 100 * (1 - _kept)], cond=[100 * _kept_c / _kept, 100 * _astra_c / (1 - _kept)],
    kj=st["glm53"]["kj"] + sj["judge_kj"] + (1 - _kept) * PREM * sj["astra_kj_est"],
    **{b: st["glm53"][b] + sj["judge_" + b] + (1 - _kept) * PREM * st["astra"]["usd"] for b in ("own", "amort", "rent")})
DERIVED["swe_joint"] = dict(glm_given_qwen_failed=_gq0, astra_given_both_failed=_Aq[(0, 0)], headline_acc=100 * _hacc)


# ---------------------------------------------------------------- TB4 redistribution (author note, 2026-09-25)
# GLM-5.3 -> GPT-6 Astra at 90% and 80% of Astra's accuracy: how much work stays off closed-source providers.
# The gate is a binormal ROC fitted to the stored glmfirst plan; Astra's success on escalated tasks that GLM-5.3 solved (pc)
# and failed (pf) is solved from that plan and Astra's accuracy; each operating point is the smallest whole number of
# escalated attempts over n_tasks x rolls that reaches its target. Parameters: T["tb4"]["redistribution"].
from statistics import NormalDist as _ND
_t4, _N = T["tb4"], _ND()
_g4, _a4 = _t4["tiers"]["glm53"]["acc"] / 100, _t4["tiers"]["astra"]["acc"] / 100
_gf = _t4["plans"]["glmfirst"]
_s0, _kc0, _ac0 = _gf["shares"][1] / 100, _gf["correct"][0] / 100, _gf["correct"][1] / 100
_f0 = 1 - _kc0 / _g4
_r0 = (_s0 - _g4 * _f0) / (1 - _g4)
_pf4 = (_ac0 - _f0 * _a4) / ((1 - _g4) * (_r0 - _f0))       # from ac0 = g f0 pc + (1-g) r0 pf and a = g pc + (1-g) pf
_pc4 = (_a4 - (1 - _g4) * _pf4) / _g4
_d4 = _N.inv_cdf(_r0) - _N.inv_cdf(_f0)
check(0 < _pf4 < 1 and 0 < _pc4 < 1, "TB4 redistribution: Astra conditional success out of range")
def _tb4_at(s):
    lo, hi = 1e-12, 1 - 1e-12
    for _ in range(200):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if _g4 * m + (1 - _g4) * _N.cdf(_d4 + _N.inv_cdf(m)) < s else (lo, m)
    f = (lo + hi) / 2
    r = _N.cdf(_d4 + _N.inv_cdf(f))
    return _g4 * (1 - f), _g4 * f * _pc4 + (1 - _g4) * r * _pf4
_kc, _ac = _tb4_at(_s0)
check(abs(100 * (_kc + _ac) - sum(_gf["correct"])) < 0.05, "TB4 redistribution: the fitted gate does not reproduce the stored plan")
_n4 = _t4["n_tasks"] * _t4["redistribution"]["rolls"]
_prem4 = {b: (_gf[b] - _t4["tiers"]["glm53"][b]) / (_s0 * _t4["tiers"]["astra"]["usd"]) for b in ("own", "amort", "rent")}
for _fr in _t4["redistribution"]["fractions_of_frontier"]:
    _k = next(k for k in range(_n4 + 1) if sum(_tb4_at(max(k, 1e-9) / _n4)) >= _fr * _a4 - 1e-12)
    _s = _k / _n4
    _kc, _ac = _tb4_at(max(_s, 1e-9))
    _t4["plans"][f"glmfirst{round(100 * _fr)}"] = dict(
        label=_gf["label"], shares=[100 * (1 - _s), 100 * _s], correct=[100 * _kc, 100 * _ac], kj=_gf["kj"],
        **{b: _t4["tiers"]["glm53"][b] + _s * _t4["tiers"]["astra"]["usd"] * _prem4[b] for b in ("own", "amort", "rent")})
DERIVED["tb4_redistribution"] = dict(fpr0=_f0, tpr0=_r0, auc=_N.cdf(_d4 / 2 ** 0.5), astra_given_glm_solved=_pc4,
                                     astra_given_glm_failed=_pf4, premium=_prem4)

secondary("swe", "SWE")
secondary("gaia", "GAIA")
secondary("tb4", "TBF")
sw = T["swe"]
mac("SWEHeadVsSol", f"{DERIVED['swe']['headline']['acc'] - sw['tiers']['sol']['acc']:.1f}")
mac("SWEHeadVsSolCost", f"{sw['tiers']['sol']['usd'] / sw['plans']['headline']['rent']:.1f}$\\times$")
mac("SWEOpenVsGLM", f"{DERIVED['swe']['openonly']['acc'] - sw['tiers']['glm53']['acc']:.1f}")
mac("SWEHeadVsAstra", f"{sw['tiers']['astra']['acc'] - DERIVED['swe']['headline']['acc']:.1f}")
mac("SWELFVsGLMFirst", f"{round(DERIVED['swe']['glmfirst']['acc'], 1) - round(DERIVED['swe']['headline']['acc'], 1):.1f}")   # from the printed values
mac("SWELFVsGLMFirstCost", f"{sw['plans']['glmfirst']['own'] / sw['plans']['headline']['own']:.1f}$\\times$")
mac("SWEHeadVsAstraOwnCost", f"{sw['tiers']['astra']['usd'] / sw['plans']['headline']['own']:.1f}$\\times$")
ga = T["gaia"]
mac("GAIAGLMFirstVsAstra", f"{DERIVED['gaia']['glmfirst']['acc'] - ga['tiers']['astra']['acc']:.1f}")
mac("GAIAGLMFirstVsAstraOwnCost", f"{ga['tiers']['astra']['usd'] / ga['plans']['glmfirst']['own']:.1f}$\\times$")
mac("GAIALFVsGLMFirst", f"{DERIVED['gaia']['glmfirst']['acc'] - DERIVED['gaia']['localfirst']['acc']:.1f}")
mac("GAIAErroredFire", f"{ga['signal']['errored_fire_pct_measured']:.1f}")
mac("GAIASigOneAUC", f"{ga['signal']['auc_answer_judge']:.2f}")
mac("GAIASigDepAUC", f"{gsig['auc_deployed']:.2f}")
mac("GAIASigFlagsAUC", f"{gsig['auc_flags']:.2f}")
mac("GAIALFFire", f"{100 * GAIA_LF['E1']:.1f}")
mac("GAIALFRandomAcc", f"{100 * GAIA_LF_RND['acc']:.1f}")
mac("GAIALFLift", f"{100 * (GAIA_LF['acc'] - GAIA_LF_RND['acc']):.1f}")
mac("GAIALFVsGLMFirstCost", f"{ga['plans']['glmfirst']['own'] / ga['plans']['localfirst']['own']:.1f}$\\times$")
t4 = T["tb4"]
mac("TBFHeadVsAstra", f"{DERIVED['tb4']['glmfirst']['acc'] - t4['tiers']['astra']['acc']:.1f}")
mac("TBFGLMFirstVsAstraOwnCost", f"{t4['tiers']['astra']['usd'] / t4['plans']['glmfirst']['own']:.1f}$\\times$")
mac("TBFAAFableOneAcc", f"{PUB['aa_tb4']['fable51_xhigh']['acc']:.1f}")
mac("TBFAAFableOneUSD", usd(PUB["aa_tb4"]["fable51_xhigh"]["usd"]))
mac("TBFGLMFirstVsFableCost", f"{PUB['aa_tb4']['fable51_xhigh']['usd'] / t4['plans']['glmfirst']['own']:.1f}$\\times$")
FA4 = PUB["aa_tb4"]["fable51_xhigh"]   # Fable 5.1 on TB4: its most accurate entry with a cost
mac("TBFAstraTMult", mult(FA4["usd"] / t4["tiers"]["astra"]["usd"]) + "$\\times$")
mac("TBFLFMult", mult(FA4["usd"] / t4["plans"]["localfirst"]["own"]) + "$\\times$")
mac("TBFGLMFirstMult", mult(FA4["usd"] / t4["plans"]["glmfirst"]["own"]) + "$\\times$")
mac("TBFGLMFirstNinetyMult", mult(FA4["usd"] / t4["plans"]["glmfirst90"]["own"]) + "$\\times$")
mac("TBFGLMFirstEightyMult", mult(FA4["usd"] / t4["plans"]["glmfirst80"]["own"]) + "$\\times$")
mac("TBFLocalSaves", f"{t4['plans']['localfirst']['kj'] - t4['plans']['glmfirst']['kj']:.0f}", "kJ the local probe adds on TB4")
mac("NBENCH", "four")

# ---------------------------------------------------------------- claim-2 frontier check (TB2.1, rented basis vs AA single models)
singles = [(v["acc"], v["usd"], v["label"]) for v in PUB["aa_tb21"].values() if v["usd"] is not None]
# the text names only models the paper shows (author, 2026-09-25: Grok 4.6 and Gemini 3.8 Flash are not named)
shown_singles = [(v["acc"], v["usd"], v["label"]) for v in PUB["aa_tb21"].values() if v["usd"] is not None and not v.get("omit_in_text")]
dom = [s for s in shown_singles if s[0] >= 100 * h["acc"] and s[1] <= h["cost"]["rent"]]
DERIVED["rented_dominators"] = dom   # reported in the text (conclusion): on rented hardware a single model is both
cheaper_more_acc = [s for s in singles if s[0] > 100 * h["acc"] and s[1] < h["cost"]["own"]]
check(not cheaper_more_acc, "a single model is more accurate and cheaper than the owned headline")
more_acc = sorted([s for s in singles if s[0] > 100 * h["acc"]], key=lambda s: s[1])
mac("CheapestAboveHeadUSD", usd(more_acc[0][1]), more_acc[0][2])
mac("CheapestAboveHeadName", more_acc[0][2])
mac("CheapestAboveHeadVsRent", f"{more_acc[0][1] / h['cost']['rent']:.1f}$\\times$")
DERIVED["aa_tb21_singles"] = singles

# ---------------------------------------------------------------- emit
lines = ["% ipj-numbers-target.tex -- GENERATED by targets/make_numbers.py from targets/targets.json.",
         "% Do not hand-edit. Every composed value is derived; see make_numbers.py.", ""]
for k in sorted(MACROS):
    lines.append(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}" if "%" not in MACROS[k] else
                 f"\\newcommand{{\\{k}}}{{{MACROS[k].split('   %')[0]}}}   %{MACROS[k].split('   %')[1]}")
(ROOT / "paper" / "ipj-numbers-target.tex").write_text("\n".join(lines) + "\n")
(ROOT / "targets" / "derived.json").write_text(json.dumps(DERIVED, indent=1, default=float))

print(f"{len(MACROS)} macros written")
print(f"s1 rule: machine {HEAD}, recall {sig['s1_recall']:.4f} at false escalation {sig['s1_fpr']:.3f};",
      {k: (round(100 * v['best']['acc'], 2), v['best']['fpr'], round(100 * v['best']['shares'][0], 2)) for k, v in RULE_PICK.items()})
for k in ("headline", "desk", "solvar", "openonly", "two_astra", "two_sol", "retry", "flashnext", "desk_open", "head_dc", "open_dc"):
    r = PLANS[k]
    print(f"{k:10s} acc {100 * r['acc']:.2f}  shares L/M/C {100 * r['shares'][0]:.1f}/{100 * r['shares'][1]:.1f}/{100 * r['shares'][2]:.1f}"
          f"  own ${r['cost']['own']:.4f} ({ANCHOR['usd'] / r['cost']['own']:.0f}x)  amort ${r['cost']['amort']:.3f} ({ANCHOR['usd'] / r['cost']['amort']:.1f}x)"
          f"  rent ${r['cost']['rent']:.3f} ({ANCHOR['usd'] / r['cost']['rent']:.1f}x)  kJ {r['kj']:.0f}")
print(f"random stage-1 at equal fire rate: {100 * rnd['acc']:.2f}  lift {100 * (h['acc'] - rnd['acc']):.1f}")
for k, m in MACH.items():
    print(f"{k:9s} acc {100 * m['acc']:.1f}  kJ/att {m['kj_wall']:.0f}  kJ/solved {m['kj_solved']:.0f}  min {m['wall_min']:.0f}  own ${m['own']:.4f}  amort ${m['amort']:.3f}")
if "--measured" in sys.argv:
    print("remaining targets: all study values (meta.status = TARGET)")
if FAILS:
    print("CHECKS FAILED:")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print(f"cross-check: H200 one agent, mean board power over an attempt {DERIVED['h200_1_mean_board_w']:.0f} W (E-C at C=1 measured 471 W, Qwen3.6 FP8)")
print("all checks passed")
