#!/usr/bin/env python
"""
M8 DEPENDENCE / OVERDISPERSION DIAGNOSTIC + DESIGN-EFFECT SENSITIVITY
(reviewer-objection #4 + the binding kill-argument WARN: A2 post-selection
independence + homogeneity is ADOPTED, not derived; P_1/P_3 partially_answered@critical).

This is the "explicit diagnostic for the post-selection independence assumption" that the
kill-argument recommends (run03 fix #2) to move P_1/P_3 toward answered. It does NOT derive
independence (still adopted); it shows the headline failure is not FRAGILE to plausible
violations, two ways:

  PART A (homogeneity half / overdispersion):
    On the certified headline operating point (P(True)-8B, tau-hat, D_cal rejected-truthful
    pairs, m_c=278), test whether the accept-hallucinated outcomes F_tau are homogeneous
    across OBSERVABLE strata (difficulty / source / category / difficulty x source). Report
    the Pearson dispersion phi (=1 under homogeneity), a G-test p-value for equality of
    proportions, and an intra-cluster-correlation (ICC) based design effect DEFF.

  PART B (independence half / effective-sample-size sensitivity):
    For EVERY cell certified at pi=0.10 (anchor seed), recompute the Clopper-Pearson psi_g
    lower bound under a design effect d (effective sample m_eff=m_c/d, s_eff=psi_hat*m_eff,
    same level delta_lower/nlow). Report psi_g LCB(d) for d in {1,1.5,2,2.5,3,4,5} and the
    BREAKDOWN d* where the LCB crosses alpha. Mechanism-agnostic: bounds the impact of any
    unmodeled positive cross-pair dependence as a variance-inflation / ESS-deflation.

  PART C (cluster vs iid-row global gate, 10k, generalizes the 1k TABLE4(a) b3 check):
    At each pi, recompute the joint-cell count when the GLOBAL rho<=alpha gate is certified
    (a) cluster-valid one-answer-per-source [headline] vs (b) iid-row over both twins
    [anti-conservative]. Shows the leak is not produced by the cluster correction at 10k.

Reuses the validated m7 machinery (split seed 7 reproduces the released headline exactly).
CPU only. Writes results/m8_dependence_results.json.
"""
import os, sys, json, csv, time
import numpy as np
from scipy.stats import binom, beta as _beta, chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
import m7_multiseed_robustness as M7   # module-level load runs once; gives byq,RAW,H,QIDS,POOLED,build_split_cache,certify_one,split

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

ALPHAS = M7.ALPHAS; FAMS = M7.FAMS; POOLED = M7.POOLED
SPLIT_SEED = 7; PREV_SEED = 2024; PI = 0.10
FLABEL = {"score_sptrue": "P(True)-8B", "score_sNLI": "ref-NLI", "score_sjudge": "LLM-judge-8B"}

# ----------------------------------------------------------------- covariates per qid
COV = {}  # qid -> dict(difficulty, source, cat_halluc)
for path, srcdef in [("data/scores/scores_strong.csv", "pqa_labeled"),
                     ("data/scores/scores_strong_artificial.csv", "pqa_artificial")]:
    for r in csv.DictReader(open(path)):
        q = int(r["qid"]); h = int(r["H"])
        d = COV.setdefault(q, {"difficulty": r.get("difficulty", "") or "NA",
                               "source": r.get("source", srcdef) or srcdef, "cat_halluc": "NA"})
        if h == 1 and (r.get("category", "") or "").strip():
            d["cat_halluc"] = r["category"].strip()

# ----------------------------------------------------------------- float Clopper-Pearson lower
def cp_lower_float(s_eff, m_eff, level):
    """One-sided lower (1-level) CP bound for a binomial proportion, FLOAT counts allowed
    (keeps psi_hat fixed while deflating m). Beta(s, m-s+1) `level` quantile."""
    if m_eff <= 0 or s_eff <= 0: return 0.0
    if s_eff >= m_eff: return float(level ** (1.0 / m_eff))
    return float(_beta.ppf(level, s_eff, m_eff - s_eff + 1.0))

# ----------------------------------------------------------------- build anchor split + headline
cache = M7.build_split_cache(POOLED, SPLIT_SEED)
nlow = cache["nlow"]; LEVEL = C.DELTA_LOWER / nlow
nj, cells = M7.certify_one(cache, PI, PREV_SEED)
# sanity: reproduce released headline
hb = cells["score_sptrue"][0.10]["best"]
assert hb is not None and hb["m_c"] == 278 and abs(hb["psi_lcb"] - 0.327) < 1e-6, f"anchor mismatch {hb}"
log(f"anchor OK: nlow={nlow}, level=delta_lower/nlow={LEVEL:.3e}; headline P(True) a=0.10 "
    f"m_c={hb['m_c']} s_c={hb['s_c']} psi_hat={hb['psi_hat']} psi_lcb={hb['psi_lcb']}")

# reconstruct cal qid order (same order tw() uses inside build_split_cache)
D_sel, D_cal, D_test = M7.split(POOLED, seed=SPLIT_SEED)
qc = [x for x in POOLED if x in D_cal]

report = {"run": "M8 dependence/overdispersion + design-effect sensitivity",
          "anchor": {"split_seed": SPLIT_SEED, "prev_seed": PREV_SEED, "pi": PI, "nlow": nlow,
                     "cp_level": LEVEL}, "partA_overdispersion": {}, "partB_design_effect": {},
          "partC_iid_vs_cluster": {}}

# =================================================================== PART A: overdispersion
def overdispersion(F, strata_key, min_k=2, min_cell=10):
    """F: 0/1 array of accept-hallucinated outcomes on the rejected-truthful set.
       strata_key: array of stratum labels aligned to F. Returns dispersion diagnostics."""
    labs = np.array(strata_key)
    groups = [g for g in sorted(set(labs.tolist())) if (labs == g).sum() >= min_cell]
    m = len(F); s = int(F.sum()); psi = s / m if m else float("nan")
    rows = []
    for g in groups:
        sel = labs == g; mk = int(sel.sum()); sk = int(F[sel].sum())
        rows.append({"stratum": str(g), "m_k": mk, "s_k": sk, "psi_k": round(sk/mk, 4)})
    K = len(rows)
    if K < min_k or psi in (0.0, 1.0):
        return {"K": K, "psi_pooled": round(psi, 4), "note": "insufficient strata or degenerate psi"}
    # Pearson dispersion phi = (1/(K-1)) sum (s_k - m_k psi)^2 / (m_k psi(1-psi))
    var0 = psi * (1 - psi)
    X2 = sum((r["s_k"] - r["m_k"]*psi)**2 / (r["m_k"]*var0) for r in rows)
    phi = X2 / (K - 1)
    g_p = float(chi2.sf(X2, K - 1))            # Pearson chi-square test of homogeneity
    # ICC (one-way ANOVA estimator on 0/1) -> clustered design effect DEFF = 1 + (m_bar-1)*ICC
    mbar = np.mean([r["m_k"] for r in rows])
    msb = sum(r["m_k"]*(r["psi_k"] - psi)**2 for r in rows) / (K - 1)
    msw = sum(r["s_k"]*(1-r["psi_k"])**2 + (r["m_k"]-r["s_k"])*(0-r["psi_k"])**2 for r in rows) / (m - K)
    m0 = (m - sum(r["m_k"]**2 for r in rows)/m) / (K - 1)
    icc = (msb - msw) / (msb + (m0 - 1)*msw) if (msb + (m0-1)*msw) > 0 else 0.0
    deff_icc = 1 + (mbar - 1)*max(icc, 0.0)
    return {"K": K, "psi_pooled": round(psi, 4), "pearson_phi": round(float(phi), 3),
            "chi2": round(float(X2), 3), "df": K-1, "homogeneity_p": round(g_p, 4),
            "icc": round(float(icc), 4), "deff_icc": round(float(deff_icc), 3),
            "deff_pearson": round(float(max(phi, 1.0)), 3), "strata": rows}

# headline P(True) operating point on D_cal rejected-truthful pairs
f = "score_sptrue"; tau = hb["tau"]
rTc = cache["rTc"][f]; rHc = cache["rHc"][f]
rej = rTc > tau
idx = np.where(rej)[0]
F = (rHc[idx] <= tau).astype(int)
qsel = [qc[i] for i in idx]
assert len(F) == hb["m_c"], f"m_c reconstruct {len(F)} != {hb['m_c']}"
diff = np.array([COV[q]["difficulty"] for q in qsel])
srcv = np.array([COV[q]["source"] for q in qsel])
catv = np.array([COV[q]["cat_halluc"] for q in qsel])
dxs = np.array([f"{COV[q]['difficulty']}|{COV[q]['source']}" for q in qsel])
report["partA_overdispersion"] = {
    "cell": f"{FLABEL[f]} alpha=0.10 @ pi={PI}", "tau": round(tau, 6),
    "m_c": int(len(F)), "s_c": int(F.sum()), "psi_hat": round(float(F.mean()), 4),
    "by_difficulty": overdispersion(F, diff),
    "by_source": overdispersion(F, srcv),
    "by_category": overdispersion(F, catv),
    "by_difficulty_x_source": overdispersion(F, dxs),
}

# =================================================================== PART B: design-effect sweep
DEFFS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
def deff_curve(m_c, s_c, alpha):
    psi = s_c / m_c
    out = []
    for d in DEFFS:
        m_eff = m_c / d; s_eff = psi * m_eff
        lcb = cp_lower_float(s_eff, m_eff, LEVEL)
        out.append({"deff": d, "m_eff": round(m_eff, 1), "psi_lcb": round(lcb, 4),
                    "survives": bool(lcb > alpha)})
    # breakdown d*: bisection on d where lcb == alpha
    lo, hi = 1.0, 200.0
    if cp_lower_float(psi*(m_c/hi), m_c/hi, LEVEL) > alpha:
        dstar = float("inf")
    else:
        for _ in range(60):
            mid = (lo+hi)/2; v = cp_lower_float(psi*(m_c/mid), m_c/mid, LEVEL)
            if v > alpha: lo = mid
            else: hi = mid
        dstar = round(lo, 2)
    return {"psi_hat": round(psi, 4), "curve": out, "breakdown_deff": dstar}

certified10 = []
for fam in FAMS:
    for a in ALPHAS:
        b = cells[fam][a]["best"]
        if b is not None:
            certified10.append((fam, a, b))
report["partB_design_effect"] = {
    "deffs": DEFFS, "level": LEVEL,
    "cells": {f"{FLABEL[fam]}|alpha={a}": dict(m_c=b["m_c"], s_c=b["s_c"], tau=round(b["tau"], 6),
                                               released_psi_lcb=b["psi_lcb"], **deff_curve(b["m_c"], b["s_c"], a))
              for (fam, a, b) in certified10},
}

# =================================================================== PART C: cluster vs iid-row global gate
# Correct generalization of the 1k TABLE4(a) b3 check to the 10k NATIVE BALANCED twin table.
# At the induced low-pi headline the deployment stream is one-answer-per-source BY CONSTRUCTION,
# so the global certificate is already iid-valid there (no cluster correction is applied and the
# cluster-vs-iid distinction is moot). The distinction only bites at the BALANCED NATIVE twin
# table, where twins are correlated; b3 asks whether the honest negative is a cluster artifact.
#   cluster-valid : subsample one answer per source (uniform, declared seed) -> iid rows.
#   iid-row       : treat ALL 2*ncal twin rows as iid (ANTI-conservative: correlated twins).
# We report, per (family x alpha), the # of rho<=alpha-certified thresholds AND the joint-cell
# count (rho_ok AND low_ok AND mass_ok) under each calibration model.
def rho_ok_native(mode, draw_seed=0):
    ncal = cache["ncal"]; rho_pool = []
    for fam in FAMS:
        rTc = cache["rTc"][fam]; rHc = cache["rHc"][fam]; grid = cache["grids"][fam]
        if mode == "cluster":                                   # one answer per source (balanced)
            rng = np.random.default_rng(draw_seed); pickH = rng.random(ncal) < 0.5
            s = np.where(pickH, rHc, rTc); h = pickH.astype(int)
        else:                                                   # iid-row: all 2N twin rows
            s = np.r_[rTc, rHc]; h = np.r_[np.zeros(ncal, int), np.ones(ncal, int)]
        for ti, tau in enumerate(grid):
            acc = s <= tau; Kc = int(acc.sum()); Sc = int(h[acc].sum())
            for a in ALPHAS: rho_pool.append((fam, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
    rej = C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL)
    return {(x[0], x[1], x[2]): bool(rej[i]) for i, x in enumerate(rho_pool)}

def summarize_native(ok):
    rho_thr = 0; joint = 0; jdetail = {}
    for fam in FAMS:
        for a in ALPHAS:
            rcert = [ti for ti in range(len(cache["grids"][fam])) if ok[(fam, a, ti)]]
            rho_thr += len(rcert)
            jc = any(ok[(fam, a, ti)] and cache["low_ok"][(fam, a, ti)] and cache["mass_ok"][(fam, ti)]
                     for ti in range(len(cache["grids"][fam])))
            jdetail[f"{FLABEL[fam]}|{a}"] = bool(jc); joint += int(jc)
    return {"rho_thresholds_certified": rho_thr, "joint_cells": joint, "joint_detail": jdetail}

# average cluster over a few subsample draws (the one-per-source pick is random); iid-row is deterministic
cluster_draws = [summarize_native(rho_ok_native("cluster", s)) for s in range(5)]
iid_native = summarize_native(rho_ok_native("iidrow"))
report["partC_iid_vs_cluster"] = {
    "setting": "NATIVE BALANCED 10k twin table (D_cal), generalizes 1k TABLE4(a) b3 check",
    "note": "At the induced low-pi headline the gate is one-answer-per-source by construction "
            "(already iid-valid); cluster-vs-iid only bites at native balance. iid-row is "
            "anti-conservative (correlated twins as iid rows).",
    "cluster_valid_over_5_draws": {
        "rho_thresholds_certified_mean": round(float(np.mean([d["rho_thresholds_certified"] for d in cluster_draws])), 2),
        "joint_cells_mean": round(float(np.mean([d["joint_cells"] for d in cluster_draws])), 2),
        "joint_cells_max": int(max(d["joint_cells"] for d in cluster_draws))},
    "iid_row_anticonservative": {
        "rho_thresholds_certified": iid_native["rho_thresholds_certified"],
        "joint_cells": iid_native["joint_cells"]},
    "verdict": "both agree at native balance (honest negative is NOT a cluster artifact at 10k)"
               if (max(d["joint_cells"] for d in cluster_draws) == 0 and iid_native["joint_cells"] == 0)
               else "DIFFER -- inspect detail"}

report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m8_dependence_results.json", "w"), indent=2, default=str)

# ----------------------------------------------------------------- console summary
print("\n" + "="*90)
print("M8 DEPENDENCE DIAGNOSTIC + DESIGN-EFFECT SENSITIVITY")
A = report["partA_overdispersion"]
print(f"\nPART A — overdispersion on headline rejected-truthful set "
      f"({A['cell']}, m_c={A['m_c']}, psi_hat={A['psi_hat']}):")
for key in ["by_difficulty", "by_source", "by_category", "by_difficulty_x_source"]:
    d = A[key]
    if "pearson_phi" in d:
        print(f"  {key:24s}: K={d['K']} phi={d['pearson_phi']} (homogeneity p={d['homogeneity_p']}) "
              f"ICC={d['icc']} DEFF_icc={d['deff_icc']} DEFF_pearson={d['deff_pearson']}")
    else:
        print(f"  {key:24s}: {d.get('note','')}")
print(f"\nPART B — psi_g LCB under design effect d (breakdown d* = LCB crosses alpha):")
for name, c in report["partB_design_effect"]["cells"].items():
    curve = " ".join(f"{x['deff']}:{x['psi_lcb']}" for x in c["curve"])
    print(f"  {name:26s} m_c={c['m_c']:4d} psi_hat={c['psi_hat']}  d*={c['breakdown_deff']}")
    print(f"       LCB(d): {curve}")
print(f"\nPART C — cluster vs iid-row global gate, NATIVE BALANCED 10k (b3 generalized):")
pc = report["partC_iid_vs_cluster"]
cv = pc["cluster_valid_over_5_draws"]; iv = pc["iid_row_anticonservative"]
print(f"  cluster-valid (5 draws): rho-thresholds~{cv['rho_thresholds_certified_mean']}, "
      f"joint cells mean={cv['joint_cells_mean']} (max {cv['joint_cells_max']})")
print(f"  iid-row (anti-cons.)   : rho-thresholds={iv['rho_thresholds_certified']}, "
      f"joint cells={iv['joint_cells']}")
print(f"  => {pc['verdict']}")
print(f"\n  wrote results/m8_dependence_results.json   ({report['elapsed_s']}s)")
print("="*90)
