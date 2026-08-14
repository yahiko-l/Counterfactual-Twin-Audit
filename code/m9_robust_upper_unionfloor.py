#!/usr/bin/env python
"""
M9 ROBUST-UPPER, UNION-FLOOR recompute (proof-checker run05, 2026-06-30).

The supplement's Lemma lem:robust-upper certifies the heterogeneity-robust upper
(safety) leaf for the D_cal-RANDOM selected-set mean psi_bar_g, so its family-wise
validity rests on the UNION BOUND at the per-candidate level delta_psiup/|C_up|
(= delta_psiup/78), NOT Holm step-down (Holm's fixed-null proof does not apply to a
random selected-set target; it is a power-only refinement that is valid ONLY where it
coincides with the union floor). The original m9_mitigation_robust_upper.py certified
the robust frontier via Holm (holm_psiup), which can relax the level above the floor.

This script recomputes the robust upper frontier + full (f,alpha,beta) gate under the
VALID union-floor criterion  p_HB <= lev_psiup = delta_psiup/nlow_psiup, everything
else identical, and compares to (i) the exact Clopper-Pearson leaf under Holm
(legitimate: exact leaf is a fixed-null family) and (ii) the original robust-Holm
numbers (self-check that we reproduce them). Output drives the corrected
tab:supp-robust-upper and the "coverage loss" / "clears the floor" wording.

CPU only, fast. Writes m9_robust_upper_unionfloor_results.json.
"""
import os, sys, json, time, math
import numpy as np
from scipy.stats import binom, beta as _beta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
import m7_multiseed_robustness as M7

E = math.e
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

# ---- robust upper leaf p-value + UCB (verbatim from m9_mitigation_robust_upper.py) ----
def hb_upper_p(s_c, m_c, beta):
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    shat = s_c / m_c
    if shat >= beta: return 1.0
    pH = math.exp(-2.0 * m_c * (beta - shat) ** 2)
    pB = E * float(binom.cdf(s_c, m_c, beta))
    return float(min(1.0, pH, pB))

def cp_upper_psi(s_c, m_c, level):
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    if s_c >= m_c: return 1.0
    if s_c <= 0:   return float(1.0 - level ** (1.0 / m_c))
    return float(_beta.ppf(1.0 - level, s_c + 1, m_c - s_c))

def hb_ucb(s_c, m_c, level):
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    shat = s_c / m_c
    ucbH = shat + math.sqrt(math.log(1.0 / level) / (2.0 * m_c))
    ucbB = cp_upper_psi(s_c, m_c, level / E)
    return float(min(ucbH, ucbB, 1.0))

# ---- setup VERBATIM ----
FAMS = M7.FAMS; ALPHAS = M7.ALPHAS; POOLED = M7.POOLED
SPLIT_SEED = 7; PREV_SEED = 2024; PI = 0.10
BETAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FLABEL = {"score_sptrue": "P(True)-8B", "score_sNLI": "ref-NLI", "score_sjudge": "LLM-judge-8B"}

cache = M7.build_split_cache(POOLED, SPLIT_SEED)
ncal = cache["ncal"]; nlow = cache["nlow"]
nj, cells = M7.certify_one(cache, PI, PREV_SEED)
hb0 = cells["score_sptrue"][0.10]["best"]
assert hb0 and hb0["m_c"] == 278, f"anchor mismatch {hb0}"
log(f"anchor OK (m_c=278). ncal={ncal}")

def stream(fam):
    rTc = cache["rTc"][fam]; rHc = cache["rHc"][fam]
    rng = np.random.default_rng(PREV_SEED); takeH = rng.random(ncal) < PI
    s = np.where(takeH, rHc, rTc); h = takeH.astype(int)
    return s, h
STREAM = {f: stream(f) for f in FAMS}

PER = {f: [] for f in FAMS}
rho_pool = []; mass_pool = []
psiup_pool_ex = {b: [] for b in BETAS}
psiup_pool_hb = {b: [] for b in BETAS}
for f in FAMS:
    grid = cache["grids"][f]; s_str, h_str = STREAM[f]
    for ti, tau in enumerate(grid):
        m_c = cache["m_c"][f][ti]; s_c = cache["s_c"][f][ti]
        acc = s_str <= tau; Kc = int(acc.sum()); Sc = int(h_str[acc].sum())
        cov = Kc / ncal
        PER[f].append(dict(ti=ti, tau=float(tau), coverage=round(cov, 4),
                           rho_hat=(round(Sc/Kc, 4) if Kc else None), Kc=Kc,
                           m_c=m_c, s_c=s_c, psi_hat=(round(s_c/m_c, 4) if m_c else None)))
        for a in ALPHAS:
            rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
        mass_pool.append((f, ti, C.minmass_p(m_c, ncal, C.GAMMA_DEFAULT)))
        for b in BETAS:
            psiup_pool_ex[b].append((f, ti, C.upper_p_psi(s_c, m_c, b)))
            psiup_pool_hb[b].append((f, ti, hb_upper_p(s_c, m_c, b)))

rho_ok = {(x[0],x[1],x[2]): bool(r) for x,r in zip(rho_pool, C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL))}
mass_ok = {(x[0],x[1]): bool(r) for x,r in zip(mass_pool, C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS))}

def holm_psiup(pool):
    ok = {}
    for b in BETAS:
        rj = C.holm_reject([x[2] for x in pool[b]], C.DELTA_PSIUP)
        for x, r in zip(pool[b], rj): ok[(b, x[0], x[1])] = bool(r)
    return ok

nlow_psiup = len(psiup_pool_ex[BETAS[0]])
lev_psiup = C.DELTA_PSIUP / nlow_psiup          # union floor = delta_psiup/|C_up|

# robust certification under the VALID UNION FLOOR: p_HB <= lev_psiup
def floor_psiup(pool):
    return {(b, x[0], x[1]): bool(x[2] <= lev_psiup) for b in BETAS for x in pool[b]}

psiup_ok_ex_holm   = holm_psiup(psiup_pool_ex)   # exact leaf, Holm  (legitimate, fixed null)
psiup_ok_hb_holm   = holm_psiup(psiup_pool_hb)   # robust leaf, Holm (original; reproduces JSON)
psiup_ok_hb_floor  = floor_psiup(psiup_pool_hb)  # robust leaf, UNION FLOOR (valid; corrected)

def best_cov(fam, predicate):
    best = None
    for ti in range(len(cache["grids"][fam])):
        if predicate(ti):
            rec = PER[fam][ti]
            if best is None or rec["coverage"] > best["coverage"]:
                best = rec
    return best

# ---- headline frontier P(True) alpha=0.10 ----
def frontier(psiup_ok):
    f = "score_sptrue"; a = 0.10; front = []
    for b in BETAS:
        g = best_cov(f, lambda ti: rho_ok[(f,a,ti)] and mass_ok[(f,ti)] and psiup_ok[(b,f,ti)])
        front.append(dict(beta=b, max_coverage=(g["coverage"] if g else 0.0),
                          certified=bool(g is not None),
                          psi_hat=(g["psi_hat"] if g else None),
                          psi_ucb=(round(hb_ucb(g["s_c"], g["m_c"], lev_psiup),4) if g else None)))
    return front
front_ex    = frontier(psiup_ok_ex_holm)
front_hb_h  = frontier(psiup_ok_hb_holm)
front_hb_f  = frontier(psiup_ok_hb_floor)

# ---- full (f,alpha,beta) gate coverage under each rule; count losses vs exact ----
def gate_cov(psiup_ok):
    out = {}
    for f in FAMS:
        for a in ALPHAS:
            for b in BETAS:
                g = best_cov(f, lambda ti: rho_ok[(f,a,ti)] and mass_ok[(f,ti)] and psiup_ok[(b,f,ti)])
                out[(f,a,b)] = (g["coverage"] if g else 0.0)
    return out
cov_ex   = gate_cov(psiup_ok_ex_holm)
cov_hb_h = gate_cov(psiup_ok_hb_holm)
cov_hb_f = gate_cov(psiup_ok_hb_floor)

def losses(cov_rb):
    L = []
    for k in cov_ex:
        if cov_rb[k] < cov_ex[k] - 1e-9:
            f,a,b = k
            L.append(dict(family=FLABEL[f], alpha=a, beta=b,
                          exact=round(cov_ex[k],4), robust=round(cov_rb[k],4),
                          drop=round(cov_ex[k]-cov_rb[k],4)))
    return L
loss_holm  = losses(cov_hb_h)
loss_floor = losses(cov_hb_f)

print("\n" + "="*92)
print("ROBUST UPPER: Holm (original) vs UNION FLOOR (valid) — P(True) alpha=0.10 headline frontier")
print("="*92)
print(f"  union floor lev_psiup = delta_psiup/{nlow_psiup} = {lev_psiup:.6e}")
print(f"\n  {'beta':>5} | {'exact cov':>9} | {'robust(Holm)':>12} {'ucb':>7} | {'robust(FLOOR)':>13} {'ucb':>7} | {'floor drop':>10}")
print("  " + "-"*82)
for i,b in enumerate(BETAS):
    ex=front_ex[i]; rh=front_hb_h[i]; rf=front_hb_f[i]
    drop = ex["max_coverage"] - rf["max_coverage"]
    print(f"  {b:>5.2f} | {ex['max_coverage']:>9.4f} | {rh['max_coverage']:>12.4f} {str(rh['psi_ucb']):>7} | "
          f"{rf['max_coverage']:>13.4f} {str(rf['psi_ucb']):>7} | {drop:>10.4f}")
print(f"\n  full 3x3x7=63-cell sweep losses vs EXACT:")
print(f"    robust under Holm  : {len(loss_holm)} cells lose coverage  -> {loss_holm}")
print(f"    robust under FLOOR : {len(loss_floor)} cells lose coverage  -> {loss_floor}")

report = {
    "run": "M9 robust-upper UNION-FLOOR recompute (proof-checker run05): robust safety leaf "
           "certified at the valid union floor delta_psiup/|C_up| instead of Holm",
    "anchor": {"split": SPLIT_SEED, "prev": PREV_SEED, "pi": PI, "nlow": nlow,
               "delta_psiup": C.DELTA_PSIUP, "nlow_psiup": nlow_psiup,
               "union_floor_level": lev_psiup},
    "betas": BETAS,
    "frontier_ptrue_alpha0.10": {
        "exact_holm": front_ex, "robust_holm": front_hb_h, "robust_unionfloor": front_hb_f,
        "robust_floor_vs_exact_drop": [round(front_ex[i]["max_coverage"]-front_hb_f[i]["max_coverage"],4)
                                       for i in range(len(BETAS))]},
    "sweep_losses_vs_exact": {"robust_holm": loss_holm, "robust_unionfloor": loss_floor},
    "elapsed_s": round(time.time()-T0,1),
}
json.dump(report, open("results/m9_robust_upper_unionfloor_results.json","w"), indent=2, default=str)
print(f"\n  wrote results/m9_robust_upper_unionfloor_results.json  ({report['elapsed_s']}s)")
print("="*92)
