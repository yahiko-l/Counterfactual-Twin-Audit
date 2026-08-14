#!/usr/bin/env python
"""
M9-ROBUST-UPPER — re-run the two-constraint mitigation gate with the HETEROGENEITY-ROBUST
upper (safety) leaf in place of the exact Clopper-Pearson upper leaf, to obtain the
DISTRIBUTION-FREE version of the mitigation coverage curve (the 0.96 -> 0.37 figure).

This is the SAFETY-side mirror of m6_robust_leaf.py (which did this for the FAILURE/lower
leaf). It swaps ONLY the psi_g upper leaf used by the two-constraint gate (m9_mitigation.py):

    exact :  upper_p_psi(s_c,m_c,beta) = P[Bin(m_c,beta) <= s_c]            (assumes A2-homog)
    robust:  hb_upper_p(s_c,m_c,beta)  = min{1, exp(-2 m (beta-psihat)^2),
                                              e * P[Bin(m_c,beta) <= s_c]}   (Poisson-binomial; no A2-homog)

Everything else is IDENTICAL to m9_mitigation.py: same split (seed 7), same deployment
stream (seed 2024, pi=0.10), same candidate grid, same rho/mass gates, same Holm step-down
within delta_psiup=0.01, same four-family union. Per supplement rem:c3 / DRAFT_robust_upper_leaf
(Lemma lem:robust-upper), the certified object becomes the selected-set MEAN psi_bar_g <= beta
(= population psi_g only under A2-homog) -- matching what the failure side already does.

Because p_HB >= p_exact pointwise (exact binomial lower tail <= Hoeffding term, and <= e*itself),
the robust gate certifies psi_bar_g<=beta at NO MORE thresholds than the exact gate, so robust
coverage <= exact coverage at every beta. The exact path here MUST reproduce
m9_mitigation_results.json (rho-only 0.9605; beta=0.10 -> 0.3703) as a self-check.

Anchor seed (7 / 2024), pooled 10k, pi=0.10. CPU only, fast.
Writes results/m9_mitigation_robust_upper_results.json.
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

# ----------------- robust upper (safety) leaf: HB lower-tail p-value -----------------
# Exact mirror of m6_robust_leaf.hb_lower_p, flipped to the lower tail (small s_c => reject).
def hb_upper_p(s_c, m_c, beta):
    """Hoeffding-Bentkus p-value for H0: mean psi_g >= beta  (reject => certify psi_bar_g <= beta).
    Small s_c => small p.  min{1, exp(-2 m (beta-shat)^2), e*P[Bin(m,beta)<=s_c]}; shat>=beta => 1."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    shat = s_c / m_c
    if shat >= beta: return 1.0
    pH = math.exp(-2.0 * m_c * (beta - shat) ** 2)        # Hoeffding lower tail
    pB = E * float(binom.cdf(s_c, m_c, beta))             # Bentkus: P[S<=s] <= e*P[Bin(m,beta)<=s]
    return float(min(1.0, pH, pB))

def cp_upper_psi(s_c, m_c, level):
    """Clopper-Pearson one-sided UPPER (1-level) confidence bound for psi_g given s_c/m_c.
    Smallest p1 with P[Bin(m_c,p1) <= s_c] <= level = Beta(s_c+1, m_c-s_c) (1-level) quantile."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    if s_c >= m_c: return 1.0
    if s_c <= 0:   return float(1.0 - level ** (1.0 / m_c))
    return float(_beta.ppf(1.0 - level, s_c + 1, m_c - s_c))

def hb_ucb(s_c, m_c, level):
    """HB UPPER confidence bound on psi_bar_g at `level` = min(Hoeffding UCB, Bentkus UCB).
    Mirror of m6_robust_leaf.hb_lcb (flipped). Bentkus UCB = CP-upper at level/e."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    shat = s_c / m_c
    ucbH = shat + math.sqrt(math.log(1.0 / level) / (2.0 * m_c))   # Hoeffding UCB
    ucbB = cp_upper_psi(s_c, m_c, level / E)                       # Bentkus UCB = CP-upper at level/e
    return float(min(ucbH, ucbB, 1.0))

# ----------------- reuse m9_mitigation.py setup VERBATIM -----------------
FAMS = M7.FAMS; ALPHAS = M7.ALPHAS; POOLED = M7.POOLED
SPLIT_SEED = 7; PREV_SEED = 2024; PI = 0.10
BETAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FLABEL = {"score_sptrue": "P(True)-8B", "score_sNLI": "ref-NLI", "score_sjudge": "LLM-judge-8B"}

cache = M7.build_split_cache(POOLED, SPLIT_SEED)
ncal = cache["ncal"]; nlow = cache["nlow"]
nj, cells = M7.certify_one(cache, PI, PREV_SEED)
hb = cells["score_sptrue"][0.10]["best"]
assert hb and hb["m_c"] == 278, f"anchor mismatch {hb}"
log(f"anchor OK (m_c=278). ncal={ncal}")

def stream(fam):
    rTc = cache["rTc"][fam]; rHc = cache["rHc"][fam]
    rng = np.random.default_rng(PREV_SEED); takeH = rng.random(ncal) < PI
    s = np.where(takeH, rHc, rTc); h = takeH.astype(int)
    return s, h
STREAM = {f: stream(f) for f in FAMS}

# per-(fam,ti): coverage, rho_hat, m_c, s_c, psi_hat, denom_mass + p-values (exact & robust)
PER = {f: [] for f in FAMS}
rho_pool = []; mass_pool = []
psiup_pool_ex = {b: [] for b in BETAS}   # exact upper leaf
psiup_pool_hb = {b: [] for b in BETAS}   # robust upper leaf
for f in FAMS:
    grid = cache["grids"][f]; s_str, h_str = STREAM[f]
    for ti, tau in enumerate(grid):
        m_c = cache["m_c"][f][ti]; s_c = cache["s_c"][f][ti]
        acc = s_str <= tau; Kc = int(acc.sum()); Sc = int(h_str[acc].sum())
        cov = Kc / ncal
        PER[f].append(dict(ti=ti, tau=float(tau), coverage=round(cov, 4),
                           rho_hat=(round(Sc/Kc, 4) if Kc else None), Kc=Kc,
                           m_c=m_c, s_c=s_c, psi_hat=(round(s_c/m_c, 4) if m_c else None),
                           denom_mass=round(m_c/ncal, 4)))
        for a in ALPHAS:
            rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
        mass_pool.append((f, ti, C.minmass_p(m_c, ncal, C.GAMMA_DEFAULT)))
        for b in BETAS:
            psiup_pool_ex[b].append((f, ti, C.upper_p_psi(s_c, m_c, b)))   # EXACT  (baseline)
            psiup_pool_hb[b].append((f, ti, hb_upper_p(s_c, m_c, b)))      # ROBUST (this run)

# Holm within each family's own budget (psiup pooled over (f,ti) per beta within delta_psiup)
rho_ok = {(x[0],x[1],x[2]): bool(r) for x,r in zip(rho_pool, C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL))}
mass_ok = {(x[0],x[1]): bool(r) for x,r in zip(mass_pool, C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS))}
def holm_psiup(pool):
    ok = {}
    for b in BETAS:
        rj = C.holm_reject([x[2] for x in pool[b]], C.DELTA_PSIUP)
        for x, r in zip(pool[b], rj): ok[(b, x[0], x[1])] = bool(r)
    return ok
psiup_ok_ex = holm_psiup(psiup_pool_ex)
psiup_ok_hb = holm_psiup(psiup_pool_hb)

def best_cov(fam, predicate):
    best = None
    for ti in range(len(cache["grids"][fam])):
        if predicate(ti):
            rec = PER[fam][ti]
            if best is None or rec["coverage"] > best["coverage"]:
                best = rec
    return best

nlow_psiup = len(psiup_pool_ex[BETAS[0]])           # candidates per beta family
lev_psiup = C.DELTA_PSIUP / nlow_psiup              # Holm floor level (for UCB reporting)

# ----------------- rho-only "certified-safe" baseline (leaf-independent) -----------------
rho_only = {}
for f in FAMS:
    rho_only[f] = {}
    for a in ALPHAS:
        g = best_cov(f, lambda ti: rho_ok[(f, a, ti)])
        rho_only[f][a] = None if g is None else dict(
            tau=round(g["tau"],6), coverage=g["coverage"], rho_hat=g["rho_hat"],
            psi_hat=g["psi_hat"], m_c=g["m_c"], leaks=bool(g["psi_hat"] is not None and g["psi_hat"] > a))

# ----------------- two-constraint gate: exact vs robust upper leaf -----------------
def gate_table(psiup_ok):
    out = {}
    for f in FAMS:
        out[FLABEL[f]] = {}
        for a in ALPHAS:
            ro = rho_only[f][a]; cov_rho = ro["coverage"] if ro else None
            row = {}
            for b in BETAS:
                g = best_cov(f, lambda ti: rho_ok[(f,a,ti)] and mass_ok[(f,ti)] and psiup_ok[(b,f,ti)])
                if g is None:
                    row[f"beta={b}"] = dict(exists=False)
                else:
                    row[f"beta={b}"] = dict(exists=True, tau=round(g["tau"],6), coverage=g["coverage"],
                        rho_hat=g["rho_hat"], psi_hat=g["psi_hat"], m_c=g["m_c"],
                        coverage_cost_vs_rho_only=(round(cov_rho-g["coverage"],4) if cov_rho is not None else None),
                        coverage_retained_frac=(round(g["coverage"]/cov_rho,4) if cov_rho else None),
                        psi_ucb=round(hb_ucb(g["s_c"], g["m_c"], lev_psiup), 4))
            out[FLABEL[f]][f"alpha={a}"] = row
    return out
gate_ex = gate_table(psiup_ok_ex)
gate_hb = gate_table(psiup_ok_hb)

# ----------------- headline frontier: P(True), alpha=0.10, beta -> max certified coverage -----------------
def frontier(psiup_ok):
    f = "score_sptrue"; a = 0.10; front = []
    for b in BETAS:
        g = best_cov(f, lambda ti: rho_ok[(f,a,ti)] and mass_ok[(f,ti)] and psiup_ok[(b,f,ti)])
        front.append(dict(beta=b, max_coverage=(g["coverage"] if g else 0.0),
                          certified=bool(g is not None), psi_hat=(g["psi_hat"] if g else None),
                          psi_ucb=(round(hb_ucb(g["s_c"], g["m_c"], lev_psiup),4) if g else None)))
    return front
front_ex = frontier(psiup_ok_ex)
front_hb = frontier(psiup_ok_hb)

# ----------------- SELF-CHECK: exact path must reproduce the published baseline -----------------
ro_ptrue = rho_only["score_sptrue"][0.10]
checks = {
    "rho_only_coverage": (ro_ptrue["coverage"], 0.9605),
    "exact_beta0.10_coverage": (front_ex[0]["max_coverage"], 0.3703),
    "exact_beta0.20_coverage": (front_ex[2]["max_coverage"], 0.5960),
}
selfcheck_ok = all(abs(got - exp) < 1e-3 for got, exp in checks.values())
for k,(got,exp) in checks.items():
    log(f"selfcheck {k}: got={got} expected={exp} {'OK' if abs(got-exp)<1e-3 else 'MISMATCH'}")
assert selfcheck_ok, f"EXACT path did not reproduce baseline: {checks}"

# ----------------- report -----------------
report = {
    "run": "M9 robust-upper: two-constraint mitigation with heterogeneity-robust (Hoeffding-Bentkus) "
           "upper safety leaf vs exact Clopper-Pearson upper leaf",
    "leaf_robust": "Hoeffding-Bentkus lower-tail p-value (Poisson-binomial, no A2-homog); "
                   "certified object = selected-set mean psi_bar_g <= beta",
    "anchor": {"split": SPLIT_SEED, "prev": PREV_SEED, "pi": PI, "nlow": nlow,
               "delta_psiup": C.DELTA_PSIUP, "holm_floor_level_psiup": round(lev_psiup, 6)},
    "betas": BETAS,
    "selfcheck": {k: {"got": got, "expected": exp} for k,(got,exp) in checks.items()},
    "rho_only_gate": {FLABEL[f]: {f"alpha={a}": rho_only[f][a] for a in ALPHAS} for f in FAMS},
    "frontier_ptrue_alpha0.10": {
        "rho_only_coverage": ro_ptrue["coverage"], "rho_only_psi_hat": ro_ptrue["psi_hat"],
        "rho_only_leaks": ro_ptrue["leaks"],
        "exact_upper_leaf": front_ex, "robust_upper_leaf": front_hb,
        "robust_vs_exact_coverage_drop": [
            round(front_ex[i]["max_coverage"] - front_hb[i]["max_coverage"], 4) for i in range(len(BETAS))]},
    "two_constraint_gate_exact": gate_ex,
    "two_constraint_gate_robust": gate_hb,
}
report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m9_mitigation_robust_upper_results.json", "w"), indent=2, default=str)

# ----------------- console -----------------
print("\n" + "="*96)
print("M9 ROBUST-UPPER — two-constraint mitigation: EXACT (Clopper-Pearson, assumes A2-homog)")
print("                  vs ROBUST (Hoeffding-Bentkus, no A2-homog) upper safety leaf")
print("="*96)
print(f"\nSelf-check (exact path reproduces published baseline): {'PASS' if selfcheck_ok else 'FAIL'}")
print(f"\nHeadline frontier: P(True)-8B, alpha=0.10, pi=0.10")
print(f"  rho-only 'certified-safe' baseline: coverage={ro_ptrue['coverage']:.4f}  "
      f"psi_hat={ro_ptrue['psi_hat']} (LEAKS at alpha=0.10)")
print(f"\n  {'beta':>5} | {'exact cov':>10} {'exact psi_ucb':>13} | {'ROBUST cov':>10} {'robust psi_ucb':>14} | {'cov drop':>8}")
print("  " + "-"*74)
for i, b in enumerate(BETAS):
    ex = front_ex[i]; rb = front_hb[i]
    drop = ex["max_coverage"] - rb["max_coverage"]
    tag = "  <== beta=alpha (headline mitigation point)" if abs(b-0.10) < 1e-9 else ""
    print(f"  {b:>5.2f} | {ex['max_coverage']:>10.4f} {str(ex['psi_ucb']):>13} | "
          f"{rb['max_coverage']:>10.4f} {str(rb['psi_ucb']):>14} | {drop:>8.4f}{tag}")
print(f"\n  Interpretation: certifying psi_bar_g <= alpha (beta=0.10) forces coverage")
print(f"    exact  leaf: {ro_ptrue['coverage']:.2f} -> {front_ex[0]['max_coverage']:.2f}  "
      f"(-{ro_ptrue['coverage']-front_ex[0]['max_coverage']:.2f})")
print(f"    ROBUST leaf: {ro_ptrue['coverage']:.2f} -> {front_hb[0]['max_coverage']:.2f}  "
      f"(-{ro_ptrue['coverage']-front_hb[0]['max_coverage']:.2f})   [distribution-free, given C1]")
print(f"\n  wrote results/m9_mitigation_robust_upper_results.json  ({report['elapsed_s']}s)")
print("="*96)
