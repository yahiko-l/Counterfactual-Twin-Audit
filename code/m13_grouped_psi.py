#!/usr/bin/env python
"""
M13 GROUP-CONDITIONAL psi_g — exercise the namesake group index g of the estimand
psi_g(tau) = P(F_tau=1 | T_tau=0, G=g). The headline is ungrouped (g==1); here we
localize the certified within-question failure to MedHallu HALLUCINATION CATEGORIES
(the medically-meaningful question/error type), i.e. WHICH kinds of hallucination
leak through a globally-certified-safe gate.

Methodology is IDENTICAL to m3_combined.py / m6_robust_leaf.py: same 1k+9k data, same
split (seed 7, 40/40/20), same orientation on D_sel, same candidate grid (n_grid=28,
gamma=0.05), same one-answer-per-source rho stream (seed 2024) to fix the headline
operating point tau_hat per (family, alpha). We then PARTITION the D_cal rejected-
truthful denominator at that fixed tau_hat by category g and certify psi_g>alpha PER
GROUP, with a SEPARATE declared multiplicity budget delta_group=0.04 Holm-corrected
over the K eligible categories (a secondary stratified certificate, disclosed as such),
reporting BOTH the exact (Clopper-Pearson, assumes A2 homogeneity) and the
heterogeneity-robust Hoeffding-Bentkus per-group lower bounds. CPU, fast.

Group label: the MedHallu category lives on the H=1 (hallucinated) row; a twin pair's
group is that category. This needs NO new data (recompute on the released scores).
"""
import os, sys, json, csv, time, math, collections
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

E = math.e
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

# ---- robust leaves (copied verbatim from m6_robust_leaf.py; Poisson-binomial valid, no A2) ----
def hb_lower_p(s_c, m_c, alpha):
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    shat = s_c / m_c
    if shat <= alpha: return 1.0
    pH = math.exp(-2.0 * m_c * (shat - alpha) ** 2)
    pB = E * float(binom.sf(s_c - 1, m_c, alpha))
    return float(min(1.0, pH, pB))

def hb_lcb(s_c, m_c, level):
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0 or s_c <= 0: return 0.0
    shat = s_c / m_c
    lcbH = shat - math.sqrt(math.log(1.0 / level) / (2.0 * m_c))
    lcbB = C.cp_lower_psi(s_c, m_c, level / E)
    return float(max(lcbH, lcbB, 0.0))

# ---- data load (identical to m3_combined.py) ----
LAB = "data/scores/scores_strong.csv"; ART = "data/scores/scores_strong_artificial.csv"
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT
PI = 0.10                     # headline / declared realistic prevalence
DELTA_GROUP = C.DELTA_LOWER   # 0.04 — separate disclosed budget for the secondary grouped certificate

def load(path, s):
    rows = list(csv.DictReader(open(path)))
    for r in rows: r.setdefault("source", s)
    return rows
rows = load(LAB, "pqa_labeled") + (load(ART, "pqa_artificial") if os.path.exists(ART) else [])
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
byq = {}
for i in range(len(rows)):
    byq.setdefault(int(qid[i]), {})[int(H[i])] = i
# group label per qid = category of the H=1 (hallucinated) row
CAT = {q: (rows[byq[q][1]]["category"].strip() or "Unlabeled") for q in byq}
QIDS = np.array(sorted(byq))
cat_tot = collections.Counter(CAT[q] for q in QIDS)
log(f"combined twins={len(QIDS)}; categories={dict(cat_tot)}")

def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids); perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

def headline_tau(qids_pool, pi):
    """Reproduce m3_combined: per (family,alpha) the joint-certified best tau_hat on D_cal.
    Returns ORI, D_cal qid order, and per-family per-alpha best cell (tau, m_c, s_c, rho_hat, psi_lcb)."""
    D_sel, D_cal, D_test = split(qids_pool)
    ORI = {}; sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
    def tw(f, qs):
        s = ORI[f]; q = [x for x in qids_pool if x in qs]
        return np.array([s[byq[x][0]] for x in q]), np.array([s[byq[x][1]] for x in q]), q
    rho_pool, low_pool, mass_pool = [], [], []; META = {}; QC = {}
    for f in FAMS:
        rTs, rHs, _ = tw(f, D_sel); grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=28)
        rTc, rHc, qc = tw(f, D_cal); ncal = len(rTc); QC[f] = (rTc, rHc, qc)
        rng = np.random.default_rng(2024); takeH = rng.random(ncal) < pi
        s_stream = np.where(takeH, rHc, rTc); h_stream = takeH.astype(int)
        META[f] = []
        for ti, tau in enumerate(grid):
            acc = s_stream <= tau; Kc = int(acc.sum()); Sc = int(h_stream[acc].sum())
            m_c, sfail = C.psi_counts(rTc, rHc, tau)
            mass_pool.append((f, ti, C.minmass_p(m_c, ncal, GAMMA)))
            META[f].append(dict(tau=float(tau), rho_hat=(round(Sc/Kc, 4) if Kc > 0 else None),
                                m_c=m_c, s_c=sfail, psi_hat=(round(sfail/m_c, 4) if m_c > 0 else None)))
            for a in ALPHAS:
                rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
                low_pool.append((f, a, ti, C.lower_p_psi(sfail, m_c, a)))
    rho_rej = C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL)
    low_rej = C.holm_reject([x[3] for x in low_pool], C.DELTA_LOWER)
    mass_rej = C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS)
    rho_ok = {(x[0],x[1],x[2]): bool(rho_rej[i]) for i,x in enumerate(rho_pool)}
    low_ok = {(x[0],x[1],x[2]): bool(low_rej[i]) for i,x in enumerate(low_pool)}
    mass_ok = {(x[0],x[1]): bool(mass_rej[i]) for i,x in enumerate(mass_pool)}
    nlow = len(low_pool); lev = C.DELTA_LOWER / nlow
    best = {}
    for f in FAMS:
        best[f] = {}
        for a in ALPHAS:
            joint = []
            for ti in range(len(META[f])):
                if rho_ok[(f,a,ti)] and low_ok[(f,a,ti)] and mass_ok[(f,ti)]:
                    r = dict(META[f][ti]); r["psi_lcb"] = round(C.cp_lower_psi(r["s_c"], r["m_c"], lev), 4); joint.append(r)
            joint.sort(key=lambda r: -(r["psi_lcb"] or 0))
            best[f][a] = joint[0] if joint else None
    return ORI, QC, best

def group_certify(rTc, rHc, qc, tau, alpha):
    """At fixed tau, partition the rejected-truthful denominator by category and certify
    psi_g>alpha per group with delta_group Holm over the K eligible (m_c>=1) categories."""
    rTc = np.asarray(rTc, float); rHc = np.asarray(rHc, float)
    rej = rTc > tau
    cats = np.array([CAT[q] for q in qc])
    groups = sorted(set(cats[rej].tolist()), key=lambda g: -int(((cats == g) & rej).sum()))
    rec = []
    for g in groups:
        gm = (cats == g) & rej
        m_g = int(gm.sum()); s_g = int((gm & (rHc <= tau)).sum())
        rec.append(dict(group=g, m_c=m_g, s_c=s_g,
                        psi_hat=(round(s_g/m_g, 4) if m_g > 0 else None),
                        p_exact=C.lower_p_psi(s_g, m_g, alpha),
                        p_hb=hb_lower_p(s_g, m_g, alpha)))
    elig = [r for r in rec if r["m_c"] >= 1]
    K = len(elig); lev = DELTA_GROUP / max(K, 1)             # Bonferroni level per group for the LCB
    ex_rej = C.holm_reject([r["p_exact"] for r in elig], DELTA_GROUP) if K else np.array([])
    hb_rej = C.holm_reject([r["p_hb"] for r in elig], DELTA_GROUP) if K else np.array([])
    by = {id(r): i for i, r in enumerate(elig)}
    for r in rec:
        if r["m_c"] >= 1:
            i = by[id(r)]
            r["lcb_exact"] = round(C.cp_lower_psi(r["s_c"], r["m_c"], lev), 4)
            r["lcb_hb"]    = round(hb_lcb(r["s_c"], r["m_c"], lev), 4)
            r["certified_exact"] = bool(ex_rej[i]); r["certified_hb"] = bool(hb_rej[i])
        else:
            r.update(lcb_exact=None, lcb_hb=None, certified_exact=False, certified_hb=False)
    return dict(K_eligible=K, delta_group=DELTA_GROUP, holm_bonf_level=round(lev, 6),
                n_cert_exact=int(sum(r["certified_exact"] for r in rec)),
                n_cert_hb=int(sum(r["certified_hb"] for r in rec)), groups=rec)

ORI, QC, BEST = headline_tau(QIDS.tolist(), PI)
out = {"run": "M13 group-conditional psi_g by MedHallu hallucination category",
       "pi": PI, "alphas": ALPHAS, "delta_group": DELTA_GROUP,
       "grouping": "MedHallu category (hallucination/question-error type), from the H=1 twin",
       "category_totals": dict(cat_tot), "cells": {}}
for f in FAMS:
    out["cells"][f] = {}
    for a in ALPHAS:
        b = BEST[f][a]
        if b is None:
            out["cells"][f][f"alpha={a}"] = dict(headline_joint_certified=False); continue
        rTc, rHc, qc = QC[f]
        gc = group_certify(rTc, rHc, qc, b["tau"], a)
        out["cells"][f][f"alpha={a}"] = dict(headline_joint_certified=True,
            tau=round(b["tau"], 6), pooled_m_c=b["m_c"], pooled_s_c=b["s_c"],
            pooled_rho_hat=b["rho_hat"], pooled_psi_lcb_exact=b["psi_lcb"], grouped=gc)
out["elapsed_s"] = round(time.time()-T0, 1)
json.dump(out, open("results/m13_grouped_psi_results.json", "w"), indent=2, default=str)

# ---- console summary (headline P(True) alpha=0.10 first) ----
print("\n" + "="*100)
print("M13 GROUP-CONDITIONAL psi_g  —  which MedHallu hallucination categories leak through a certified-safe gate")
print("="*100)
for f in FAMS:
    for a in ALPHAS:
        cell = out["cells"][f][f"alpha={a}"]
        if not cell.get("headline_joint_certified"): continue
        g = cell["grouped"]
        tag = "HEADLINE" if (f == "score_sptrue" and a == 0.10) else ""
        print(f"\n[{f}  alpha={a}]  tau={cell['tau']:.4f}  pooled m_c={cell['pooled_m_c']} "
              f"psi_lcb(exact)={cell['pooled_psi_lcb_exact']}  {tag}")
        print(f"   delta_group={g['delta_group']}  K_eligible={g['K_eligible']}  Holm-Bonf level={g['holm_bonf_level']}  "
              f"| certified>alpha: exact={g['n_cert_exact']}  robust(HB)={g['n_cert_hb']}  of {g['K_eligible']}")
        print(f"   {'category':46s} {'m_c':>4s} {'s_c':>4s} {'psi^':>6s} {'LCBex':>6s} {'LCBhb':>6s}  cert(ex/hb)")
        for r in g["groups"]:
            mark = ("Y" if r["certified_exact"] else "n") + "/" + ("Y" if r["certified_hb"] else "n")
            le = f"{r['lcb_exact']}" if r["lcb_exact"] is not None else "  -  "
            lh = f"{r['lcb_hb']}" if r["lcb_hb"] is not None else "  -  "
            print(f"   {r['group'][:46]:46s} {r['m_c']:>4d} {r['s_c']:>4d} {str(r['psi_hat']):>6s} {le:>6s} {lh:>6s}    {mark}")
print("\n  wrote results/m13_grouped_psi_results.json")
print("="*100)
