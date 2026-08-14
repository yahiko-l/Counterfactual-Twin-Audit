#!/usr/bin/env python
"""
M6 ROBUST LEAF — heterogeneity-robust re-certification of the headline psi_g>alpha failure
certificate, addressing kill-argument P3. The exact leaf of Lemma L1 targets the POPULATION
psi_g under the i.i.d. clause; this robust leaf drops the conditional-homogeneity half of (A2)
and bounds the SELECTED-SET MEAN psi_g_bar instead, via the supplement's Poisson-binomial-aware
Hoeffding/Bentkus construction. Homogeneity is therefore needed only to equate the two objects,
not to underwrite the failure certificate.

Swaps ONLY the psi_g (delta_lower) leaf — exact `lower_p_psi`/`cp_lower_psi` -> Hoeffding-Bentkus
(and empirical-Bernstein for reference). Everything else identical to m3_combined.py: same data,
same one-answer-per-source rho stream (seed 2024), same split (seed 7), same candidate grid, same
Holm within delta_lower=0.04, same union of four families, same cluster-valid global + mass leaves
(those are on iid one-answer-per-source rows / question-level counts, so they do NOT assume (A2)
and are kept exact). This is exactly the rem:c3 prescription: "re-run Holm (L3) and union (L5)
unchanged." CPU, fast.

Reports, per prevalence and family x alpha: exact psi_lcb vs robust HB psi_lcb vs EB psi_lcb, and
whether each still certifies psi_g>alpha (lcb>alpha), plus exact-vs-robust joint-cell counts.
"""
import os, sys, json, csv, time, math
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

E = math.e
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

# ---------- robust leaves (valid for Poisson-binomial: independent heterogeneous Bernoulli, NO (A2)) ----------
def hb_lower_p(s_c, m_c, alpha):
    """Hoeffding-Bentkus p-value for H0: mean psi_g <= alpha (reject => certify psi_g>alpha)."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    shat = s_c / m_c
    if shat <= alpha: return 1.0
    pH = math.exp(-2.0 * m_c * (shat - alpha) ** 2)            # Hoeffding upper tail
    pB = E * float(binom.sf(s_c - 1, m_c, alpha))              # Bentkus: P[S>=s] <= e*P[Bin(m,alpha)>=s]
    return float(min(1.0, pH, pB))

def hb_lcb(s_c, m_c, level):
    """HB lower confidence bound on psi_g at `level` = max(Hoeffding LCB, Bentkus LCB)."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0 or s_c <= 0: return 0.0
    shat = s_c / m_c
    lcbH = shat - math.sqrt(math.log(1.0 / level) / (2.0 * m_c))      # Hoeffding LCB
    lcbB = C.cp_lower_psi(s_c, m_c, level / E)                        # Bentkus LCB = CP at level/e
    return float(max(lcbH, lcbB, 0.0))

def eb_lcb(s_c, m_c, level):
    """Empirical-Bernstein LCB (tighter; uses Bernoulli sample variance). Reference only."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 1 or s_c <= 0: return 0.0
    shat = s_c / m_c
    var = shat * (1.0 - shat) * m_c / (m_c - 1.0)
    ln = math.log(3.0 / level)                                       # 3/level: split over two EB terms + safety
    eb = shat - math.sqrt(2.0 * var * ln / m_c) - 3.0 * ln / (m_c - 1.0)
    return float(max(eb, 0.0))

# ---------- data load (identical to m3_combined.py) ----------
LAB = "data/scores/scores_strong.csv"; ART = "data/scores/scores_strong_artificial.csv"
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT
PI_PRIMARY = 0.10; PIS = [0.50, 0.20, 0.10, 0.05]

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
QIDS = np.array(sorted(byq))
log(f"combined twins={len(QIDS)}")

def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids); perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

def certify_at_prevalence(qids_pool, pi, label):
    D_sel, D_cal, D_test = split(qids_pool)
    ORI = {}
    sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
    def tw(f, qs):
        s = ORI[f]; q = [x for x in qids_pool if x in qs]
        return np.array([s[byq[x][0]] for x in q]), np.array([s[byq[x][1]] for x in q]), q
    rho_pool, low_pool_exact, low_pool_hb, mass_pool = [], [], [], []
    META = {}
    for f in FAMS:
        rTs, rHs, _ = tw(f, D_sel); grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=28)
        rTc, rHc, _ = tw(f, D_cal); ncal = len(rTc)
        rng = np.random.default_rng(2024); takeH = rng.random(ncal) < pi
        s_stream = np.where(takeH, rHc, rTc); h_stream = takeH.astype(int)
        META[f] = []
        for ti, tau in enumerate(grid):
            acc = s_stream <= tau; Kc = int(acc.sum()); Sc = int(h_stream[acc].sum())
            m_c, sfail = C.psi_counts(rTc, rHc, tau)
            mass_pool.append((f, ti, C.minmass_p(m_c, ncal, GAMMA)))
            META[f].append(dict(tau=round(float(tau), 6), rho_hat=(round(Sc/Kc, 4) if Kc > 0 else None),
                                m_c=m_c, s_c=sfail, psi_hat=(round(sfail/m_c, 4) if m_c > 0 else None),
                                denom_mass=round(m_c/ncal, 4)))
            for a in ALPHAS:
                rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
                low_pool_exact.append((f, a, ti, C.lower_p_psi(sfail, m_c, a)))
                low_pool_hb.append((f, a, ti, hb_lower_p(sfail, m_c, a)))    # ROBUST leaf p-value
    rho_rej = C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL)
    mass_rej = C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS)
    low_rej_exact = C.holm_reject([x[3] for x in low_pool_exact], C.DELTA_LOWER)
    low_rej_hb = C.holm_reject([x[3] for x in low_pool_hb], C.DELTA_LOWER)    # ROBUST Holm
    rho_ok = {(x[0],x[1],x[2]): bool(rho_rej[i]) for i,x in enumerate(rho_pool)}
    mass_ok = {(x[0],x[1]): bool(mass_rej[i]) for i,x in enumerate(mass_pool)}
    low_ok_ex = {(x[0],x[1],x[2]): bool(low_rej_exact[i]) for i,x in enumerate(low_pool_exact)}
    low_ok_hb = {(x[0],x[1],x[2]): bool(low_rej_hb[i]) for i,x in enumerate(low_pool_hb)}
    nlow = len(low_pool_exact); lev = C.DELTA_LOWER / nlow
    out = {}
    for f in FAMS:
        out[f] = {}
        for a in ALPHAS:
            jex, jhb = [], []
            for ti in range(len(META[f])):
                base = rho_ok[(f,a,ti)] and mass_ok[(f,ti)]
                if base and low_ok_ex[(f,a,ti)]:
                    r = dict(META[f][ti]); r["psi_lcb"] = round(C.cp_lower_psi(r["s_c"], r["m_c"], lev), 4); jex.append(r)
                if base and low_ok_hb[(f,a,ti)]:
                    r = dict(META[f][ti])
                    r["psi_lcb_hb"] = round(hb_lcb(r["s_c"], r["m_c"], lev), 4)
                    r["psi_lcb_eb"] = round(eb_lcb(r["s_c"], r["m_c"], lev), 4)
                    r["psi_lcb_exact"] = round(C.cp_lower_psi(r["s_c"], r["m_c"], lev), 4)
                    jhb.append(r)
            jex.sort(key=lambda r: -(r["psi_lcb"] or 0)); jhb.sort(key=lambda r: -(r["psi_lcb_hb"] or 0))
            out[f][a] = dict(exact_joint=len(jex) > 0, exact_best=(jex[0] if jex else None),
                             robust_joint=len(jhb) > 0, robust_best=(jhb[0] if jhb else None))
    njx = sum(1 for f in FAMS for a in ALPHAS if out[f][a]["exact_joint"])
    njr = sum(1 for f in FAMS for a in ALPHAS if out[f][a]["robust_joint"])
    return dict(label=label, pi=pi, n_twins=len(qids_pool), holm_level=round(lev, 6),
                exact_joint_cells=njx, robust_joint_cells=njr, cells=out)

POOLED = QIDS.tolist()
report = {"run": "M6 robust-leaf (Hoeffding-Bentkus / empirical-Bernstein psi_g leaf; addresses kill-argument P3 / supplement rem:c3)",
          "leaf": "Hoeffding-Bentkus (valid for Poisson-binomial, no (A2) homogeneity)",
          "delta_lower": C.DELTA_LOWER, "alphas": ALPHAS, "pi_primary": PI_PRIMARY,
          "pooled": {f"pi={pi}": certify_at_prevalence(POOLED, pi, f"pooled pi={pi}") for pi in PIS}}
report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m6_robust_leaf_results.json", "w"), indent=2, default=str)

print("\n" + "="*92)
print("M6 ROBUST LEAF — exact (Clopper-Pearson, assumes A2) vs robust (Hoeffding-Bentkus, no A2) psi_g>alpha")
print("="*92)
for pi in PIS:
    r = report["pooled"][f"pi={pi}"]
    print(f"\nπ={pi}:  exact joint cells = {r['exact_joint_cells']}/9   robust(HB) joint cells = {r['robust_joint_cells']}/9   (Holm level={r['holm_level']})")
    for f in FAMS:
        for a in ALPHAS:
            c = r["cells"][f][a]
            if c["exact_joint"] or c["robust_joint"]:
                eb_ = c["exact_best"]; rb = c["robust_best"]
                ex = f"exactLCB={eb_['psi_lcb']}" if eb_ else "exact=NONE"
                if rb:
                    rr = f"HB-LCB={rb['psi_lcb_hb']} EB-LCB={rb['psi_lcb_eb']} (exactLCB={rb['psi_lcb_exact']}) survives={rb['psi_lcb_hb']>a}"
                else:
                    rr = "robust=NOT certified"
                tag = "✓" if c["robust_joint"] else "✗"
                print(f"   [{f} α={a}] {tag} m_c={(eb_ or rb)['m_c']} ψ̂={(eb_ or rb)['psi_hat']} | {ex} | {rr}")
print("\n  wrote results/m6_robust_leaf_results.json")
print("="*92)
