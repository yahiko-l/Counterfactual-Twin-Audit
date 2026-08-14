#!/usr/bin/env python
"""R067 — single-detector psi_g certification for a PUBLISHED-GATE reproduction.

IDENTICAL protocol to the MedHallu headline (m3_combined.py): source-level 40/40/20 split (seed 7),
28-pt D_sel quantile grid, prevalence-draw seed 2024, four-family Holm budget, min-mass floor gamma.
But for ONE detector score column (a single published gate), so the joint-cell count is out of 3
(the three alpha levels). A joint cell is certified iff some tau passes  rho<=alpha (cluster-valid
one-answer-per-source LTT on the pi-mixture)  AND  psi_g>alpha (exact conditional binomial)  AND
denom>=gamma.  Question: does a globally certified-safe PUBLISHED gate still leak psi_g at low pi?

Usage: python m_pubgate_cert.py <scores_csv> <score_col> <out_json> <label>   (CPU)"""
import os, sys, json, csv
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

CSVP = sys.argv[1]; COL = sys.argv[2]; OUTP = sys.argv[3]; LABEL = sys.argv[4] if len(sys.argv) > 4 else "pubgate"
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT; PIS = [0.50, 0.20, 0.10, 0.05]
rows = list(csv.DictReader(open(CSVP)))
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
RAW = np.array([float(r[COL]) for r in rows])
byq = {}
for i in range(len(rows)): byq.setdefault(int(qid[i]), {})[int(H[i])] = i
QIDS = [q for q in sorted(byq) if set(byq[q]) == {0, 1}]

def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids); perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

def certify_at(pi):
    D_sel, D_cal, D_test = split(QIDS)
    sel = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
    _, fl = C.orient(H[sel], RAW[sel]); ORI = (-RAW if fl else RAW)
    def tw(qs):
        q = [x for x in QIDS if x in qs]; return np.array([ORI[byq[x][0]] for x in q]), np.array([ORI[byq[x][1]] for x in q])
    rTs, rHs = tw(D_sel); grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=28)
    rTc, rHc = tw(D_cal); ncal = len(rTc)
    takeH = np.random.default_rng(2024).random(ncal) < pi
    s_stream = np.where(takeH, rHc, rTc); h_stream = takeH.astype(int)
    rho_pool = []; low_pool = []; mass_pool = []; META = []
    for ti, tau in enumerate(grid):
        acc = s_stream <= tau; Kc = int(acc.sum()); Sc = int(h_stream[acc].sum())
        m_c, sfail = C.psi_counts(rTc, rHc, tau)
        mass_pool.append(C.minmass_p(m_c, ncal, GAMMA))
        META.append(dict(tau=round(float(tau), 6), rho_hat=(round(Sc/Kc, 4) if Kc > 0 else None),
                         m_c=m_c, s_c=sfail, psi_hat=(round(sfail/m_c, 4) if m_c > 0 else None)))
        for a in ALPHAS:
            rho_pool.append((a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
            low_pool.append((a, ti, C.lower_p_psi(sfail, m_c, a)))
    rr = C.holm_reject([x[2] for x in rho_pool], C.DELTA_GLOBAL)
    lr = C.holm_reject([x[2] for x in low_pool], C.DELTA_LOWER)
    mr = C.holm_reject(mass_pool, C.DELTA_MASS)
    rho_ok = {(x[0], x[1]): bool(rr[i]) for i, x in enumerate(rho_pool)}
    low_ok = {(x[0], x[1]): bool(lr[i]) for i, x in enumerate(low_pool)}
    mass_ok = {ti: bool(mr[ti]) for ti in range(len(grid))}
    nlow = len(low_pool); cells = {}
    for a in ALPHAS:
        joint = [ti for ti in range(len(grid)) if rho_ok[(a, ti)] and low_ok[(a, ti)] and mass_ok[ti]]
        best = None
        if joint:
            bi = max(joint, key=lambda ti: C.cp_lower_psi(META[ti]["s_c"], META[ti]["m_c"], C.DELTA_LOWER/nlow))
            best = dict(META[bi]); best["psi_lcb"] = round(C.cp_lower_psi(best["s_c"], best["m_c"], C.DELTA_LOWER/nlow), 4)
        cells[a] = dict(joint_certified=bool(joint), best=best)
    return dict(pi=pi, joint_cells=sum(cells[a]["joint_certified"] for a in ALPHAS), cells=cells)

def paired_acc():
    D_sel, _, _ = split(QIDS); sel = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
    _, fl = C.orient(H[sel], RAW[sel]); ORI = (-RAW if fl else RAW)
    rT = np.array([ORI[byq[q][0]] for q in QIDS]); rH = np.array([ORI[byq[q][1]] for q in QIDS])
    return round(float(np.mean(rH > rT) + 0.5*np.mean(rH == rT)), 4)

from sklearn.metrics import roc_auc_score
_, gfl = C.orient(H, RAW)
rep = dict(run=f"R067 published-gate psi_g cert ({LABEL})", label=LABEL, col=COL, n_twins=len(QIDS),
           risk_auroc=round(float(roc_auc_score(H, (-RAW if gfl else RAW))), 4), paired_acc=paired_acc(),
           split_seed=7, prev_seed=2024, n_grid=28, gamma=GAMMA, alphas=ALPHAS, sweep={}, pooled={})
for pi in PIS:
    r = certify_at(pi); rep["pooled"][f"pi={pi}"] = r; rep["sweep"][f"pi={pi}"] = r["joint_cells"]
json.dump(rep, open(OUTP, "w"), indent=2, default=str)
print(f"[{LABEL}] AUROC={rep['risk_auroc']} paired_acc={rep['paired_acc']}  joint-cell sweep (of 3 alphas): "
      + "  ".join(f"{pi}:{rep['sweep'][pi]}" for pi in rep['sweep']))
prim = rep["pooled"]["pi=0.1"]
for a in ALPHAS:
    b = prim["cells"][a]["best"]
    if b: print(f"    [a={a}] tau={b['tau']} rho={b['rho_hat']} m_c={b['m_c']} psi_hat={b['psi_hat']} psi_lcb={b['psi_lcb']}")
print(f"wrote {OUTP}")
