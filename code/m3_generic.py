#!/usr/bin/env python
"""Generic psi_g joint-cell certification for a SECOND dataset (cross-distribution stress test).
Same protocol as m3_combined.py (split seed 7, prevalence seed 2024, 28-pt grid, gamma=0.05, four-family
Holm budget) but reads ONE scores CSV (idx,qid,H,...,score_sptrue,score_sjudge,score_sNLI) and reports
the joint-cell count across the induced-prevalence sweep pi in {0.50,0.20,0.10,0.05}.

Usage:  python m3_generic.py <scores_csv> <out_json> [dataset_label]
A joint cell (family x alpha) is certified iff some tau passes  rho<=alpha (cluster-valid LTT on the
one-answer-per-source pi-mixture)  AND  psi_g>alpha (exact conditional binomial)  AND  denom>=gamma.
The headline question for the second dataset: does the robust core reproduce at low pi?  CPU only."""
import os, sys, json, csv, time
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

CSVP = sys.argv[1] if len(sys.argv) > 1 else "data/scores/scores_halueval.csv"
OUTP = sys.argv[2] if len(sys.argv) > 2 else "results/m3_halueval_results.json"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "second-dataset"
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]; ALPHAS = [0.05, 0.10, 0.20]
GAMMA = C.GAMMA_DEFAULT; PIS = [0.50, 0.20, 0.10, 0.05]; PI_PRIMARY = 0.10

rows = list(csv.DictReader(open(CSVP)))
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
byq = {}
for i in range(len(rows)): byq.setdefault(int(qid[i]), {})[int(H[i])] = i
QIDS = sorted(q for q in byq if set(byq[q]) == {0, 1})    # complete twins only
print(f"[{LABEL}] complete twins={len(QIDS)} (rows={len(rows)})")

def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids); perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

def certify_at_prevalence(qids_pool, pi):
    D_sel, D_cal, D_test = split(qids_pool)
    sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    ORI = {}
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
    def tw(f, qs):
        s = ORI[f]; q = [x for x in qids_pool if x in qs]
        return np.array([s[byq[x][0]] for x in q]), np.array([s[byq[x][1]] for x in q])
    rho_pool, low_pool, mass_pool = [], [], []; META = {}
    for f in FAMS:
        rTs, rHs = tw(f, D_sel); grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=28)
        rTc, rHc = tw(f, D_cal); ncal = len(rTc)
        u = np.random.default_rng(2024).random(ncal); takeH = u < pi
        s_stream = np.where(takeH, rHc, rTc); h_stream = takeH.astype(int)
        META[f] = []
        for ti, tau in enumerate(grid):
            acc = s_stream <= tau; Kc = int(acc.sum()); Sc = int(h_stream[acc].sum())
            m_c, sfail = C.psi_counts(rTc, rHc, tau)
            mass_pool.append((f, ti, C.minmass_p(m_c, ncal, GAMMA)))
            META[f].append(dict(tau=round(float(tau), 6), rho_hat=(round(Sc/Kc, 4) if Kc > 0 else None),
                                m_c=m_c, s_c=sfail, psi_hat=(round(sfail/m_c, 4) if m_c > 0 else None)))
            for a in ALPHAS:
                rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
                low_pool.append((f, a, ti, C.lower_p_psi(sfail, m_c, a)))
    rr = C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL); lr = C.holm_reject([x[3] for x in low_pool], C.DELTA_LOWER)
    mr = C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS)
    rho_ok = {(x[0], x[1], x[2]): bool(rr[i]) for i, x in enumerate(rho_pool)}
    low_ok = {(x[0], x[1], x[2]): bool(lr[i]) for i, x in enumerate(low_pool)}
    mass_ok = {(x[0], x[1]): bool(mr[i]) for i, x in enumerate(mass_pool)}
    nlow = len(low_pool); cells = {}
    for f in FAMS:
        cells[f] = {}
        for a in ALPHAS:
            joint = []
            for ti in range(len(META[f])):
                if rho_ok[(f, a, ti)] and low_ok[(f, a, ti)] and mass_ok[(f, ti)]:
                    rec = dict(META[f][ti]); rec["psi_lcb"] = round(C.cp_lower_psi(rec["s_c"], rec["m_c"], C.DELTA_LOWER/nlow), 4)
                    joint.append(rec)
            joint.sort(key=lambda r: -(r["psi_lcb"] or 0))
            cells[f][a] = dict(joint_certified=len(joint) > 0, best=(joint[0] if joint else None))
    njoint = sum(1 for f in FAMS for a in ALPHAS if cells[f][a]["joint_certified"])
    survivors = [f"{f.replace('score_s','')}/a{a}" for f in FAMS for a in ALPHAS if cells[f][a]["joint_certified"]]
    return dict(pi=pi, joint_cells=njoint, survivors=survivors, cells=cells)

def paired_acc(qids_pool):
    D_sel, _, _ = split(qids_pool); sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    _, fl = C.orient(H[sel_idx], RAW["score_sjudge"][sel_idx]); s = (-RAW["score_sjudge"] if fl else RAW["score_sjudge"])
    rT = np.array([s[byq[q][0]] for q in qids_pool]); rH = np.array([s[byq[q][1]] for q in qids_pool])
    return round(float(np.mean(rH > rT) + 0.5*np.mean(rH == rT)), 4)

report = dict(run=f"M3 generic ({LABEL})", dataset=LABEL, n_twins=len(QIDS), gamma=GAMMA,
              alphas=ALPHAS, pis=PIS, judge_paired_acc=paired_acc(QIDS),
              config=dict(split_seed=7, split_fracs=[0.40, 0.40, 0.20], prev_seed=2024, n_grid=28,
                          gamma=GAMMA, alphas=ALPHAS, pis=PIS, exchangeable_unit="source question (twin pair)",
                          delta=dict(total=C.DELTA_TOTAL, global_rate=C.DELTA_GLOBAL, mass=C.DELTA_MASS,
                                     psi_up=C.DELTA_PSIUP, lower=C.DELTA_LOWER),
                          note="identical protocol to the MedHallu headline (m3_combined.py): same source-level "
                               "40/40/20 split (seed 7), 28-point D_sel quantile grid, prevalence-draw seed 2024, "
                               "four-family Holm budget summing to 0.10, min-mass floor gamma"),
              pooled={f"pi={pi}": certify_at_prevalence(QIDS, pi) for pi in PIS})
report["sweep"] = {f"pi={pi}": report["pooled"][f"pi={pi}"]["joint_cells"] for pi in PIS}
json.dump(report, open(OUTP, "w"), indent=2, default=str)
print(f"[{LABEL}] joint-cell sweep (of 9):  " + "  ".join(f"pi={pi}:{report['sweep'][f'pi={pi}']}" for pi in PIS))
print(f"[{LABEL}] survivors at pi=0.10: {report['pooled']['pi=0.1']['survivors']}")
prim = report["pooled"][f"pi={PI_PRIMARY}"]
for f in FAMS:
    for a in ALPHAS:
        b = prim["cells"][f][a]["best"]
        if b: print(f"    [{f} a={a}] tau={b['tau']} rho={b['rho_hat']} m_c={b['m_c']} psi_hat={b['psi_hat']} psi_lcb={b['psi_lcb']}")
print(f"wrote {OUTP}")
