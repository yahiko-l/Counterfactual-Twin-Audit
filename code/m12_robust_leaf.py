#!/usr/bin/env python
"""
M12 ROBUST LEAF — heterogeneity-robust (Hoeffding-Bentkus) re-certification of the SECOND-DATASET
reproductions (HaluEval-QA, MedQuAD-subtle, MedQuAD-blatant), so the external evidence is presented
with the SAME assumption-aligned robust leaf as the MedHallu headline (m6_robust_leaf.py).

Reviewer ask (GPT-5.5 Pro, 2026-06-20, "high-value non-blocking #2"): the MedHallu headline is
robust-leaf certified (drops the (A2) homogeneity clause, keeps only the post-selection independence
DERIVED from C1), but the external reproductions in Table 5 are presented only through the exact /
homogeneity-conditional binomial leaf. This script swaps ONLY the psi_g (delta_lower) leaf
exact `lower_p_psi`/`cp_lower_psi` -> Hoeffding-Bentkus, EXACTLY as m6 does for MedHallu, and keeps
everything else identical to m3_generic.py: same single-CSV complete-twins loader, same split (seed 7),
same one-answer-per-source rho stream (prevalence seed 2024), same 28-pt D_sel grid, same gamma, same
Holm within delta_lower=0.04, same union of four families, same cluster-valid global + mass leaves
(those are on iid one-answer-per-source rows / question-level counts -> they do NOT assume (A2) and stay
exact). The exact leaf is recomputed alongside as a REPRODUCTION GUARD: exact_joint_cells must match the
published m3_*_results.json sweeps (0/3/6/9 HaluEval, 0/1/3/6 MedQuAD-subtle, 0/0/2/2 MedQuAD-blatant).

Reports, per dataset / prevalence / family x alpha: exact psi_lcb vs robust HB psi_lcb vs EB psi_lcb, and
whether each still certifies psi_g>alpha (lcb>alpha), plus exact-vs-robust joint-cell counts. CPU, fast.
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
# Byte-for-byte the same definitions used in m6_robust_leaf.py for the MedHallu headline.
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

FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT
PI_PRIMARY = 0.10; PIS = [0.50, 0.20, 0.10, 0.05]

# (csv, label, published exact sweep for the reproduction guard)  -- blatant uses scores_medquad.csv (orig style)
DATASETS = [
    ("data/scores/scores_halueval.csv",       "HaluEval-QA",     {0.05: 9, 0.10: 6, 0.20: 3, 0.50: 0}),
    ("data/scores/scores_medquad_subtle.csv",  "MedQuAD-subtle",  {0.05: 6, 0.10: 3, 0.20: 1, 0.50: 0}),
    ("data/scores/scores_medquad.csv",         "MedQuAD-blatant", {0.05: 2, 0.10: 2, 0.20: 0, 0.50: 0}),
]

def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids); perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

def certify_at_prevalence(rows, byq, QIDS, pi):
    qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
    RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
    qids_pool = QIDS
    D_sel, D_cal, D_test = split(qids_pool)
    sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    ORI = {}
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
    def tw(f, qs):
        s = ORI[f]; q = [x for x in qids_pool if x in qs]
        return np.array([s[byq[x][0]] for x in q]), np.array([s[byq[x][1]] for x in q])
    rho_pool, low_pool_exact, low_pool_hb, mass_pool = [], [], [], []
    META = {}
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
    return dict(pi=pi, holm_level=round(lev, 6), exact_joint_cells=njx, robust_joint_cells=njr, cells=out)

def run_dataset(csvp, label, guard):
    rows = list(csv.DictReader(open(csvp)))
    byq = {}
    for i in range(len(rows)): byq.setdefault(int(rows[i]["qid"]), {})[int(rows[i]["H"])] = i
    QIDS = sorted(q for q in byq if set(byq[q]) == {0, 1})    # complete twins only (identical to m3_generic.py)
    log(f"[{label}] complete twins={len(QIDS)} (rows={len(rows)}) from {os.path.basename(csvp)}")
    res = {f"pi={pi}": certify_at_prevalence(rows, byq, QIDS, pi) for pi in PIS}
    # reproduction guard: exact-leaf joint counts must match the published m3_*_results.json sweep
    guard_ok = all(res[f"pi={pi}"]["exact_joint_cells"] == guard[pi] for pi in PIS)
    return dict(dataset=label, csv=os.path.basename(csvp), n_twins=len(QIDS),
                exact_sweep={f"pi={pi}": res[f"pi={pi}"]["exact_joint_cells"] for pi in PIS},
                robust_sweep={f"pi={pi}": res[f"pi={pi}"]["robust_joint_cells"] for pi in PIS},
                published_exact_sweep={f"pi={pi}": guard[pi] for pi in PIS},
                exact_reproduces_published=guard_ok, pooled=res)

report = {"run": "M12 robust-leaf (Hoeffding-Bentkus psi_g leaf) on second-dataset reproductions; "
                 "assumption-aligns Table 5 with the MedHallu headline (m6_robust_leaf.py).",
          "leaf": "Hoeffding-Bentkus (valid for Poisson-binomial, no (A2) homogeneity; bounds selected-set mean psi_bar_g)",
          "delta_lower": C.DELTA_LOWER, "alphas": ALPHAS, "pis": PIS, "pi_primary": PI_PRIMARY, "gamma": GAMMA,
          "config": dict(split_seed=7, split_fracs=[0.40, 0.40, 0.20], prev_seed=2024, n_grid=28,
                         note="exact leaf swapped to Hoeffding-Bentkus ONLY; global+mass+Holm+union identical to m3_generic.py"),
          "datasets": {}}
for csvp, label, guard in DATASETS:
    report["datasets"][label] = run_dataset(csvp, label, guard)
report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m12_robust_leaf_results.json", "w"), indent=2, default=str)

print("\n" + "="*100)
print("M12 ROBUST LEAF — exact (Clopper-Pearson, assumes A2) vs robust (Hoeffding-Bentkus, no A2) on Table-5 benchmarks")
print("="*100)
for _, label, _ in DATASETS:
    d = report["datasets"][label]
    guard = "OK" if d["exact_reproduces_published"] else "*** MISMATCH ***"
    print(f"\n### {label}  (n_twins={d['n_twins']})   exact reproduces published sweep: {guard}")
    print(f"    exact  joint sweep (of 9):  " + "  ".join(f"pi={pi}:{d['exact_sweep'][f'pi={pi}']}" for pi in PIS))
    print(f"    robust joint sweep (of 9):  " + "  ".join(f"pi={pi}:{d['robust_sweep'][f'pi={pi}']}" for pi in PIS))
    prim = d["pooled"][f"pi={PI_PRIMARY}"]
    print(f"    -- representative cells at pi={PI_PRIMARY} (Holm level={prim['holm_level']}):")
    for f in FAMS:
        for a in ALPHAS:
            c = prim["cells"][f][a]
            if c["exact_joint"] or c["robust_joint"]:
                eb_ = c["exact_best"]; rb = c["robust_best"]
                base = eb_ or rb
                ex = f"exactLCB={eb_['psi_lcb']}" if eb_ else "exact=NONE"
                if rb:
                    rr = f"HB-LCB={rb['psi_lcb_hb']} EB-LCB={rb['psi_lcb_eb']} survives(>a)={rb['psi_lcb_hb']>a}"
                else:
                    rr = "robust=NOT certified"
                tag = "OK " if c["robust_joint"] else "-- "
                fam = f.replace("score_s", "")
                print(f"       [{tag}{fam:7s} a={a}] rho={base['rho_hat']} m_c={base['m_c']} psi_hat={base['psi_hat']} | {ex} | {rr}")
print("\n  wrote results/m12_robust_leaf_results.json")
print("="*100)
