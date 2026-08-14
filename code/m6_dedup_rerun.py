#!/usr/bin/env python
"""
m6 DE-DUP RE-RUN — re-certify the headline psi_g>alpha failure on the C1-de-duplicated pool
(drop the single shared-abstract twin so the HB-independence lemma's source-question
independence premise holds literally, not just at 99.99%).

Logic is copied VERBATIM from m6_robust_leaf.py (same data, same orient, same split seed 7,
same prevalence seed 2024, same 28-pt grid, same four-family Holm, same HB leaf) and
parametrized only by an exclusion set DROP applied to the pooled twin list. As a transcription
guard it first runs on the FULL pool (DROP=empty): that MUST reproduce m6's published headline
(P(True) pi=0.10 alpha=0.10: exact LCB 0.327, HB LCB 0.320, psi_hat 0.4317, m_c 278) before the
de-dup number is trusted.

CPU, fast. Writes m6_dedup_rerun_results.json.
"""
import os, sys, json, csv, time, math
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

E = math.e
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

# ---------- robust leaves (verbatim from m6_robust_leaf.py) ----------
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

# ---------- data load (verbatim from m6_robust_leaf.py) ----------
from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
LAB = artifact("scores_strong.csv"); ART = artifact("scores_strong_artificial.csv")
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

def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids); perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

def certify_at_prevalence(qids_pool, pi, label, folds=None):
    D_sel, D_cal, D_test = folds if folds is not None else split(qids_pool)
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
                low_pool_hb.append((f, a, ti, hb_lower_p(sfail, m_c, a)))
    rho_rej = C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL)
    mass_rej = C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS)
    low_rej_exact = C.holm_reject([x[3] for x in low_pool_exact], C.DELTA_LOWER)
    low_rej_hb = C.holm_reject([x[3] for x in low_pool_hb], C.DELTA_LOWER)
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
                    r["psi_lcb_exact"] = round(C.cp_lower_psi(r["s_c"], r["m_c"], lev), 4)
                    jhb.append(r)
            jex.sort(key=lambda r: -(r["psi_lcb"] or 0)); jhb.sort(key=lambda r: -(r["psi_lcb_hb"] or 0))
            out[f][a] = dict(exact_joint=len(jex) > 0, exact_best=(jex[0] if jex else None),
                             robust_joint=len(jhb) > 0, robust_best=(jhb[0] if jhb else None))
    njx = sum(1 for f in FAMS for a in ALPHAS if out[f][a]["exact_joint"])
    njr = sum(1 for f in FAMS for a in ALPHAS if out[f][a]["robust_joint"])
    return dict(label=label, pi=pi, n_twins=len(qids_pool), holm_level=round(lev, 6),
                exact_joint_cells=njx, robust_joint_cells=njr, cells=out)

# ---------- run FULL then DE-DUP ----------
DROP = set(json.load(open(artifact("m_c1_dedup_results.json")))["drop_qids"])
ALL = sorted(byq)
POOL_FULL = [q for q in ALL]
POOL_DEDUP = [q for q in ALL if q not in DROP]
log(f"full pool={len(POOL_FULL)}  drop={sorted(DROP)}  dedup pool={len(POOL_DEDUP)}")

def headline(rep):
    c = rep["cells"]["score_sptrue"][0.10]["robust_best"]
    e = rep["cells"]["score_sptrue"][0.10]["exact_best"]
    return dict(m_c=c["m_c"], s_c=c["s_c"], psi_hat=c["psi_hat"],
                hb_lcb=c["psi_lcb_hb"], exact_lcb=(e["psi_lcb"] if e else None),
                robust_cells=rep["robust_joint_cells"], exact_cells=rep["exact_joint_cells"])

# Fixed published seed-7 folds on the FULL pool; de-dup = remove the dropped twin from
# whichever fold it lands in (isolates the MARGINAL effect of the drop, no reshuffle).
FULL_FOLDS = split(POOL_FULL)
where = [name for name, S in zip(("D_sel","D_cal","D_test"), FULL_FOLDS) if DROP & S]
log(f"dropped twin {sorted(DROP)} lands in fold(s): {where or ['(not in any fold?)']}")
DEDUP_FIXED_FOLDS = tuple(S - DROP for S in FULL_FOLDS)

report = {"run": "m6 de-dup re-run (drop 1 shared-abstract twin; verbatim m6 HB leaf)",
          "drop_qids": sorted(DROP), "drop_lands_in_fold": where}
for tag, pool in [("full", POOL_FULL), ("dedup_resplit", POOL_DEDUP)]:
    report[tag] = {f"pi={pi}": certify_at_prevalence(pool, pi, f"{tag} pi={pi}") for pi in PIS}
# fixed-split de-dup: same published folds on the full pool, just minus the dropped twin
report["dedup_fixed"] = {f"pi={pi}": certify_at_prevalence(POOL_FULL, pi, f"dedup_fixed pi={pi}",
                                                           folds=DEDUP_FIXED_FOLDS) for pi in PIS}

hf = headline(report["full"][f"pi={PI_PRIMARY}"])
hd = headline(report["dedup_resplit"][f"pi={PI_PRIMARY}"])
hx = headline(report["dedup_fixed"][f"pi={PI_PRIMARY}"])
report["headline_compare"] = {
    "full": hf, "dedup_resplit": hd, "dedup_fixed": hx,
    "fixed_hb_lcb_delta": round(hx["hb_lcb"] - hf["hb_lcb"], 5),
    "fixed_exact_lcb_delta": (round(hx["exact_lcb"] - hf["exact_lcb"], 5)
                              if hf["exact_lcb"] and hx["exact_lcb"] else None),
    "resplit_hb_lcb_delta": round(hd["hb_lcb"] - hf["hb_lcb"], 5)}
report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open(artifact("m6_dedup_rerun_results.json"), "w"), indent=2, default=str)

print("\n" + "="*92)
print("m6 DE-DUP RE-RUN — headline P(True) pi=0.10 alpha=0.10  (full vs C1-de-duplicated pool)")
print("="*92)
print(f"  GUARD (full pool must reproduce published m6): "
      f"m_c={hf['m_c']} s_c={hf['s_c']} psi_hat={hf['psi_hat']} HB-LCB={hf['hb_lcb']} exact-LCB={hf['exact_lcb']}")
print(f"  expected published: m_c=278 s_c=120 psi_hat=0.4317 HB-LCB=0.320 exact-LCB=0.327")
print(f"  DE-DUP fixed-split (drop {sorted(DROP)} from published folds, lands in {report['drop_lands_in_fold']}):")
print(f"     m_c={hx['m_c']} s_c={hx['s_c']} psi_hat={hx['psi_hat']} HB-LCB={hx['hb_lcb']} exact-LCB={hx['exact_lcb']}")
print(f"     Δ HB-LCB = {report['headline_compare']['fixed_hb_lcb_delta']:+.5f}   "
      f"Δ exact-LCB = {report['headline_compare']['fixed_exact_lcb_delta']}")
print(f"  DE-DUP re-split (reshuffled folds, for reference): "
      f"m_c={hd['m_c']} HB-LCB={hd['hb_lcb']} exact-LCB={hd['exact_lcb']} (Δ HB={report['headline_compare']['resplit_hb_lcb_delta']:+.5f})")
print(f"  robust joint cells: full={hf['robust_cells']}/9  dedup-fixed={hx['robust_cells']}/9  dedup-resplit={hd['robust_cells']}/9")
print("="*92)
