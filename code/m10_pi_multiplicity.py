#!/usr/bin/env python
"""
M10 — PREVALENCE-MULTIPLICITY-BUDGETED certification (reviewer-protection for the low-π headline).

The released headline (m3_combined.py) pre-specifies pi=0.10 as primary and reports the
pi-sweep {0.05,0.10,0.20,0.50} as DESCRIPTIVE sensitivity, with Holm applied WITHIN each
fixed pi over (family x alpha x tau). A strict reviewer can object: "you scanned 4 prevalences
and the effect vanishes at 0.50 — did you protect the pi search inside the multiplicity budget?"

This script answers that by putting pi INSIDE the candidate family and re-Holm-correcting.
Key statistical fact (verified against certify.py): the psi-failure tail lower_p_psi(s_c,m_c,alpha)
and the min-mass tail minmass_p(m_c,ncal,gamma) depend ONLY on the split (rT,rH,tau) — they are
pi-INDEPENDENT. pi enters ONLY through the global-rate gate rho(tau,pi)=P(H=1|accept) computed on
the one-answer-per-source pi-mixture stream. So we report TWO clearly-labeled chargings:

  (BASE)  per-pi Holm — reproduces the released paper numbers (integrity check; must match 8/6/2/0).
  (A) pi-as-search-dimension (PRINCIPLED): enlarge ONLY the rho (delta_global) family to
      (f x alpha x tau x pi); psi & mass families unchanged (their hypotheses are pi-independent,
      so duplicating them across pi would test the SAME hypothesis 4x — an incorrect over-charge).
  (B) pi-in-every-family (CONSERVATIVE upper bound): enlarge ALL families to (... x pi). This
      deliberately over-charges psi/mass (duplicates identical p-values 4x) and is the most
      conservative reading of "control FWER over everything you scanned, including pi."

If the headline P(True)-8B pi=0.10 alpha=0.10 cell survives BOTH A and B, the prevalence-fishing
objection is dead. Same data / split seed 7 / prevalence seed 2024 / 28-pt grid / gamma=0.05 as M3.
CPU only.
"""
import os, sys, json, csv, time
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

LAB = "data/scores/scores_strong.csv"; ART = "data/scores/scores_strong_artificial.csv"
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT
PIS = [0.50, 0.20, 0.10, 0.05]; PI_PRIMARY = 0.10
SPLIT_SEED = 7; PREV_SEED = 2024; NGRID = 28

def load(path, src):
    rows = list(csv.DictReader(open(path)))
    for r in rows: r.setdefault("source", src)
    return rows
rows = load(LAB, "pqa_labeled") + (load(ART, "pqa_artificial") if os.path.exists(ART) else [])
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
byq = {}
for i in range(len(rows)):
    q, h = int(qid[i]), int(H[i])
    byq.setdefault(q, {})[h] = i
for q, d in byq.items(): assert set(d) == {0, 1}, f"qid {q} incomplete"
QIDS = sorted(byq)
log(f"combined twins={len(QIDS)}")

def split(qids, seed=SPLIT_SEED, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids); perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

# ----- build all per-(f,alpha,tau,pi) p-values once (pi-independent pieces computed once) -----
def build_pools(qids_pool):
    D_sel, D_cal, D_test = split(qids_pool)
    ORI = {}
    sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
    def tw(f, qs):
        s = ORI[f]; q = [x for x in qids_pool if x in qs]
        return np.array([s[byq[x][0]] for x in q]), np.array([s[byq[x][1]] for x in q])
    GRID = {}; META = {}
    rho_p = {}; low_p = {}; mass_p = {}        # keyed: rho[(f,a,ti,pi)], low[(f,a,ti)], mass[(f,ti)]
    for f in FAMS:
        rTs, rHs = tw(f, D_sel); GRID[f] = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=NGRID)
        rTc, rHc = tw(f, D_cal); ncal = len(rTc)
        u = np.random.default_rng(PREV_SEED).random(ncal)      # common random numbers across pi (mirror M3)
        META[f] = []
        for ti, tau in enumerate(GRID[f]):
            m_c, sfail = C.psi_counts(rTc, rHc, tau)
            mass_p[(f, ti)] = C.minmass_p(m_c, ncal, GAMMA)
            rec = dict(tau=round(float(tau), 6), m_c=m_c, s_c=sfail,
                       psi_hat=(round(sfail/m_c, 4) if m_c > 0 else None), denom_mass=round(m_c/ncal, 4), rho={})
            for a in ALPHAS:
                low_p[(f, a, ti)] = C.lower_p_psi(sfail, m_c, a)
            for pi in PIS:
                takeH = u < pi; s_stream = np.where(takeH, rHc, rTc); h_stream = takeH.astype(int)
                acc = s_stream <= tau; Kc = int(acc.sum()); Sc = int(h_stream[acc].sum())
                rec["rho"][pi] = dict(rho_K=Kc, rho_S=Sc, rho_hat=(round(Sc/Kc, 4) if Kc > 0 else None))
                for a in ALPHAS:
                    rho_p[(f, a, ti, pi)] = float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0
            META[f].append(rec)
    return GRID, META, rho_p, low_p, mass_p, len(qids_pool)

def holm_ok(pool_keys, pool_pvals, delta):
    rej = C.holm_reject(pool_pvals, delta)
    return {k: bool(rej[i]) for i, k in enumerate(pool_keys)}

def joint_count(GRID, rho_ok, low_ok, mass_ok, pi, enlarged_psi_mass):
    """count + detail of jointly-certified (f,alpha) cells at prevalence pi.
    enlarged_psi_mass=True ⇒ low/mass keyed with pi (variant B); else pi-independent keys (A/BASE)."""
    cells = {}; n = 0
    for f in FAMS:
        cells[f] = {}
        for a in ALPHAS:
            hit = []
            for ti in range(len(GRID[f])):
                lk = (f, a, ti, pi) if enlarged_psi_mass else (f, a, ti)
                mk = (f, ti, pi) if enlarged_psi_mass else (f, ti)
                if rho_ok.get((f, a, ti, pi)) and low_ok.get(lk) and mass_ok.get(mk):
                    hit.append(ti)
            cells[f][a] = hit
            if hit: n += 1
    return n, cells

GRID, META, rho_p, low_p, mass_p, npool = build_pools(QIDS)

# ---- BASE: per-pi Holm (reproduce released paper) ----
def per_pi_base():
    out = {}
    for pi in PIS:
        rk = [(f, a, ti, pi) for f in FAMS for a in ALPHAS for ti in range(len(GRID[f]))]
        lk = [(f, a, ti) for f in FAMS for a in ALPHAS for ti in range(len(GRID[f]))]
        mk = [(f, ti) for f in FAMS for ti in range(len(GRID[f]))]
        rho_ok = holm_ok(rk, [rho_p[k] for k in rk], C.DELTA_GLOBAL)
        low_ok = holm_ok(lk, [low_p[k] for k in lk], C.DELTA_LOWER)
        mass_ok = holm_ok(mk, [mass_p[k] for k in mk], C.DELTA_MASS)
        n, cells = joint_count(GRID, rho_ok, low_ok, mass_ok, pi, False)
        out[pi] = (n, cells)
    return out

# ---- A: enlarge ONLY rho to (f,a,ti,pi); psi & mass pi-independent (single family) ----
def variant_A():
    rk = [(f, a, ti, pi) for f in FAMS for a in ALPHAS for ti in range(len(GRID[f])) for pi in PIS]
    lk = [(f, a, ti) for f in FAMS for a in ALPHAS for ti in range(len(GRID[f]))]
    mk = [(f, ti) for f in FAMS for ti in range(len(GRID[f]))]
    rho_ok = holm_ok(rk, [rho_p[k] for k in rk], C.DELTA_GLOBAL)
    low_ok = holm_ok(lk, [low_p[k] for k in lk], C.DELTA_LOWER)
    mass_ok = holm_ok(mk, [mass_p[k] for k in mk], C.DELTA_MASS)
    return {pi: joint_count(GRID, rho_ok, low_ok, mass_ok, pi, False) for pi in PIS}, rho_ok, low_ok, mass_ok

# ---- B: enlarge EVERY family to include pi (conservative over-charge) ----
def variant_B():
    rk = [(f, a, ti, pi) for f in FAMS for a in ALPHAS for ti in range(len(GRID[f])) for pi in PIS]
    lk = [(f, a, ti, pi) for f in FAMS for a in ALPHAS for ti in range(len(GRID[f])) for pi in PIS]
    mk = [(f, ti, pi) for f in FAMS for ti in range(len(GRID[f])) for pi in PIS]
    rho_ok = holm_ok(rk, [rho_p[k] for k in rk], C.DELTA_GLOBAL)
    low_ok = holm_ok(lk, [low_p[(f, a, ti)] for (f, a, ti, pi) in lk], C.DELTA_LOWER)
    mass_ok = holm_ok(mk, [mass_p[(f, ti)] for (f, ti, pi) in mk], C.DELTA_MASS)
    return {pi: joint_count(GRID, rho_ok, low_ok, mass_ok, pi, True) for pi in PIS}, rho_ok, low_ok, mass_ok

base = per_pi_base()
A, A_rho, A_low, A_mass = variant_A()
B, B_rho, B_low, B_mass = variant_B()

def cell_label(f, a): return f"{f.replace('score_s','')}/a{a}"
def headline_status(rho_ok, low_ok, mass_ok, enlarged):
    f, a, pi = "score_sptrue", 0.10, 0.10
    ti_hd = next((ti for ti, r in enumerate(META[f]) if abs(r["tau"]-0.222824) < 1e-4), None)
    lk = (f, a, ti_hd, pi) if enlarged else (f, a, ti_hd)
    mk = (f, ti_hd, pi) if enlarged else (f, ti_hd)
    return dict(ti=ti_hd, tau=META[f][ti_hd]["tau"], m_c=META[f][ti_hd]["m_c"], s_c=META[f][ti_hd]["s_c"],
                rho_hat=META[f][ti_hd]["rho"][pi]["rho_hat"], psi_hat=META[f][ti_hd]["psi_hat"],
                rho_p=rho_p[(f, a, ti_hd, pi)], low_p=low_p[(f, a, ti_hd)],
                rho_ok=bool(rho_ok.get((f, a, ti_hd, pi))), low_ok=bool(low_ok.get(lk)), mass_ok=bool(mass_ok.get(mk)))

report = dict(run="M10 prevalence-multiplicity-budgeted certification", split_seed=SPLIT_SEED,
              prev_seed=PREV_SEED, n_grid=NGRID, gamma=GAMMA, pis=PIS, alphas=ALPHAS,
              delta=dict(total=C.DELTA_TOTAL, global_=C.DELTA_GLOBAL, mass=C.DELTA_MASS, lower=C.DELTA_LOWER),
              family_sizes=dict(
                  base_rho_per_pi=sum(len(GRID[f]) for f in FAMS)*len(ALPHAS),
                  A_rho=sum(len(GRID[f]) for f in FAMS)*len(ALPHAS)*len(PIS),
                  A_low=sum(len(GRID[f]) for f in FAMS)*len(ALPHAS),
                  B_low=sum(len(GRID[f]) for f in FAMS)*len(ALPHAS)*len(PIS)))
report["sweep_counts"] = dict(
    BASE_per_pi={f"pi={pi}": base[pi][0] for pi in PIS},
    A_pi_in_rho_only={f"pi={pi}": A[pi][0] for pi in PIS},
    B_pi_in_all_families={f"pi={pi}": B[pi][0] for pi in PIS})
report["surviving_cells"] = {}
for tag, V in [("BASE", base), ("A", A), ("B", B)]:
    report["surviving_cells"][tag] = {f"pi={pi}": [cell_label(f, a) for f in FAMS for a in ALPHAS if V[pi][1][f][a]]
                                      for pi in PIS}
report["headline_cell_ptrue_pi0.10_a0.10"] = dict(
    BASE=headline_status({k: True for k in rho_p}, {k: True for k in low_p}, {k: True for k in mass_p}, False),  # raw p's
    A=headline_status(A_rho, A_low, A_mass, False),
    B=headline_status(B_rho, B_low, B_mass, True))
report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m10_pi_multiplicity_results.json", "w"), indent=2, default=str)

print("\n" + "="*92)
print("M10 — PREVALENCE-MULTIPLICITY-BUDGETED CERTIFICATION (pooled 10k)")
print("="*92)
print(f"{'pi':>6} | {'BASE (per-pi, released)':>24} | {'A (pi in rho only)':>20} | {'B (pi in ALL families)':>22}")
print("-"*92)
for pi in PIS:
    print(f"{pi:>6} | {str(base[pi][0])+'/9':>24} | {str(A[pi][0])+'/9':>20} | {str(B[pi][0])+'/9':>22}")
print("-"*92)
print(f"Holm family sizes: rho per-pi={report['family_sizes']['base_rho_per_pi']} -> A/B rho={report['family_sizes']['A_rho']}"
      f" ({len(PIS)}x); psi low base={report['family_sizes']['A_low']} -> B={report['family_sizes']['B_low']}")
print(f"\nReleased paper baseline sweep should be 8/6/2/0 at pi=0.05/0.10/0.20/0.50 — "
      f"GOT {base[0.05][0]}/{base[0.10][0]}/{base[0.20][0]}/{base[0.50][0]}  "
      f"{'[MATCH ✓]' if (base[0.05][0],base[0.10][0],base[0.20][0],base[0.50][0])==(8,6,2,0) else '[MISMATCH ✗]'}")
hd = report["headline_cell_ptrue_pi0.10_a0.10"]
print(f"\nHEADLINE cell P(True)-8B pi=0.10 alpha=0.10 (tau={hd['A']['tau']}, m_c={hd['A']['m_c']}, "
      f"rho_hat={hd['A']['rho_hat']}, psi_hat={hd['A']['psi_hat']}):")
print(f"   rho p-value = {hd['A']['rho_p']:.2e} (vs Holm), psi lower-tail p = {hd['A']['low_p']:.2e}")
print(f"   survives A (pi in rho): rho_ok={hd['A']['rho_ok']} low_ok={hd['A']['low_ok']} mass_ok={hd['A']['mass_ok']} "
      f"=> {'CERTIFIED ✓' if all([hd['A']['rho_ok'],hd['A']['low_ok'],hd['A']['mass_ok']]) else 'NOT ✗'}")
print(f"   survives B (pi in all):  rho_ok={hd['B']['rho_ok']} low_ok={hd['B']['low_ok']} mass_ok={hd['B']['mass_ok']} "
      f"=> {'CERTIFIED ✓' if all([hd['B']['rho_ok'],hd['B']['low_ok'],hd['B']['mass_ok']]) else 'NOT ✗'}")
print("\n  wrote results/m10_pi_multiplicity_results.json")
print("="*92)
