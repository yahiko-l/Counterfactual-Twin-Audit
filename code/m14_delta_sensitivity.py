#!/usr/bin/env python
"""
M14 — total-budget (delta_total) sensitivity + displayed-cell coverage provenance.

Answers the pre-submission reviewer ask "is delta_total=0.10 doing load-bearing work?"
and provides the provenance for the coverage column of the headline joint table.

Reruns the IDENTICAL m3_combined.py certification path (same released CSVs, same split
seed 7, same prevalence-draw seed 2024, same D_sel-frozen 28-point quantile grids, same
four-family structure) at TWO total budgets:

    delta_total = 0.10  (released):  (d_glob, d_mass, d_psiup, d_low) = (.025,.025,.01,.04)
    delta_total = 0.05  (halved)  :  (.0125,.0125,.005,.02)   — every family halved

and certifies BOTH psi_g failure leaves at each budget:

    exact leaf  : per-candidate exact conditional binomial, Holm step-down within d_low
                  (as m3_combined.py); LCB displayed at the union floor d_low/n_low.
    robust leaf : Hoeffding-Bentkus on the selected-set mean, certified at the UNION
                  FLOOR p_hb <= d_low/n_low (the paper's operative criterion for the
                  D_cal-random target; Holm is a power-only refinement and is NOT used
                  for the robust decision here).

Every displayed (best) cell also records the gate coverage on the induced one-answer
stream, coverage = rho_K / n_cal — the provenance for the coverage column of the joint
table (released-budget branch).

Guards: the delta_total=0.10 branch must reproduce the released
m3_combined_results.json cells exactly (tau, m_c, s_c, exact psi_lcb, joint counts) and
the released m6_robust_leaf_results.json robust_best HB LCBs at pi=0.10; the script
asserts this before writing the report.

Also emits the labeled-only (1k expert subset) robust-HB companions at the released
budget for the subset-robustness table.

CPU only; consumes the released scores_strong{,_artificial}.csv. No GPU, no network.
"""
import os, sys, json, csv, time, math
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

E = math.e
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

# ---------- robust HB leaf (identical formulas to m6_robust_leaf.py) ----------
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

# ---------- data load (identical to m3_combined.py) ----------
LAB = "data/scores/scores_strong.csv"; ART = "data/scores/scores_strong_artificial.csv"
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT
PI_PRIMARY = 0.10; PIS = [0.50, 0.20, 0.10, 0.05]
BUDGETS = {"0.10": dict(glob=0.025, mass=0.025, psiup=0.01, low=0.04),
           "0.05": dict(glob=0.0125, mass=0.0125, psiup=0.005, low=0.02)}

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

def certify_at_prevalence(qids_pool, pi, label, d):
    """m3_combined.py's certification path, parameterized by the family budgets `d`;
    adds the robust HB leaf at the union floor and per-cell coverage."""
    D_sel, D_cal, D_test = split(qids_pool)
    sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    ORI = {}
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
    def tw(f, qs):
        s = ORI[f]; q = [x for x in qids_pool if x in qs]
        return np.array([s[byq[x][0]] for x in q]), np.array([s[byq[x][1]] for x in q]), q
    rho_pool, low_pool, hb_pool, mass_pool = [], [], [], []
    META = {}
    ncal = None
    for f in FAMS:
        rTs, rHs, _ = tw(f, D_sel); grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=28)
        rTc, rHc, _ = tw(f, D_cal); ncal = len(rTc)
        rng = np.random.default_rng(2024)
        takeH = rng.random(ncal) < pi
        s_stream = np.where(takeH, rHc, rTc); h_stream = takeH.astype(int)
        META[f] = []
        for ti, tau in enumerate(grid):
            acc = s_stream <= tau; Kc = int(acc.sum()); Sc = int(h_stream[acc].sum())
            m_c, sfail = C.psi_counts(rTc, rHc, tau)
            mass_pool.append((f, ti, C.minmass_p(m_c, ncal, GAMMA)))
            META[f].append(dict(tau=round(float(tau), 6), rho_K=Kc, rho_S=Sc,
                                rho_hat=(round(Sc/Kc, 4) if Kc > 0 else None),
                                coverage=(round(Kc/ncal, 4) if ncal > 0 else None),
                                m_c=m_c, s_c=sfail,
                                psi_hat=(round(sfail/m_c, 4) if m_c > 0 else None),
                                denom_mass=round(m_c/ncal, 4)))
            for a in ALPHAS:
                rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
                low_pool.append((f, a, ti, C.lower_p_psi(sfail, m_c, a)))
                hb_pool.append((f, a, ti, hb_lower_p(sfail, m_c, a)))
    rho_rej = C.holm_reject([x[3] for x in rho_pool], d["glob"])
    low_rej = C.holm_reject([x[3] for x in low_pool], d["low"])
    mass_rej = C.holm_reject([x[2] for x in mass_pool], d["mass"])
    nlow = len(low_pool); lev = d["low"] / nlow                      # union floor level
    rho_ok = {(x[0], x[1], x[2]): bool(rho_rej[i]) for i, x in enumerate(rho_pool)}
    low_ok = {(x[0], x[1], x[2]): bool(low_rej[i]) for i, x in enumerate(low_pool)}
    hb_ok = {(x[0], x[1], x[2]): (x[3] <= lev) for x in hb_pool}     # UNION-FLOOR decision
    mass_ok = {(x[0], x[1]): bool(mass_rej[i]) for i, x in enumerate(mass_pool)}
    cells = {}
    for f in FAMS:
        cells[f] = {}
        for a in ALPHAS:
            jex, jhb = [], []
            for ti in range(len(META[f])):
                base = rho_ok[(f, a, ti)] and mass_ok[(f, ti)]
                if base and low_ok[(f, a, ti)]:
                    r = dict(META[f][ti]); r["psi_lcb"] = round(C.cp_lower_psi(r["s_c"], r["m_c"], lev), 4)
                    jex.append(r)
                if base and hb_ok[(f, a, ti)]:
                    r = dict(META[f][ti])
                    r["psi_lcb_hb"] = round(hb_lcb(r["s_c"], r["m_c"], lev), 4)
                    r["psi_lcb_exact"] = round(C.cp_lower_psi(r["s_c"], r["m_c"], lev), 4)
                    jhb.append(r)
            jex.sort(key=lambda r: -(r["psi_lcb"] or 0)); jhb.sort(key=lambda r: -(r["psi_lcb_hb"] or 0))
            cells[f][a] = dict(exact_joint=len(jex) > 0, n_joint_exact=len(jex),
                               exact_best=(jex[0] if jex else None),
                               robust_joint=len(jhb) > 0, n_joint_robust=len(jhb),
                               robust_best=(jhb[0] if jhb else None))
    njx = sum(1 for f in FAMS for a in ALPHAS if cells[f][a]["exact_joint"])
    njr = sum(1 for f in FAMS for a in ALPHAS if cells[f][a]["robust_joint"])
    return dict(label=label, pi=pi, n_twins=len(qids_pool), n_cal=ncal, n_low=nlow,
                union_floor_level=lev, exact_joint_cells=njx, robust_joint_cells=njr, cells=cells)

POOLED = QIDS.tolist(); LABELED = [q for q in QIDS if q < 1000000]

report = {"run": "M14 delta_total sensitivity (0.10 released vs 0.05 halved) + displayed-cell coverage",
          "budgets": BUDGETS, "alphas": ALPHAS, "gamma": GAMMA, "pi_primary": PI_PRIMARY,
          "robust_decision": "union floor p_hb <= d_low/n_low (paper's operative criterion; Holm power-only)",
          "coverage_def": "coverage = rho_K / n_cal on the induced one-answer-per-source stream at tau"}
for dl, d in BUDGETS.items():
    log(f"certifying pooled sweep at delta_total={dl} ...")
    report[f"pooled_delta={dl}"] = {f"pi={pi}": certify_at_prevalence(POOLED, pi, f"pooled pi={pi} dtot={dl}", d) for pi in PIS}
    report[f"labeled_delta={dl}"] = {f"pi={PI_PRIMARY}": certify_at_prevalence(LABELED, PI_PRIMARY, f"labeled pi={PI_PRIMARY} dtot={dl}", d)}

# ---------------- guards: delta=0.10 branch must reproduce the released results ----------------
log("guard: reproducing released m3_combined_results.json / m6_robust_leaf_results.json ...")
rel3 = json.load(open("results/m3_combined_results.json"))
rel6 = json.load(open("results/m6_robust_leaf_results.json"))
akey = {0.05: "0.05", 0.10: "0.1", 0.20: "0.2"}
guards = []
for pi in PIS:
    mine = report["pooled_delta=0.10"][f"pi={pi}"]; rel = rel3["pooled"][f"pi={pi}"]
    assert mine["exact_joint_cells"] == rel["joint_cells"], (pi, mine["exact_joint_cells"], rel["joint_cells"])
    guards.append(f"pi={pi}: exact joint {mine['exact_joint_cells']} == released {rel['joint_cells']}")
    for f in FAMS:
        for a in ALPHAS:
            mb = mine["cells"][f][a]["exact_best"]; rb = rel["cells"][f][akey[a]]["best"]
            assert (mb is None) == (rb is None), (pi, f, a)
            if mb:
                for k in ("tau", "m_c", "s_c", "psi_lcb"):
                    assert abs(float(mb[k]) - float(rb[k])) < 1e-9, (pi, f, a, k, mb[k], rb[k])
                assert mb["rho_K"] == rb["rho_K"], (pi, f, a, mb["rho_K"], rb["rho_K"])
mine = report["labeled_delta=0.10"][f"pi={PI_PRIMARY}"]; rel = rel3["labeled_only"][f"pi={PI_PRIMARY}"]
assert mine["exact_joint_cells"] == rel["joint_cells"]
guards.append(f"labeled pi=0.1: exact joint {mine['exact_joint_cells']} == released {rel['joint_cells']}")
r6 = rel6["pooled"][f"pi={PI_PRIMARY}"]["cells"]
mineP = report["pooled_delta=0.10"][f"pi={PI_PRIMARY}"]["cells"]
for f in FAMS:
    for a in ALPHAS:
        rb = r6[f][akey[a]].get("robust_best"); mb = mineP[f][a]["robust_best"]
        rj = bool(r6[f][akey[a]].get("robust_joint")); mj = mineP[f][a]["robust_joint"]
        assert mj == rj, (f, a, "robust joint set differs from released m6", mj, rj)
        if mb and rb:
            assert abs(float(mb["psi_lcb_hb"]) - float(rb["psi_lcb_hb"])) < 1e-9, (f, a, mb["psi_lcb_hb"], rb["psi_lcb_hb"])
guards.append("pi=0.1 robust(HB) cell set + best LCBs == released m6 (union floor)")
report["reproduction_guards"] = guards

# ---------------- verdict: what changes at delta_total=0.05 ----------------
def cellset(branch, pi, leaf):
    r = report[f"pooled_delta={branch}"][f"pi={pi}"]
    return sorted((f, a) for f in FAMS for a in ALPHAS if r["cells"][f][a][f"{leaf}_joint"])
verd = {}
for pi in PIS:
    s10e, s05e = cellset("0.10", pi, "exact"), cellset("0.05", pi, "exact")
    s10r, s05r = cellset("0.10", pi, "robust"), cellset("0.05", pi, "robust")
    verd[f"pi={pi}"] = {
        "exact_cells_delta0.10": len(s10e), "exact_cells_delta0.05": len(s05e),
        "exact_dropped": [f"{f}@a={a}" for (f, a) in s10e if (f, a) not in s05e],
        "exact_added": [f"{f}@a={a}" for (f, a) in s05e if (f, a) not in s10e],
        "robust_cells_delta0.10": len(s10r), "robust_cells_delta0.05": len(s05r),
        "robust_dropped": [f"{f}@a={a}" for (f, a) in s10r if (f, a) not in s05r],
        "robust_added": [f"{f}@a={a}" for (f, a) in s05r if (f, a) not in s10r]}
hl10 = report["pooled_delta=0.10"][f"pi={PI_PRIMARY}"]["cells"]["score_sptrue"][0.10]
hl05 = report["pooled_delta=0.05"][f"pi={PI_PRIMARY}"]["cells"]["score_sptrue"][0.10]
verd["headline_ptrue_a0.10"] = {
    "delta0.10": {"exact_lcb": hl10["exact_best"]["psi_lcb"], "robust_lcb_hb": hl10["robust_best"]["psi_lcb_hb"],
                  "coverage": hl10["exact_best"]["coverage"]},
    "delta0.05": {"exact_lcb": hl05["exact_best"]["psi_lcb"], "robust_lcb_hb": hl05["robust_best"]["psi_lcb_hb"],
                  "coverage": hl05["exact_best"]["coverage"]}}
report["verdict"] = verd
report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m14_delta_sensitivity_results.json", "w"), indent=2, default=str)

print("\n" + "="*96)
print("M14 DELTA_TOTAL SENSITIVITY — released 0.10 vs halved 0.05 (both leaves; robust = union floor)")
print("="*96)
for g in guards: print("  GUARD OK:", g)
print()
for pi in PIS:
    v = verd[f"pi={pi}"]
    print(f"pi={pi}:  exact {v['exact_cells_delta0.10']}->{v['exact_cells_delta0.05']}"
          f" (dropped {v['exact_dropped'] or 'none'}; added {v['exact_added'] or 'none'})"
          f"   robust {v['robust_cells_delta0.10']}->{v['robust_cells_delta0.05']}"
          f" (dropped {v['robust_dropped'] or 'none'}; added {v['robust_added'] or 'none'})")
h = verd["headline_ptrue_a0.10"]
print(f"\nheadline P(True) a=0.10: exact LCB {h['delta0.10']['exact_lcb']} -> {h['delta0.05']['exact_lcb']}; "
      f"robust HB LCB {h['delta0.10']['robust_lcb_hb']} -> {h['delta0.05']['robust_lcb_hb']}; coverage {h['delta0.10']['coverage']}")
print("\ncoverage at the released displayed cells (pi=0.10, delta=0.10):")
for f in FAMS:
    for a in ALPHAS:
        b = report["pooled_delta=0.10"][f"pi={PI_PRIMARY}"]["cells"][f][a]["exact_best"]
        if b: print(f"  [{f} a={a}] tau={b['tau']} coverage={b['coverage']} (rho_K={b['rho_K']}/{report['pooled_delta=0.10'][f'pi={PI_PRIMARY}']['n_cal']})")
print("\nlabeled-only (1k) pi=0.10 released budget: robust HB companions for the subset table:")
for f in FAMS:
    for a in ALPHAS:
        c = report["labeled_delta=0.10"][f"pi={PI_PRIMARY}"]["cells"][f][a]
        if c["exact_joint"] or c["robust_joint"]:
            eb_ = c["exact_best"]; rb = c["robust_best"]
            print(f"  [{f} a={a}] exactLCB={eb_ and eb_['psi_lcb']} m_c={(eb_ or rb)['m_c']} | "
                  f"robust={'HB-LCB=%s' % rb['psi_lcb_hb'] if rb else 'NOT certified at union floor'}")
print("\n  wrote results/m14_delta_sensitivity_results.json")
print("="*96)
