#!/usr/bin/env python
"""
M2 + M3 certification on STRONG-judge scores (CPU; consumes data/scores/scores_strong.csv).
Maps EXPERIMENT_TRACKER R020/R021 (M2) + R030–R033 (M3) + R042 (negative controls).

  B1 (R032) HEADLINE joint cell: at a τ where the global gate is CLUSTER-VALID certified safe
       (ρ(τ)≤α via one-answer-per-source, δ_global), is ψ_g>α ALSO certified (δ_lower exact
       conditional-binomial, pair=unit) on a min-mass denominator (δ_mass)?  All within δ_total.
  B2 (R033): per-family global risk-AUROC vs paired-acc; η=P̂(r_H<r_T); stratified ψ_g.
       K4 gate: strong-judge paired-acc < 0.5 with cluster-bootstrap CI excluding 0.5?
  M2 baseline (R020): Conformal-Abstention-style MARGINAL gate (control marginal accepted
       hallucination rate ≤ α) → its realized conditional ψ_g (the "marginal mirage").
  Negative controls (R042): TwinFAR=P(accept A_H|accept A_T) trivial bound; Mondrian ρ_g by difficulty.

CRITICAL: H comes only from MedHallu; every score (judge/selfcheck/ptrue/NLI) is a SCORE UNDER
AUDIT, never a label. Orientation locked on D_sel only (D_cal-independent).
"""
import os, sys, json, csv, time
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
CSV = os.environ.get("CSV", "data/scores/scores_strong.csv")
MAIN = ["score_sptrue", "score_selfcheck", "score_sjudge"]   # weak-UQ / answer-conditioned / strong-judge
ALL_F = ["score_sjudge", "score_sptrue", "score_selfcheck", "score_sNLI"]
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT
DISP = {"score_sptrue": "P(True)-7B", "score_selfcheck": "SelfCheck-NLI", "score_sjudge": "strong-judge-7B", "score_sNLI": "ref-NLI"}

rows = list(csv.DictReader(open(CSV)))
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
diff = np.array([r["difficulty"] for r in rows]); cat = np.array([r.get("category", "") for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in ALL_F}
byq = {}
for i in range(len(rows)):
    q, h = int(qid[i]), int(H[i])
    assert (q not in byq) or (h not in byq[q]), f"dup (qid={q},H={h})"
    byq.setdefault(q, {})[h] = i
assert len(rows) == 2*len(byq), f"{len(rows)} != 2x{len(byq)}"
for q, d in byq.items(): assert set(d) == {0, 1}, f"qid {q} incomplete: {set(d)}"
QIDS = np.array(sorted(byq))

def group_split(fracs=(0.40, 0.40, 0.20), seed=7):
    perm = np.random.default_rng(seed).permutation(len(QIDS)); n = len(QIDS)
    a = int(n*fracs[0]); b = a + int(n*fracs[1])
    return set(QIDS[perm[:a]].tolist()), set(QIDS[perm[a:b]].tolist()), set(QIDS[perm[b:]].tolist())
D_sel, D_cal, D_test = group_split()
sel_idx = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
ORI = {}; FLIP = {}
for f in ALL_F:
    _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f]); FLIP[f] = bool(fl)

def twins(f, qset, arr=None):
    s = ORI[f] if arr is None else arr; qs = [q for q in QIDS if q in qset]
    return (np.array([s[byq[q][0]] for q in qs]), np.array([s[byq[q][1]] for q in qs]), qs)

report = {"run": "M2+M3 certify (strong judge)", "model_scores": CSV, "alphas": ALPHAS, "gamma": GAMMA,
          "delta": {"total": C.DELTA_TOTAL, "global": C.DELTA_GLOBAL, "mass": C.DELTA_MASS,
                    "psiup": C.DELTA_PSIUP, "lower": C.DELTA_LOWER},
          "split": {"sel": len(D_sel), "cal": len(D_cal), "test": len(D_test)},
          "orientation": {f: {"flipped": FLIP[f], "risk_auc": round(float(roc_auc_score(H, ORI[f])), 4)} for f in ALL_F}}

# ===================== B2 / K4 — paired-acc, AUROC, η, stratified (R033) =====================
def cluster_bootstrap_pairedacc(rT, rH, B=2000, seed=1):
    """Cluster (twin) bootstrap CI for paired-acc=P(r_H>r_T). Each pair is the resampling unit."""
    n = len(rT); rng = np.random.default_rng(seed); pa = (rH > rT).astype(float) + 0.5*(rH == rT)
    stats = [pa[rng.integers(0, n, n)].mean() for _ in range(B)]
    lo, hi = np.percentile(stats, [2.5, 97.5]); return float(pa.mean()), float(lo), float(hi)
b2 = {}
for f in ALL_F:
    rT, rH, _ = twins(f, set(QIDS.tolist()))
    pa, lo, hi = cluster_bootstrap_pairedacc(rT, rH)
    b2[f] = dict(risk_auc=round(float(roc_auc_score(H, ORI[f])), 4), paired_acc=round(pa, 4),
                 paired_acc_CI=[round(lo, 4), round(hi, 4)], eta_rH_lt_rT=round(float(np.mean(rH < rT)), 4),
                 ci_excludes_0p5=bool(hi < 0.5 or lo > 0.5))
report["B2_paired"] = b2
# answer-conditioned non-degeneracy (codex): the pilot q-level SE gave identical twin scores
# (mean|Δ|=0, all-equal) ⇒ paired-acc 0.5. Show SelfCheck-NLI is answer-specific.
nd = {}
for f in ["score_selfcheck", "score_sNLI"]:
    rT, rH, _ = twins(f, set(QIDS.tolist()))
    nd[f] = dict(mean_abs_twin_delta=round(float(np.mean(np.abs(rH - rT))), 4),
                 frac_twins_equal=round(float(np.mean(rH == rT)), 4),
                 twin_corr=round(float(np.corrcoef(rT, rH)[0, 1]), 4))
report["answer_conditioned_nondegeneracy"] = nd
# K4: strong judge paired-acc < 0.5 with CI excluding 0.5 ?
sj = b2["score_sjudge"]
K4_pathology = bool(sj["paired_acc"] < 0.5 and sj["paired_acc_CI"][1] < 0.5)
report["K4"] = {"strong_judge_paired_acc": sj["paired_acc"], "CI": sj["paired_acc_CI"],
                "pathology_holds (paired<0.5, CI excludes .5)": K4_pathology,
                "verdict": ("C2 headline HOLDS: strong judge ranks hallucination safer than truth"
                            if K4_pathology else
                            "C2 pilot-specific: strong judge paired-acc≥0.5 → reframe around weaker detectors / demote C2")}

# ===================== B1 — cluster-valid ρ≤α  AND  ψ_g>α  joint cell (R032) =================
# Candidate grid frozen on D_sel. Each δ-family is Holm-controlled ONCE over its FULL pooled
# candidate set across all (family × α × τ) — pooling, not per-(family,α), so the δ_lower / δ_global
# budgets are not spent 9× (codex MAJOR). ρ is the ONE-ANSWER-PER-SOURCE marginal ρ_1perQ (cluster-
# valid: exchangeable unit = source question); ψ_g uses the full pair set; min-mass is α-free (dedup
# per (family,τ)). Union of the 4 δ-families ⇒ P(any false claim) ≤ δ_total.
from scipy.stats import binom as _b
GRID = {}; OPS = {}; META = {}
rho_pool = []   # (f, a, ti, p)
low_pool = []   # (f, a, ti, p)
mass_pool = []  # (f, ti, p)   -- α-free, one per (f,τ)
for f in MAIN:
    rT_sel, rH_sel, _ = twins(f, D_sel)
    grid = C.candidate_grid(np.r_[rT_sel, rH_sel], gamma=GAMMA, n_grid=28); GRID[f] = grid
    rT_c, rH_c, _ = twins(f, D_cal); n_cal = len(rT_c)
    pickH = np.random.default_rng(0).random(n_cal) < 0.5     # one-answer-per-source (declared seed)
    s_ops = np.where(pickH, rH_c, rT_c); h_ops = pickH.astype(int)
    META[f] = []
    for ti, tau in enumerate(grid):
        accs = s_ops <= tau; Kc = int(accs.sum()); Sc = int(h_ops[accs].sum())
        m_c, sfail = C.psi_counts(rT_c, rH_c, tau)
        mass_pool.append((f, ti, C.minmass_p(m_c, n_cal, GAMMA)))
        META[f].append(dict(tau=round(float(tau), 6), rho_K=Kc, rho_S=Sc,
                            rho_hat=(round(Sc/Kc, 4) if Kc > 0 else None), m_c=m_c, s_c=sfail,
                            psi_hat=(round(sfail/m_c, 4) if m_c > 0 else None), denom_mass=round(m_c/n_cal, 4)))
        for a in ALPHAS:
            rho_pool.append((f, a, ti, float(_b.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))   # certify ρ_1perQ≤α
            low_pool.append((f, a, ti, C.lower_p_psi(sfail, m_c, a)))                  # certify ψ_g>α
# one Holm per δ-family over the FULL pooled candidate set
rho_rej  = C.holm_reject([x[3] for x in rho_pool],  C.DELTA_GLOBAL)
low_rej   = C.holm_reject([x[3] for x in low_pool],  C.DELTA_LOWER)
mass_rej = C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS)
rho_ok  = {(x[0], x[1], x[2]): bool(rho_rej[i])  for i, x in enumerate(rho_pool)}
low_ok  = {(x[0], x[1], x[2]): bool(low_rej[i])  for i, x in enumerate(low_pool)}
mass_ok = {(x[0], x[1]):        bool(mass_rej[i]) for i, x in enumerate(mass_pool)}
n_low = len(low_pool)
joint_cells = {}; rho_only = {}
for f in MAIN:
    joint_cells[f] = {}; rho_only[f] = {}
    for a in ALPHAS:
        rho_only[f][a] = int(sum(rho_ok[(f, a, ti)] for ti in range(len(GRID[f]))))
        joint = []
        for ti in range(len(GRID[f])):
            if rho_ok[(f, a, ti)] and low_ok[(f, a, ti)] and mass_ok[(f, ti)]:
                rec = dict(META[f][ti]); rec["psi_lcb"] = round(C.cp_lower_psi(rec["s_c"], rec["m_c"], C.DELTA_LOWER/n_low), 4)
                joint.append(rec)
        best_rho = None
        for ti in range(len(GRID[f])):
            if rho_ok[(f, a, ti)] and (best_rho is None or META[f][ti]["rho_K"] > best_rho[0]):
                best_rho = (META[f][ti]["rho_K"], ti)
        joint_cells[f][a] = dict(joint_certified=len(joint) > 0, n_joint=len(joint), joint=joint[:5],
                                 best_rho_tau=(META[f][best_rho[1]] if best_rho else None),
                                 best_rho_psi_certified=(bool(low_ok[(f, a, best_rho[1])] and mass_ok[(f, best_rho[1])]) if best_rho else None))
report["B1_joint"] = joint_cells
report["B1_rho_certified_counts"] = rho_only
n_joint_cells = sum(1 for f in MAIN for a in ALPHAS if joint_cells[f][a]["joint_certified"])
any_rho = sum(rho_only[f][a] for f in MAIN for a in ALPHAS)
report["B1_summary"] = {"joint_certified_cells": n_joint_cells, "total_cells": len(MAIN)*len(ALPHAS),
                        "any_rho_certified": any_rho > 0,
                        "headline": ("C1 HEADLINE HOLDS: cluster-valid certified-safe gate (ρ≤α) STILL fails ψ_g>α"
                                     if n_joint_cells > 0 else
                                     ("ρ≤α now certifiable but no joint ψ_g>α cell" if any_rho > 0 else
                                      "ρ≤α STILL not cluster-certifiable even with strong judge → joint headline not demonstrable; report ψ_g separation as feasibility/benchmark"))}

# ===================== M2 baseline — Conformal-Abstention marginal gate (R020) ===============
# Marginal control: largest τ with marginal accepted hallucination rate ≤ α (conformal/LTT style),
# then its REALIZED conditional ψ_g (the marginal mirage). Calibrate on D_cal, evaluate on D_test.
def marginal_gate_tau(scores, Hh, alpha, delta):
    """Largest τ certifying marginal P(H=1 | accept) ≤ α by exact-binomial over a grid (Bonferroni)."""
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 60))); from scipy.stats import binom as _b
    m = len(grid); best = None
    for tau in grid:
        acc = scores <= tau; K = int(acc.sum())
        if K == 0: continue
        if float(_b.cdf(int(Hh[acc].sum()), K, alpha)) <= delta/m:
            if best is None or K > best[1]: best = (float(tau), K)
    return best[0] if best else None
m2 = {}
for f in MAIN:
    sc_cal, H_cal = ORI[f][np.array([byq[q][h] for q in QIDS if q in D_cal for h in (0,1)])], \
                    H[np.array([byq[q][h] for q in QIDS if q in D_cal for h in (0,1)])]
    rT_t, rH_t, _ = twins(f, D_test); m2[f] = {}
    for a in ALPHAS:
        th = marginal_gate_tau(sc_cal, H_cal, a, C.DELTA_GLOBAL)
        if th is None: m2[f][a] = {"marginal_tau": None}; continue
        psi, m_c, s_c = C.psi_hat(rT_t, rH_t, th); rho_t, K = C.rho_hat_at(rT_t, rH_t, th)
        m2[f][a] = {"marginal_tau": round(th, 6), "test_rho_hat": (round(rho_t,4) if not np.isnan(rho_t) else None),
                    "test_psi_hat": (round(psi,4) if not np.isnan(psi) else None),
                    "test_denom_m_c": m_c, "marginal_mirage": bool(not np.isnan(psi) and psi > a)}
report["M2_conformal_abstention_baseline"] = m2

# ===================== Negative controls (R042) — TwinFAR + Mondrian =========================
nc = {"TwinFAR": {}, "Mondrian_by_difficulty": {}}
for f in MAIN:
    rT, rH, _ = twins(f, set(QIDS.tolist())); nc["TwinFAR"][f] = {}
    grid = np.unique(np.quantile(np.r_[rT, rH], np.linspace(0.05, 0.95, 19)))
    for a in ALPHAS:
        # TwinFAR(τ)=P(accept A_H | accept A_T): trivial bound ≤ α/(1−α). Use τ at ~80% truthful accept.
        thr = np.quantile(rT, 0.80)
        accT = rT <= thr; tw = (rH[accT] <= thr).mean() if accT.sum() else float("nan")
        nc["TwinFAR"][f][a] = dict(twinFAR_at_80=round(float(tw), 4), trivial_bound=round(a/(1-a), 4))
# Mondrian: per-difficulty realized ρ_g under a global τ (group≠pair); use strong judge
f = "score_sjudge"; rT, rH, qs = twins(f, set(QIDS.tolist()))
gT = np.array([diff[byq[q][0]] for q in qs]); gH = np.array([diff[byq[q][1]] for q in qs])
thr = np.quantile(np.r_[rT, rH], 0.30)   # a strict-ish global τ
nc["Mondrian_by_difficulty"] = {str(g): {"rho_g": (round(v[0],4) if not np.isnan(v[0]) else None), "K_g": v[1]}
                                for g, v in C.mondrian_rho_by_group(rT, rH, thr, gT, gH).items()}
report["negative_controls"] = nc

report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m3_results.json", "w"), indent=2, default=str)
print("\n" + "="*84)
print(f"M3 CERTIFY (strong judge {report['orientation']})")
print(f"\nB2/K4 paired-acc (risk orientation):")
for f in ALL_F:
    v = b2[f]; print(f"  {DISP[f]:16s} AUROC={v['risk_auc']:.3f} paired-acc={v['paired_acc']:.3f} "
                     f"CI={v['paired_acc_CI']} η(rH<rT)={v['eta_rH_lt_rT']:.3f}")
print(f"\nK4: {report['K4']['verdict']}")
print(f"\nB1 headline: {report['B1_summary']['headline']}")
print(f"  joint cells certified: {n_joint_cells}/{len(MAIN)*len(ALPHAS)};  any ρ≤α cluster-certified: {report['B1_summary']['any_rho_certified']}")
print("  wrote results/m3_results.json")
print("="*84)
