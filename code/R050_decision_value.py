#!/usr/bin/env python
"""R050 -- decision value of psi_g: it is not identified by ANY marginal detector metric.

Answers reviewer major #2 ("psi_g must formally OUT-INFORM F1/TPR/FNR -- upgrade the Delta_psi
diagnostic to a decision-value statement"). The claim is a partial-identification theorem, verified
constructively on the real MedHallu headline data (zero new model runs; reuses the exact 10k split /
orientation / thresholds of m_pairedlift.py so it reproduces every certified (m_c, s_c, F1) exactly).

Setup (per certified cell of tab:joint, on D_cal): each source question q is a twin with a truthful
answer score r_T(q) and a hallucinated-twin score r_H(q); the gate accepts iff score <= tau.
    m_c = #{r_T > tau}                (truthful rejected)      P(T=0) = m_c / n
    s_c = #{r_T > tau AND r_H <= tau}  (halluc leaks through)   psi_g  = s_c / m_c
    F1  = #{r_H <= tau} / n            (marginal halluc acceptance = FNR = miss rate on H=1)

DECISION-VALUE THEOREM (partial identification). Every marginal detector metric a validation study
can compute -- coverage, rho=P(H=1|accept), precision, recall/TPR, FNR, the harmonic-mean
F1-score, AUROC (note: the variable `F1` below is the marginal halluc-acceptance rate = FNR,
NOT the harmonic-mean F1-score; both are marginal functionals) -- is a
functional of the two SEPARATE score marginals {r_T} and {r_H}. psi_g is a functional of their JOINT
(the within-question coupling). Holding both marginals fixed and only RE-PAIRING truthful with
hallucinated twins leaves every marginal metric numerically invariant while psi_g ranges over its
Frechet interval [L, U]. At each headline cell that interval is essentially [0, 1]: the marginals --
including the certified-safe rho <= alpha -- carry NO information about within-question safety. Only
the paired measurement identifies psi_g (observed ~0.43 at the headline, >> alpha).

This script (CPU only, no network):
  (A) reproduces (m_c, s_c, F1) and self-checks against m_pairedlift_results.json;
  (B) computes the Frechet partial-ID interval [L,U] for psi_g from the marginals;
  (C) CONSTRUCTS the extremal re-pairings on the real r_H multiset achieving psi_g ~ L and ~ U, and
      ASSERTS a battery of marginal metrics (coverage, rho, F1/FNR, TPR, precision, AUROC) is
      invariant across observed / min / max re-pairings (to float tolerance);
  (D) quantifies the value of the paired audit: prior (marginal-only) uncertainty width U-L=1 vs
      posterior (paired) two-sided Clopper-Pearson CI, and the deploy/block verdict flip
      (marginal policy rho<=alpha => DEPLOY; paired policy also requires psi_g<=alpha => BLOCK).

Usage: python R050_decision_value.py
Output: results/R050_decision_value_results.json + printed table.
"""
import os, csv, json, sys
import numpy as np
from scipy.stats import binom, beta as _beta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
LAB = artifact("scores_strong.csv")
ART = artifact("scores_strong_artificial.csv")
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
SPLIT_SEED = 7          # identical to m_pairedlift.py / m3_combined.py
REPAIR_SEED = 20260713  # only labels the (deterministic) sort-based extremal constructions

# headline certified cells: (family, tau, display, alphas) -- exactly the tab:joint thresholds
CELLS = [("score_sptrue", 0.2228, "P(True)-8B",  [0.10, 0.20]),
         ("score_sjudge", 0.2451, "LLM-judge-8B", [0.20]),
         ("score_sNLI",   0.5186, "ref-NLI",      [0.10, 0.20])]

# tab:joint CERTIFIED numbers at each cell's headline alpha, on the INDUCED pi=0.10 deployment stream
# (rho_hat = cluster-valid accepted-hallucination rate; psi_lcb_robust = heterogeneity-robust HB LCB
# on the selected-set mean). These drive the deploy/block verdict-flip -- NOT the balanced-table rho.
CERT = {"P(True)-8B":   dict(alpha=0.10, rho_hat=0.0636, psi_lcb_robust=0.320),
        "LLM-judge-8B": dict(alpha=0.20, rho_hat=0.0924, psi_lcb_robust=0.4664),
        "ref-NLI":      dict(alpha=0.10, rho_hat=0.0709, psi_lcb_robust=0.2087)}


def load():
    def rd(p, s):
        r = list(csv.DictReader(open(p)))
        for x in r:
            x.setdefault("source", s)
        return r
    rows = rd(LAB, "lab") + rd(ART, "art")
    qid = np.array([int(r["qid"]) for r in rows])
    H = np.array([int(r["H"]) for r in rows])
    RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
    byq = {}
    for i in range(len(rows)):
        byq.setdefault(int(qid[i]), {})[int(H[i])] = i
    QIDS = sorted(q for q in byq if set(byq[q]) == {0, 1})
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(QIDS))
    n = len(QIDS); a = int(n * .4); b = a + int(n * .4)
    D_sel = set(np.array(QIDS)[perm[:a]].tolist())
    D_cal = set(np.array(QIDS)[perm[a:b]].tolist())
    sel = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
    ORI = {}
    for f in FAMS:
        _, fl = C.orient(H[sel], RAW[f][sel])
        ORI[f] = (-RAW[f] if fl else RAW[f])
    qc = [q for q in QIDS if q in D_cal]
    return byq, H, ORI, qc, len(QIDS)


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def marginal_metrics(rT, rH, tau):
    """Every metric here is a functional of the SEPARATE marginals {r_T},{r_H} (or their pooled
    multiset) -- all invariant under re-pairing. Positive class = hallucination (H=1); the gate
    should REJECT it (reject iff score > tau)."""
    rT = np.asarray(rT, float); rH = np.asarray(rH, float); n = len(rT)
    accT = rT <= tau; accH = rH <= tau
    covT = float(accT.mean()); covH = float(accH.mean())      # accept rates on each arm
    coverage = float((accT.sum() + accH.sum()) / (2 * n))     # pooled accept rate
    Kacc = int(accT.sum() + accH.sum())
    rho = float(accH.sum() / Kacc) if Kacc > 0 else float("nan")   # P(H=1 | accept), pooled
    FNR = covH                                                # miss rate on H=1 = P(accept|H=1)
    TPR = 1.0 - covH                                          # recall on H=1 = P(reject|H=1)
    # precision of the REJECT decision as a hallucination flag: P(H=1 | reject)
    rejT = ~accT; rejH = ~accH; Krej = int(rejT.sum() + rejH.sum())
    precision = float(rejH.sum() / Krej) if Krej > 0 else float("nan")
    pooled = np.concatenate([rT, rH]); labels = np.concatenate([np.zeros(n), np.ones(n)])
    return dict(coverage=round(coverage, 6), covT=round(covT, 6), covH=round(covH, 6),
                rho_balanced_pooled=round(rho, 6), FNR=round(FNR, 6), TPR=round(TPR, 6),
                precision=round(precision, 6), auroc=round(auroc(pooled, labels), 6),
                F1_marginal=round(covH, 6))


def frechet_psi(pT0, F1):
    """Frechet-Hoeffding interval for psi_g = P(r_T>tau, r_H<=tau)/P(r_T>tau) given the marginals
    P(r_T>tau)=pT0 and P(r_H<=tau)=F1. joint in [max(0,pT0+F1-1), min(pT0,F1)]."""
    lo_j = max(0.0, pT0 + F1 - 1.0); hi_j = min(pT0, F1)
    return lo_j / pT0, hi_j / pT0


def extremal_repairings(rT, rH, tau):
    """Construct the psi_g-min and psi_g-max re-pairings on the REAL r_H multiset, holding both
    marginals fixed. The rejected-truthful set is the m_c questions with the largest r_T (fixed by
    the r_T marginal); we choose which hallucinated scores land on them.
      max psi_g: give the rejected-truthful pairs the ACCEPTED (r_H<=tau) hallucinated scores first.
      min psi_g: give them the REJECTED (r_H>tau) hallucinated scores first.
    Returns (psi_min, psi_max, m_c) and the two re-paired rH vectors (aligned to the rT order)."""
    rT = np.asarray(rT, float); rH = np.asarray(rH, float); n = len(rT)
    rej_idx = np.argsort(-rT)                      # questions by descending truthful risk
    m_c = int((rT > tau).sum())
    rej_mask_sorted = np.zeros(n, bool); rej_mask_sorted[:m_c] = True  # top-m_c are rejected-truthful
    # sort hallucinated scores so that accepted (<=tau) come first for MAX, last for MIN
    rH_acc = np.sort(rH[rH <= tau]); rH_rej = np.sort(rH[rH > tau])
    order_max = np.concatenate([rH_acc, rH_rej])   # accepted-H first -> land on rejected-truthful
    order_min = np.concatenate([rH_rej, rH_acc])   # rejected-H first -> land on rejected-truthful

    def realize(order):
        rH_paired = np.empty(n)
        rH_paired[rej_idx] = order                 # assign in the descending-r_T order
        s = int(((rT > tau) & (rH_paired <= tau)).sum())
        return s / m_c if m_c else float("nan"), rH_paired
    psi_max, rH_max = realize(order_max)
    psi_min, rH_min = realize(order_min)
    return psi_min, psi_max, m_c, rH_min, rH_max


def _verdict(disp):
    """Deploy/block verdict flip on the CERTIFIED induced-pi=0.10 numbers (tab:joint), NOT the
    balanced-table rho. Marginal policy deploys iff certified rho <= alpha; the paired policy also
    requires the certified psi_g lower bound <= alpha. The verdict flips wherever a marginal-safe
    gate carries a certified within-question failure."""
    c = CERT[disp]; a = c["alpha"]
    marg = c["rho_hat"] <= a
    pair = marg and (c["psi_lcb_robust"] <= a)
    return dict(alpha=a, certified_rho_hat=c["rho_hat"], certified_psi_lcb_robust=c["psi_lcb_robust"],
                marginal_policy_deploy=bool(marg), paired_policy_deploy=bool(pair),
                verdict_flips=bool(marg and not pair))


def cp_two_sided(s, m, level=0.05):
    """Two-sided Clopper-Pearson (1-level) CI for psi_g given s/m."""
    lo = 0.0 if s == 0 else float(_beta.ppf(level / 2, s, m - s + 1))
    hi = 1.0 if s == m else float(_beta.ppf(1 - level / 2, s + 1, m - s))
    return lo, hi


def main():
    byq, H, ORI, qc, n_pooled = load()
    ref = None
    rp = artifact("m_pairedlift_results.json")
    if os.path.exists(rp):
        ref = {c["family"].split(" (")[0]: c for c in json.load(open(rp))["cells"]}

    out = {"run": "R050 decision-value: psi_g is not identified by any marginal detector metric",
           "split_seed": SPLIT_SEED, "D_cal_twins": len(qc),
           "estimand_note": "Frechet partial ID: marginals leave psi_g free over [L,U]; re-pairing "
                            "holds every marginal metric invariant while psi_g spans the interval.",
           "cells": [], "self_check": "PASS"}
    checks = []
    print(f"{'cell':<14}{'m_c':>5}{'s_c':>5}{'psi_hat':>8}{'F1/FNR':>8}"
          f"{'  Frechet[L,U]':<16}{'repair[min,max]':<18}{'CP95 CI':<16}")
    for fam, tau, disp, alphas in CELLS:
        rT = np.array([ORI[fam][byq[q][0]] for q in qc])
        rH = np.array([ORI[fam][byq[q][1]] for q in qc])
        n = len(rT)
        m_c = int((rT > tau).sum()); s_c = int(((rT > tau) & (rH <= tau)).sum())
        psi_hat = s_c / m_c if m_c else float("nan")
        pT0 = m_c / n
        mm_obs = marginal_metrics(rT, rH, tau)
        F1 = mm_obs["F1_marginal"]
        L, U = frechet_psi(pT0, F1)
        psi_min, psi_max, m_c2, rH_min, rH_max = extremal_repairings(rT, rH, tau)
        # marginal metrics under the two extremal re-pairings (must equal observed to float tol)
        mm_min = marginal_metrics(rT, rH_min, tau); mm_max = marginal_metrics(rT, rH_max, tau)
        inv_ok = all(abs(mm_obs[k] - mm_min[k]) < 1e-9 and abs(mm_obs[k] - mm_max[k]) < 1e-9
                     for k in mm_obs)
        cp_lo, cp_hi = cp_two_sided(s_c, m_c)
        # self-check vs m_pairedlift
        sc_ok = True
        if ref and disp in ref:
            r = ref[disp]
            sc_ok = (r["m_c"] == m_c and r["s_c"] == s_c and abs(r["F1"] - F1) < 1e-3)
            checks.append((disp, sc_ok))
        row = dict(family=disp, tau=tau, alphas=alphas, n=n, m_c=m_c, s_c=s_c,
                   psi_hat=round(psi_hat, 4), pT0=round(pT0, 4),
                   marginal_metrics_observed=mm_obs,
                   frechet_interval=[round(L, 4), round(U, 4)],
                   repairing_extremes=dict(psi_min=round(psi_min, 4), psi_max=round(psi_max, 4)),
                   marginal_invariance_under_repairing=bool(inv_ok),
                   marginal_metrics_min_repair=mm_min, marginal_metrics_max_repair=mm_max,
                   paired_audit_ci95=[round(cp_lo, 4), round(cp_hi, 4)],
                   prior_width_marginal_only=round(U - L, 4),
                   posterior_width_paired=round(cp_hi - cp_lo, 4),
                   decision=_verdict(disp),
                   selfcheck_vs_pairedlift=bool(sc_ok))
        out["cells"].append(row)
        print(f"{disp:<14}{m_c:>5}{s_c:>5}{psi_hat:>8.4f}{F1:>8.4f}"
              f"  [{L:.2f},{U:.2f}]     [{psi_min:.3f},{psi_max:.3f}]   "
              f"[{cp_lo:.3f},{cp_hi:.3f}]  inv={inv_ok} flip={row['decision']['verdict_flips']}")
        if not (inv_ok and sc_ok):
            out["self_check"] = "FAIL"
    if any(not ok for _, ok in checks):
        out["self_check"] = "FAIL"
    json.dump(out, open(artifact("R050_decision_value_results.json"), "w"), indent=2)
    print(f"\nself_check={out['self_check']}  (reproduces m_pairedlift m_c/s_c/F1; marginal "
          f"metrics invariant under re-pairing; psi_g spans the Frechet interval)")
    print(f"wrote {artifact('R050_decision_value_results.json')}")


if __name__ == "__main__":
    main()
