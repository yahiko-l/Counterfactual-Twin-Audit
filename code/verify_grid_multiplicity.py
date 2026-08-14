#!/usr/bin/env python
"""Provenance for the §4 threshold-grid multiplicity note (the "28 -> 26 / 234 / 78" documentation).

A reviewer asked why the paper's advertised 28-point quantile grid yields Holm families of 234 (lower/global)
and 78 (upper safety) rather than the nominal 3x28x3=252 and 3x28=84. The reduction is produced by the
min-mass gamma-floor pre-filter inside certify.candidate_grid(): of the 28 quantile thresholds per family it
keeps only those whose D_sel pooled rejection rate lies in [gamma, 0.995] (so the rejected-truthful denominator
can plausibly clear the applicability floor and the accept-almost-nothing tail is dropped). This script
reproduces that count on the REAL headline D_sel and confirms:
  (1) each family's 28-point quantile grid survives np.unique intact (28 unique) and the gamma-floor filter
      then drops EXACTLY 2 (the extreme thresholds outside [gamma, rate_max]) -> 26 kept/family, so the
      reduction is attributable to the filter alone (recorded per-family in per_family_trace); hence
      |C_low| = 3*26*3 = 234 (lower/global) and |C_up| = 3*26 = 78 (upper safety, per beta);
  (2) recomputing the headline psi_g>alpha FAILURE leaf (the binding leaf of the joint cells) over the NOMINAL
      unfiltered lower count (nlow=252) still certifies every headline cell under BOTH the exact
      (Clopper-Pearson) and heterogeneity-robust (Hoeffding-Bentkus) leaves -- so the documented gap is a
      documentation matter, not a numerical one. (The 84 upper count is the separate mitigation family; it is
      reported as a count, not re-certified here.)

Mirrors m3_combined.py's load/split/orientation exactly (same scores, seed 7 split, 28-point grid, gamma).
CPU only, no GPU, no network. Output: results/grid_multiplicity_check.json
"""
import os, sys, csv, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
LAB = artifact("scores_strong.csv")
ART = artifact("scores_strong_artificial.csv")
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
ALPHAS = [0.05, 0.10, 0.20]
GAMMA = C.GAMMA_DEFAULT
E = math.e
# headline FAILURE-leaf cells (name, s_c, m_c, alpha) from tab:joint
HEADLINE = [("P(True)@.05", 179, 2423, 0.05), ("P(True)@.10", 120, 278, 0.10),
            ("LLM-judge@.20", 198, 347, 0.20), ("ref-NLI@.10", 138, 488, 0.10)]


def hb_lcb(s, m, level):
    if m <= 0 or s <= 0:
        return 0.0
    sh = s / m
    return max(sh - math.sqrt(math.log(1.0 / level) / (2.0 * m)), C.cp_lower_psi(s, m, level / E), 0.0)


def load():
    def rd(p, src):
        r = list(csv.DictReader(open(p)))
        for x in r:
            x.setdefault("source", src)
        return r
    rows = rd(LAB, "lab") + rd(ART, "art")
    qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
    RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
    byq = {}
    for i in range(len(rows)):
        byq.setdefault(int(qid[i]), {})[int(H[i])] = i
    qids = sorted(q for q in byq if set(byq[q]) == {0, 1})
    perm = np.random.default_rng(7).permutation(len(qids)); n = len(qids); a = int(n * .4)
    D_sel = set(np.array(qids)[perm[:a]].tolist())
    return byq, H, RAW, qids, D_sel


def main():
    byq, H, RAW, qids, D_sel = load()
    sel_idx = np.array([byq[q][h] for q in qids if q in D_sel for h in (0, 1)])
    per_family = {}; trace = {}
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ori = (-RAW[f] if fl else RAW[f])
        rTs = np.array([ori[byq[q][0]] for q in qids if q in D_sel])
        rHs = np.array([ori[byq[q][1]] for q in qids if q in D_sel])
        pooled = np.r_[rTs, rHs]
        # replicate certify.candidate_grid() step by step so the 28->26 reduction is ATTRIBUTED:
        qraw = np.quantile(pooled, np.linspace(0.01, 0.99, 28))   # (i) 28 nominal quantile points
        qs = np.unique(qraw)                                       # (ii) after np.unique
        kept_taus = []; excluded = []
        for tau in qs:
            rate = float((pooled > tau).mean())                   # D_sel rejection rate at tau
            if GAMMA <= rate <= 0.995:
                kept_taus.append(round(float(tau), C.TAU_ROUND))
            else:
                excluded.append(dict(tau=round(float(tau), 6), rate=round(rate, 4),
                                     reason=("rate<gamma" if rate < GAMMA else "rate>rate_max")))
        kept_unique = sorted(set(kept_taus))                      # (iii) after gamma-floor filter + rounding
        # cross-check: this inline replication must equal the shipped candidate_grid()
        assert len(kept_unique) == len(C.candidate_grid(pooled, gamma=GAMMA, n_grid=28)), \
            f"inline grid replication != certify.candidate_grid for {f}"
        per_family[f] = len(kept_unique)
        trace[f] = dict(n_quantile_points=28, n_after_np_unique=int(len(qs)),
                        n_kept_after_gamma_filter=len(kept_unique),
                        reduction_is_gamma_filter=bool(len(qs) == 28 and len(kept_unique) == 26),
                        excluded_thresholds=excluded)
    kept = per_family                      # thresholds kept per family after the gamma-floor filter
    c_low = sum(kept.values()) * len(ALPHAS)
    c_up = sum(kept.values())              # per beta
    all26 = all(v == 26 for v in kept.values())
    # the reduction is attributable to the gamma-floor filter alone iff np.unique kept all 28 and
    # exactly 2 were dropped by the [gamma, rate_max] condition, for EVERY family
    gamma_filter_only = all(t["reduction_is_gamma_filter"] and len(t["excluded_thresholds"]) == 2
                            for t in trace.values())

    # robustness: headline cells at the NOMINAL unfiltered nlow=252 (3*28*3) still certify?
    nlow_nominal = 3 * 28 * len(ALPHAS)    # 252
    robustness = []
    for name, s, m, alpha in HEADLINE:
        lev = C.DELTA_LOWER / nlow_nominal
        ex = round(C.cp_lower_psi(s, m, lev), 4); rb = round(hb_lcb(s, m, lev), 4)
        robustness.append(dict(cell=name, alpha=alpha, nlow=nlow_nominal,
                               exact_lcb=ex, robust_lcb=rb, clears=bool(ex > alpha and rb > alpha)))
    all_clear = all(r["clears"] for r in robustness)

    out = dict(run="grid-multiplicity provenance (28 -> 26 gamma-floor reduction; 234/78; lower-leaf 252-robustness)",
               gamma=GAMMA, rate_max=0.995, n_grid_nominal=28, kept_per_family=kept,
               all_families_keep_26=all26, reduction_is_gamma_filter_only=gamma_filter_only,
               per_family_trace=trace, C_low=c_low, C_up=c_up,
               arithmetic=dict(C_low="3x26x3=234 (lower/global)", C_up="3x26=78 (upper, per beta)",
                               nominal_unfiltered_counts="3x28x3=252 (lower) / 3x28=84 (upper, per beta)"),
               headline_failure_leaf_robust_at_nominal_252=robustness,
               headline_failure_leaf_all_clear_at_252=all_clear,
               scope_note="252-robustness is checked for the headline psi_g>alpha FAILURE leaf only "
                          "(the binding leaf of the joint cells); the 84 upper-safety count pertains to the "
                          "separate mitigation family (sec:res-mitigate) and is reported here as a count, not "
                          "re-certified at 84.",
               verdict="PASS" if (all26 and gamma_filter_only and c_low == 234 and c_up == 78 and all_clear)
                       else "CHECK")
    json.dump(out, open(artifact("grid_multiplicity_check.json"), "w"), indent=2)
    print(f"kept per family (of 28): {kept}  -> all 26? {all26}  (reduction is gamma-filter only? {gamma_filter_only})")
    for f, t in trace.items():
        print(f"  {f}: 28 quantiles -> {t['n_after_np_unique']} after np.unique -> "
              f"{t['n_kept_after_gamma_filter']} kept; dropped {len(t['excluded_thresholds'])}: "
              f"{[(e['rate'], e['reason']) for e in t['excluded_thresholds']]}")
    print(f"|C_low| = 3x26x3 = {c_low}   |C_up| = 3x26 = {c_up}")
    print(f"headline psi_g>alpha failure leaf certified at NOMINAL unfiltered nlow=252 (both exact + robust)?")
    for r in robustness:
        print(f"  {r['cell']:<14} exact {r['exact_lcb']:.4f} / robust {r['robust_lcb']:.4f}  > α={r['alpha']}  {'OK' if r['clears'] else 'NO'}")
    print(f"VERDICT: {out['verdict']}")
    print(f"wrote {artifact('grid_multiplicity_check.json')}")


if __name__ == "__main__":
    main()
