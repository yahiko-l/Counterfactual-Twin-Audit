#!/usr/bin/env python
"""R063 — matched-marginal policy comparison (revision Major-3; prespecified in
refine-logs/analysis_plan_v2_20260717.md Section 5.1).

Question: do policies (family, tau) that are MATCHED on the marginal detector profile
(coverage, accepted-set rate rho, marginal hallucinated-acceptance F1, family AUROC)
still differ in the paired rate psi_hat? If yes, the paired estimand carries decision
information that no marginal profile pins down (empirical companion to the Frechet
non-identifiability result).

Prespecified (frozen before running):
  candidates  = all (family, tau) grid points of the RELEASED Table-4 pipeline
                (m3_combined: scores_strong + scores_strong_artificial, split seed 7,
                D_sel orientation, 28-point gamma-filtered grid), with m_c > 0 on D_cal
  metrics     = coverage & rho_hat on the induced pi=0.10 one-answer stream
                (u = rng(2024).random(ncal) mirror), F1 = P(accept A_H) on D_cal pairs,
                family AUROC (threshold-free, D_cal both twins), psi_hat = s_c/m_c
  calipers    = |d cov| <= 0.03, |d rho| <= 0.01, |d F1| <= 0.03, |d AUROC| <= 0.05
  matching    = optimal one-to-one min-total-standardized-distance over caliper-eligible
                pairs (networkx blossom; greedy fallback); ALL matched pairs reported
  inference   = paired question-level bootstrap (B=2000, shared multinomial weights)
                95% percentile CI for d psi_hat; policy-flip flag at beta_ref = 0.10
  feasibility = if < 3 matched pairs: report verbatim infeasibility, never relax calipers

Self-check (abort on failure): the four Table-4 distinct thresholds must reproduce the
released (m_c, s_c, psi_hat, F1, rho_hat, coverage) exactly.

Output: R063_matched_marginal_results.json (committable; counts and metrics only).
"""
import os, sys, json, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
LAB = artifact("scores_strong.csv"); ART = artifact("scores_strong_artificial.csv")
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
NICE = {"score_sptrue": "P(True)-8B", "score_sjudge": "LLM-judge-8B", "score_sNLI": "ref-NLI"}
GAMMA = C.GAMMA_DEFAULT
PI = 0.10; PREV_SEED = 2024; SPLIT_SEED = 7; NGRID = 28
CALIPERS = dict(cov=0.03, rho=0.01, F1=0.03, auroc=0.05)
BETA_REF = 0.10; B_BOOT = 2000; BOOT_SEED = 20260717
EXPECT = {  # released Table-4 distinct thresholds (self-check)
    ("score_sptrue", 0.000123): dict(m_c=2423, s_c=179, psi=0.0739, F1=0.1295, rho=0.0311, cov=0.370),
    ("score_sptrue", 0.222824): dict(m_c=278, s_c=120, psi=0.4317, F1=0.6015, rho=0.0636, cov=0.900),
    ("score_sjudge", 0.245085): dict(m_c=347, s_c=198, psi=0.5706, F1=0.8852, rho=0.0924, cov=0.909),
    ("score_sNLI", 0.518631): dict(m_c=488, s_c=138, psi=0.2828, F1=0.657, rho=0.0709, cov=0.854),
}


def load(path, srcname):
    rows = list(csv.DictReader(open(path)))
    for r in rows: r.setdefault("source", srcname)
    return rows


def main():
    rows = load(LAB, "pqa_labeled") + load(ART, "pqa_artificial")
    qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
    RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
    byq = {}
    for i in range(len(rows)): byq.setdefault(int(qid[i]), {})[int(H[i])] = i
    QIDS = np.array(sorted(byq))
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(QIDS)); n = len(QIDS)
    a, b = int(n * .40), int(n * .40) + int(n * .40)
    D_sel = set(QIDS[perm[:a]].tolist()); D_cal = set(QIDS[perm[a:b]].tolist())
    sel_idx = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
    cal_q = [q for q in QIDS if q in D_cal]; ncal = len(cal_q)
    print(f"[data] twins={n} ncal={ncal}")

    cands = []
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI = (-RAW[f] if fl else RAW[f])
        rTs = np.array([ORI[byq[q][0]] for q in QIDS if q in D_sel])
        rHs = np.array([ORI[byq[q][1]] for q in QIDS if q in D_sel])
        grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=NGRID)
        rTc = np.array([ORI[byq[q][0]] for q in cal_q]); rHc = np.array([ORI[byq[q][1]] for q in cal_q])
        u = np.random.default_rng(PREV_SEED).random(ncal)          # mirror M3/m10 CRN
        takeH = u < PI; s_stream = np.where(takeH, rHc, rTc)
        both = np.r_[rTc, rHc]; lab = np.r_[np.zeros(ncal, int), np.ones(ncal, int)]
        order = np.argsort(np.argsort(both)); ranks = order + 1.0   # average-tie-free ranks OK here
        auroc = (ranks[lab == 1].sum() - ncal * (ncal + 1) / 2) / (ncal * ncal)
        for tau in grid:
            m_c, s_c = C.psi_counts(rTc, rHc, tau)
            if m_c <= 0: continue
            acc = s_stream <= tau; cov = float(acc.mean())
            rho = float((takeH & acc).sum() / max(acc.sum(), 1))
            cands.append(dict(family=f, tau=round(float(tau), 6), m_c=int(m_c), s_c=int(s_c),
                              psi=float(s_c / m_c), F1=float((rHc <= tau).mean()), cov=cov,
                              rho=rho, auroc=float(auroc),
                              tT=(rTc > tau), aH=(rHc <= tau)))
    print(f"[candidates] {len(cands)} (family x gamma-filtered grid, m_c>0)")

    # ---- self-check against released Table-4 thresholds ----
    ok_all = True
    for (f, tau), exp in EXPECT.items():
        hit = [c for c in cands if c["family"] == f and abs(c["tau"] - tau) < 5e-6]
        ok = bool(hit) and hit[0]["m_c"] == exp["m_c"] and hit[0]["s_c"] == exp["s_c"] and \
             abs(hit[0]["F1"] - exp["F1"]) < 6e-4 and abs(hit[0]["rho"] - exp["rho"]) < 6e-4 and \
             abs(hit[0]["cov"] - exp["cov"]) < 6e-4
        ok_all &= ok
        print(f"[self-check] {NICE[f]} tau={tau}: {'PASS' if ok else 'FAIL'}"
              + ("" if ok else f"  got={{m_c:{hit[0]['m_c'] if hit else None}}}"))
    if not ok_all:
        print("ABORT: pipeline does not reproduce Table 4."); sys.exit(1)

    # ---- caliper-eligible pairs + optimal one-to-one matching ----
    edges = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            ci, cj = cands[i], cands[j]
            gaps = dict(cov=abs(ci["cov"] - cj["cov"]), rho=abs(ci["rho"] - cj["rho"]),
                        F1=abs(ci["F1"] - cj["F1"]), auroc=abs(ci["auroc"] - cj["auroc"]))
            if all(gaps[k] <= CALIPERS[k] for k in CALIPERS):
                dist = sum(gaps[k] / CALIPERS[k] for k in CALIPERS)
                edges.append((i, j, dist, gaps))
    print(f"[eligible] {len(edges)} caliper-satisfying pairs")
    try:
        import networkx as nx
        G = nx.Graph(); G.add_weighted_edges_from([(i, j, d) for i, j, d, _ in edges])
        matching = nx.min_weight_matching(G)
        matched = sorted(tuple(sorted(e)) for e in matching)
        method = "networkx blossom min_weight_matching"
    except Exception:
        used, matched = set(), []
        for i, j, d, _ in sorted(edges, key=lambda e: e[2]):
            if i in used or j in used: continue
            matched.append((i, j)); used |= {i, j}
        method = "greedy fallback"
    gap_of = {(i, j): g for i, j, _, g in edges}

    # ---- paired question-level bootstrap (shared weights) ----
    W = np.random.default_rng(BOOT_SEED).multinomial(ncal, np.full(ncal, 1 / ncal), size=B_BOOT)
    def psi_boot(c):
        num = W @ (c["tT"] & c["aH"]).astype(float); den = W @ c["tT"].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    pairs_out, flips, cross = [], 0, 0
    for (i, j) in matched:
        ci, cj = cands[i], cands[j]
        lo, hi = (ci, cj) if ci["psi"] <= cj["psi"] else (cj, ci)
        d = hi["psi"] - lo["psi"]
        db = psi_boot(hi) - psi_boot(lo)
        ci95 = [float(np.nanpercentile(db, 2.5)), float(np.nanpercentile(db, 97.5))]
        flip = (lo["psi"] <= BETA_REF) and (hi["psi"] > BETA_REF)
        flips += int(flip); cross += int(ci["family"] != cj["family"])
        g = gap_of[(i, j)]
        pairs_out.append(dict(
            policy_lo=dict(family=NICE[lo["family"]], tau=lo["tau"], psi=round(lo["psi"], 4), m_c=lo["m_c"]),
            policy_hi=dict(family=NICE[hi["family"]], tau=hi["tau"], psi=round(hi["psi"], 4), m_c=hi["m_c"]),
            cross_family=ci["family"] != cj["family"],
            balance=dict(d_cov=round(g["cov"], 4), d_rho=round(g["rho"], 4),
                         d_F1=round(g["F1"], 4), d_auroc=round(g["auroc"], 4)),
            delta_psi=round(d, 4), delta_psi_ci95=[round(x, 4) for x in ci95],
            flip_at_beta_ref=bool(flip)))
    pairs_out.sort(key=lambda p: -p["delta_psi"])

    feasible = len(matched) >= 3
    summary = dict(
        n_candidates=len(cands), n_eligible_pairs=len(edges), n_matched_pairs=len(matched),
        n_cross_family=cross, matching_method=method,
        delta_psi_min=round(min((p["delta_psi"] for p in pairs_out), default=float("nan")), 4),
        delta_psi_median=round(float(np.median([p["delta_psi"] for p in pairs_out])), 4) if pairs_out else None,
        delta_psi_max=round(max((p["delta_psi"] for p in pairs_out), default=float("nan")), 4),
        n_ci_excl_zero=sum(1 for p in pairs_out if p["delta_psi_ci95"][0] > 0),
        n_flip_at_beta_ref=flips, beta_ref=BETA_REF, feasible=feasible,
        infeasibility_statement=(None if feasible else
            "the available policy family did not support a sufficiently matched empirical comparison"))
    out = dict(run="R063 matched-marginal policy comparison (prespecified: analysis_plan_v2 S5.1)",
               prereg="refine-logs/analysis_plan_v2_20260717.md (commit 013ff04)",
               pi=PI, split_seed=SPLIT_SEED, prev_seed=PREV_SEED, n_grid=NGRID,
               calipers=CALIPERS, bootstrap=dict(B=B_BOOT, seed=BOOT_SEED, scheme="paired multinomial over D_cal questions"),
               summary=summary, matched_pairs=pairs_out)
    with open(artifact("R063_matched_marginal_results.json"), "w") as fo:
        json.dump(out, fo, indent=2)
    print(f"[matched] {len(matched)} pairs ({cross} cross-family) via {method}")
    if pairs_out:
        print(f"[delta_psi] min/med/max = {summary['delta_psi_min']}/{summary['delta_psi_median']}/{summary['delta_psi_max']}"
              f"  CI-excl-0: {summary['n_ci_excl_zero']}  flips@beta_ref={BETA_REF}: {flips}")
    print("[wrote] R063_matched_marginal_results.json")


if __name__ == "__main__":
    main()
