#!/usr/bin/env python
"""
M7 MULTI-SEED ROBUSTNESS (reviewer-objection #2: "is 8/6/2/0 a seed artifact?").

The released headline (m3_combined.py) fixes TWO seeds:
  * calibration/sel/test SPLIT seed = 7      (m3_combined.split, default seed=7)
  * deployment PREVALENCE-draw seed = 2024   (rng = np.random.default_rng(2024))
and reports a single joint-cell count per prevalence (8/6/2/0 for pi=0.05/0.10/0.20/0.50).

This driver re-runs the *identical* certification path over a grid of (split_seed, prev_seed)
pairs and reports the DISTRIBUTION of the headline quantities, so the paper can state the
result is not an artifact of the two fixed seeds. It does NOT touch the released code path:
it imports certify.py (the shared substrate) and reuses m3_combined's load + certify logic
verbatim, with only the two seeds lifted to parameters.

STRUCTURAL NOTE exploited for speed (and worth stating in the paper):
  - m_c, s_c, the psi lower-failure certificate (low_ok) and the min-mass certificate (mass_ok)
    depend ONLY on the SPLIT seed (they are computed on D_cal twins, no prevalence draw).
  - ONLY the global rho<=alpha gate (rho_ok) depends on the prevalence draw (and pi).
  - joint cell = rho_ok AND low_ok AND mass_ok.
  => varying prev_seed stresses the "certified-safe gate" side; varying split_seed stresses
     the psi-failure side. We vary both and decompose.

Output: results/m7_multiseed_results.json
CPU only; no GPU, no network.

Usage:
  python m7_multiseed_robustness.py            # full grid (40 split x 100 prev) x 4 pi
  python m7_multiseed_robustness.py 5 5        # quick: 5 split x 5 prev (validation)
"""
import os, sys, json, csv, time
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# ----------------------------------------------------------------- config (mirror m3_combined)
LAB = "data/scores/scores_strong.csv"
ART = "data/scores/scores_strong_artificial.csv"
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
ALPHAS = [0.05, 0.10, 0.20]
GAMMA = C.GAMMA_DEFAULT
PIS = [0.50, 0.20, 0.10, 0.05]
N_GRID = 28

# anchor = released headline seeds
SPLIT_SEED_ANCHOR = 7
PREV_SEED_ANCHOR = 2024

# ----------------------------------------------------------------- data load (mirror m3_combined)
def load(path, src):
    rows = list(csv.DictReader(open(path)))
    for r in rows: r.setdefault("source", src)
    return rows

rows = load(LAB, "pqa_labeled") + (load(ART, "pqa_artificial") if os.path.exists(ART) else [])
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
src = np.array([r["source"] for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
byq = {}
for i in range(len(rows)):
    q, h = int(qid[i]), int(H[i])
    assert (q not in byq) or (h not in byq[q]), f"dup (qid={q},H={h})"
    byq.setdefault(q, {})[h] = i
for q, d in byq.items(): assert set(d) == {0, 1}, f"qid {q} incomplete"
QIDS = np.array(sorted(byq))
POOLED = QIDS.tolist()
log(f"combined twins={len(QIDS)} (labeled={int((src=='pqa_labeled').sum())//2}, "
    f"artificial={int((src=='pqa_artificial').sum())//2})")

def split(qids, seed, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids)
    perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

# ----------------------------------------------------------------- split-level cache (psi side)
# Everything here depends ONLY on split_seed. Computed once per split_seed.
def build_split_cache(qids_pool, split_seed):
    D_sel, D_cal, D_test = split(qids_pool, seed=split_seed)
    sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    ORI = {}
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
    def tw(f, qs):
        s = ORI[f]; q = [x for x in qids_pool if x in qs]
        return (np.array([s[byq[x][0]] for x in q]),
                np.array([s[byq[x][1]] for x in q]), q)
    cache = {"grids": {}, "rTc": {}, "rHc": {}, "ncal": None,
             "m_c": {}, "s_c": {}, "low_pool": [], "mass_pool": []}
    for f in FAMS:
        rTs, rHs, _ = tw(f, D_sel)
        grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=N_GRID)
        rTc, rHc, _ = tw(f, D_cal); ncal = len(rTc)
        cache["ncal"] = ncal
        cache["grids"][f] = grid; cache["rTc"][f] = rTc; cache["rHc"][f] = rHc
        cache["m_c"][f] = []; cache["s_c"][f] = []
        for ti, tau in enumerate(grid):
            m_c, sfail = C.psi_counts(rTc, rHc, tau)
            cache["m_c"][f].append(m_c); cache["s_c"][f].append(sfail)
            cache["mass_pool"].append((f, ti, C.minmass_p(m_c, ncal, GAMMA)))
            for a in ALPHAS:
                cache["low_pool"].append((f, a, ti, C.lower_p_psi(sfail, m_c, a)))
    # Holm on the split-only families (identical for every prev_seed/pi)
    low_rej = C.holm_reject([x[3] for x in cache["low_pool"]], C.DELTA_LOWER)
    mass_rej = C.holm_reject([x[2] for x in cache["mass_pool"]], C.DELTA_MASS)
    cache["low_ok"] = {(x[0], x[1], x[2]): bool(low_rej[i]) for i, x in enumerate(cache["low_pool"])}
    cache["mass_ok"] = {(x[0], x[1]): bool(mass_rej[i]) for i, x in enumerate(cache["mass_pool"])}
    cache["nlow"] = len(cache["low_pool"])
    # precompute the display psi_lcb per candidate (split-only)
    cache["psi_lcb"] = {}
    for f in FAMS:
        for ti in range(len(cache["grids"][f])):
            sfail = cache["s_c"][f][ti]; m_c = cache["m_c"][f][ti]
            cache["psi_lcb"][(f, ti)] = round(C.cp_lower_psi(sfail, m_c, C.DELTA_LOWER/cache["nlow"]), 4)
    return cache

# ----------------------------------------------------------------- rho side (prevalence draw)
def certify_one(cache, pi, prev_seed):
    """Return (joint_cells, cells) for a fixed split cache + (pi, prev_seed).
    Mirrors m3_combined.certify_at_prevalence rho-stream + joint-intersection logic exactly."""
    ncal = cache["ncal"]
    rho_pool = []
    rho_hat = {}  # (f,ti) -> stream rho_hat (prevalence-dependent)
    for f in FAMS:
        rTc = cache["rTc"][f]; rHc = cache["rHc"][f]; grid = cache["grids"][f]
        rng = np.random.default_rng(prev_seed)          # << prev seed (was 2024)
        takeH = rng.random(ncal) < pi
        s_stream = np.where(takeH, rHc, rTc); h_stream = takeH.astype(int)
        for ti, tau in enumerate(grid):
            acc = s_stream <= tau; Kc = int(acc.sum()); Sc = int(h_stream[acc].sum())
            rho_hat[(f, ti)] = (round(Sc/Kc, 4) if Kc > 0 else None)
            for a in ALPHAS:
                rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
    rho_rej = C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL)
    rho_ok = {(x[0], x[1], x[2]): bool(rho_rej[i]) for i, x in enumerate(rho_pool)}
    cells = {}
    for f in FAMS:
        cells[f] = {}
        for a in ALPHAS:
            joint = []
            for ti in range(len(cache["grids"][f])):
                if rho_ok[(f, a, ti)] and cache["low_ok"][(f, a, ti)] and cache["mass_ok"][(f, ti)]:
                    joint.append(dict(ti=ti, tau=float(cache["grids"][f][ti]),
                                      rho_hat=rho_hat[(f, ti)], m_c=cache["m_c"][f][ti],
                                      s_c=cache["s_c"][f][ti],
                                      psi_hat=(round(cache["s_c"][f][ti]/cache["m_c"][f][ti], 4)
                                               if cache["m_c"][f][ti] > 0 else None),
                                      psi_lcb=cache["psi_lcb"][(f, ti)]))
            joint.sort(key=lambda r: -(r["psi_lcb"] or 0))
            cells[f][a] = dict(certified=len(joint) > 0, best=(joint[0] if joint else None))
    njoint = sum(1 for f in FAMS for a in ALPHAS if cells[f][a]["certified"])
    return njoint, cells

# ----------------------------------------------------------------- run grid
def main():
    n_split = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_prev = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    split_seeds = list(range(n_split))
    prev_seeds = list(range(n_prev))

    # ---- VALIDATION: anchor (7, 2024) must reproduce the released headline exactly
    anchor_cache = build_split_cache(POOLED, SPLIT_SEED_ANCHOR)
    anchor = {}
    for pi in PIS:
        nj, cells = certify_one(anchor_cache, pi, PREV_SEED_ANCHOR)
        anchor[pi] = (nj, cells)
    got = {pi: anchor[pi][0] for pi in PIS}
    exp = {0.05: 8, 0.10: 6, 0.20: 2, 0.50: 0}
    hb = anchor[0.10][1]["score_sptrue"][0.10]["best"]
    log(f"ANCHOR (split=7, prev=2024): joint by pi = "
        f"{{0.05:{got[0.05]}, 0.10:{got[0.10]}, 0.20:{got[0.20]}, 0.50:{got[0.50]}}}")
    log(f"ANCHOR headline P(True) a=0.10 best: tau={hb['tau']:.6f} rho_hat={hb['rho_hat']} "
        f"m_c={hb['m_c']} psi_hat={hb['psi_hat']} psi_lcb={hb['psi_lcb']}")
    ok_counts = (got == exp)
    ok_cell = (hb['m_c'] == 278 and abs(hb['psi_lcb'] - 0.327) < 1e-6 and hb['rho_hat'] == 0.0636)
    assert ok_counts, f"ANCHOR joint counts mismatch: got {got} expected {exp}"
    assert ok_cell, f"ANCHOR headline cell mismatch: {hb}"
    log("ANCHOR reproduces released headline EXACTLY. Proceeding to grid.")

    # ---- full grid (build each split cache once, sweep prev x pi)
    # records[pi] = list of dicts per (split_seed, prev_seed)
    records = {pi: [] for pi in PIS}
    HEAD = ("score_sptrue", 0.10)   # the headline cell, audited at pi=0.10
    t_last = time.time()
    for si, ss in enumerate(split_seeds):
        cache = build_split_cache(POOLED, ss)
        for ps in prev_seeds:
            for pi in PIS:
                nj, cells = certify_one(cache, pi, ps)
                rec = {"split": ss, "prev": ps, "joint": nj,
                       "cert": {f"{f}|{a}": bool(cells[f][a]["certified"]) for f in FAMS for a in ALPHAS}}
                hcell = cells[HEAD[0]][HEAD[1]]["best"]
                if hcell is not None:
                    rec["head"] = {"tau": round(hcell["tau"], 6), "rho_hat": hcell["rho_hat"],
                                   "m_c": hcell["m_c"], "psi_hat": hcell["psi_hat"],
                                   "psi_lcb": hcell["psi_lcb"]}
                else:
                    rec["head"] = None
                records[pi].append(rec)
        if time.time() - t_last > 20:
            log(f"  split {si+1}/{len(split_seeds)} done"); t_last = time.time()

    # ---- aggregate
    def pct(a, q): return round(float(np.percentile(a, q)), 4)
    def summ(a):
        a = np.asarray(a, float)
        return dict(n=len(a), mean=round(float(a.mean()), 4), std=round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
                    min=round(float(a.min()), 4), p5=pct(a, 5), p25=pct(a, 25), median=pct(a, 50),
                    p75=pct(a, 75), p95=pct(a, 95), max=round(float(a.max()), 4))

    report = {
        "run": "M7 multi-seed robustness (split x prevalence-draw)",
        "grid": {"n_split": len(split_seeds), "n_prev": len(prev_seeds),
                 "split_seeds": f"0..{len(split_seeds)-1}", "prev_seeds": f"0..{len(prev_seeds)-1}",
                 "n_combos_per_pi": len(split_seeds)*len(prev_seeds)},
        "anchor_released": {"split_seed": SPLIT_SEED_ANCHOR, "prev_seed": PREV_SEED_ANCHOR,
                            "joint_by_pi": {str(pi): got[pi] for pi in PIS},
                            "headline_cell": hb},
        "delta": {"total": C.DELTA_TOTAL}, "alphas": ALPHAS, "gamma": GAMMA, "n_grid": N_GRID,
        "per_pi": {}, "headline_cell_pi0.10": {}, "decomposition_pi0.10": {},
        "elapsed_s": None,
    }
    CELLKEYS = [f"{f}|{a}" for f in FAMS for a in ALPHAS]
    for pi in PIS:
        recs = records[pi]
        joints = [r["joint"] for r in recs]
        hist = {k: int(sum(1 for j in joints if j == k)) for k in range(10)}
        cellfrac = {ck: round(float(np.mean([r["cert"][ck] for r in recs])), 4) for ck in CELLKEYS}
        report["per_pi"][str(pi)] = {
            "joint_cells": summ(joints),
            "frac_at_least_one_joint": round(float(np.mean([j >= 1 for j in joints])), 4),
            "joint_count_histogram": hist,
            "per_cell_certified_frac": cellfrac,
        }
    # headline cell deep-dive at pi=0.10
    recs10 = records[0.10]
    head_recs = [r["head"] for r in recs10 if r["head"] is not None]
    head_cert_frac = round(float(np.mean([r["cert"]["score_sptrue|0.1"] for r in recs10])), 4)
    report["headline_cell_pi0.10"] = {
        "cell": "score_sptrue (P(True)-8B), alpha=0.10",
        "certified_frac": head_cert_frac,
        "psi_lcb": summ([h["psi_lcb"] for h in head_recs]) if head_recs else None,
        "m_c": summ([h["m_c"] for h in head_recs]) if head_recs else None,
        "rho_hat": summ([h["rho_hat"] for h in head_recs if h["rho_hat"] is not None]) if head_recs else None,
        "psi_hat": summ([h["psi_hat"] for h in head_recs if h["psi_hat"] is not None]) if head_recs else None,
        "note": "psi_lcb/m_c summarized over the subset of (split,prev) combos where the cell is jointly certified",
    }
    # variance decomposition for pi=0.10 joint count: between-split vs between-prev
    J = np.zeros((len(split_seeds), len(prev_seeds)))
    idx = {(r["split"], r["prev"]): r["joint"] for r in recs10}
    for i, ss in enumerate(split_seeds):
        for j, ps in enumerate(prev_seeds):
            J[i, j] = idx[(ss, ps)]
    report["decomposition_pi0.10"] = {
        "overall_mean": round(float(J.mean()), 4), "overall_std": round(float(J.std(ddof=1)), 4),
        "between_split_std_of_splitmeans": round(float(J.mean(axis=1).std(ddof=1)), 4),
        "between_prev_std_of_prevmeans": round(float(J.mean(axis=0).std(ddof=1)), 4),
        "mean_within_split_std_over_prev": round(float(J.std(axis=1, ddof=1).mean()), 4),
        "mean_within_prev_std_over_split": round(float(J.std(axis=0, ddof=1).mean()), 4),
        "note": "J[i,j] = joint-cell count at split_seed i, prev_seed j, pi=0.10",
    }
    report["elapsed_s"] = round(time.time()-T0, 1)
    out = "results/m7_multiseed_results.json"
    json.dump(report, open(out, "w"), indent=2, default=str)

    # ---- console summary
    print("\n" + "="*92)
    print(f"M7 MULTI-SEED ROBUSTNESS — {len(split_seeds)} split x {len(prev_seeds)} prev = "
          f"{len(split_seeds)*len(prev_seeds)} combos/pi  ({report['elapsed_s']}s)")
    print(f"ANCHOR (released, split=7/prev=2024): "
          f"8/6/2/0  (pi=0.05/0.10/0.20/0.50)  [reproduced exactly]")
    print(f"\nJoint-cell count (of 9) distribution over {len(split_seeds)*len(prev_seeds)} reseeds:")
    print(f"  {'pi':>5} {'mean':>6} {'std':>5} {'min':>4} {'p5':>4} {'med':>4} {'p95':>4} {'max':>4}  "
          f"{'P(>=1 cell)':>11}")
    for pi in PIS:
        s = report["per_pi"][str(pi)]["joint_cells"]; f1 = report["per_pi"][str(pi)]["frac_at_least_one_joint"]
        print(f"  {pi:>5} {s['mean']:>6} {s['std']:>5} {int(s['min']):>4} {int(s['p5']):>4} "
              f"{int(s['median']):>4} {int(s['p95']):>4} {int(s['max']):>4}  {f1:>11}")
    h = report["headline_cell_pi0.10"]
    print(f"\nHeadline cell P(True)-8B alpha=0.10 @ pi=0.10:")
    print(f"  certified in {h['certified_frac']*100:.1f}% of reseeds; "
          f"psi_lcb mean={h['psi_lcb']['mean']} [p5={h['psi_lcb']['p5']}, p95={h['psi_lcb']['p95']}] "
          f"(released 0.327); m_c mean={h['m_c']['mean']} (released 278)")
    d = report["decomposition_pi0.10"]
    print(f"\npi=0.10 joint-count decomposition: overall {d['overall_mean']}+-{d['overall_std']}; "
          f"between-split std {d['between_split_std_of_splitmeans']}, "
          f"between-prev std {d['between_prev_std_of_prevmeans']}")
    print(f"  wrote {out}")
    print("="*92)

if __name__ == "__main__":
    main()
