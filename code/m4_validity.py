#!/usr/bin/env python
"""
M4 — validity + label robustness (CPU; consumes data/scores/scores_strong.csv).
Maps EXPERIMENT_TRACKER R040 (B3) + R041 (B4 protocol/band) + R042 (already in m3_certify).

  B3 (R040): cluster-valid (one-answer-per-source) vs NAIVE iid-row ρ certification. Demonstrates
       iid-row OVER-certifies (anti-conservative: counts both correlated twin answers as independent
       ⇒ inflated effective n ⇒ certifies ρ≤α at τ where the cluster-valid certificate cannot).
       + γ-sensitivity of the min-mass floor; + α-sweep operating band of ψ_g.
  B4 (R041): blinded label-validity. We CANNOT run human adjudication here, so we (a) emit the
       pre-registered stratified ~300-twin blind sheet (shuffled, NO labels/scores), and (b) propagate
       a SIMULATED label-noise band η∈{0,.05,.10,.15} into the headline ψ_g lower certificate
       (worst-case: flip up to η of the failure pairs) to show the robustness envelope. The real
       human η replaces the simulated band later; results are explicitly marked PENDING-HUMAN.
"""
import os, sys, json, csv, time
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
CSV = os.environ.get("CSV", "data/scores/scores_strong.csv")
MAIN = ["score_sptrue", "score_selfcheck", "score_sjudge"]
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT
rows = list(csv.DictReader(open(CSV)))
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
diff = np.array([r["difficulty"] for r in rows]); cat = np.array([r.get("category", "") for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in MAIN}
byq = {}
for i in range(len(rows)): byq.setdefault(int(qid[i]), {})[int(H[i])] = i
QIDS = np.array(sorted(byq))
perm = np.random.default_rng(7).permutation(len(QIDS)); n = len(QIDS)
D_sel = set(QIDS[perm[:int(.4*n)]].tolist()); D_cal = set(QIDS[perm[int(.4*n):int(.8*n)]].tolist())
sel_idx = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
ORI = {}
for f in MAIN:
    _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
def twins(f, qset):
    s = ORI[f]; qs = [q for q in QIDS if q in qset]
    return np.array([s[byq[q][0]] for q in qs]), np.array([s[byq[q][1]] for q in qs])
report = {"run": "M4 validity + label band", "alphas": ALPHAS, "gamma": GAMMA}

# ===================== B3 — cluster-valid vs iid-row ρ certification (R040) ==================
log("B3 — cluster-valid (one-per-source) vs iid-row ρ certification")
b3 = {}
for f in MAIN:
    rT_s, rH_s = twins(f, D_sel); grid = C.candidate_grid(np.r_[rT_s, rH_s], gamma=GAMMA, n_grid=30)
    rT_c, rH_c = twins(f, D_cal); n_cal = len(rT_c)
    # iid-row: treat all 2*n_cal answers as independent (anti-conservative)
    s_iid = np.r_[rT_c, rH_c]; h_iid = np.r_[np.zeros(n_cal, int), np.ones(n_cal, int)]
    # cluster-valid: one answer per source
    pickH = np.random.default_rng(0).random(n_cal) < 0.5
    s_ops = np.where(pickH, rH_c, rT_c); h_ops = pickH.astype(int)
    b3[f] = {}
    for a in ALPHAS:
        def n_cert(scores, h):
            m = len(grid); c = 0
            for tau in grid:
                acc = scores <= tau; K = int(acc.sum())
                if K > 0 and float(binom.cdf(int(h[acc].sum()), K, a)) <= C.DELTA_GLOBAL/m: c += 1
            return c
        ci = n_cert(s_iid, h_iid); cc = n_cert(s_ops, h_ops)
        b3[f][a] = dict(iid_row_certified_taus=ci, cluster_valid_certified_taus=cc,
                        iid_over_certifies=bool(ci > cc))
report["B3_cluster_vs_iid"] = b3
# γ-sensitivity (min-mass floor) on the strong judge at α=.10
f = "score_sjudge"; rT_c, rH_c = twins(f, D_cal); n_cal = len(rT_c)
grid = C.candidate_grid(np.r_[twins(f, D_sel)[0], twins(f, D_sel)[1]], gamma=GAMMA, n_grid=30)
gs = {}
for g in (0.02, 0.05, 0.10, 0.15):
    cnt = 0
    for tau in grid:
        m_c, _ = C.psi_counts(rT_c, rH_c, tau)
        if C.minmass_p(m_c, n_cal, g) <= C.DELTA_MASS: cnt += 1
    gs[g] = cnt
report["B3_gamma_sensitivity_judge_a10"] = {str(k): v for k, v in gs.items()}

# ===================== B4 — blind sheet + SIMULATED label-noise band (R041) ==================
log("B4 — emit stratified blind sheet + simulated η-band on ψ_g lower certificate")
# pre-registered stratified ~300-twin sample (difficulty × hallucination category) on D_cal-source qs
cal_qs = [q for q in QIDS if q in D_cal]
strata = {}
for q in cal_qs:
    d = diff[byq[q][1]]; c = cat[byq[q][1]] or "NA"; strata.setdefault((d, c), []).append(int(q))
rng = np.random.default_rng(11); target = 300; pick = []
keys = list(strata);
per = max(1, target // max(1, len(keys)))
for k in keys:
    qs = strata[k]; rng.shuffle(qs); pick += qs[:per]
pick = pick[:target]
# blind sheet: shuffled rows, NO H / NO score (adjudicator sees only Q + the two answers in random order)
import csv as _csv
sheet = "results/B4_blind_sheet.csv"
with open(sheet, "w", newline="") as fh:
    w = _csv.writer(fh); w.writerow(["item_id", "qid", "answer_A", "answer_B", "which_is_hallucinated(blank for human)"])
    # we only have scores here, not the answer TEXT; emit qids + slots (text joined later from MedHallu)
    for it, q in enumerate(pick): w.writerow([it, q, "<answerA_text>", "<answerB_text>", ""])
report["B4_blind_sheet"] = {"path": sheet, "n_twins": len(pick), "strata": len(keys),
                            "note": "PENDING-HUMAN: adjudicator fills column 5 blind to MedHallu labels; "
                                    "answer text to be joined from MedHallu before handing out."}
# simulated band: worst-case ψ_g lower-CB when up to η of failure pairs are label-flipped (s_c -> s_c*(1-η))
f = "score_sjudge"; rT_c, rH_c = twins(f, D_cal); n_cal = len(rT_c)
grid = C.candidate_grid(np.r_[twins(f, D_sel)[0], twins(f, D_sel)[1]], gamma=GAMMA, n_grid=30)
band = {}
for eta in (0.0, 0.05, 0.10, 0.15):
    best = None
    for a in ALPHAS:
        for tau in grid:
            m_c, s_c = C.psi_counts(rT_c, rH_c, tau)
            s_wc = int(np.floor(s_c * (1 - eta)))                  # worst-case: η of failures were label noise
            if m_c > 0 and C.minmass_p(m_c, n_cal, GAMMA) <= C.DELTA_MASS:
                p = C.lower_p_psi(s_wc, m_c, a)
                if p <= C.DELTA_LOWER / (len(grid)*len(ALPHAS)*len(MAIN)):
                    lcb = C.cp_lower_psi(s_wc, m_c, C.DELTA_LOWER/(len(grid)*len(ALPHAS)*len(MAIN)))
                    if best is None or lcb > best[0]: best = (lcb, a, round(float(tau), 4), m_c, s_wc)
    band[str(eta)] = (dict(psi_lcb=round(best[0], 4), alpha=best[1], tau=best[2], m_c=best[3], s_c_wc=best[4])
                      if best else {"psi_lcb": None, "note": "no certified cell at this η"})
report["B4_simulated_label_noise_band_judge"] = band
report["B4_K3_note"] = ("K3: ψ_g>α failure " + ("SURVIVES" if band.get("0.1", {}).get("psi_lcb") else "VANISHES")
                        + " a simulated η=0.10 label-noise band (worst-case). Replace with real human η.")

report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m4_results.json", "w"), indent=2, default=str)
print("\n" + "="*78)
print("M4 VALIDITY")
print("B3 cluster-valid vs iid-row (certified-τ counts; iid over-certifies if >):")
for f in MAIN:
    print(f"  {f:16s} " + "  ".join(f"α={a}: iid={b3[f][a]['iid_row_certified_taus']} clu={b3[f][a]['cluster_valid_certified_taus']}" for a in ALPHAS))
print(f"B3 γ-sensitivity (judge,α=.10) certified-mass τ count: {report['B3_gamma_sensitivity_judge_a10']}")
print(f"B4 simulated η-band (judge) ψ_g lower-CB: " + ", ".join(f"η={k}:{v.get('psi_lcb')}" for k, v in band.items()))
print(f"  {report['B4_K3_note']}")
print("  wrote results/m4_results.json + B4_blind_sheet.csv")
print("="*78)
