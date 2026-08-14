#!/usr/bin/env python
"""
M1 KILL-GATE PREFLIGHT — certify ψ_g>α on EXISTING pilot scores (CPU, ZERO new GPU).
Reuses idea-stage/pilot/scores.csv. Maps EXPERIMENT_TRACKER R010–R013.

  R010 build the ψ_g certifier on the 3 main score families (ptrue / nli / judge-proxy),
       group-split by SOURCE question into D_sel (freeze C) / D_cal (certify) / D_test (descriptive)
  R011 K1 : does a certified ψ_g>α separation EXIST?  (exact conditional-binomial lower
       certificate, Holm over the D_sel-frozen δ_lower candidates, min-mass δ_mass denominator)
  R012 K2 : does the judge-proxy keep a certifiable non-trivial denominator P̂(T_τ=0)≥γ
       across the α-sweep (incl. at the strict global-ρ-certified operating τ̂)?
  R013 6-family appendix diagnostic (nli/kov/tfidf/judge/ptrue/sement)

STOP/GO (EXPERIMENT_TRACKER §Stop/Go):
  K1 GO ⇔ ≥1 (family,α) cell with certified ψ_g>α on a min-mass-certified denominator.
  K2 GO ⇔ judge-proxy retains a min-mass-certified denominator at the operating band.
  K1 GO ∧ K2 GO ⇒ GPU milestones M2/M3 authorized (NOT headline evidence — pilot judge is a
  Qwen2.5-1.5B proxy; the true strong judge is M3 + K4). Either fails ⇒ feasibility/negative
  note, M2–M5 KILLED, do NOT burn 8–24 GPU-day.
"""
import os, sys, json, csv, time
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
CSV = os.environ.get("CSV", "idea-stage/pilot/scores.csv")
MAIN = ["score_ptrue", "score_nli", "score_judge"]      # weak-UQ / answer-NLI / strong-judge proxy
ALL6 = ["score_nli", "score_kov", "score_tfidf", "score_judge", "score_ptrue", "score_sement"]
ALPHAS = [0.05, 0.10, 0.20]
GAMMA = C.GAMMA_DEFAULT

# --------------------------------------------------------------------- load + split + orient
rows = list(csv.DictReader(open(CSV)))
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in ALL6}
# pair integrity — FAIL CLOSED (a duplicate or half twin would silently corrupt ψ_g counts)
byq = {}
for i in range(len(rows)):
    q, h = int(qid[i]), int(H[i])
    assert (q not in byq) or (h not in byq[q]), f"duplicate (qid={q}, H={h}) row"
    byq.setdefault(q, {})[h] = i
assert len(rows) == 2 * len(byq), f"{len(rows)} rows != 2×{len(byq)} qids"
for q, d in byq.items():
    assert set(d) == {0, 1}, f"qid {q} is not a complete truthful+hallucinated twin: {set(d)}"
QIDS = np.array(sorted(byq))

def group_split(fracs=(0.40, 0.40, 0.20), seed=7):
    """Group-split SOURCE questions (twin = atomic unit, never crosses folds)."""
    perm = np.random.default_rng(seed).permutation(len(QIDS))
    n = len(QIDS); a = int(n*fracs[0]); b = a + int(n*fracs[1])
    return (set(QIDS[perm[:a]].tolist()), set(QIDS[perm[a:b]].tolist()), set(QIDS[perm[b:]].tolist()))
D_sel, D_cal, D_test = group_split()

# ORIENTATION LOCK on D_sel ONLY (candidate/score direction must be D_cal-independent;
# using D_cal/D_test labels to pick the flip would leak calibration labels — codex CRITICAL).
sel_idx = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
ORI = {}; FLIP = {}
for f in ALL6:
    _, fl = C.orient(H[sel_idx], RAW[f][sel_idx])      # decide sign on D_sel answers only
    ORI[f] = (-RAW[f] if fl else RAW[f]); FLIP[f] = bool(fl)

def twins(fam, qset):
    """(r_T, r_H) arrays over the source questions in qset, in locked risk orientation."""
    s = ORI[fam]; qs = [q for q in QIDS if q in qset]
    rT = np.array([s[byq[q][0]] for q in qs]); rH = np.array([s[byq[q][1]] for q in qs])
    return rT, rH
def answers(fam, qset):
    """(scores, H) over individual answers (both twins) — for the global-ρ iid-row preview."""
    s = ORI[fam]; idx = [byq[q][h] for q in QIDS if q in qset for h in (0, 1)]
    return s[np.array(idx)], H[np.array(idx)]

log(f"loaded {len(rows)} rows / {len(QIDS)} twins; split |D_sel|={len(D_sel)} |D_cal|={len(D_cal)} |D_test|={len(D_test)}")
report = {"run": "M1 kill-gate preflight", "delta": {"total": C.DELTA_TOTAL, "lower": C.DELTA_LOWER,
          "mass": C.DELTA_MASS, "psiup": C.DELTA_PSIUP, "global": C.DELTA_GLOBAL}, "gamma": GAMMA,
          "alphas": ALPHAS, "split": {"sel": len(D_sel), "cal": len(D_cal), "test": len(D_test)},
          "orientation": {f: {"flipped": bool(FLIP[f]), "risk_auc": round(float(roc_auc_score(H, ORI[f])), 4)} for f in ALL6},
          "note_judge": "judge-proxy = Qwen2.5-1.5B (M1 only); strong judge ≥7-8B/API is M3+K4. M1 GO is necessary-not-sufficient."}

# ============================ R010 freeze candidate set C on D_sel (D_cal-independent) =====
# Per main family: operating-band τ-grid (frozen on D_sel) + global-ρ-certified τ̂(α) preview.
CAND = {}                       # fam -> list of dict(tau, source)
RHO_PREVIEW = {}                # fam -> {alpha: tau_hat or None}  (iid-row PREVIEW; cluster-valid is M2)
for f in MAIN:
    rT_sel, rH_sel = twins(f, D_sel)
    grid = C.candidate_grid(np.r_[rT_sel, rH_sel], gamma=GAMMA, n_grid=24)
    cand = [{"tau": float(t), "source": "grid"} for t in grid]
    sc_sel, h_sel = answers(f, D_sel)
    RHO_PREVIEW[f] = {}
    for a in ALPHAS:
        th, cov = C.rho_ltt_tau(sc_sel, h_sel, a, C.DELTA_GLOBAL, grid)
        RHO_PREVIEW[f][a] = (None if th is None else float(th))
        if th is not None and all(abs(th - cc["tau"]) > 1e-12 for cc in cand):
            cand.append({"tau": float(th), "source": f"rho_tauhat_a{a}"})
    CAND[f] = sorted(cand, key=lambda d: d["tau"])
report["rho_preview_tauhat"] = RHO_PREVIEW

# ============================ R011 K1 — certify ψ_g>α (Holm over δ_lower + min-mass δ_mass) ==
# Build the GLOBAL δ_lower candidate pool (every displayed ψ_g>α claim) and the δ_mass pool.
lower_items = []   # (fam, alpha, tau, m_c, s_c, p)
mass_items  = []   # (fam, tau, m_c, s_c, p)  -- denominator non-triviality, computed once per (fam,tau)
mass_index  = {}
for f in MAIN:
    rT_cal, rH_cal = twins(f, D_cal); n_cal = len(rT_cal)
    for cc in CAND[f]:
        tau = cc["tau"]; m_c, s_c = C.psi_counts(rT_cal, rH_cal, tau)
        mass_index[(f, tau)] = len(mass_items)
        mass_items.append([f, tau, m_c, s_c, C.minmass_p(m_c, n_cal, GAMMA)])
        for a in ALPHAS:
            lower_items.append([f, a, tau, m_c, s_c, C.lower_p_psi(s_c, m_c, a)])
# Holm within each δ-family over its full candidate pool
lower_rej = C.holm_reject([it[5] for it in lower_items], C.DELTA_LOWER)
mass_rej  = C.holm_reject([it[4] for it in mass_items],  C.DELTA_MASS)
mass_ok   = {(mass_items[i][0], mass_items[i][1]): bool(mass_rej[i]) for i in range(len(mass_items))}
n_lower = len(lower_items)
bonf_level = C.DELTA_LOWER / max(1, n_lower)     # conservative simultaneous level for reported CP-LCB

# Assemble per-(family,α) cells: certified ⇔ ∃τ with Holm-rejected ψ_g>α AND min-mass-certified denom
cells = {}
for f in MAIN:
    cells[f] = {}
    for a in ALPHAS:
        best = None
        for k, it in enumerate(lower_items):
            if it[0] != f or it[1] != a: continue
            fam, al, tau, m_c, s_c, p = it
            certified = bool(lower_rej[k]) and mass_ok.get((f, tau), False)
            psi_hat = (s_c / m_c) if m_c > 0 else float("nan")
            lcb = C.cp_lower_psi(s_c, m_c, bonf_level)          # simultaneous CP lower bound on ψ_g
            cand_rec = dict(tau=round(tau, 6), m_c=m_c, s_c=s_c,
                            psi_hat=(round(psi_hat, 4) if m_c > 0 else None),
                            psi_lcb=round(lcb, 4), denom_mass=round(m_c/len(twins(f, D_cal)[0]), 4),
                            lower_p=p, certified=certified)
            # prefer a certified candidate with the largest certified lower-CB
            key = (1 if certified else 0, lcb if certified else -1, psi_hat if m_c > 0 else -1)
            if best is None or key > best[0]:
                best = (key, cand_rec)
        cells[f][a] = best[1] if best else None
report["K1_cells"] = cells
n_cert = sum(1 for f in MAIN for a in ALPHAS if cells[f][a] and cells[f][a]["certified"])
K1_GO = n_cert >= 1
report["K1"] = {"certified_cells": n_cert, "total_cells": len(MAIN)*len(ALPHAS), "GO": bool(K1_GO),
                "bonferroni_lcb_level": bonf_level, "n_lower_candidates": n_lower}

# Headline JOINT-cell PREVIEW: at the global-ρ τ̂(α) (iid-row PREVIEW, anti-conservative; the
# cluster-valid ρ certificate is an M2/B3 deliverable), is ψ_g>α also certified? rho_preview_passed
# is NOT a δ_global certificate — it only records whether a candidate-safe operating point exists.
joint = {}
for f in MAIN:
    joint[f] = {}
    for a in ALPHAS:
        th = RHO_PREVIEW[f][a]
        if th is None: joint[f][a] = {"rho_preview_passed": False}; continue
        rT_cal, rH_cal = twins(f, D_cal); m_c, s_c = C.psi_counts(rT_cal, rH_cal, th)
        certified = False
        for k, it in enumerate(lower_items):
            if it[0]==f and it[1]==a and abs(it[2]-th)<1e-9:
                certified = bool(lower_rej[k]) and mass_ok.get((f, th), False); break
        joint[f][a] = {"rho_preview_passed": True, "tau_hat": round(th,6), "m_c": m_c, "s_c": s_c,
                       "psi_hat": (round(s_c/m_c,4) if m_c>0 else None),
                       "psi_gt_alpha_certified": certified}
report["joint_cell_preview"] = joint

# ============================ R012 K2 — judge-proxy denominator mass across α-sweep =========
f = "score_judge"; rT_cal, rH_cal = twins(f, D_cal); n_cal = len(rT_cal)
k2_band = []
for cc in CAND[f]:
    tau = cc["tau"]; m_c, s_c = C.psi_counts(rT_cal, rH_cal, tau)
    k2_band.append(dict(tau=round(tau,6), denom_mass=round(m_c/n_cal,4), m_c=m_c,
                        minmass_certified=mass_ok.get((f, tau), False), source=cc["source"]))
# at the strict global-ρ operating τ̂(α): does the judge keep denom ≥ γ?
k2_at_tauhat = {}
for a in ALPHAS:
    th = RHO_PREVIEW[f][a]
    if th is None: k2_at_tauhat[a] = {"rho_preview_passed": False}; continue
    m_c, s_c = C.psi_counts(rT_cal, rH_cal, th)
    k2_at_tauhat[a] = {"tau_hat": round(th,6), "denom_mass": round(m_c/n_cal,4), "m_c": m_c,
                       "minmass_certified": mass_ok.get((f, th), False),   # Holm-valid mass claim
                       "lcb_rate_pointwise": round(C.cp_lower_rate(m_c, n_cal, C.DELTA_MASS), 4)}  # descriptive only
judge_band_ok = any(b["minmass_certified"] for b in k2_band)
judge_tauhat_ok = any(v.get("minmass_certified") for v in k2_at_tauhat.values())
K2_GO = bool(judge_band_ok)     # judge keeps a min-mass-certified denominator in the operating band
report["K2"] = {"judge_band": k2_band, "judge_at_rho_tauhat": k2_at_tauhat,
                "denom_certified_in_band": judge_band_ok, "denom_certified_at_tauhat": judge_tauhat_ok,
                "GO": K2_GO}

# ============================ R013 6-family appendix diagnostic ============================
appx = {}
for f in ALL6:
    rT_c, rH_c = twins(f, D_cal); n_c = len(rT_c)
    grid = C.candidate_grid(np.r_[twins(f, D_sel)[0], twins(f, D_sel)[1]], gamma=GAMMA, n_grid=24)
    best = {"max_psi_hat_massok": None, "tau": None, "m_c": None, "s_c": None}
    for tau in grid:
        m_c, s_c = C.psi_counts(rT_c, rH_c, tau)
        if m_c > 0 and (m_c/n_c) >= GAMMA:
            ph = s_c/m_c
            if best["max_psi_hat_massok"] is None or ph > best["max_psi_hat_massok"]:
                best = {"max_psi_hat_massok": round(ph,4), "tau": round(float(tau),6), "m_c": m_c, "s_c": s_c}
    rT_all, rH_all = twins(f, set(QIDS.tolist()))
    appx[f] = dict(flipped=bool(FLIP[f]), risk_auc=round(float(roc_auc_score(H, ORI[f])),4),
                   paired_acc=round(C.paired_acc(rT_all, rH_all),4),
                   eta_rH_lt_rT=round(float(np.mean(rH_all < rT_all)),4), **best)
report["R013_appendix_6family"] = appx

# ============================ D_test descriptive confirmation (NOT in budget) ===============
desc = {}
for f in MAIN:
    rT_t, rH_t = twins(f, D_test); desc[f] = {}
    for a in ALPHAS:
        cc = cells[f][a]
        if cc and cc["certified"]:
            m_c, s_c = C.psi_counts(rT_t, rH_t, cc["tau"])
            desc[f][a] = dict(tau=cc["tau"], m_c=m_c, s_c=s_c,
                              psi_hat=(round(s_c/m_c,4) if m_c>0 else None))
report["D_test_descriptive"] = desc

# ----------------------------------------------------------------------------- verdict
report["GATE"] = {"K1_GO": bool(K1_GO), "K2_GO": bool(K2_GO),
                  "BOTH_GO": bool(K1_GO and K2_GO),
                  "decision": ("GO M2/M3 (GPU authorized; necessary-not-sufficient)"
                               if (K1_GO and K2_GO) else
                               "STOP — feasibility/negative note; M2–M5 KILLED; do NOT burn GPU")}
report["elapsed_s"] = round(time.time()-T0, 1)
os.makedirs("results", exist_ok=True)
json.dump(report, open("results/m1_results.json", "w"), indent=2, default=str)

# ---- ψ̂_g lower-CB vs α table (3 main families) → markdown fragment for EXPERIMENT_RESULTS
lines = []
lines.append("| Family | α | best τ | m_c (denom) | s_c (fail) | denom mass | ψ̂_g | ψ_g lower-CB | ψ_g>α certified | global ρ≤α gate (iid preview) |")
lines.append("|--------|---|--------|-------------|------------|------------|------|--------------|-----------------|-------------------------------|")
fam_disp = {"score_ptrue": "P(True)/weak-UQ", "score_nli": "answer-NLI", "score_judge": "judge-proxy(1.5B)"}
for f in MAIN:
    for a in ALPHAS:
        cc = cells[f][a]
        if cc is None: continue
        rho_flag = "no safe gate (ρ✗)" if not joint[f][a].get("rho_preview_passed") else "preview ρ✓"
        lines.append(f"| {fam_disp[f]} | {a:.2f} | {cc['tau']:.4f} | {cc['m_c']} | {cc['s_c']} | "
                     f"{cc['denom_mass']:.3f} | {('%.3f'%cc['psi_hat']) if cc['psi_hat'] is not None else 'NA'} | "
                     f"{cc['psi_lcb']:.3f} | {'CERT' if cc['certified'] else 'no'} | {rho_flag} |")
open("results/m1_table.md", "w").write("\n".join(lines) + "\n")

print("\n" + "="*82)
print(f"M1 PREFLIGHT — K1 GO={K1_GO} ({n_cert}/{len(MAIN)*len(ALPHAS)} cells)  K2 GO={K2_GO}  "
      f"⇒ BOTH_GO={K1_GO and K2_GO}")
print("ψ̂_g lower-CB vs α (3 main families):\n")
print("\n".join(lines))
print("\nDecision:", report["GATE"]["decision"])
print("  wrote results/m1_results.json + m1_table.md")
print("="*82)
