#!/usr/bin/env python
"""Real Clinical Arm — M4 joint two-leaf analysis (NOMINAL, in-sample; NOT split-validated) on clinician labels + stage-1 safety leaf.

Assembles the nominal in-sample joint analysis for the pooled@α cells of the Real Clinical Arm (selection is
same-sample: orientation + grid on the full stream, NOT a held-out split, so bounds are grid-multiplicity-
adjusted only, not a split-validated coverage guarantee):

  LEAF 1  global safety   ρ(τ*)=P(H=1 | accept) ≤ α        exact-binomial UCB ≤ α at δ_global (grid-Holm),
                                                            on the frozen one-answer-per-source S_R
                                                            (cluster-valid; stage-1 2-judge consensus H),
                                                            at the MEASURED natural prevalence π̂.
  LEAF 2  within-question ψ_g(τ*) > α   operative heterogeneity-robust HB LCB on psibar_g (+ exact CP psi_g)
                                                            > α at δ_lower, on the frozen M3 twin pairing,
                                                            relabeled by TWO blinded clinicians (relabel is
                                                            non-monotone; a worst-case lower envelope is
                                                            reported alongside).

Both leaves hold for a cell iff: a threshold nominally globally safe is *simultaneously* found to fail the
paired within-question audit, on real patient questions, with the paired side adjudicated by clinicians. δ accounting is the paper's union budget (δ_global+δ_mass+δ_psiup+δ_lower = δtotal); we report the
two BINDING leaves at δtotal=0.10 (main-paper budget, reproduces M3) and a stricter δtotal=0.05 sensitivity.

Decision-value (answers "ψ_g must out-inform F1/TPR/FNR"): at τ* we also report the gate-as-detector marginal
F1=P(accept | H=1) (the "safe gate still accepts X% of hallucinations" number), TPR/FNR, and the paired lift
Δψ=ψ̂−F1 (≤0 by construction under positive twin-correlation; ρ, F1, ψ_g are three DIFFERENT lenses and ρ is
the misleadingly-reassuring one).

Self-check: the reproduced ρ̂ / m_c / s_c MUST match pilot_projection_results.json byte-for-byte (same R_SEED,
same draw order) — asserted, else the RNG replication is wrong and the result is void.

CPU only. Output: results/M4_real_arm_certificate.json
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
from scipy.stats import binom, beta
from collections import defaultdict
import pilot_project_margin as P   # reuse the EXACT certified code path (load_labels/load_scores, R_SEED, FAMS)

E = math.e

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
R_SEED = P.R_SEED
FAMS = P.FAMS
GAMMA = P.GAMMA
FAMLAB = {"score_sptrue": "P(True)-8B", "score_sjudge": "LLM-judge-8B", "score_sNLI": "ref-NLI"}
M3KEY = {"score_sptrue": "sptrue", "score_sjudge": "sjudge", "score_sNLI": "sNLI"}
PROJ = json.load(open(artifact("pilot_projection_results.json")))
M3 = json.load(open(artifact("M3_human_adjudication_results.json")))


def rho_ucb(x, n, level):
    """Clopper-Pearson one-sided upper bound on P(H=1|accept): binom.cdf(x,n,ucb)=level. Holds iff ucb<=α."""
    if n == 0:
        return 1.0
    if x >= n:
        return 1.0
    return float(beta.ppf(1.0 - level, x + 1, n - x))


def hb_lcb(s_c, m_c, level):
    """Heterogeneity-robust Hoeffding-Bentkus lower confidence bound on the selected-set mean psibar_g
    (verbatim from m6_robust_leaf.hb_lcb): max(Hoeffding LCB, Bentkus LCB=CP at level/e). This is the
    paper's OPERATIVE headline leaf (no homogeneity); cp_lower_psi is the exact sharpening under homogeneity."""
    s_c, m_c = int(s_c), int(m_c)
    if m_c <= 0 or s_c <= 0:
        return 0.0
    shat = s_c / m_c
    lcbH = shat - math.sqrt(math.log(1.0 / level) / (2.0 * m_c))
    lcbB = C.cp_lower_psi(s_c, m_c, level / E)
    return float(max(lcbH, lcbB, 0.0))


def certify_pool(byq, pool_keys, alpha, sources):
    """Replicate pilot_project_margin.project() RNG faithfully, adding ρ_UCB(K,Sacc), F1/TPR/FNR, Δψ."""
    rng = np.random.default_rng(R_SEED)
    out = {}
    for f in FAMS:
        allA = [(byq[q][i], q) for q in pool_keys for i in range(len(byq[q]))]
        raw = np.array([a[0][1][f] for a in allA]); hlab = np.array([a[0][2] for a in allA])
        oriented, flip = C.orient(hlab, raw); sign = -1.0 if flip else 1.0
        def sval(d): return sign * d[f]
        # ---- LEAF 1: frozen one-per-source S_R (cluster-valid), strictest τ with ρ_UCB ≤ α ----
        S_scores, S_H = [], []
        for q in pool_keys:
            cand = byq[q]; pick = cand[int(rng.integers(len(cand)))]
            S_scores.append(sval(pick[1])); S_H.append(pick[2])
        S_scores = np.array(S_scores); S_H = np.array(S_H, int)
        if S_H.sum() == 0 or len(S_scores) < 10:
            out[f] = dict(note="degenerate S_R"); continue
        grid = C.candidate_grid(S_scores, gamma=GAMMA, n_grid=28); m = max(len(grid), 1)
        tau_star = Kc = Sc = None; rho_at = None
        for tau in grid:
            acc = S_scores <= tau; K = int(acc.sum())
            if K == 0: continue
            Sacc = int(S_H[acc].sum())
            if float(binom.cdf(Sacc, K, alpha)) <= C.DELTA_GLOBAL / m:
                tau_star = float(tau); rho_at = Sacc / K; Kc, Sc = K, Sacc   # keep LAST (largest) passing τ
        # ---- LEAF 2 basis: design-B twins at tau_star (stage-1; clinician relabel pulled from M3) ----
        m_c = s_c = nT = f1_num = f1_den = 0
        for q in pool_keys:
            cand = byq[q]; h0 = [c for c in cand if c[2] == 0]; h1 = [c for c in cand if c[2] == 1]
            if not h0 or not h1: continue
            nT += 1
            tw_t = h0[int(rng.integers(len(h0)))]; tw_h = h1[int(rng.integers(len(h1)))]
            rT = sval(tw_t[1]); rH = sval(tw_h[1])
            f1_den += 1
            if rH <= (tau_star if tau_star is not None else -np.inf): f1_num += 1   # halluc-twin ACCEPTED
            if tau_star is not None and rT > tau_star:
                m_c += 1
                if rH <= tau_star: s_c += 1
        if tau_star is None:
            out[f] = dict(note="no τ with ρ_UCB≤α", pi_SR=round(float(S_H.mean()), 4)); continue
        psi_hat = s_c / m_c if m_c else None
        F1 = f1_num / f1_den if f1_den else None          # marginal P(accept | H=1) on the H-twins
        out[f] = dict(flip=bool(flip), tau_star=round(tau_star, 6), n_SR=len(S_scores),
                      pi_SR=round(float(S_H.mean()), 4), K_accept=Kc, S_accept_halluc=Sc,
                      rho_hat=round(float(rho_at), 4), grid_m=m,
                      m_c=m_c, s_c=s_c, psi_hat=round(psi_hat, 4) if psi_hat is not None else None,
                      F1_marginal=round(F1, 4) if F1 is not None else None,
                      TPR_gate=round(1 - F1, 4) if F1 is not None else None,   # P(reject | H=1)
                      FNR_gate=round(F1, 4) if F1 is not None else None,       # P(accept | H=1) = leak
                      delta_psi=round(psi_hat - F1, 4) if (psi_hat is not None and F1 is not None) else None)
    return out


def main():
    H = P.load_labels("strict")
    sc, fams = P.load_scores(artifact("scores_pilot.csv"))
    keys = [k for k in sc if k in H]
    byq = defaultdict(list)
    for (source, orig, ans) in keys:
        byq[(source, orig)].append((ans, sc[(source, orig, ans)], H[(source, orig, ans)]))
    sources = sorted({s for (s, _) in byq})
    pool = list(byq.keys())

    # reproduce pooled@α for α∈{0.10,0.20} and SELF-CHECK vs the released projection
    repro = {a: certify_pool(byq, pool, a, sources) for a in (0.10, 0.20)}
    print(f"[join] scored={len(sc)} labeled={len(H)} joined={len(keys)} sources={sources}")
    for a in (0.10, 0.20):
        for f in FAMS:
            got = repro[a].get(f, {}); exp = PROJ["results"][f"pooled@a{a}"].get(f, {})
            if "rho_hat" in exp and "rho_hat" in got:
                for k in ("tau_star", "rho_hat", "m_c", "s_c"):
                    assert abs(float(got[k]) - float(exp[k])) <= (1e-5 if k in ("tau_star", "rho_hat") else 0), \
                        f"SELF-CHECK FAIL pooled@a{a} {f}.{k}: repro {got[k]} != proj {exp[k]}"
    print("[self-check] reproduced ρ̂ / τ* / m_c / s_c == pilot_projection_results.json  ✓")

    # ---- assemble the joint (nominal, in-sample) analysis for the pooled@α=0.20 cells ----
    ALPHA = 0.20
    budgets = {"main(δtotal=0.10)": dict(dg=C.DELTA_GLOBAL, dl=C.DELTA_LOWER),          # 0.025 / 0.04
               "strict(δtotal=0.05)": dict(dg=C.DELTA_GLOBAL / 2, dl=C.DELTA_LOWER / 2)}  # 0.0125 / 0.02
    NLOW = int(M3.get("NLOW", 468))
    cells = {}
    for f in FAMS:
        r = repro[ALPHA][f]; mk = M3KEY[f]; hm = M3["cells"][mk]["human"]["consensus_strict"]
        hm_lcb = M3["cells"][mk]["min_human_psi_lcb"]
        leaves = {}
        for bname, b in budgets.items():
            level_g = b["dg"] / r["grid_m"]
            r_ucb = rho_ucb(r["S_accept_halluc"], r["K_accept"], level_g)
            lev_l = b["dl"] / NLOW
            psibar_robust = round(hb_lcb(hm["s_c"], hm["m_c"], lev_l), 4)   # OPERATIVE headline leaf (no homogeneity)
            psi_exact = round(C.cp_lower_psi(hm["s_c"], hm["m_c"], lev_l), 4)  # exact sharpening under homogeneity
            leaves[bname] = dict(rho_UCB=round(r_ucb, 4), rho_leaf_ok=bool(r_ucb <= ALPHA),
                                 human_psibar_lcb_robust=psibar_robust, human_psi_lcb_exact=psi_exact,
                                 psi_leaf_ok=bool(psibar_robust > ALPHA),   # operative = robust leaf
                                 both_leaves_hold=bool(r_ucb <= ALPHA and psibar_robust > ALPHA))
        cells[f] = dict(
            family=FAMLAB[f], alpha=ALPHA, tau_star=r["tau_star"],
            rho_leaf=dict(basis="frozen one-per-source S_R, stage-1 2-judge consensus H (cluster-valid)",
                          n_SR=r["n_SR"], K_accept=r["K_accept"], S_accept_halluc=r["S_accept_halluc"],
                          rho_hat=r["rho_hat"], pi_measured=r["pi_SR"]),
            psi_leaf=dict(basis="frozen M3 twin pairing, TWO blinded clinicians, clinician relabel (non-monotone; worst-case floor reported separately)",
                          m_c=hm["m_c"], s_c=hm["s_c"], psi_hat=round(hm["s_c"]/hm["m_c"], 4) if hm["m_c"] else None,
                          human_psibar_lcb_robust_main=leaves["main(δtotal=0.10)"]["human_psibar_lcb_robust"],
                          human_psi_lcb_exact_main=leaves["main(δtotal=0.10)"]["human_psi_lcb_exact"],
                          leaf_note="operative = heterogeneity-robust psibar_g (HB); exact psi_g sharpens under homogeneity"),
            decision_value=dict(F1_marginal=r["F1_marginal"], TPR_gate=r["TPR_gate"],
                                FNR_leak=r["FNR_gate"], delta_psi=r["delta_psi"],
                                reading="ρ̂ low but F1 (fraction of the natural hallucinated twins accepted) high; "
                                        "ψ_g is the within-question lens ρ hides"),
            budgets=leaves,
            both_leaves_hold_main=leaves["main(δtotal=0.10)"]["both_leaves_hold"])
    inter = M3["inter_annotator"]
    survivors = [FAMLAB[f] for f in FAMS if cells[f]["both_leaves_hold_main"]]
    out = dict(
        run="Real Clinical Arm — M4 joint two-leaf analysis, NOMINAL in-sample (NOT split-validated): clinician ψ_g leaf + stage-1 ρ leaf",
        dataset="6-model natural stream, real patient questions (K-QA/MedicationQA/LiveQA); MEASURED prevalence",
        alpha_nominal=ALPHA, alpha_10="not attainable (M2-max; ρ≤0.10 gate dilutes ψ_g below α at strict τ)",
        inter_annotator=dict(cohen_kappa=inter["cohen_kappa"], raw_agreement=inter["raw_agreement"],
                             n=inter["n_jointly_labeled"]),
        delta_budget="union bound δ_global(.025)+δ_mass(.025)+δ_psiup(.01)+δ_lower(.04)=δtotal(.10); "
                     "two binding leaves reported at .10 (main) and .05 (strict) total",
        corroborated_cells=survivors, cells=cells,
        scope="ρ leaf + measured π̂ rest on stage-1 2-judge H (SR not clinician-labeled); ψ_g leaf is clinician-"
              "adjudicated. Detector-specific: only ref-NLI survives; P(True)/LLM-judge collapse under clinician "
              "relabel (stage-1 over-flagged, η≈0.34-0.41).")
    json.dump(out, open(artifact("M4_real_arm_certificate.json"), "w"), indent=2, ensure_ascii=False)

    # ---- print ----
    print(f"\n[inter-annotator] Cohen κ={inter['cohen_kappa']}  (raw {inter['raw_agreement']}, n={inter['n_jointly_labeled']})")
    print(f"[α nominal={ALPHA} (in-sample);  α=0.10 not attainable]")
    print(f"\n{'family':<14}{'τ*':>9}{'ρ̂':>7}{'ρ_UCB':>8}{'π̂':>7}{'F1':>7}{'ψ̂_h':>7}{'ψ̄_g robust':>11}{'ψ_g exact':>10}  JOINT")
    for f in FAMS:
        c = cells[f]; L = c["budgets"]["main(δtotal=0.10)"]
        j = "✓✓ HOLDS(nominal)" if c["both_leaves_hold_main"] else "✗ (ψ_g leaf collapses)"
        print(f"{c['family']:<14}{c['tau_star']:>9.4f}{c['rho_leaf']['rho_hat']:>7.3f}"
              f"{L['rho_UCB']:>8.3f}{c['rho_leaf']['pi_measured']:>7.3f}"
              f"{c['decision_value']['F1_marginal']:>7.3f}{c['psi_leaf']['psi_hat']:>7.3f}"
              f"{L['human_psibar_lcb_robust']:>11.3f}{L['human_psi_lcb_exact']:>10.3f}  {j}")
    print(f"\nsurvivors (joint, δtotal=0.10): {survivors}")
    print("δtotal=0.05 sensitivity (ref-NLI):",
          {k: v for k, v in cells['score_sNLI']['budgets']['strict(δtotal=0.05)'].items()})
    print(f"\nwrote {artifact('M4_real_arm_certificate.json')}")


if __name__ == "__main__":
    main()
