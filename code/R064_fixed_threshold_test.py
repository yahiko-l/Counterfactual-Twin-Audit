#!/usr/bin/env python
"""R064 — fixed-threshold held-out test-set reporting (reviewer Q2; PRESPECIFIED in
refine-logs/analysis_plan_v2_20260717.md Section 5.4).

For the three distinct operating thresholds that survive BOTH delta_total budgets
(T2: P(True) tau=0.222824; T3: LLM-judge tau=0.245085; T4: ref-NLI tau=0.518631), report on
D_test — completely, without any threshold re-selection: m_test, s_test, psi_hat_test, the exact
two-sided 95% Clopper–Pearson interval for the population psi_g (i.i.d. source-question reading),
the heterogeneity-robust Hoeffding–Bentkus 95% LCB for the realized test-set mean psi-bar_g,test,
the induced-stream global rate and coverage at pi=0.10 (documented fresh seed), and the
rejected-truthful mass. Locked disclosure: this is a held-out, fixed-threshold SENSITIVITY with
complete reporting of all frozen operating points — NOT a preregistered confirmation, because the
prior manuscript already descriptively inspected a P(True) test result. No new certificates, no
multiplicity spending.

Output: R064_fixed_threshold_test_results.json (committable).
"""
import os, sys, json, csv, math
import numpy as np
from scipy.stats import beta as sbeta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C


def hb_lcb(s_c, m_c, level):
    """HB lower confidence bound on the selected-set mean (inlined from m6_robust_leaf.hb_lcb,
    which cannot be imported because that module loads data at import time)."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0 or s_c <= 0:
        return 0.0
    shat = s_c / m_c
    lcbH = shat - math.sqrt(math.log(1.0 / level) / (2.0 * m_c))
    lcbB = C.cp_lower_psi(s_c, m_c, level / math.e)
    return float(max(lcbH, lcbB, 0.0))

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
LAB = artifact("scores_strong.csv"); ART = artifact("scores_strong_artificial.csv")
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
NICE = {"score_sptrue": "P(True)-8B", "score_sjudge": "LLM-judge-8B", "score_sNLI": "ref-NLI"}
THRESH = [("score_sptrue", 0.222824, "T2"), ("score_sjudge", 0.245085, "T3"), ("score_sNLI", 0.518631, "T4")]
PI = 0.10; SPLIT_SEED = 7; GAMMA = C.GAMMA_DEFAULT
TEST_PREV_SEED = 20260718   # documented fresh arm seed for the descriptive test-stream read
EXPECT_CAL = {("score_sptrue", 0.222824): (278, 120), ("score_sjudge", 0.245085): (347, 198),
              ("score_sNLI", 0.518631): (488, 138)}


def cp2(s, m, conf=0.95):
    lo = 0.0 if s == 0 else float(sbeta.ppf((1 - conf) / 2, s, m - s + 1))
    hi = 1.0 if s == m else float(sbeta.ppf(1 - (1 - conf) / 2, s + 1, m - s))
    return [round(lo, 4), round(hi, 4)]


def main():
    rows = ([dict(r, source="pqa_labeled") for r in csv.DictReader(open(LAB))] +
            [dict(r, source="pqa_artificial") for r in csv.DictReader(open(ART))])
    qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
    RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
    byq = {}
    for i in range(len(rows)): byq.setdefault(int(qid[i]), {})[int(H[i])] = i
    QIDS = np.array(sorted(byq))
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(QIDS)); n = len(QIDS)
    a, b = int(n * .40), int(n * .40) + int(n * .40)
    D_sel = set(QIDS[perm[:a]].tolist()); D_cal = set(QIDS[perm[a:b]].tolist())
    D_test = set(QIDS[perm[b:]].tolist())
    sel_idx = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
    cal_q = [q for q in QIDS if q in D_cal]; test_q = [q for q in QIDS if q in D_test]
    n_test = len(test_q)
    print(f"[data] twins={n}  ncal={len(cal_q)}  ntest={n_test}")

    out_cells = []
    for f, tau, tag in THRESH:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI = (-RAW[f] if fl else RAW[f])
        rTc = np.array([ORI[byq[q][0]] for q in cal_q]); rHc = np.array([ORI[byq[q][1]] for q in cal_q])
        mc, sc = C.psi_counts(rTc, rHc, tau)
        assert (mc, sc) == EXPECT_CAL[(f, tau)], ("cal self-check failed", f, tau, mc, sc)
        rTt = np.array([ORI[byq[q][0]] for q in test_q]); rHt = np.array([ORI[byq[q][1]] for q in test_q])
        m_t, s_t = C.psi_counts(rTt, rHt, tau)
        psi_t = s_t / m_t if m_t else None
        u = np.random.default_rng(TEST_PREV_SEED).random(n_test)
        takeH = u < PI; s_stream = np.where(takeH, rHt, rTt)
        acc = s_stream <= tau; cov = float(acc.mean())
        rho = float((takeH & acc).sum() / max(acc.sum(), 1))
        cell = dict(tag=tag, family=NICE[f], tau=tau,
                    cal_selfcheck=dict(m_c=int(mc), s_c=int(sc)),
                    m_test=int(m_t), s_test=int(s_t),
                    psi_hat_test=round(psi_t, 4) if psi_t is not None else None,
                    psi_pop_cp95=cp2(s_t, m_t),
                    psibar_test_hb_lcb95=round(float(hb_lcb(s_t, m_t, 0.05)), 4),
                    stream_rho_hat=round(rho, 4), stream_coverage=round(cov, 4),
                    mass=round(m_t / n_test, 4), mass_ge_gamma=bool(m_t / n_test >= GAMMA))
        out_cells.append(cell)
        print(f"  {tag} {NICE[f]:<13} tau={tau}: m_t={m_t} s_t={s_t} psi={cell['psi_hat_test']} "
              f"CP95={cell['psi_pop_cp95']} HB-LCB={cell['psibar_test_hb_lcb95']} "
              f"rho={cell['stream_rho_hat']} cov={cell['stream_coverage']} mass={cell['mass']}")

    out = dict(run="R064 fixed-threshold held-out test reporting (prespecified S5.4)",
               prereg="refine-logs/analysis_plan_v2_20260717.md S5.4",
               disclosure=("held-out, fixed-threshold sensitivity with complete reporting of all frozen "
                           "operating points; NOT a preregistered confirmation (the prior manuscript "
                           "descriptively inspected a P(True) test result)"),
               pi=PI, split_seed=SPLIT_SEED, test_prev_seed=TEST_PREV_SEED, n_test=n_test,
               cells=out_cells)
    with open(artifact("R064_fixed_threshold_test_results.json"), "w") as fo:
        json.dump(out, fo, indent=2)
    print("[wrote] R064_fixed_threshold_test_results.json")


if __name__ == "__main__":
    main()
