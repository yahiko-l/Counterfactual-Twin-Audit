#!/usr/bin/env python
"""R071 — fixed-assignment companion + deterministic pair-loss leaf (SENSITIVITY).

Answers 2026-07-23 review Q1 (probability space of the global exact leaf) and 2026-07-22 (c)
(present the deterministic pair-loss leaf L_i as a parallel leaf). Prespecified in
refine-logs/analysis_plan_v3_amendment_20260723.md §9. RUN FROM REPO ROOT. CPU only.

At the 4 frozen headline thresholds, tabulate three global-rate (rho<=alpha) leaves:
  (E) EXACT arm-randomization leaf p_glob = P[Bin(K,alpha) <= S] (the released/Table-4 path);
  (Z) FIXED-arm-assignment bounded-loss Hoeffding-Bentkus companion, conditional on the archived
      seed's realized assignment z (run16 lem:companion (a-ii); Y_i = alpha + A_i(z_i-alpha) in [0,1]);
  (L) DETERMINISTIC pair-loss leaf L_i = pi(1-alpha)F_i - alpha(1-pi)T_i, with the exact algebraic
      identity E[L]<=0 <=> rho_pi<=alpha; bounded-loss HB on Y_i=(L_i-L_min)/range.
L is the arm-randomization EXPECTATION of the fixed-z loss (E_z[A_i(z_i-alpha)] = L_i), i.e. the
de-randomized parallel leaf. All three are conservative relative to (E) by construction; a headline
cell where a conservative leaf does not clear the nominal reference level is expected and reported.
SENSITIVITY: no budget consumed, Table 4 untouched. Self-check: the exact leaf certifies rho<=alpha
at all 4 cells (Table-4 ground truth); L's sign matches (rho_pi - alpha) at every cell.
"""
import os, sys, json, time, math
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
import m7_multiseed_robustness as M7

E = math.e
FAMS = M7.FAMS; POOLED = M7.POOLED; SPLIT_SEED = 7; PREV_SEED = 2024; PI = 0.10
FLABEL = {"score_sptrue": "P(True)-8B", "score_sNLI": "ref-NLI", "score_sjudge": "LLM-judge-8B"}
DELTA_REF = C.DELTA_GLOBAL   # 0.025 nominal single-cell reference (SENSITIVITY, not a budgeted family)
# 4 frozen headline cells: (family, exact tau, primary alpha)
CELLS = [("score_sptrue", 0.000123, 0.05), ("score_sptrue", 0.222824, 0.10),
         ("score_sjudge", 0.245085, 0.20), ("score_sNLI", 0.518631, 0.10)]

def hb_bounded_p(n, ybar, mu0):
    """One-sided Hoeffding-Bentkus lower-tail p for bounded [0,1] mean ybar under H0: E[Y] >= mu0."""
    if n <= 0 or ybar >= mu0: return 1.0
    pH = math.exp(-2.0 * n * (mu0 - ybar) ** 2)
    pB = E * float(binom.cdf(int(math.floor(n * ybar)), n, mu0))
    return float(min(1.0, pH, pB))

cache = M7.build_split_cache(POOLED, SPLIT_SEED)
ncal = cache["ncal"]
rng = np.random.default_rng(PREV_SEED); Z = (rng.random(ncal) < PI).astype(int)   # archived assignment

rows = []; selfcheck_err = []
for fam, tau, alpha in CELLS:
    rTc = cache["rTc"][fam]; rHc = cache["rHc"][fam]
    T = (rTc <= tau).astype(int); F = (rHc <= tau).astype(int)          # per-pair accept indicators
    A = np.where(Z == 1, F, T)                                          # stream accept at archived z
    K = int(A.sum()); S = int((A * Z).sum())                           # accepted / accepted-hallucinated
    rho_hat_stream = S / K if K else None
    # (E) exact arm-randomization leaf
    p_exact = float(binom.cdf(S, K, alpha)) if K > 0 else 1.0
    # (Z) fixed-assignment bounded-loss HB companion: Y_i = alpha + A_i(z_i - alpha) in [0,1], mu0 = alpha
    Yz = alpha + A * (Z - alpha); p_fixedz = hb_bounded_p(ncal, float(Yz.mean()), alpha)
    # (L) deterministic pair-loss leaf
    fhat = float(F.mean()); that = float(T.mean())
    Lbar = PI * (1 - alpha) * fhat - alpha * (1 - PI) * that
    rho_pi = (PI * fhat) / (PI * fhat + (1 - PI) * that) if (PI * fhat + (1 - PI) * that) > 0 else None
    Lmin = -alpha * (1 - PI); Lmax = PI * (1 - alpha); Rng = Lmax - Lmin
    L = PI * (1 - alpha) * F - alpha * (1 - PI) * T
    Ybar = float(((L - Lmin) / Rng).mean()); mu0 = (0 - Lmin) / Rng    # E[L]<=0 <=> E[Y]<=mu0
    p_detL = hb_bounded_p(ncal, Ybar, mu0)
    # identity + de-randomization checks
    ED_over_z = float((PI * (1 - alpha) * F - alpha * (1 - PI) * T).sum())   # E_z[sum A_i(z_i-alpha)] = sum L_i
    sumL = float(L.sum())
    L_sign_ok = ((Lbar <= 0) == (rho_pi is not None and rho_pi <= alpha))
    if not L_sign_ok: selfcheck_err.append(f"{FLABEL[fam]}@{tau}: L-sign != (rho_pi<=alpha)")
    if p_exact > DELTA_REF: selfcheck_err.append(f"{FLABEL[fam]}@{tau}: exact leaf p={p_exact:.2e} does NOT certify (>{DELTA_REF})")
    rows.append(dict(family=FLABEL[fam], tau=tau, alpha=alpha, K=K, S=S,
                     rho_hat_stream=(round(rho_hat_stream, 4) if rho_hat_stream is not None else None),
                     rho_pi_deterministic=(round(rho_pi, 4) if rho_pi is not None else None),
                     fhat=round(fhat, 4), that=round(that, 4), Lbar=round(Lbar, 5),
                     p_exact=p_exact, p_fixedz_HB=p_fixedz, p_detL_HB=p_detL,
                     certifies_exact=bool(p_exact <= DELTA_REF), certifies_fixedz=bool(p_fixedz <= DELTA_REF),
                     certifies_detL=bool(p_detL <= DELTA_REF),
                     identity_Lbar_le0_iff_rhopi_le_alpha=bool(L_sign_ok),
                     derandomization_sumL_eq_Ez_sumD=bool(abs(sumL - ED_over_z) < 1e-9)))

out = dict(run="R071: fixed-assignment companion + deterministic pair-loss leaf (SENSITIVITY; Q1 / 2026-07-22(c))",
           amendment="analysis_plan_v3_amendment_20260723.md §9", note=(
             "Three global-rate leaves at the 4 frozen headline cells: (E) exact arm-randomization "
             "binomial (Table-4 path), (Z) fixed-assignment bounded-loss HB companion, (L) deterministic "
             "pair-loss L_i=pi(1-a)F_i-a(1-pi)T_i (E[L]<=0 iff rho_pi<=a; = E_z of the fixed-z loss). "
             "(Z)/(L) are conservative vs (E) by construction; SENSITIVITY only, Table 4 unchanged."),
           delta_ref=DELTA_REF, pi=PI, ncal=ncal, cells=rows,
           selfcheck=("PASS" if not selfcheck_err else selfcheck_err))
if selfcheck_err:
    print("SELF-CHECK FAILED:"); [print("  -", x) for x in selfcheck_err]; sys.exit(1)
json.dump(out, open("results/R071_results.json", "w"), indent=2, default=str)

print("=" * 100)
print("R071 — global-rate leaves at the 4 frozen headline cells (SENSITIVITY; conservative vs exact)")
print(f"{'cell':22s} {'rho_hat':>8s} {'rho_pi':>7s} {'Lbar':>9s} {'p_exact':>10s} {'p_fixedzHB':>11s} {'p_detL_HB':>11s}  cert(E/Z/L)")
for r in rows:
    print(f"{r['family']+'@'+str(r['tau']):22s} {str(r['rho_hat_stream']):>8s} {str(r['rho_pi_deterministic']):>7s} "
          f"{r['Lbar']:>9.5f} {r['p_exact']:>10.2e} {r['p_fixedz_HB']:>11.2e} {r['p_detL_HB']:>11.2e}  "
          f"{int(r['certifies_exact'])}/{int(r['certifies_fixedz'])}/{int(r['certifies_detL'])}")
print(f"\nidentity L<=0 <=> rho_pi<=alpha : {all(r['identity_Lbar_le0_iff_rhopi_le_alpha'] for r in rows)}")
print(f"de-randomization sum L = E_z[sum A(z-a)] : {all(r['derandomization_sumL_eq_Ez_sumD'] for r in rows)}")
print(f"self-check: {out['selfcheck']}")
print("  wrote results/R071_results.json")
print("=" * 100)
