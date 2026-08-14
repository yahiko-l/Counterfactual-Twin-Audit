#!/usr/bin/env python
"""R055 -- faithful published-abstention baselines: a marginally-certified-safe gate still leaks psi_g.

Answers AIM reviewer #7 ("no faithful abstention-baseline impl"). The decision-value theorem R050
proved that NO marginal detector metric identifies psi_g; this script makes that OPERATIONAL against
two REAL published selective-risk / abstention certificates: each method is calibrated by ITS OWN
rule (blind to psi_g), its marginal target is checked out-of-sample, and the within-question
psi_bar_g lower bound is then reported at each rule's coverage-maximising audit-eligible threshold (see the
DEPLOY-DOMAIN NOTE below). Zero new model runs --
reuses the released 10k headline scores exactly (scores_strong.csv + scores_strong_artificial.csv;
same seed-7 split, same candidate grid, same induced-pi one-answer-per-source deployment stream, same
heterogeneity-robust Hoeffding-Bentkus psi_bar_g leaf as m6_robust_leaf.py). MedHalu is NOT used
(its scores are request-gated / gitignored, so a committable-reproducible artifact cannot ship them).

DEPLOY-DOMAIN NOTE (important for honest reading): the coverage-max threshold below is the
coverage-maximiser over the gamma-eligible audit grid (certify.candidate_grid keeps only thresholds whose
D_sel POOLED rejection rate >= gamma=0.05); a cell is counted AUDIT-ELIGIBLE only when that selected
threshold in turn clears the rejected-TRUTHFUL min-mass floor (so a within-question audit is well-posed).
Over the UNRESTRICTED domain a marginal rule at alpha>=pi would instead pick near-accept-all (rho ~ pi is
trivially safe, m_c -> 0, psi_g undefined) -- the low-prevalence vacuousness the audit exists to expose. So
the claim is NOT "the method's unrestricted deploy point leaks"; it is "a marginal certificate CAN leave
psi_bar_g uncontrolled at abstaining, audit-eligible operating points" (4 of 16 primary cells here).

TWO FAITHFULLY-IMPLEMENTED published calibration rules (each = a marginal certificate + a coverage-max
rule over the gamma-eligible audit grid; NEITHER looks at the within-question pairing r_T vs r_H):
  * SRC-LTT  -- Selective risk control by exact-binomial Learn-then-Test (angelopoulos_ltt), the
                calibration underlying risk-controlling selective prediction (bates_rcps) and sharing
                the accepted-set target of COIN (coin2025): certify rho(tau)=P(H=1|accept) <= alpha at
                FWER delta over the grid (Bonferroni delta/m), then DEPLOY at the coverage-maximizing
                certified-safe tau. The induced-pi stream is one-answer-per-source, so the exact-binomial
                LTT is cluster-valid (certify.rho_ltt_tau). This is an LTT/RCPS-lineage implementation,
                NOT a claim to reproduce every COIN/RCPS algorithm verbatim.
  * CRC     -- Conformal Risk Control (angelopoulos_crc), the calibration used by conformal abstention
                (conformalabstention2024): bound the marginal accepted-hallucination rate E[loss] <= alpha
                with loss=1{accept a hallucination}, via the finite-sample inversion
                (n*Rhat(tau)+1)/(n+1) <= alpha; deploy at the coverage-max tau under that bound.
The MCQA forced-choice reduction of MedAbstain and the e-value / linear-expectation methods
(medabstain2026 / score2026 / lec2025) are NOT implemented here: MedAbstain's arms are mechanically
coupled (r_T+r_H=1) and are disclosed only as a scope boundary (5_results.tex, sec:res-pubgate);
LEC/SCoRE are discussed in related work, not run.

For every (method, detector, alpha) at the primary pi=0.10 (and a 0.05/0.20 sweep):
  (1) CALIBRATE the coverage-max tau on D_cal by the method's own rule (blind to psi_g). The method's
      calibration criterion is met on D_cal (LTT high-probability at 1-delta; CRC in expectation). Also
      record tau_minrisk and the whole certified-safe set.
  (2) OUT-OF-SAMPLE STABILITY CHECK on the held-out D_test questions: over 200 INDEPENDENT prevalence-
      role assignments (disjoint questions AND independent randomness), report the median realized
      marginal risk, its median one-sided upper Clopper-Pearson interval, and the empirical fraction of
      draws below target. This is a stability corroboration on one fixed D_test, NOT a second formal
      certificate (a median of CP upper bounds is not itself a 95% bound); CRC's target is an
      expectation, for which the mean realized loss is also reported.
  (3) REPORT psi_bar_g at tau_deploy (seed-independent -- it uses the twin pairs, not the stream): the
      heterogeneity-robust Hoeffding-Bentkus LCB (the operative headline object) at the full headline
      union-floor multiplicity (delta_lower/234, conservative and directly comparable to tab:joint) and
      the baseline's own grid multiplicity, the homogeneity-conditional exact Clopper-Pearson companion,
      and an out-of-sample D_test single-threshold HB-LCB. The rejected-truthful denominator is checked
      against the gamma=0.05 min-mass floor (raw and the actual 78-hypothesis step-down Holm family).
  (4) SAFE-SET PROFILE: the fraction of the method's certified-safe thresholds whose psi_bar_g HB-LCB
      (at the headline multiplicity) exceeds alpha -- how pervasively the marginal certificate fails to
      exclude within-question leakage.

SUCCESS CRITERION (plan Block 8): a faithful baseline's own marginal certificate holds while
psi_bar_g,LCB > alpha at its coverage-maximising audit-eligible threshold (denominator clears the gamma floor).

Usage: python R055_faithful_baseline.py     (CPU only, no network, ~seconds)
Output: results/R055_faithful_baseline_results.json + printed table.
"""
import os, sys, csv, json, math, time
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>


# ---------------------------------------------------------------------------
# heterogeneity-robust psi_bar_g leaves -- COPIED VERBATIM from m6_robust_leaf.py so the baseline uses
# the IDENTICAL operative headline object (Hoeffding-Bentkus, valid for Poisson-binomial, no homogeneity).
# Inlined (not imported) because importing m6_robust_leaf re-runs its module-level M6 certification and
# rewrites the released m6_robust_leaf_results.json (elapsed_s timestamp) as a side effect.
def hb_lower_p(s_c, m_c, alpha):
    """Hoeffding-Bentkus p-value for H0: mean psi_g <= alpha (reject => certify psi_g>alpha)."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0: return 1.0
    shat = s_c / m_c
    if shat <= alpha: return 1.0
    pH = math.exp(-2.0 * m_c * (shat - alpha) ** 2)
    pB = math.e * float(binom.sf(s_c - 1, m_c, alpha))
    return float(min(1.0, pH, pB))


def hb_lcb(s_c, m_c, level):
    """HB lower confidence bound on psi_g at `level` = max(Hoeffding LCB, Bentkus LCB)."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0 or s_c <= 0: return 0.0
    shat = s_c / m_c
    lcbH = shat - math.sqrt(math.log(1.0 / level) / (2.0 * m_c))
    lcbB = C.cp_lower_psi(s_c, m_c, level / math.e)
    return float(max(lcbH, lcbB, 0.0))


def eb_lcb(s_c, m_c, level):
    """Empirical-Bernstein LCB (tighter; reference only)."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 1 or s_c <= 0: return 0.0
    shat = s_c / m_c
    var = shat * (1.0 - shat) * m_c / (m_c - 1.0)
    ln = math.log(3.0 / level)
    eb = shat - math.sqrt(2.0 * var * ln / m_c) - 3.0 * ln / (m_c - 1.0)
    return float(max(eb, 0.0))
LAB = artifact("scores_strong.csv")
ART = artifact("scores_strong_artificial.csv")
MEDAB = artifact("scores_medabstain.csv")
E = math.e
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

FAMS = {"score_sptrue": "P(True)-8B", "score_sjudge": "LLM-judge-8B", "score_sNLI": "ref-NLI"}
ALPHAS = [0.05, 0.10, 0.20]
PIS = [0.05, 0.10, 0.20]
PI_PRIMARY = 0.10
GAMMA = C.GAMMA_DEFAULT
SPLIT_SEED = 7
PREV_SEED = 2024            # D_cal induced-prevalence draw -- IDENTICAL to the headline (m3/m6), so
                            # calibration reproduces tab:joint's stream and self-checks against certify.rho_ltt_tau
N_OOS_SEEDS = 200           # D_test marginal STABILITY-checked over MANY INDEPENDENT prevalence draws (disjoint
                            # questions AND independent randomness), reporting the median + the fraction
                            # of draws meeting the target -- a single shared seed would rank-correlate cal/test.
OOS_SEED0 = 90210           # base for the independent D_test prevalence seeds (distinct from PREV_SEED)
NGRID = 28
NLOW_HEADLINE = 3 * 26 * 3          # 234: the headline lower/global Holm family (gamma-floored 26/fam)
LEV_HEADLINE = C.DELTA_LOWER / NLOW_HEADLINE


# --------------------------------------------------------------------------- data
def load_pool(paths, need_cols):
    """Load + pool the given CSVs, extracting only `need_cols` score columns (they must exist in
    every pooled row; e.g. score_selfcheck exists only in the 1k labeled file so is not requested)."""
    rows = []
    for p, src in paths:
        rr = list(csv.DictReader(open(p)))
        for x in rr:
            x.setdefault("source", src)
        rows += rr
    qid = np.array([int(r["qid"]) for r in rows])
    H = np.array([int(r["H"]) for r in rows])
    RAW = {c: np.array([float(r[c]) for r in rows]) for c in need_cols}
    byq = {}
    for i in range(len(rows)):
        byq.setdefault(int(qid[i]), {})[int(H[i])] = i
    QIDS = sorted(q for q in byq if set(byq[q]) == {0, 1})
    return byq, H, RAW, QIDS


def split(qids, seed=SPLIT_SEED, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids)
    perm = np.random.default_rng(seed).permutation(len(qids))
    n = len(qids); a = int(n * fr[0]); b = a + int(n * fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())


def oriented(byq, H, RAW, col, QIDS, D_sel):
    """Orientation FIT ON D_sel only (identical to m3_combined / m6), applied globally."""
    sel = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
    _, fl = C.orient(H[sel], RAW[col][sel])
    return (-RAW[col] if fl else RAW[col])


def twins(ori, byq, QIDS, subset):
    q = [x for x in QIDS if x in subset]
    return (np.array([ori[byq[x][0]] for x in q]),      # r_T
            np.array([ori[byq[x][1]] for x in q]))       # r_H


def pi_stream(rT, rH, pi, seed=PREV_SEED):
    """Induced-prevalence deployment stream: ONE answer per source question (cluster-valid)."""
    rng = np.random.default_rng(seed)
    takeH = rng.random(len(rT)) < pi
    s = np.where(takeH, rH, rT)
    h = takeH.astype(int)
    return s, h


# --------------------------------------------------------------------------- baseline threshold rules
def src_ltt_safe_set(s_stream, h_stream, alpha, delta, grid):
    """Selective-risk-control certified-safe set: taus whose exact-binomial LTT certifies
    rho(tau)=P(H=1|accept) <= alpha at Bonferroni level delta/|grid|. Returns list of
    (tau, K, S, rho_hat, coverage, ltt_p) for the certified-safe taus. BLIND to psi_g."""
    m = len(grid); out = []
    for tau in grid:
        acc = s_stream <= tau
        K = int(acc.sum())
        if K == 0:
            continue
        S = int(h_stream[acc].sum())
        p = float(binom.cdf(S, K, alpha))            # P[Bin(K,alpha) <= S]; small => certify rho<=alpha
        if p <= delta / m:
            out.append(dict(tau=round(float(tau), C.TAU_ROUND), K=K, S=S,
                            rho_hat=round(S / K, 6), coverage=round(K / len(s_stream), 6), ltt_p=p))
    return out


def crc_safe_set(s_stream, h_stream, alpha, grid):
    """Conformal Risk Control certified-safe set: taus whose finite-sample CRC bound on the marginal
    accepted-hallucination rate (n*Rhat+1)/(n+1) <= alpha (loss=1{accept & H=1}). BLIND to psi_g."""
    n = len(s_stream); out = []
    for tau in grid:
        acc = s_stream <= tau
        loss = (h_stream * acc).sum()                # # accepted hallucinations
        Rhat = loss / n
        crc_bound = (n * Rhat + 1.0) / (n + 1.0)     # Angelopoulos CRC upper bound, B=1
        if crc_bound <= alpha:
            K = int(acc.sum())
            out.append(dict(tau=round(float(tau), C.TAU_ROUND), K=K, S=int(loss),
                            crc_risk_hat=round(float(Rhat), 6), crc_bound=round(float(crc_bound), 6),
                            coverage=round(K / n, 6)))
    return out


def cp_upper_rate(k, n, level=0.05):
    """One-sided upper (1-level) Clopper-Pearson bound on a rate k/n."""
    from scipy.stats import beta as _b
    if n == 0:
        return 1.0
    return 1.0 if k == n else float(_b.ppf(1 - level, k + 1, n - k))


def psi_at(rTc, rHc, tau, lev_head=LEV_HEADLINE, lev_grid=None):
    """psi_bar_g summary at a fixed tau: counts + robust HB LCB (headline & grid multiplicity) +
    exact CP companion. Identical machinery to m6_robust_leaf.py."""
    m_c, s_c = C.psi_counts(rTc, rHc, tau)
    d = dict(m_c=m_c, s_c=s_c, psi_hat=round(s_c / m_c, 4) if m_c > 0 else None,
             hb_lcb_headline_mult=round(hb_lcb(s_c, m_c, lev_head), 4),
             exact_lcb_headline_mult=round(C.cp_lower_psi(s_c, m_c, lev_head), 4))
    if lev_grid is not None:
        d["hb_lcb_grid_mult"] = round(hb_lcb(s_c, m_c, lev_grid), 4)
    return d


def build_mass_holm(byq, H, RAW, QIDS):
    """The ACTUAL step-down Holm min-mass family used by the headline (m6_robust_leaf.py:103/112):
    the 3 detector families x their gamma-floored grids = 78 mass hypotheses, p = minmass_p(m_c, ncal, gamma)
    with m_c=#(r_T>tau) on D_cal (independent of pi and alpha). Runs certify.holm_reject at DELTA_MASS and
    returns {(col, tau_rounded): rejected}. A deploy threshold clears the headline min-mass certificate iff
    its (col, tau) is Holm-rejected here."""
    D_sel, D_cal, _ = split(QIDS)
    pool = []
    for col in FAMS:
        ori = oriented(byq, H, RAW, col, QIDS, D_sel)
        rTs, rHs = twins(ori, byq, QIDS, D_sel)
        grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=NGRID)
        rTc, rHc = twins(ori, byq, QIDS, D_cal); ncal = len(rTc)
        for tau in grid:
            m_c, _ = C.psi_counts(rTc, rHc, tau)
            pool.append((col, round(float(tau), C.TAU_ROUND), float(C.minmass_p(m_c, ncal, GAMMA))))
    rej = C.holm_reject([p for _, _, p in pool], C.DELTA_MASS)
    return {(c, t): bool(rej[i]) for i, (c, t, _) in enumerate(pool)}, len(pool)


def oos_marginal(rTt, rHt, tau, pi, method, alpha, n_seeds=N_OOS_SEEDS):
    """OUT-OF-SAMPLE STABILITY CHECK of the baseline's marginal target on the held-out D_test twins, over
    n_seeds INDEPENDENT prevalence draws (each disjoint-question stream gets its own randomness; a single
    shared seed would rank-correlate the calibration and test H-assignments). For SRC-LTT the marginal is
    the selective risk rho=P(H=1|accept); for CRC it is the accepted-hallucination rate E[loss]. Returns
    the median realized risk, the mean (CRC's expectation target), the median one-sided upper 95% CP
    interval, the median coverage, and the fraction of draws whose realized risk <= alpha. The stability
    flag is method-appropriate: mean<=alpha for CRC (expectation guarantee), median<=alpha for SRC-LTT.
    This is a stability corroboration on one fixed D_test, NOT a second formal certificate."""
    risks, uppers, covs = [], [], []
    n = len(rTt)
    for j in range(n_seeds):
        rng = np.random.default_rng(OOS_SEED0 + j)
        takeH = rng.random(n) < pi
        s = np.where(takeH, rHt, rTt); h = takeH.astype(int)
        acc = s <= tau; K = int(acc.sum()); S = int(h[acc].sum())
        if method == "CRC":
            k = int((h * acc).sum())
            risks.append(k / n); uppers.append(cp_upper_rate(k, n)); covs.append(K / n)
        else:
            risks.append((S / K) if K > 0 else float("nan"))
            uppers.append(cp_upper_rate(S, K) if K > 0 else 1.0); covs.append(K / n)
    risks = np.array(risks, float)
    med = float(np.nanmedian(risks)); mean = float(np.nanmean(risks))
    stat = mean if method == "CRC" else med    # CRC target is an expectation; SRC-LTT a risk level
    return dict(median=round(med, 4), mean=round(mean, 4),   # mean = CRC's expectation target
                frac_meeting=round(float(np.mean(risks <= alpha)), 4),
                upper95_median=round(float(np.nanmedian(uppers)), 4),
                cov_median=round(float(np.median(covs)), 4),
                target_met_oos=bool(stat <= alpha))    # DESCRIPTIVE stability flag (method-appropriate), not a formal CI


# --------------------------------------------------------------------------- main
def run():
    byq, H, RAW, QIDS = load_pool([(LAB, "pqa_labeled"), (ART, "pqa_artificial")], list(FAMS))
    log(f"pooled twins = {len(QIDS)}  detector families = {list(FAMS)}")
    mass_holm, mass_family_size = build_mass_holm(byq, H, RAW, QIDS)      # actual 78-hypothesis step-down Holm
    log(f"min-mass Holm family = {mass_family_size} hypotheses (headline protocol)")

    report = dict(
        run="R055 faithful published-abstention baselines: marginal certificate met yet psi_bar_g LCB > alpha at abstaining tau",
        note=("Each baseline is calibrated on D_cal by its own marginal rule (blind to psi_g); its calibration "
              "criterion is met on D_cal (LTT high-probability at 1-delta; CRC in expectation). The coverage-max "
              "tau is chosen over the gamma-eligible audit grid (candidate_grid pooled-rejection prefilter), NOT "
              "the unrestricted domain (where alpha>=pi trivially picks near-accept-all); a cell is audit-eligible "
              "only when that tau also clears the rejected-truthful min-mass floor. D_test is an OUT-OF-SAMPLE "
              "STABILITY CHECK over 200 independent prevalence draws (median realized risk + fraction below "
              "target), NOT a second formal certificate. psi_bar_g robust-HB LCB (leaf identical to "
              "m6_robust_leaf.py, Hoeffding-Bentkus, no homogeneity) is reported at that tau."),
        mass_family_size=mass_family_size,
        split_seed=SPLIT_SEED, prev_seed=PREV_SEED, n_grid=NGRID, gamma=GAMMA,
        delta_global=C.DELTA_GLOBAL, delta_lower=C.DELTA_LOWER,
        headline_lower_multiplicity=NLOW_HEADLINE, headline_lower_level=round(LEV_HEADLINE, 8),
        alphas=ALPHAS, pis=PIS, pi_primary=PI_PRIMARY,
        methods={}, self_check="PASS")

    checks = []

    def eval_method(method, detectors, pi):
        """detectors: list of (byq,H,RAW,QIDS,col,disp) tuples (one per detector family)."""
        cells = []
        for (bq, HH, RW, QI, col, disp) in detectors:
            D_sel, D_cal, D_test = split(QI)
            ori = oriented(bq, HH, RW, col, QI, D_sel)
            rTs, rHs = twins(ori, bq, QI, D_sel)
            grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=NGRID)
            rTc, rHc = twins(ori, bq, QI, D_cal)
            rTt, rHt = twins(ori, bq, QI, D_test)
            ncal = len(rTc)
            s_cal, h_cal = pi_stream(rTc, rHc, pi, seed=PREV_SEED)   # D_cal stream == headline (seed 2024)
            lev_grid = C.DELTA_LOWER / max(1, len(grid))
            for a in ALPHAS:
                # ---- baseline's own certified-safe set + deploy rule (BLIND to psi_g) ----
                if method == "CRC":
                    safe = crc_safe_set(s_cal, h_cal, a, grid)
                    target_name = "E[accept & H=1] (CRC, expectation guarantee)"
                else:  # SRC-LTT
                    safe = src_ltt_safe_set(s_cal, h_cal, a, C.DELTA_GLOBAL, grid)
                    target_name = "rho=P(H=1|accept) (selective risk, high-prob LTT)"
                if not safe:
                    cells.append(dict(detector=disp, alpha=a, pi=pi, certified_safe_set_size=0,
                                      deploy=None, note="baseline certifies no safe threshold at this alpha"))
                    continue
                # deploy = coverage-max certified-safe tau (accept as much as you safely can)
                dep = max(safe, key=lambda r: r["coverage"])
                # most-conservative certified-safe tau (lowest realized marginal risk)
                riskkey = "crc_risk_hat" if method == "CRC" else "rho_hat"
                minr = min(safe, key=lambda r: r[riskkey])
                tau_d = dep["tau"]

                # ---- out-of-sample marginal STABILITY CHECK on D_test over MANY INDEPENDENT prevalence draws ----
                # (disjoint questions AND independent randomness; ONE shared seed would rank-correlate cal/test).
                oos = oos_marginal(rTt, rHt, tau_d, pi, method, a)

                # ---- psi_bar_g at the deploy threshold (D_cal, the certification fold; seed-INDEPENDENT) ----
                psi_dep = psi_at(rTc, rHc, tau_d, lev_grid=lev_grid)
                mass_p = float(C.minmass_p(psi_dep["m_c"], ncal, GAMMA))     # P[Bin(ncal,gamma) >= m_c]
                mass_floor_ok = bool(mass_p <= C.DELTA_MASS)                 # clears the raw gamma=0.05 applicability floor
                # whether it clears the ACTUAL 78-hypothesis step-down Holm min-mass family (headline protocol)
                mass_floor_ok_headline = bool(mass_holm.get((col, round(tau_d, C.TAU_ROUND)), False))
                # out-of-sample single-threshold psi_bar_g on D_test (tau_d frozen on D_cal; NOT prevalence-dependent)
                m_ct, s_ct = C.psi_counts(rTt, rHt, tau_d)
                psi_dep_test = dict(m_c=m_ct, s_c=s_ct, psi_hat=round(s_ct / m_ct, 4) if m_ct > 0 else None,
                                    hb_lcb=round(hb_lcb(s_ct, m_ct, C.DELTA_LOWER), 4))
                psi_minr = psi_at(rTc, rHc, minr["tau"], lev_grid=lev_grid)

                # ---- safe-set profile: among the method's certified-safe taus, the fraction whose psi_bar_g
                #      HB-LCB (at the conservative HEADLINE multiplicity delta_lower/234) exceeds alpha,
                #      split by whether the threshold also clears the raw gamma min-mass floor ----
                prof_all, prof_massok = [], []
                for r in safe:
                    mc, sc = C.psi_counts(rTc, rHc, r["tau"])
                    leaks = hb_lcb(sc, mc, LEV_HEADLINE) > a
                    prof_all.append(leaks)
                    if C.minmass_p(mc, ncal, GAMMA) <= C.DELTA_MASS:
                        prof_massok.append(leaks)
                frac_leak_all = round(float(np.mean(prof_all)), 4) if prof_all else None
                frac_leak_massok = round(float(np.mean(prof_massok)), 4) if prof_massok else None

                leaks_psi = bool(psi_dep["hb_lcb_headline_mult"] > a)        # deploy tau leaks (mass aside)
                success = bool(oos["target_met_oos"] and leaks_psi and mass_floor_ok)
                cells.append(dict(
                    detector=disp, alpha=a, pi=pi, marginal_target=target_name,
                    certified_safe_set_size=len(safe),
                    deploy=dict(tau=tau_d, coverage_cal=dep["coverage"],
                                marginal_risk_hat_cal=dep.get(riskkey),
                                oos_marginal_median=oos["median"], oos_marginal_mean=oos["mean"],
                                oos_marginal_frac_meeting=oos["frac_meeting"],
                                oos_marginal_upper95_median=oos["upper95_median"], oos_coverage_median=oos["cov_median"],
                                oos_stable=oos["target_met_oos"],
                                mass_p=round(mass_p, 6), mass_floor_ok=mass_floor_ok,
                                mass_floor_ok_headline_holm=mass_floor_ok_headline,
                                leaks_psi_bar_g=leaks_psi, **psi_dep),
                    deploy_oos_psi_bar_g=psi_dep_test,
                    minrisk=dict(tau=minr["tau"], marginal_risk_hat_cal=minr.get(riskkey), **psi_minr),
                    frac_certified_safe_leaking_psi_bar_g_all=frac_leak_all,
                    frac_certified_safe_leaking_psi_bar_g_massclearing=frac_leak_massok,
                    success_faithful_baseline_leaks=success))
                checks.append((f"{method}/{disp}/a={a}/pi={pi}",
                               # self-check: SRC-LTT deploy reproduces certify.rho_ltt_tau coverage-max
                               method == "CRC" or _reproduces_certify(s_cal, h_cal, a, grid, tau_d)))
        return cells

    # detector tuples for the strong pool. NOTE: the MCQA forced-choice reduction of MedAbstain
    # (scores_medabstain.csv) is NOT run as a faithful certificate here -- its two arms are mechanically
    # coupled (r_T + r_H = 1) so psi_hat saturates at 1 by construction; the paper reports it only as a
    # disclosed scope boundary (5_results.tex, sec:res-pubgate). The two faithful published decision rules
    # are SRC-LTT (selective risk) and CRC.
    strong_dets = [(byq, H, RAW, QIDS, col, disp) for col, disp in FAMS.items()]

    for method, dets in [("SRC-LTT", strong_dets), ("CRC", strong_dets)]:
        report["methods"][method] = {}
        for pi in PIS:
            report["methods"][method][f"pi={pi}"] = eval_method(method, dets, pi)
        log(f"{method}: done")

    if any(not ok for _, ok in checks):
        report["self_check"] = "FAIL"
        report["failed_checks"] = [k for k, ok in checks if not ok]
    json.dump(report, open(artifact("R055_faithful_baseline_results.json"), "w"), indent=2, default=str)

    # ------------------------------------------------- printed summary (primary pi)
    print("\n" + "=" * 108)
    print(f"R055 FAITHFUL ABSTENTION BASELINES  (primary pi={PI_PRIMARY}; psi_bar_g robust HB-LCB @ headline mult delta_lower/{NLOW_HEADLINE})")
    print("  each rule's calibration criterion is met on D_cal (LTT high-prob / CRC expectation); psi_bar_g,LCB > alpha at its coverage-max audit-eligible tau")
    print("=" * 108)
    hdr = (f"{'method':<9}{'detector':<14}{'a':>5}{'tau*':>10}{'covCal':>8}{'oosRisk~':>9}{'oosUB~':>8}"
           f"{'met%':>6}{'m_c':>6}{'psi_h':>7}{'psiLCB':>8}{'>a':>4}{'gfloor':>7}{'leakAll':>8}")
    print(hdr); print("-" * len(hdr))
    for method in report["methods"]:
        for c in report["methods"][method][f"pi={PI_PRIMARY}"]:
            if not c.get("deploy"):
                print(f"{method:<9}{c['detector']:<14}{c['alpha']:>5}   (no certified-safe threshold)")
                continue
            d = c["deploy"]
            leak = "" if c["frac_certified_safe_leaking_psi_bar_g_all"] is None else f"{100*c['frac_certified_safe_leaking_psi_bar_g_all']:.0f}%"
            print(f"{method:<9}{c['detector']:<14}{c['alpha']:>5.2f}{d['tau']:>10.4f}{d['coverage_cal']:>8.3f}"
                  f"{d['oos_marginal_median']:>9.4f}{d['oos_marginal_upper95_median']:>8.4f}"
                  f"{100*d['oos_marginal_frac_meeting']:>5.0f}%"
                  f"{d['m_c']:>6}{(d['psi_hat'] or 0):>7.3f}{d['hb_lcb_headline_mult']:>8.4f}"
                  f"{'Y' if d['leaks_psi_bar_g'] else 'n':>4}{'Y' if d['mass_floor_ok'] else 'n':>7}{leak:>8}")
    n_succ = sum(1 for m in report["methods"] for c in report["methods"][m][f"pi={PI_PRIMARY}"]
                 if c.get("success_faithful_baseline_leaks"))
    n_tot = sum(1 for m in report["methods"] for c in report["methods"][m][f"pi={PI_PRIMARY}"] if c.get("deploy"))
    print("-" * len(hdr))
    print(f"self_check = {report['self_check']}   |   audit-eligible cells: D_cal calibration met + D_test-stable AND "
          f"psi_bar_g,LCB>alpha AND clears gamma-floor: {n_succ}/{n_tot} (primary pi={PI_PRIMARY})")
    print("  (oosRisk~/oosUB~ = MEDIAN over 200 independent D_test prevalence draws (stability check, not a formal CI);")
    print("   met% = draws with realized risk<=alpha; leakAll = %% of the method's certified-safe thresholds with")
    print("   psi_bar_g HB-LCB(headline mult)>alpha; gfloor = clears raw gamma=0.05 min-mass floor)")
    print(f"wrote {artifact('R055_faithful_baseline_results.json')}")


def _reproduces_certify(s_stream, h_stream, alpha, grid, tau_expected):
    """Self-check: our SRC-LTT coverage-max deploy tau equals certify.rho_ltt_tau's (independent path)."""
    th, cov = C.rho_ltt_tau(s_stream, h_stream, alpha, C.DELTA_GLOBAL, grid)
    if th is None:
        return True     # no-safe-set case handled separately
    return abs(round(float(th), C.TAU_ROUND) - tau_expected) < 1e-9


if __name__ == "__main__":
    run()
