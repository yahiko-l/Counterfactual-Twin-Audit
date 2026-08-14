#!/usr/bin/env python
"""
ψ_g pair-rejection conditional-hallucination certification library.
Shared substrate for M0 (sanity / metric-correctness) and M1 (kill-gate preflight),
reused downstream by M2–M5. CPU only; no GPU, no network.

Estimand (FINAL_PROPOSAL §1; EXPERIMENT_PLAN B1):
    ψ_g(τ) = P( F_τ = 1 | T_τ = 0, G = g )
           = P( accept hallucinated twin | gate rejects truthful twin , group g )
with, per source question (a "twin pair", the exchangeable unit):
    r_T = r(Q, A_truthful)   (the H=0 answer)   T_τ = 1{ r_T ≤ τ }   (accept truthful)
    r_H = r(Q, A_halluc)     (the H=1 answer)   F_τ = 1{ r_H ≤ τ }   (accept hallucinated)
    denominator pairs : T_τ = 0  ⇔  r_T > τ   (truthful rejected)
    failure pairs     : T_τ = 0 ∧ F_τ = 1  ⇔  r_T > τ ∧ r_H ≤ τ
Per-candidate exact conditional-binomial (pair = unit; conditional on the realized
denominator m_c = #rejected-truthful pairs; s_c = #of those with F=1; s_c ~ Bin(m_c, ψ_g)):

    lower_p(s_c, m_c, α) = P[ Bin(m_c, α) ≥ s_c ]   → certify ψ_g > α   (HEADLINE, δ_lower)
    upper_p(s_c, m_c, β) = P[ Bin(m_c, β) ≤ s_c ]   → certify ψ_g ≤ β   (δ_ψup)
    minmass_p(m_c, n, γ) = P[ Bin(n , γ) ≥ m_c ]    → certify P(T_τ=0) ≥ γ (δ_mass)
    m_c = 0  → all p = 1  (nothing certifiable; NEVER ψ_g=0 / "safe")

SCORE ORIENTATION (M0 / R004 lock): every family is oriented so that
"higher r ⇒ more like a hallucination ⇒ should abstain", because the gate accepts the
LOW-risk tail (accept iff r ≤ τ). Operationally: flip the raw score iff AUROC(H, raw) < 0.5.
This is the only defensible convention for *building a gate*; ψ_g is always computed in it.

Multiplicity: every displayed certified claim belongs to exactly one δ-family; within a
family Holm (primary, step-down) controls FWER over the D_sel-frozen candidate set C; the
four families share a single declared budget δ_total = δ_global+δ_mass+δ_ψup+δ_lower, and a
union bound gives P(any false claim) ≤ δ_total.
"""
import numpy as np
from scipy.stats import binom

# ----------------------------------------------------------------------------- budget
DELTA_TOTAL  = 0.10
DELTA_GLOBAL = 0.025   # ρ(τ) ≤ α  (global gate; iid-row PREVIEW here, cluster-valid is M2/B3)
DELTA_MASS   = 0.025   # min-mass  P(T_τ=0) ≥ γ
DELTA_PSIUP  = 0.01    # ψ_g ≤ β   (upper safety certificate, only where a gate passes)
DELTA_LOWER  = 0.04    # ψ_g > α   (HEADLINE pair-failure certificate)
assert abs((DELTA_GLOBAL + DELTA_MASS + DELTA_PSIUP + DELTA_LOWER) - DELTA_TOTAL) < 1e-12
GAMMA_DEFAULT = 0.05   # min-mass floor on the rejected-truthful denominator rate

# ----------------------------------------------------------------- exact one-sided tests
def lower_p_psi(s_c, m_c, alpha):
    """p-value for H0: ψ_g ≤ α  (reject ⇒ certify ψ_g > α). Large s_c ⇒ small p.
    p = P[Bin(m_c, α) ≥ s_c] = sf(s_c-1). m_c=0 ⇒ 1."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0:
        return 1.0
    return float(binom.sf(s_c - 1, m_c, alpha))      # P[X ≥ s_c]

def upper_p_psi(s_c, m_c, beta):
    """p-value for H0: ψ_g ≥ β  (reject ⇒ certify ψ_g ≤ β). Small s_c ⇒ small p.
    p = P[Bin(m_c, β) ≤ s_c] = cdf(s_c). m_c=0 ⇒ 1."""
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0:
        return 1.0
    return float(binom.cdf(s_c, m_c, beta))           # P[X ≤ s_c]

def minmass_p(m_c, n, gamma):
    """p-value for H0: P(T_τ=0) < γ  (reject ⇒ certify rate ≥ γ). Large m_c ⇒ small p.
    p = P[Bin(n, γ) ≥ m_c] = sf(m_c-1). n=0 ⇒ 1."""
    m_c = int(m_c); n = int(n)
    if n <= 0:
        return 1.0
    return float(binom.sf(m_c - 1, n, gamma))         # P[X ≥ m_c]

# ----------------------------------------------- inverted exact one-sided CIs (reporting)
def cp_lower_psi(s_c, m_c, level):
    """Clopper–Pearson one-sided LOWER (1-level) confidence bound for ψ_g given s_c/m_c.
    Largest p0 with P[Bin(m_c,p0) ≥ s_c] ≤ level, i.e. the Beta(s_c, m_c-s_c+1) `level`
    quantile (0 when s_c=0). This is exactly the bound that exceeds α iff lower_p ≤ level."""
    from scipy.stats import beta as _beta
    s_c = int(s_c); m_c = int(m_c)
    if m_c <= 0 or s_c <= 0:
        return 0.0
    if s_c >= m_c:
        return float(level ** (1.0 / m_c))            # degenerate upper-edge case
    return float(_beta.ppf(level, s_c, m_c - s_c + 1))

def cp_lower_rate(m_c, n, level):
    """Clopper–Pearson one-sided LOWER (1-level) bound for the denominator RATE P(T_τ=0)."""
    from scipy.stats import beta as _beta
    m_c = int(m_c); n = int(n)
    if n <= 0 or m_c <= 0:
        return 0.0
    if m_c >= n:
        return float(level ** (1.0 / n))
    return float(_beta.ppf(level, m_c, n - m_c + 1))

# ------------------------------------------------------------------------- Holm step-down
def holm_reject(pvals, delta):
    """Holm–Bonferroni step-down at FWER ≤ delta over the family of |pvals| hypotheses.
    Returns a boolean array aligned to the INPUT order (True = reject = certified)."""
    p = np.asarray(pvals, float)
    m = len(p)
    out = np.zeros(m, bool)
    if m == 0:
        return out
    order = np.argsort(p, kind="mergesort")           # ascending, stable
    thresh = delta / (m - np.arange(m))               # δ/m, δ/(m-1), …, δ/1
    passed = p[order] <= thresh
    # reject the longest leading run of passes (step-down stops at first failure)
    k = int(np.argmin(passed)) if not passed.all() else m
    if passed.all():
        out[:] = True
    else:
        out[order[:k]] = True
    return out

def holm_crit_level(rank, m, delta):
    """Per-hypothesis Holm critical level for the hypothesis at ascending `rank` (0-based)."""
    return delta / (m - rank)

# -------------------------------------------------------------- score orientation (R004)
def orient(H, raw):
    """Lock risk orientation: return (oriented, flipped) so AUROC(H, oriented) ≥ 0.5.
    Flip (negate) iff AUROC(H, raw) < 0.5. Ties / degenerate scores are left unflipped."""
    from sklearn.metrics import roc_auc_score
    raw = np.asarray(raw, float)
    try:
        auc = roc_auc_score(H, raw)
    except Exception:
        return raw, False
    if auc < 0.5:
        return -raw, True
    return raw, False

def paired_acc(rT, rH):
    """P(r_H > r_T) over twins in the GIVEN orientation (ties count 0.5).
    In risk orientation this is the rate of CORRECT pairwise ranking (hallucinated riskier)."""
    rT = np.asarray(rT, float); rH = np.asarray(rH, float)
    return float(np.mean(rH > rT) + 0.5 * np.mean(rH == rT))

# ------------------------------------------------------------------- twin / ψ_g mechanics
def psi_counts(rT, rH, tau):
    """(m_c, s_c) at threshold τ: m_c = #(r_T > τ)  [truthful rejected];
       s_c = #(r_T > τ ∧ r_H ≤ τ)  [hallucinated accepted among them]."""
    rT = np.asarray(rT, float); rH = np.asarray(rH, float)
    rej_T = rT > tau
    m_c = int(rej_T.sum())
    s_c = int((rej_T & (rH <= tau)).sum())
    return m_c, s_c

def psi_hat(rT, rH, tau):
    m_c, s_c = psi_counts(rT, rH, tau)
    return (s_c / m_c) if m_c > 0 else float("nan"), m_c, s_c

# ----------------------------------------------------- global ρ(τ)=P(H=1|accept) ≤ α LTT
# NOTE: iid-row exact-binomial LTT, reused from idea-stage/pilot/pilot_ltt.py. On the twin
# table the two answers of a question are correlated, so this is ANTI-CONSERVATIVE
# (over-certifies; run02 lesson). It is used ONLY as a PREVIEW to locate the operating τ̂
# for the headline joint cell; the cluster-valid certificate (one-answer-per-source /
# cluster-LEC-LTT) is an M2/B3 deliverable and is NOT claimed here.
def rho_ltt_tau(scores, H, alpha, delta, grid):
    """Largest-coverage τ on `grid` with exact-binomial certificate ρ(τ)≤α (Bonferroni/m),
    accept iff score ≤ τ, ρ=P(H=1|accept). Returns (tau_hat or None, certified_coverage)."""
    scores = np.asarray(scores, float); H = np.asarray(H, int)
    m = len(grid); th = None; best = -1.0
    for tau in grid:
        acc = scores <= tau
        K = int(acc.sum())
        if K == 0:
            continue
        p = float(binom.cdf(int(H[acc].sum()), K, alpha))   # P[Bin(K,α) ≤ S]; certify ρ≤α
        if p <= delta / m:
            cov = K / len(scores)
            if cov > best:
                best = cov; th = float(tau)
    return th, (best if best >= 0 else 0.0)

# --------------------------------------------------------------------- candidate τ-grid
TAU_ROUND = 9   # canonical τ precision: identical floats across all candidate-keyed lookups

def rho_ltt_one_per_source(rT, rH, alpha, delta, grid, seed=0):
    """CLUSTER-VALID global ρ(τ)=P(H=1|accept)≤α certificate via ONE-ANSWER-PER-SOURCE
    (the LEC/COIN substrate; the exchangeable unit is the source question, not the row).
    From each twin pick exactly one answer (uniform, declared seed) ⇒ rows iid across
    questions ⇒ exact-binomial LTT is finite-sample valid (no iid-row over-certification).
    Returns (tau_hat or None, certified_coverage, n_eff)."""
    rT = np.asarray(rT, float); rH = np.asarray(rH, float)
    pick_H = np.random.default_rng(seed).random(len(rT)) < 0.5     # True ⇒ take hallucinated (H=1)
    s = np.where(pick_H, rH, rT); h = pick_H.astype(int)
    th, cov = rho_ltt_tau(s, h, alpha, delta, grid)
    return th, cov, int(len(s))

def rho_hat_at(rT, rH, tau):
    """Realized accepted-set hallucination rate ρ̂(τ)=P(H=1|accept) over BOTH twins (descriptive)."""
    rT = np.asarray(rT, float); rH = np.asarray(rH, float)
    accT = rT <= tau; accH = rH <= tau; K = int(accT.sum() + accH.sum())
    return (int(accH.sum()) / K) if K > 0 else float("nan"), K

def mondrian_rho_by_group(rT, rH, tau, groups_T, groups_H):
    """Per-group realized ρ_g under a GLOBAL τ (negative control: Mondrian bounds group-avg ρ_g,
    NOT the paired contrast). Returns {group: (rho_g, K_g)} over accepted answers of both twins."""
    rT = np.asarray(rT, float); rH = np.asarray(rH, float)
    gT = np.asarray(groups_T); gH = np.asarray(groups_H)
    out = {}
    for g in sorted(set(gT.tolist()) | set(gH.tolist())):
        mT = (gT == g) & (rT <= tau); mH = (gH == g) & (rH <= tau)
        K = int(mT.sum() + mH.sum())
        out[g] = ((int(mH.sum()) / K) if K > 0 else float("nan"), K)
    return out

def candidate_grid(scores_sel, gamma=GAMMA_DEFAULT, n_grid=24, rate_max=0.995):
    """Freeze a τ-grid on D_sel scores (pooled answers): a quantile grid kept where the
    D_sel pooled rejection rate P(r>τ) is ≥ γ (a necessary pre-filter so the rejected-
    truthful denominator can plausibly clear the min-mass floor) and not the degenerate
    accept-almost-nothing tail (rate ≤ rate_max). The strict-τ end (large rate) IS kept —
    that is where a ρ≤α gate operates. Frozen on D_sel ⇒ D_cal-independent ⇒ Holm /
    conditional-binomial conditioning stay valid. τ rounded to a canonical precision so the
    same value keys every downstream lookup (no float-equality false-negatives)."""
    scores_sel = np.asarray(scores_sel, float)
    qs = np.unique(np.quantile(scores_sel, np.linspace(0.01, 0.99, n_grid)))
    keep = []
    for tau in qs:
        rate = float((scores_sel > tau).mean())
        if gamma <= rate <= rate_max:
            keep.append(round(float(tau), TAU_ROUND))
    return np.array(sorted(set(keep))) if keep else np.round(qs, TAU_ROUND)
