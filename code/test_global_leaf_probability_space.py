#!/usr/bin/env python
"""Unit tests for the global-leaf probability space (Q1 fix, AIM revision 2026-07-17).

Backs supp lem:companion (a-i)/(a-ii) and rem:condspace; prespecified in
refine-logs/analysis_plan_v2_20260717.md Section 7. Four groups:

  G1  Exact randomization leaf, exhaustive enumeration (small n): under iid
      source questions x iid Bernoulli(pi) arms -- with STRONGLY CORRELATED
      (T,F) within a question -- the law of S | K=k is exactly Bin(k, rho_pi),
      and p_glob = P[Bin(K,alpha) <= S] is super-uniform at the null boundary.
  G2  Boundary Monte Carlo (n=400): empirical P(p_glob <= u) <= u (+MC slack),
      randomizing BOTH questions and arms (the space the paper claims).
  G3  Fixed-assignment counterexample (rem:condspace verbatim): conditional on
      z, the binomial p-value is anti-conservative (P(p<=0.75 | z) = 1 > 0.75).
  G4  Fixed-assignment bounded-loss Hoeffding / Hoeffding-Bentkus leaf IS valid
      conditional on z with heterogeneous per-question acceptance probabilities
      (size <= u at the null boundary); plus the K=0 => p=1 convention and a
      Holm family-wise check over independent boundary-null candidates.

CPU-only, deterministic seeds, ~1 min. Run:
  python test_global_leaf_probability_space.py
Exit 0 iff all tests pass.
"""
import sys
import numpy as np
from scipy.stats import binom

RNG = np.random.default_rng(20260717)
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- G1: enumeration
def g1_enumeration():
    # per-question joint law of (T,F): strongly positively correlated
    pTF = {(1, 1): 0.35, (1, 0): 0.25, (0, 1): 0.15, (0, 0): 0.25}
    pi, n = 0.3, 5
    pT = sum(v for (t, f), v in pTF.items() if t == 1)  # 0.60
    pF = sum(v for (t, f), v in pTF.items() if f == 1)  # 0.50
    a, b = pi * pF, (1 - pi) * pT
    rho = a / (a + b)

    # exact joint pmf of (S, K) by full enumeration of (T,F,Z) per question
    # state: dict (s, k) -> prob, convolved question by question
    dist = {(0, 0): 1.0}
    cell = []  # per-question outcome distribution over (ds, dk)
    for (t, f), pv in pTF.items():
        for z, pz in ((1, pi), (0, 1 - pi)):
            aacc = f if z == 1 else t          # accepted?
            ds, dk = (z * aacc), aacc
            cell.append(((ds, dk), pv * pz))
    for _ in range(n):
        nxt = {}
        for (s, k), p0 in dist.items():
            for (ds, dk), pc in cell:
                key = (s + ds, k + dk)
                nxt[key] = nxt.get(key, 0.0) + p0 * pc
        dist = nxt

    # (1) S | K=k  ==  Bin(k, rho) exactly, for every k with P(K=k) > 0
    max_err = 0.0
    for k in range(n + 1):
        pk = sum(p for (s, kk), p in dist.items() if kk == k)
        if pk < 1e-15:
            continue
        for s in range(k + 1):
            cond = sum(p for (ss, kk), p in dist.items() if kk == k and ss == s) / pk
            max_err = max(max_err, abs(cond - binom.pmf(s, k, rho)))
    check("G1a S|K=k is exactly Bin(k, rho_pi) under correlated twins",
          max_err < 1e-12, f"max|diff|={max_err:.2e}, rho={rho:.5f}")

    # (2) super-uniformity of p_glob at the boundary alpha = rho:
    #     P(p <= u) <= u for every achievable u
    alpha = rho
    pvals = {}
    for (s, k), p in dist.items():
        pv = 1.0 if k == 0 else float(binom.cdf(s, k, alpha))
        pvals[pv] = pvals.get(pv, 0.0) + p
    worst = 0.0
    for u in sorted(pvals):
        cum = sum(p for pv, p in pvals.items() if pv <= u + 1e-15)
        worst = max(worst, cum - u)
    check("G1b p_glob super-uniform at boundary (exact enumeration)",
          worst < 1e-12, f"max[P(p<=u)-u]={worst:.2e}")


# ---------------------------------------------------------------- G2: boundary MC
def g2_boundary_mc(reps=20000, n=400):
    # marginals chosen so rho_pi == alpha exactly: P(T)=P(F)=0.6, pi=alpha=0.1
    pi, alpha = 0.1, 0.1
    cells = np.array([(1, 1), (1, 0), (0, 1), (0, 0)])
    probs = np.array([0.45, 0.15, 0.15, 0.25])       # marginals 0.6 / 0.6, correlated
    idx = RNG.choice(4, size=(reps, n), p=probs)
    T, F = cells[idx, 0], cells[idx, 1]
    Z = (RNG.random((reps, n)) < pi).astype(int)
    A = np.where(Z == 1, F, T)
    S, K = (A * Z).sum(1), A.sum(1)
    p = binom.cdf(S, K, alpha)                        # cdf(s,0,a)=1 handles K=0
    ok, msgs = True, []
    for u in (0.01, 0.025, 0.05, 0.10, 0.20, 0.50):
        emp = float((p <= u).mean())
        slack = 4.0 * np.sqrt(u * (1 - u) / reps)
        if emp > u + slack:
            ok = False
        msgs.append(f"u={u}:{emp:.4f}")
    check("G2  boundary MC size (joint over questions AND arms)",
          ok, " ".join(msgs))


# ------------------------------------------------- G3: fixed-z counterexample
def g3_counterexample():
    # z = (1, 0), f = t = 1, alpha = 1/2  ->  K=2, S=1 deterministically
    alpha = 0.5
    p = float(binom.cdf(1, 2, alpha))                 # = 0.75
    # conditional on z the outcome is deterministic: P(p <= 0.75 | z) = 1 > 0.75
    check("G3  fixed-assignment counterexample (rem:condspace)",
          abs(p - 0.75) < 1e-12 and 1.0 > p,
          f"p_glob={p} deterministic; P(p<=0.75|z)=1 > 0.75 => not super-uniform given z")


# ------------------------------------- G4: fixed-z bounded-loss leaf + extras
def g4_fixed_z_hoeffding(reps=20000, n=200, n_H=60):
    z = np.zeros(n, dtype=int); z[:n_H] = 1
    f_i = RNG.uniform(0.2, 0.9, size=n_H)             # heterogeneous, fixed
    t_i = RNG.uniform(0.2, 0.9, size=n - n_H)
    alpha = f_i.sum() / (f_i.sum() + t_i.sum())       # rho_z boundary exactly
    accH = RNG.random((reps, n_H)) < f_i
    accT = RNG.random((reps, n - n_H)) < t_i
    S = accH.sum(1)
    K = S + accT.sum(1)
    diff = alpha * K - S                              # = -(sum of D_i)
    pH = np.where(S < alpha * K, np.exp(-2.0 * diff**2 / n), 1.0)
    Ybar = alpha + (S - alpha * K) / n
    bent = np.e * binom.cdf(np.floor(n * Ybar).astype(int), n, alpha)
    pHB = np.where(Ybar < alpha, np.minimum(1.0, np.minimum(pH, bent)), 1.0)
    okH = okHB = True
    msgs = []
    for u in (0.01, 0.05, 0.10, 0.25):
        eH, eHB = float((pH <= u).mean()), float((pHB <= u).mean())
        slack = 4.0 * np.sqrt(u * (1 - u) / reps)
        okH &= eH <= u + slack
        okHB &= eHB <= u + slack
        msgs.append(f"u={u}:H={eH:.4f},HB={eHB:.4f}")
    check("G4a fixed-z Hoeffding leaf size (heterogeneous, boundary)", okH, " ".join(msgs))
    check("G4b fixed-z Hoeffding-Bentkus leaf size", okHB)


def g4_k0_and_holm(reps=5000, n=200, m=5):
    # K = 0 convention
    check("G4c K=0 gives p=1", float(binom.cdf(0, 0, 0.1)) == 1.0)
    # Holm FWER over m independent boundary-null candidates at delta = 0.10
    pi, alpha, delta = 0.1, 0.1, 0.10
    cells = np.array([(1, 1), (1, 0), (0, 1), (0, 0)])
    probs = np.array([0.45, 0.15, 0.15, 0.25])
    any_rej = np.zeros(reps, dtype=bool)
    P = np.empty((reps, m))
    for j in range(m):                                # independent candidates
        idx = RNG.choice(4, size=(reps, n), p=probs)
        T, F = cells[idx, 0], cells[idx, 1]
        Z = (RNG.random((reps, n)) < pi).astype(int)
        A = np.where(Z == 1, F, T)
        P[:, j] = binom.cdf((A * Z).sum(1), A.sum(1), alpha)
    Psort = np.sort(P, axis=1)
    # literal Holm step-down: under the all-null configuration, Holm rejects at
    # least one hypothesis iff the smallest p-value clears its first step delta/m
    any_rej = Psort[:, 0] <= delta / m
    emp = float(any_rej.mean())
    slack = 4.0 * np.sqrt(delta * (1 - delta) / reps)
    check("G4d Holm family-wise error over boundary nulls <= delta",
          emp <= delta + slack, f"FWER_hat={emp:.4f} vs delta={delta}")


if __name__ == "__main__":
    g1_enumeration()
    g2_boundary_mc()
    g3_counterexample()
    g4_fixed_z_hoeffding()
    g4_k0_and_holm()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {FAILURES}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)
