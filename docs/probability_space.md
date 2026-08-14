# Probability space of the global exact-binomial leaf (Q1 resolution)

> **Code-only release: the manuscript is under review.** The score files, the generated results
> and the de-identified clinician tables this document refers to are added on acceptance, so
> `data/`, `results/` and `adjudication/` are not in this repository yet. `RESULT_SHA256SUMS.txt`
> commits now to the bytes that will arrive. Editors and reviewers hold the complete artifact.

Revision artifact (2026-07-17), companion to Supplement `ass:cluster`,
`lem:companion`(a-i)/(a-ii), and `rem:condspace`. Prespecified in
`refine-logs/analysis_plan_v2_20260717.md` Section 7.

## The implemented test

For a D_sel-frozen candidate tau, the released code computes

    p_glob(tau) = P[ Bin(K_tau, alpha) <= S_tau ]

where K_tau / S_tau are the accepted / accepted-hallucinated counts of the
one-answer-per-source induced stream.

## The sampler (code audit, 2026-07-17)

The audited arm is drawn PER QUESTION, i.i.d. Bernoulli(pi), independent of all
records and scores:

- `m10_pi_multiplicity.py` line ~88: `takeH = u < pi` with
  `u = np.random.default_rng(PREV_SEED=2024).random(ncal)`;
- `certify.py` line ~184: same pattern (`rng.random(n) < 0.5`);
- `m3_combined.py` `certify_at_prevalence`: same mirror (seed 2024).

It is NOT a fixed-quota or permutation design. The archived seed records one
realization of this randomized design and is not a conditioning event.

## Validity claim (exact randomization leaf, lem:companion(a-i))

Probabilities are taken JOINTLY over i.i.d. source-question sampling (A6) and
the Bernoulli(pi) arm randomization, conditional only on K_tau — never on the
realized assignment vector. Multinomial thinning gives, exactly and for
arbitrarily dependent within-question twin indicators,

    (S, K−S, n−K) ~ Multinomial(n; a, b, 1−a−b),   a = pi·P(F=1), b = (1−pi)·P(T=1)
    S | K=k ~ Bin(k, a/(a+b)),                     a/(a+b) = P(H=1 | A=1) = rho_pi(tau)

so p_glob is super-uniform under H0: rho_pi(tau) >= alpha (stochastic dominance
in the conditional success rate; p := 1 on {K=0}; average over K).

## What is NOT claimed

Conditional on the realized assignment z, the law of S given K is Fisher
noncentral hypergeometric, not binomial; the exact test is anti-conservative in
that conditional space (minimal counterexample: n_H = n_T = 1, f = t = 1,
alpha = 1/2 gives deterministic p = 3/4 with P(p <= 3/4 | z) = 1). Conditional
on z, the paper claims only the bounded-loss Hoeffding–Bentkus leaf
(lem:companion(a-ii)), and that leaf certifies the assignment-specific ratio
rho_z — a companion object that generally differs from rho_pi at finite n and
converges to it under the i.i.d. design. No conditional-on-assignment
exact-binomial claim is made anywhere in the paper.

## Machine checks

`test_global_leaf_probability_space.py` (run by `reproduce.py --verify-only`):

- G1: exhaustive small-n enumeration under strongly correlated twins —
  S | K=k equals Bin(k, rho_pi) to 1e-12 and p_glob is exactly super-uniform
  at the null boundary;
- G2: boundary Monte Carlo in the joint space (n=400) — empirical size below
  nominal at every tested u;
- G3: the fixed-assignment counterexample reproduces exactly (p = 0.75
  deterministic);
- G4: fixed-z Hoeffding / Hoeffding–Bentkus leaves are conservative at the
  rho_z boundary with heterogeneous per-question acceptance probabilities;
  literal Holm FWER at the all-null configuration; K=0 => p=1 convention.

Zero certificate numbers changed under this resolution.
