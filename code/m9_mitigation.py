#!/usr/bin/env python
"""
M9 MITIGATION (reviewer-objection #5: "a diagnostic negative is nice -- what should a
deployer DO?"). Upgrades the paper from pure diagnostic to problem + actionable control.

Two naive fixes FAIL (we show / cite):
  (i)  "use a better detector": the 11-judge battery already shows the best judge
       (paired acc 0.969) STILL certifies 5 psi_g>alpha failure cells at pi=0.10 --
       improving within-pair discrimination does NOT eliminate the leak (battery_summary.json).
  (ii) "tighten the global target alpha": rho<=alpha is a MARGINAL constraint; it cannot
       see the paired contrast (the paper's whole point).

The deployable fix is a TWO-CONSTRAINT gate that uses the paper's OWN four-family budget:
certify BOTH the global rate rho<=alpha (delta_global) AND the paired UPPER safety
certificate psi_g<=beta (delta_psiup), plus the min-mass floor, all Holm-controlled. The
threshold is selected offline on twin calibration data, then deployed as an ordinary
per-answer gate. We ask: does such a gate EXIST, and at what COVERAGE COST vs the rho-only
"certified-safe" gate the deployer would otherwise pick?

Anchor seed (7/2024), pooled 10k, primary pi=0.10. Reuses validated M7 machinery.
Writes results/m9_mitigation_results.json. CPU only.
"""
import os, sys, json, time
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
import m7_multiseed_robustness as M7

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

FAMS = M7.FAMS; ALPHAS = M7.ALPHAS; POOLED = M7.POOLED
SPLIT_SEED = 7; PREV_SEED = 2024; PI = 0.10
BETAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FLABEL = {"score_sptrue": "P(True)-8B", "score_sNLI": "ref-NLI", "score_sjudge": "LLM-judge-8B"}

cache = M7.build_split_cache(POOLED, SPLIT_SEED)
ncal = cache["ncal"]; nlow = cache["nlow"]
# sanity
nj, cells = M7.certify_one(cache, PI, PREV_SEED)
hb = cells["score_sptrue"][0.10]["best"]
assert hb and hb["m_c"] == 278, f"anchor mismatch {hb}"
log(f"anchor OK (m_c=278). ncal={ncal}")

# ----- deployment stream (anchor prev seed) per family: coverage, rho on the stream
def stream(fam):
    rTc = cache["rTc"][fam]; rHc = cache["rHc"][fam]
    rng = np.random.default_rng(PREV_SEED); takeH = rng.random(ncal) < PI
    s = np.where(takeH, rHc, rTc); h = takeH.astype(int)
    return s, h
STREAM = {f: stream(f) for f in FAMS}

# ----- per-(fam,ti): coverage, rho_hat, m_c, s_c, psi_hat, denom_mass + p-values
PER = {f: [] for f in FAMS}
rho_pool = []; mass_pool = []
psiup_pool = {b: [] for b in BETAS}
for f in FAMS:
    grid = cache["grids"][f]; s_str, h_str = STREAM[f]
    for ti, tau in enumerate(grid):
        m_c = cache["m_c"][f][ti]; s_c = cache["s_c"][f][ti]
        acc = s_str <= tau; Kc = int(acc.sum()); Sc = int(h_str[acc].sum())
        cov = Kc / ncal
        PER[f].append(dict(ti=ti, tau=float(tau), coverage=round(cov, 4),
                           rho_hat=(round(Sc/Kc, 4) if Kc else None), Kc=Kc,
                           m_c=m_c, s_c=s_c, psi_hat=(round(s_c/m_c, 4) if m_c else None),
                           denom_mass=round(m_c/ncal, 4)))
        for a in ALPHAS:
            rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
        mass_pool.append((f, ti, C.minmass_p(m_c, ncal, C.GAMMA_DEFAULT)))
        for b in BETAS:
            psiup_pool[b].append((f, ti, C.upper_p_psi(s_c, m_c, b)))
# Holm rejections (each family's own budget; psiup pooled over (f,ti) per beta within delta_psiup)
rho_ok = {(x[0],x[1],x[2]): bool(r) for x,r in zip(rho_pool, C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL))}
mass_ok = {(x[0],x[1]): bool(r) for x,r in zip(mass_pool, C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS))}
psiup_ok = {}
for b in BETAS:
    rj = C.holm_reject([x[2] for x in psiup_pool[b]], C.DELTA_PSIUP)
    for x, r in zip(psiup_pool[b], rj): psiup_ok[(b, x[0], x[1])] = bool(r)

def best_cov(fam, predicate):
    """Most permissive (max coverage) ti satisfying predicate(ti); None if none."""
    best = None
    for ti in range(len(cache["grids"][fam])):
        if predicate(ti):
            rec = PER[fam][ti]
            if best is None or rec["coverage"] > best["coverage"]:
                best = rec
    return best

report = {"run": "M9 mitigation: two-constraint (rho<=alpha AND psi_g<=beta) gate + coverage cost",
          "anchor": {"split": SPLIT_SEED, "prev": PREV_SEED, "pi": PI, "nlow": nlow},
          "betas": BETAS, "rho_only_gate": {}, "two_constraint_gate": {},
          "frontier_ptrue": {}, "naive_fix_detector_quality": {}}

# ===== rho-only "certified-safe" gate (the conventional deployer choice): max coverage s.t. rho<=alpha
for f in FAMS:
    report["rho_only_gate"][FLABEL[f]] = {}
    for a in ALPHAS:
        g = best_cov(f, lambda ti: rho_ok[(f, a, ti)])
        report["rho_only_gate"][FLABEL[f]][f"alpha={a}"] = (
            None if g is None else dict(tau=round(g["tau"],6), coverage=g["coverage"],
                rho_hat=g["rho_hat"], psi_hat=g["psi_hat"], m_c=g["m_c"], denom_mass=g["denom_mass"],
                leaks=bool(g["psi_hat"] is not None and g["psi_hat"] > a)))

# ===== two-constraint gate: max coverage s.t. rho<=alpha AND psi_g<=beta (certified) AND mass>=gamma
for f in FAMS:
    report["two_constraint_gate"][FLABEL[f]] = {}
    for a in ALPHAS:
        ro = report["rho_only_gate"][FLABEL[f]][f"alpha={a}"]
        cov_rho = ro["coverage"] if ro else None
        row = {}
        for b in BETAS:
            g = best_cov(f, lambda ti: rho_ok[(f,a,ti)] and mass_ok[(f,ti)] and psiup_ok[(b,f,ti)])
            if g is None:
                row[f"beta={b}"] = dict(exists=False)
            else:
                row[f"beta={b}"] = dict(exists=True, tau=round(g["tau"],6), coverage=g["coverage"],
                    rho_hat=g["rho_hat"], psi_hat=g["psi_hat"], m_c=g["m_c"],
                    coverage_cost_vs_rho_only=(round(cov_rho - g["coverage"], 4) if cov_rho is not None else None),
                    coverage_retained_frac=(round(g["coverage"]/cov_rho, 4) if cov_rho else None))
        report["two_constraint_gate"][FLABEL[f]][f"alpha={a}"] = row

# ===== safety frontier for the headline P(True) family at alpha=0.10: beta -> max certified coverage
f = "score_sptrue"; a = 0.10
ro = report["rho_only_gate"][FLABEL[f]][f"alpha={a}"]
front = []
for b in BETAS:
    g = best_cov(f, lambda ti: rho_ok[(f,a,ti)] and mass_ok[(f,ti)] and psiup_ok[(b,f,ti)])
    front.append(dict(beta=b, max_coverage=(g["coverage"] if g else 0.0),
                      certified=bool(g is not None), psi_hat=(g["psi_hat"] if g else None)))
report["frontier_ptrue"] = {"alpha": a, "rho_only_coverage": ro["coverage"], "rho_only_psi_hat": ro["psi_hat"],
                            "rho_only_leaks": ro["leaks"], "frontier": front}

# ===== naive fix (ii): detector quality does not substitute -- reframe the battery
try:
    bat = json.load(open("results/battery_summary.json"))
    rows = sorted(bat["rows"], key=lambda r: r["paired_acc"])
    report["naive_fix_detector_quality"] = {
        "paired_acc_span": bat["paired_acc_span"],
        "best_judge_paired_acc": rows[-1]["paired_acc"],
        "best_judge_failure_cells_pi0.10": rows[-1]["cells"]["pi=0.1"],
        "min_failure_cells_pi0.10_over_battery": min(r["cells"]["pi=0.1"] for r in rows),
        "note": "even the best within-pair discriminator still certifies psi_g>alpha cells at "
                "pi=0.10; detector quality does not eliminate the leak (so a two-constraint gate, "
                "not a better score, is the fix)."}
except Exception as e:
    report["naive_fix_detector_quality"] = {"error": str(e)}

report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m9_mitigation_results.json", "w"), indent=2, default=str)

# ----- console
print("\n" + "="*92)
print("M9 MITIGATION — two-constraint (rho<=alpha AND certified psi_g<=beta) gate")
print(f"\n[A] rho-only 'certified-safe' gate (max coverage s.t. rho<=alpha), pi={PI}:")
for f in FAMS:
    for a in ALPHAS:
        g = report["rho_only_gate"][FLABEL[f]][f"alpha={a}"]
        if g: print(f"  {FLABEL[f]:14s} a={a}: cov={g['coverage']:.3f} rho={g['rho_hat']} "
                    f"psi_hat={g['psi_hat']} m_c={g['m_c']}  LEAKS={g['leaks']}")
print(f"\n[B] two-constraint gate for headline P(True)-8B (alpha=0.10) across beta:")
ro = report["rho_only_gate"]["P(True)-8B"]["alpha=0.1"]
print(f"  rho-only baseline: coverage={ro['coverage']:.3f}, psi_hat={ro['psi_hat']} (leaks at alpha=0.10)")
for b in BETAS:
    r = report["two_constraint_gate"]["P(True)-8B"]["alpha=0.1"][f"beta={b}"]
    if r["exists"]:
        print(f"  beta={b}: cov={r['coverage']:.3f} (retain {r['coverage_retained_frac']*100:.0f}%, "
              f"cost {r['coverage_cost_vs_rho_only']:.3f}) psi_hat={r['psi_hat']} m_c={r['m_c']}")
    else:
        print(f"  beta={b}: NO certified two-constraint threshold exists")
print(f"\n[C] naive 'better detector' fix: best judge paired_acc="
      f"{report['naive_fix_detector_quality'].get('best_judge_paired_acc')} STILL has "
      f"{report['naive_fix_detector_quality'].get('best_judge_failure_cells_pi0.10')} failure cells; "
      f"battery min={report['naive_fix_detector_quality'].get('min_failure_cells_pi0.10_over_battery')}")
print(f"\n  wrote results/m9_mitigation_results.json  ({report['elapsed_s']}s)")
print("="*92)
