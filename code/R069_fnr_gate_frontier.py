#!/usr/bin/env python
"""R069 — marginal-FNR (q_H)-constrained gate vs the paired psi_g gate (AIM 2026-07-23 Major 2).

Prespecified in refine-logs/analysis_plan_v3_amendment_20260723.md
(SHA256 19659f661ed1d01379d4d16624a81ee521f473c8c6702c148a9e06b6e18b7a43). RUN FROM REPO ROOT. CPU only.

q_H(tau) = P(F_tau=1) = detector's hallucination-detection FNR (the "F1" column of m_pairedlift),
n_H = ncal i.i.d. hallucinated D_cal twins. FNR safety leaf = upper_p_psi(s_H,n_H,beta_F); FNR
exceedance = lower_p_psi(s_H,n_H,beta_F) (certify.py, no new inequality, same C1). Frozen rho/mass/
psi certificates are REUSED (read-only reconstruction). New budget delta_T12=0.05 (Fup .025/Flow .025),
Holm within each leg over (3 fam x 26 grid x 3 beta_F). Self-check reproduces m_pairedlift/m9 anchors
BEFORE emitting any new number, then computes:
  (A) certified safety-coverage frontiers: native rho+q_H (PRIMARY), rho+mass+q_H (common-support
      sensitivity), frozen psi_g gate 0.3703 + rho-only 0.9605 references;
  (B) exhaustive 3x3 psi/FNR status map at matched cap (beta_F*,beta*)=(0.10,0.10);
  (C) multi-policy + equi-coverage crosswalk; and reads off locked conclusions S1-S4.
"""
import os, sys, json, time, hashlib
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
import m7_multiseed_robustness as M7

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

FAMS = M7.FAMS; ALPHAS = M7.ALPHAS; POOLED = M7.POOLED
SPLIT_SEED = 7; PREV_SEED = 2024; PI = 0.10
FLABEL = {"score_sptrue": "P(True)-8B", "score_sNLI": "ref-NLI", "score_sjudge": "LLM-judge-8B"}
GAMMA = C.GAMMA_DEFAULT
# ---- frozen amendment budgets & tolerances ----
DELTA_FUP = 0.025; DELTA_FLOW = 0.025
BETA_F_STAR = 0.10; BETA_F_ALL = [0.05, 0.10, 0.20]     # confirmatory 0.10 + sensitivity {0.05,0.20}
BETA_STAR = 0.10; ALPHA_PRIMARY = 0.10; DCOV_MATERIAL = 0.10
AMEND_SHA = "19659f661ed1d01379d4d16624a81ee521f473c8c6702c148a9e06b6e18b7a43"
HEADLINE_TAUS = [0.000123, 0.222824, 0.245085, 0.518631]

cache = M7.build_split_cache(POOLED, SPLIT_SEED)
ncal = cache["ncal"]; nlow = cache["nlow"]

# ===================== per-(fam,ti) quantities =====================
def stream(f):
    rTc = cache["rTc"][f]; rHc = cache["rHc"][f]
    rng = np.random.default_rng(PREV_SEED); takeH = rng.random(ncal) < PI
    return np.where(takeH, rHc, rTc), takeH.astype(int)
STREAM = {f: stream(f) for f in FAMS}

PER = {f: [] for f in FAMS}
rho_pool = []; mass_pool = []; psiup_pool = []
fnrup_pool = {b: [] for b in BETA_F_ALL}; fnrlow_pool = {b: [] for b in BETA_F_ALL}
for f in FAMS:
    grid = cache["grids"][f]; s_str, h_str = STREAM[f]; rTc = cache["rTc"][f]; rHc = cache["rHc"][f]
    for ti, tau in enumerate(grid):
        m_c = cache["m_c"][f][ti]; s_c = cache["s_c"][f][ti]
        acc = s_str <= tau; Kc = int(acc.sum()); Sc = int(h_str[acc].sum()); cov = Kc / ncal
        s_H = int((rHc <= tau).sum()); qH = s_H / ncal                 # marginal FNR, n_H = ncal
        p_T = float((rTc <= tau).mean())                                # truthful acceptance (for Frechet)
        PER[f].append(dict(ti=ti, tau=float(tau), coverage=round(cov, 4),
                           rho_hat=(round(Sc/Kc, 4) if Kc else None), Kc=Kc, m_c=m_c, s_c=s_c,
                           psi_hat=(round(s_c/m_c, 4) if m_c else None), denom_mass=round(m_c/ncal, 4),
                           s_H=s_H, q_H=round(qH, 4), p_T=round(p_T, 4)))
        for a in ALPHAS:
            rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
        mass_pool.append((f, ti, C.minmass_p(m_c, ncal, GAMMA)))
        psiup_pool.append((f, ti, C.upper_p_psi(s_c, m_c, BETA_STAR)))
        for bF in BETA_F_ALL:
            fnrup_pool[bF].append((f, ti, C.upper_p_psi(s_H, ncal, bF)))
            fnrlow_pool[bF].append((f, ti, C.lower_p_psi(s_H, ncal, bF)))

# ===================== Holm rejections =====================
rho_ok = {(x[0], x[1], x[2]): bool(r) for x, r in zip(rho_pool, C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL))}
mass_ok = {(x[0], x[1]): bool(r) for x, r in zip(mass_pool, C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS))}
psiup_ok = {(x[0], x[1]): bool(r) for x, r in zip(psiup_pool, C.holm_reject([x[2] for x in psiup_pool], C.DELTA_PSIUP))}
low_ok = {(f, ti): bool(cache["low_ok"][(f, ALPHA_PRIMARY, ti)]) for f in FAMS for ti in range(len(cache["grids"][f]))}
# NEW FNR legs: Holm pooled over (f,ti,beta_F) within each leg's own delta
fu_keys = [(bF, x[0], x[1]) for bF in BETA_F_ALL for x in fnrup_pool[bF]]
fnrup_ok = {k: bool(r) for k, r in zip(fu_keys, C.holm_reject([x[2] for bF in BETA_F_ALL for x in fnrup_pool[bF]], DELTA_FUP))}
fl_keys = [(bF, x[0], x[1]) for bF in BETA_F_ALL for x in fnrlow_pool[bF]]
fnrlow_ok = {k: bool(r) for k, r in zip(fl_keys, C.holm_reject([x[2] for bF in BETA_F_ALL for x in fnrlow_pool[bF]], DELTA_FLOW))}

def best_cov(f, pred):
    best = None
    for ti in range(len(cache["grids"][f])):
        if pred(ti):
            rec = PER[f][ti]
            if best is None or rec["coverage"] > best["coverage"]: best = rec
    return best

# ===================== SELF-CHECK (abort before any new number) =====================
def selfcheck():
    e = []
    if ncal != 4000: e.append(f"ncal={ncal}!=4000")
    if nlow != 234: e.append(f"nlow={nlow}!=234")
    for f in FAMS:
        if len(cache["grids"][f]) != 26: e.append(f"grid[{f}]={len(cache['grids'][f])}!=26")
    qH = lambda f, tau: round(float((cache["rHc"][f] <= tau).mean()), 4)
    for f, tau, exp in [("score_sptrue", 0.2228, 0.6015), ("score_sptrue", 0.000123, 0.1295),
                        ("score_sjudge", 0.2451, 0.8852), ("score_sNLI", 0.5186, 0.657)]:
        if abs(qH(f, tau) - exp) > 5e-4: e.append(f"F1[{f}@{tau}]={qH(f,tau)}!={exp}")
    mc, sf = C.psi_counts(cache["rTc"]["score_sptrue"], cache["rHc"]["score_sptrue"], 0.222824)
    if mc != 278 or sf != 120: e.append(f"anchor m_c/s_c={mc}/{sf}!=278/120")
    ro = best_cov("score_sptrue", lambda ti: rho_ok[("score_sptrue", 0.10, ti)])
    if not ro or ro["coverage"] != 0.9605: e.append(f"rho_only={ro['coverage'] if ro else None}!=0.9605")
    tc = best_cov("score_sptrue", lambda ti: rho_ok[("score_sptrue", 0.10, ti)] and mass_ok[("score_sptrue", ti)] and psiup_ok[("score_sptrue", ti)])
    if not tc or tc["coverage"] != 0.3703: e.append(f"two_constraint={tc['coverage'] if tc else None}!=0.3703")
    return e

errs = selfcheck()
grid_hash = hashlib.sha256((";".join(f"{f}:" + ",".join(f"{t:.9f}" for t in cache["grids"][f]) for f in FAMS)).encode()).hexdigest()[:16]
if errs:
    print("SELF-CHECK FAILED — aborting before any new number:"); [print("  -", x) for x in errs]; sys.exit(1)
log(f"self-check PASS (ncal={ncal} nlow={nlow} grids=26x3; m_pairedlift F1 + m_c=278 + m9 0.9605/0.3703 reproduced). provenance grid_hash={grid_hash}")

# ===================== (A) frontiers (headline P(True)-8B, alpha=0.10) =====================
f = "score_sptrue"; a = ALPHA_PRIMARY
ro = best_cov(f, lambda ti: rho_ok[(f, a, ti)])
A = dict(rho_only=dict(coverage=ro["coverage"], psi_hat=ro["psi_hat"], q_H=ro["q_H"], tau=round(ro["tau"], 6),
                       leaks_psi=bool(ro["psi_hat"] and ro["psi_hat"] > a)),
         native_fnr={}, common_support_fnr={}, psi_gate_frozen={})
for bF in BETA_F_ALL:
    g = best_cov(f, lambda ti: rho_ok[(f, a, ti)] and fnrup_ok[(bF, f, ti)])
    A["native_fnr"][bF] = None if g is None else dict(coverage=g["coverage"], tau=round(g["tau"], 6), q_H=g["q_H"], psi_hat=g["psi_hat"], m_c=g["m_c"])
    g2 = best_cov(f, lambda ti: rho_ok[(f, a, ti)] and mass_ok[(f, ti)] and fnrup_ok[(bF, f, ti)])
    A["common_support_fnr"][bF] = None if g2 is None else dict(coverage=g2["coverage"], tau=round(g2["tau"], 6), q_H=g2["q_H"], psi_hat=g2["psi_hat"], m_c=g2["m_c"])
gp = best_cov(f, lambda ti: rho_ok[(f, a, ti)] and mass_ok[(f, ti)] and psiup_ok[(f, ti)])
A["psi_gate_frozen"][BETA_STAR] = dict(coverage=gp["coverage"], tau=round(gp["tau"], 6), psi_hat=gp["psi_hat"], m_c=gp["m_c"])
C_FNR_star = A["native_fnr"][BETA_F_STAR]["coverage"] if A["native_fnr"][BETA_F_STAR] else None
C_psi = gp["coverage"]

# ===================== (B) exhaustive 3x3 map at (beta_F*,beta*)=(0.10,0.10) =====================
def psi_status(f, ti):  return "safe" if psiup_ok[(f, ti)] else ("unsafe" if low_ok[(f, ti)] else "undet")
def fnr_status(f, ti):  return "safe" if fnrup_ok[(BETA_F_STAR, f, ti)] else ("unsafe" if fnrlow_ok[(BETA_F_STAR, f, ti)] else "undet")
STAT = ["safe", "unsafe", "undet"]
def make_map(elig):
    m = {ps: {fs: 0 for fs in STAT} for ps in STAT}; pts = {}
    for f in FAMS:
        for ti in range(len(cache["grids"][f])):
            if not elig(f, ti): continue
            ps, fs = psi_status(f, ti), fnr_status(f, ti); m[ps][fs] += 1
            pts.setdefault(f"psi-{ps}_FNR-{fs}", []).append(dict(family=FLABEL[f], tau=round(PER[f][ti]["tau"], 6),
                coverage=PER[f][ti]["coverage"], q_H=PER[f][ti]["q_H"], psi_hat=PER[f][ti]["psi_hat"], m_c=PER[f][ti]["m_c"]))
    return m, pts
elig_infer = lambda f, ti: rho_ok[(f, 0.10, ti)] and mass_ok[(f, ti)]     # rho+mass eligible (psi asserted throughout)
map_infer, pts_infer = make_map(elig_infer)
map_all, _ = make_map(lambda f, ti: True)
R_psi_notF = map_infer["safe"]["unsafe"]      # psi-safe & FNR-unsafe (witness)
n_Fs_psiu = map_infer["unsafe"]["safe"]       # FNR-safe & psi-unsafe
n_Fs_psiundet = map_infer["undet"]["safe"]    # FNR-safe & psi-undetermined
both_safe = map_infer["safe"]["safe"]
B = dict(cap=dict(beta_F=BETA_F_STAR, beta=BETA_STAR), eligible_universe="rho+mass certified (alpha=0.10)",
         status_3x3_inferential=map_infer, status_3x3_all_grid_descriptive=map_all,
         witness_psi_safe_FNR_unsafe=R_psi_notF, n_FNRsafe_psi_unsafe=n_Fs_psiu,
         n_FNRsafe_psi_undet=n_Fs_psiundet, both_safe=both_safe, witness_points=pts_infer.get("psi-safe_FNR-unsafe", []))

# ===================== (C) multi-policy + equi-coverage crosswalk =====================
gfr = best_cov(f, lambda ti: rho_ok[(f, a, ti)] and (min(1.0, PER[f][ti]["q_H"] / (1 - PER[f][ti]["p_T"]) if PER[f][ti]["p_T"] < 1 else 1.0) <= 0.10))
psi_tau = gp["tau"]; psi_ti = [ti for ti in range(len(cache["grids"][f])) if abs(PER[f][ti]["tau"] - psi_tau) < 1e-12][0]
Cx = dict(policies_at_cap_0_10=dict(
            rho_only=A["rho_only"]["coverage"], psi_gate=C_psi, native_fnr_gate=C_FNR_star,
            frechet_upper_descriptive=(gfr["coverage"] if gfr else None)),
          equi_coverage_at_psi_gate=dict(coverage=C_psi, tau=round(psi_tau, 6),
            q_H_there=PER[f][psi_ti]["q_H"], q_H_certifies_le_0_10=bool(fnrup_ok[(BETA_F_STAR, f, psi_ti)]),
            note="at the psi_g-gate operating point, is the marginal FNR itself <=0.10 certified?"))

# ===================== locked conclusions S1-S4 (from actual numbers) =====================
def s2():
    if C_FNR_star is None: return "null (no native FNR gate at beta_F*=0.10; see S4)"
    d = round(C_psi - C_FNR_star, 4)
    if d >= DCOV_MATERIAL: return f"psi-gate retains MATERIALLY more coverage ({C_psi} vs {C_FNR_star}, +{d})"
    if -DCOV_MATERIAL < d < DCOV_MATERIAL: return f"comparable coverage ({C_psi} vs {C_FNR_star}, d={d})"
    return f"native FNR gate MATERIALLY cheaper ({C_FNR_star} vs {C_psi}) -> downgrade psi_g to audit quantity"
def s3():
    if n_Fs_psiu == 0 and n_Fs_psiundet == 0: return "certified-set CONTAINMENT: FNR gate = conservative surrogate for paired control (matched cap, this benchmark)"
    if n_Fs_psiu > 0: return f"certified COUNTEREXAMPLE (n_FNRsafe&psi-unsafe={n_Fs_psiu}): FNR gate is NOT a surrogate at the matched cap"
    return f"UNRESOLVED (no counterexample; n_FNRsafe&psi-undet={n_Fs_psiundet}>0)"
S = dict(
    S1_dissociation=(f"witness R_psi-safe&FNR-unsafe={R_psi_notF} (>0 => paired criterion met where marginal FNR certifiably exceeded; "
                     f"criteria NOT interchangeable at matched cap)" if R_psi_notF > 0 else f"no witness on this frozen set (R=0)"),
    S2_coverage=s2(), S3_surrogate_direction=s3(),
    S4_null=("null: no native FNR gate certifies at beta_F*=0.10; matched-cap coverage comparison is null"
             if C_FNR_star is None else "n/a (native FNR gate certifies at beta_F*=0.10)"))

# q_H range summary (context)
qH_summary = {FLABEL[f2]: dict(min=min(r["q_H"] for r in PER[f2]), max=max(r["q_H"] for r in PER[f2])) for f2 in FAMS}

report = dict(run="R069: marginal-FNR (q_H)-constrained gate vs paired psi_g gate (AIM 2026-07-23 Major 2)",
              amendment_sha=AMEND_SHA, provenance_grid_hash=grid_hash,
              anchor=dict(split=SPLIT_SEED, prev=PREV_SEED, pi=PI, ncal=ncal, nlow=nlow, gamma=GAMMA,
                          n_H=ncal, headline_taus=HEADLINE_TAUS),
              budget=dict(delta_Fup=DELTA_FUP, delta_Flow=DELTA_FLOW, beta_F=BETA_F_ALL, beta_star=BETA_STAR,
                          claim_level_union_bounds=dict(native_fnr_gate=0.05, common_support_fnr=0.075,
                              psi_gate_frozen=0.06, witness=0.085, full_map=0.15, paper_level_simple_union=0.20)),
              q_H_range=qH_summary, A_frontiers_ptrue=A, B_status_map=B, C_crosswalk=Cx,
              locked_conclusions=S, elapsed_s=round(time.time() - T0, 1))
json.dump(report, open("results/R069_results.json", "w"), indent=2, default=str)

print("\n" + "=" * 96)
print("R069 — marginal-FNR (q_H)-constrained gate vs paired psi_g gate  (headline P(True)-8B, alpha=0.10)")
print(f"\nq_H (FNR) range on grid: {qH_summary}")
print(f"\n[A] frontiers @ cap 0.10:  rho-only cov={A['rho_only']['coverage']} (psi_hat={A['rho_only']['psi_hat']}, leaks={A['rho_only']['leaks_psi']})")
print(f"    psi_g-gate (frozen)      cov={C_psi}")
print(f"    native rho+q_H gate      beta_F=0.10 -> {A['native_fnr'][0.10]}")
print(f"                             beta_F=0.05 -> {A['native_fnr'][0.05]};  beta_F=0.20 -> {A['native_fnr'][0.20]}")
print(f"    common-support rho+mass+q_H beta_F=0.10 -> {A['common_support_fnr'][0.10]}")
print(f"\n[B] 3x3 status map (rho+mass eligible), rows=psi / cols=FNR:")
for ps in STAT: print(f"    psi-{ps:6s}: " + "  ".join(f"FNR-{fs}={map_infer[ps][fs]}" for fs in STAT))
print(f"    -> witness psi-safe&FNR-unsafe = {R_psi_notF} ; FNR-safe&psi-unsafe = {n_Fs_psiu} ; FNR-safe&psi-undet = {n_Fs_psiundet} ; both-safe = {both_safe}")
print(f"\n[C] policies @ cap 0.10: {Cx['policies_at_cap_0_10']}")
print(f"    equi-coverage @ psi-gate: {Cx['equi_coverage_at_psi_gate']}")
print(f"\n[S] locked conclusions:")
for k, v in S.items(): print(f"    {k}: {v}")
print(f"\n  wrote results/R069_results.json  ({report['elapsed_s']}s)")
print("=" * 96)
