#!/usr/bin/env python
"""
M3 COMBINED — DOMINANT-claim settle with the pqa_artificial denominator expansion (R-C).
Consumes scores_strong.csv (1k pqa_labeled, has selfcheck) + scores_strong_artificial.csv
(9k pqa_artificial, judge/ptrue/refNLI). CPU.

Headline test (C1): does a CLUSTER-VALID certified-safe gate (ρ(τ)≤α) STILL have certified
ψ_g(τ)>α on a non-trivial rejected-truthful denominator — at a realistic DEPLOYMENT PREVALENCE?

Deployment stream (cluster-valid by construction): for each source question draw ONE answer —
hallucinated w.p. π, else truthful — so rows are iid across questions ⇒ exact-binomial ρ LTT is
finite-sample valid. ρ certified at τ̂ on this stream; ψ_g audited on the FULL twins at the same τ̂.
PRIMARY π=0.10 (declared realistic prevalence); native(0.50) + {0.05,0.20} as sensitivity. Each
δ-family Holm-pooled over (family×α×τ) at a FIXED π; union of 4 families ⇒ ≤ δ_total. Pooled (10k)
and pqa_labeled-only (1k) reported (plan: stratified sensitivity). Judge = score under audit, never
a label; H from MedHallu only.
"""
import os, sys, json, csv, time
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
LAB = "data/scores/scores_strong.csv"; ART = "data/scores/scores_strong_artificial.csv"
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]   # common to both files (selfcheck = labeled-only)
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT
PI_PRIMARY = 0.10; PIS = [0.50, 0.20, 0.10, 0.05]

def load(path, src):
    rows = list(csv.DictReader(open(path)))
    for r in rows: r.setdefault("source", src)
    return rows
rows = load(LAB, "pqa_labeled") + (load(ART, "pqa_artificial") if os.path.exists(ART) else [])
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
src = np.array([r["source"] for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
byq = {}
for i in range(len(rows)):
    q, h = int(qid[i]), int(H[i])
    assert (q not in byq) or (h not in byq[q]), f"dup (qid={q},H={h})"
    byq.setdefault(q, {})[h] = i
for q, d in byq.items(): assert set(d) == {0, 1}, f"qid {q} incomplete"
QIDS = np.array(sorted(byq))
log(f"combined twins={len(QIDS)} (labeled={sum(src=='pqa_labeled')//2}, artificial={sum(src=='pqa_artificial')//2})")

def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids)
    perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

def certify_at_prevalence(qids_pool, pi, label):
    D_sel, D_cal, D_test = split(qids_pool)
    sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    ORI = {}
    for f in FAMS:
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); ORI[f] = (-RAW[f] if fl else RAW[f])
    def tw(f, qs):
        s = ORI[f]; q = [x for x in qids_pool if x in qs]
        return np.array([s[byq[x][0]] for x in q]), np.array([s[byq[x][1]] for x in q]), q
    rho_pool, low_pool, mass_pool = [], [], []
    META = {}
    for f in FAMS:
        rTs, rHs, _ = tw(f, D_sel); grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=28)
        rTc, rHc, qc = tw(f, D_cal); ncal = len(rTc)
        # deployment stream: one answer per question, hallucinated w.p. π (declared seed)
        rng = np.random.default_rng(2024)
        takeH = rng.random(ncal) < pi
        s_stream = np.where(takeH, rHc, rTc); h_stream = takeH.astype(int)
        META[f] = []
        for ti, tau in enumerate(grid):
            acc = s_stream <= tau; Kc = int(acc.sum()); Sc = int(h_stream[acc].sum())
            m_c, sfail = C.psi_counts(rTc, rHc, tau)
            mass_pool.append((f, ti, C.minmass_p(m_c, ncal, GAMMA)))
            META[f].append(dict(tau=round(float(tau), 6), rho_K=Kc, rho_S=Sc,
                                rho_hat=(round(Sc/Kc, 4) if Kc > 0 else None), m_c=m_c, s_c=sfail,
                                psi_hat=(round(sfail/m_c, 4) if m_c > 0 else None), denom_mass=round(m_c/ncal, 4)))
            for a in ALPHAS:
                rho_pool.append((f, a, ti, float(binom.cdf(Sc, Kc, a)) if Kc > 0 else 1.0))
                low_pool.append((f, a, ti, C.lower_p_psi(sfail, m_c, a)))
    rho_rej = C.holm_reject([x[3] for x in rho_pool], C.DELTA_GLOBAL)
    low_rej = C.holm_reject([x[3] for x in low_pool], C.DELTA_LOWER)
    mass_rej = C.holm_reject([x[2] for x in mass_pool], C.DELTA_MASS)
    rho_ok = {(x[0], x[1], x[2]): bool(rho_rej[i]) for i, x in enumerate(rho_pool)}
    low_ok = {(x[0], x[1], x[2]): bool(low_rej[i]) for i, x in enumerate(low_pool)}
    mass_ok = {(x[0], x[1]): bool(mass_rej[i]) for i, x in enumerate(mass_pool)}
    nlow = len(low_pool); cells = {}
    for f in FAMS:
        cells[f] = {}
        for a in ALPHAS:
            joint = []
            for ti in range(len(META[f])):
                if rho_ok[(f, a, ti)] and low_ok[(f, a, ti)] and mass_ok[(f, ti)]:
                    rec = dict(META[f][ti]); rec["psi_lcb"] = round(C.cp_lower_psi(rec["s_c"], rec["m_c"], C.DELTA_LOWER/nlow), 4)
                    joint.append(rec)
            joint.sort(key=lambda r: -(r["psi_lcb"] or 0))
            cells[f][a] = dict(joint_certified=len(joint) > 0, n_joint=len(joint), best=(joint[0] if joint else None),
                               rho_certified_any=int(sum(rho_ok[(f, a, ti)] for ti in range(len(META[f])))))
    njoint = sum(1 for f in FAMS for a in ALPHAS if cells[f][a]["joint_certified"])
    return dict(label=label, pi=pi, n_twins=len(qids_pool), joint_cells=njoint, cells=cells)

report = {"run": "M3 combined (denominator-expanded, prevalence-conditioned)", "pi_primary": PI_PRIMARY,
          "alphas": ALPHAS, "gamma": GAMMA, "delta": {"total": C.DELTA_TOTAL}}
# K4 on the combined judge set
def paired_stats(f, qids_pool):
    D_sel, _, _ = split(qids_pool); sel_idx = np.array([byq[q][h] for q in qids_pool if q in D_sel for h in (0, 1)])
    _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); s = (-RAW[f] if fl else RAW[f])
    rT = np.array([s[byq[q][0]] for q in qids_pool]); rH = np.array([s[byq[q][1]] for q in qids_pool])
    rng = np.random.default_rng(1); pa = (rH > rT) + 0.5*(rH == rT)
    bs = [pa[rng.integers(0, len(pa), len(pa))].mean() for _ in range(2000)]
    from sklearn.metrics import roc_auc_score
    return dict(paired_acc=round(float(pa.mean()), 4), CI=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                risk_auc=round(float(roc_auc_score(H, s)), 4), n=len(qids_pool))
report["K4_combined_judge"] = paired_stats("score_sjudge", QIDS.tolist())

POOLED = QIDS.tolist(); LABELED = [q for q in QIDS if q < 1000000]
log("certifying joint cell across prevalences (pooled 10k)...")
report["pooled"] = {f"pi={pi}": certify_at_prevalence(POOLED, pi, f"pooled pi={pi}") for pi in PIS}
log("certifying joint cell at primary prevalence (pqa_labeled only, sensitivity)...")
report["labeled_only"] = {f"pi={PI_PRIMARY}": certify_at_prevalence(LABELED, PI_PRIMARY, f"labeled pi={PI_PRIMARY}")}

prim = report["pooled"][f"pi={PI_PRIMARY}"]
report["verdict"] = {
    "C1_joint_certified_at_primary_pi": prim["joint_cells"] > 0,
    "headline": (f"C1 HOLDS at deployment prevalence π={PI_PRIMARY}: a cluster-valid certified-safe gate "
                 f"(ρ≤α) still has certified ψ_g>α on a non-trivial denominator (denominator-expanded)."
                 if prim["joint_cells"] > 0 else
                 f"C1 NOT certified even at π={PI_PRIMARY} with the 10k denominator expansion — report as feasibility/negative."),
    "C2_status": "DEMOTED (K4): strong-judge paired-acc≥0.5 — pathology pilot-specific."}
report["elapsed_s"] = round(time.time()-T0, 1)
json.dump(report, open("results/m3_combined_results.json", "w"), indent=2, default=str)

print("\n" + "="*86)
print(f"M3 COMBINED — {len(QIDS)} twins; K4 judge paired-acc={report['K4_combined_judge']['paired_acc']} "
      f"CI={report['K4_combined_judge']['CI']} (≥0.5 ⇒ C2 demoted)")
print(f"\nJoint cell (ρ≤α ∧ ψ_g>α ∧ denom≥γ) certified count by prevalence (pooled 10k):")
for pi in PIS:
    r = report["pooled"][f"pi={pi}"]; print(f"  π={pi}: {r['joint_cells']}/9 joint cells", end="")
    for f in FAMS:
        for a in ALPHAS:
            c = r["cells"][f][a]
            if c["joint_certified"]:
                b = c["best"]; print(f"\n      [{f} α={a}] τ̂={b['tau']} ρ̂={b['rho_hat']} m_c={b['m_c']} ψ̂={b['psi_hat']} ψ_lcb={b['psi_lcb']}", end="")
    print()
print(f"\nVERDICT: {report['verdict']['headline']}")
print("  wrote results/m3_combined_results.json")
print("="*86)
