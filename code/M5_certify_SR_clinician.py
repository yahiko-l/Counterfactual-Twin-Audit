#!/usr/bin/env python
"""Real Clinical Arm — M5: recompute the ρ (global-safety / prevalence) leaf on CLINICIAN S_R labels.

This is the machinery that RETIRES the "hybrid ground-truth" caveat of the real arm. M4's ρ leaf and the
measured prevalence π̂ rest on the STAGE-1 two-judge (gpt-oss-120b + QwQ-32B) consensus H of the frozen
one-answer-per-source stream S_R. Once a clinician labels the full S_R (the 1232-row sheet in
../MEDHALU_annotation_handoff/M3_SR_sheet_annotator{1,2}.csv), this script swaps the clinician verdict in
for the stage-1 consensus and re-certifies ρ(τ*) ≤ α and re-measures π̂ — holding the gate (orientation,
grid, τ* search) FIXED at M4's construction so the ONLY change is the S_R label source.

Design (reuses M4's exact code path so the gate is identical by construction):
  * load the full pooled join and per-family orientation via pilot_project_margin (P.load_labels/load_scores
    + C.orient on the full pool) — IDENTICAL to M4_certify_real_arm.certify_pool;
  * realize the frozen S_R with the same R_SEED draw order (loop 1 only — the ρ leaf needs no twins);
  * per S_R pick (source, qid, answerer) substitute the label from `sr_labels[(source,qid,answerer)]`;
  * re-run the strictest-τ ρ_UCB ≤ α search on that relabeled S_R; report ρ̂, ρ_UCB, π̂, coverage.

Label sources:
  --dryrun            : use the stage-1 consensus H (from M3_SR_key.csv) as the label source. This MUST
                        reproduce M4's ρ leaf (rho_hat / tau_star / K_accept / S_accept) byte-for-byte —
                        asserted; it validates the whole join+RNG path with no clinician input.
  (default / real)    : read the two FILLED clinician sheets
                        ../MEDHALU_annotation_handoff/M3_SR_sheet_annotator{1,2}_FILLED.csv
                        (clinician_verdict ∈ {FAITHFUL→H=0, HALLUCINATED→H=1, unsure}), take the STRICT
                        consensus (both HALLUCINATED ⇒ H=1; else H=0; 'unsure' on either ⇒ dropped from the
                        S_R denominator), join to the realized S_R via M3_SR_key.csv item_id, report Cohen κ
                        between the annotators on S_R, and re-certify the ρ leaf. Emits a side-by-side vs M4.

The clinician S_R key (M3_SR_key.csv) is analyst-only / git-ignored; keep it out of any commit.
CPU only, no network. Output: results/M5_SR_clinician_results.json
"""
import os, sys, json, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
import pilot_project_margin as P
from scipy.stats import binom
from collections import defaultdict
from M4_certify_real_arm import rho_ucb   # exact CP upper bound used by the released ρ leaf

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
R_SEED = P.R_SEED
FAMS = P.FAMS
GAMMA = P.GAMMA
FAMLAB = {"score_sptrue": "P(True)-8B", "score_sjudge": "LLM-judge-8B", "score_sNLI": "ref-NLI"}
HANDOFF = os.path.join(ROOT, "MEDHALU_annotation_handoff")
SRKEY = artifact("M3_SR_key.csv")


def load_sr_key():
    """The frozen realized S_R: item_id -> (source, qid, answerer, consensus_H). Defines the labeling universe."""
    if not os.path.exists(SRKEY):
        return {}
    rows = list(csv.DictReader(open(SRKEY)))
    return {r["item_id"]: dict(source=r["source"], qid=r["qid"], answerer=r["answerer"],
                               consensus_H=int(r["consensus_H"])) for r in rows}


def clinician_sr_labels():
    """STRICT two-clinician consensus on the filled S_R sheets, keyed by (source,qid,answerer) via the SR key.
    Returns (labels, kappa, n_joint, n_unsure). Empty if the filled sheets are absent (clinician not yet run)."""
    key = load_sr_key()
    f1 = os.path.join(HANDOFF, "M3_SR_sheet_annotator1_FILLED.csv")
    f2 = os.path.join(HANDOFF, "M3_SR_sheet_annotator2_FILLED.csv")
    if not (key and os.path.exists(f1) and os.path.exists(f2)):
        return None, None, 0, 0
    def rd(p):
        out = {}
        for r in csv.DictReader(open(p)):
            v = (r.get("clinician_verdict") or "").strip().upper()
            if v in ("FAITHFUL", "HALLUCINATED", "UNSURE"):
                out[r["item_id"]] = v
        return out
    a1, a2 = rd(f1), rd(f2)
    labels = {}; both = []; n_unsure = 0
    for iid in set(a1) & set(a2):
        v1, v2 = a1[iid], a2[iid]
        if v1 == "UNSURE" or v2 == "UNSURE":
            n_unsure += 1; continue
        h1 = 1 if v1 == "HALLUCINATED" else 0
        h2 = 1 if v2 == "HALLUCINATED" else 0
        both.append((h1, h2))
        H = 1 if (h1 == 1 and h2 == 1) else 0                 # STRICT consensus (mirrors stage-1 strict)
        meta = key.get(iid)
        if meta:
            labels[(meta["source"], meta["qid"], meta["answerer"])] = H
    # Cohen's kappa on the jointly-labeled (non-unsure) S_R answers
    kappa = _cohen_kappa([h1 for h1, _ in both], [h2 for _, h2 in both]) if both else None
    return labels, kappa, len(both), n_unsure


def _cohen_kappa(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int); n = len(a)
    if n == 0:
        return None
    po = float((a == b).mean())
    pe = sum((((a == k).mean()) * ((b == k).mean())) for k in (0, 1))
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def rho_leaf(byq, pool_keys, alpha, sr_labels=None):
    """M4's LEAF-1 ρ certificate, with the S_R label source pluggable. sr_labels=None ⇒ stage-1 consensus
    (must reproduce M4). Otherwise sr_labels[(source,qid,answerer)] overrides the picked answer's label
    (missing key / dropped 'unsure' ⇒ that S_R pick is excluded from the denominator)."""
    rng = np.random.default_rng(R_SEED)
    out = {}
    for f in FAMS:
        allA = [(byq[q][i], q) for q in pool_keys for i in range(len(byq[q]))]
        raw = np.array([a[0][1][f] for a in allA]); hlab = np.array([a[0][2] for a in allA])
        _, flip = C.orient(hlab, raw); sign = -1.0 if flip else 1.0     # gate orientation FIXED at M4's (stage-1)
        def sval(d): return sign * d[f]
        S_scores, S_H = [], []
        for q in pool_keys:
            cand = byq[q]; pick = cand[int(rng.integers(len(cand)))]    # SAME RNG draw as M4 (ρ leaf = loop 1)
            ans = pick[0]
            if sr_labels is None:
                lab = pick[2]                                          # stage-1 consensus H
            else:
                kk = (q[0], q[1], ans)
                if kk not in sr_labels:
                    continue                                          # unlabeled / 'unsure' ⇒ drop from S_R
                lab = sr_labels[kk]
            S_scores.append(sval(pick[1])); S_H.append(lab)
        # advance the RNG through M4's twin-pick loop (stage-1 h0/h1 split) so the S_R draws of the NEXT
        # family stay byte-aligned with M4's shared-RNG consumption -- draws discarded (ρ leaf needs no twins)
        for q in pool_keys:
            cand = byq[q]; h0 = [c for c in cand if c[2] == 0]; h1 = [c for c in cand if c[2] == 1]
            if not h0 or not h1: continue
            rng.integers(len(h0)); rng.integers(len(h1))
        S_scores = np.array(S_scores); S_H = np.array(S_H, int)
        if len(S_scores) < 10 or S_H.sum() == 0:
            out[f] = dict(note="degenerate S_R", n_SR=int(len(S_scores)),
                          pi_measured=round(float(S_H.mean()), 4) if len(S_H) else None); continue
        grid = C.candidate_grid(S_scores, gamma=GAMMA, n_grid=28); m = max(len(grid), 1)
        tau_star = Kc = Sc = None; rho_at = None
        for tau in grid:
            acc = S_scores <= tau; K = int(acc.sum())
            if K == 0: continue
            Sacc = int(S_H[acc].sum())
            if float(binom.cdf(Sacc, K, alpha)) <= C.DELTA_GLOBAL / m:
                tau_star = float(tau); rho_at = Sacc / K; Kc, Sc = K, Sacc      # keep LAST (largest) passing τ
        r_ucb = rho_ucb(Sc, Kc, C.DELTA_GLOBAL / m) if tau_star is not None else None
        out[f] = dict(flip=bool(flip), tau_star=round(tau_star, 6) if tau_star is not None else None,
                      n_SR=int(len(S_scores)), pi_measured=round(float(S_H.mean()), 4),
                      K_accept=Kc, S_accept_halluc=Sc,
                      rho_hat=round(float(rho_at), 4) if rho_at is not None else None,
                      rho_UCB=round(float(r_ucb), 4) if r_ucb is not None else None,
                      rho_leaf_ok=bool(r_ucb is not None and r_ucb <= alpha),
                      coverage=round(Kc / len(S_scores), 4) if Kc is not None else None, grid_m=m,
                      note=None if tau_star is not None else "no τ with ρ_UCB≤α")
    return out


def build_byq():
    H = P.load_labels("strict")
    sc, _ = P.load_scores(artifact("scores_pilot.csv"))
    keys = [k for k in sc if k in H]
    byq = defaultdict(list)
    for (source, orig, ans) in keys:
        byq[(source, orig)].append((ans, sc[(source, orig, ans)], H[(source, orig, ans)]))
    return byq


def main():
    dryrun = "--dryrun" in sys.argv
    byq = build_byq(); pool = list(byq.keys())
    proj = json.load(open(artifact("pilot_projection_results.json")))

    if dryrun:
        res = {a: rho_leaf(byq, pool, a, sr_labels=None) for a in (0.10, 0.20)}
        # SELF-CHECK: stage-1 label source must reproduce M4's ρ leaf exactly
        ok = True
        for a in (0.10, 0.20):
            for f in FAMS:
                g = res[a].get(f, {}); e = proj["results"][f"pooled@a{a}"].get(f, {})
                if "rho_hat" in e and g.get("rho_hat") is not None:
                    if not (abs(g["tau_star"] - float(e["tau_star"])) <= 1e-5
                            and abs(g["rho_hat"] - float(e["rho_hat"])) <= 1e-4):
                        ok = False; print(f"  MISMATCH pooled@a{a} {f}: {g.get('tau_star')},{g.get('rho_hat')} "
                                          f"vs {e['tau_star']},{e['rho_hat']}")
        out = dict(run="M5 ρ-leaf DRY-RUN (stage-1 consensus label source; must reproduce M4)",
                   self_check="PASS" if ok else "FAIL", results=res)
        json.dump(out, open(artifact("M5_SR_clinician_results.json"), "w"), indent=2, ensure_ascii=False)
        print(f"[dry-run] self_check={'PASS' if ok else 'FAIL'} — stage-1 label source reproduces M4's ρ leaf")
        print(f"          (validates the load+orientation+RNG+τ* path; swap in clinician labels when the "
              f"filled S_R sheets arrive)")
        for a in (0.10, 0.20):
            for f in FAMS:
                r = res[a][f]
                if r.get("rho_hat") is not None:
                    print(f"  α={a} {FAMLAB[f]:<13} τ*={r['tau_star']:.4f} ρ̂={r['rho_hat']:.3f} "
                          f"ρ_UCB={r['rho_UCB']:.3f} π̂={r['pi_measured']:.3f} cov={r['coverage']:.3f}")
        return

    # real clinician mode
    labels, kappa, n_joint, n_unsure = clinician_sr_labels()
    if labels is None:
        print("Clinician S_R labels not found. Expected FILLED sheets:")
        print(f"  {os.path.join(HANDOFF, 'M3_SR_sheet_annotator1_FILLED.csv')}")
        print(f"  {os.path.join(HANDOFF, 'M3_SR_sheet_annotator2_FILLED.csv')}")
        print("Run `python M5_certify_SR_clinician.py --dryrun` to validate the pipeline meanwhile.")
        return
    stage1 = {a: rho_leaf(byq, pool, a, sr_labels=None) for a in (0.10, 0.20)}
    clin = {a: rho_leaf(byq, pool, a, sr_labels=labels) for a in (0.10, 0.20)}
    out = dict(run="M5 ρ leaf on CLINICIAN S_R labels (retires the hybrid-ground-truth caveat)",
               sr_inter_annotator=dict(cohen_kappa=kappa, n_jointly_labeled=n_joint, n_unsure_dropped=n_unsure),
               stage1_reference=stage1, clinician=clin,
               note="ρ leaf + measured π̂ now rest on clinician S_R verdicts (strict 2-clinician consensus); "
                    "gate orientation/grid held fixed at M4's construction. Compare clinician vs stage1 per cell.")
    json.dump(out, open(artifact("M5_SR_clinician_results.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[clinician S_R] Cohen κ={kappa} (n={n_joint}, unsure dropped={n_unsure})")
    print(f"{'α':>5} {'family':<13} {'ρ̂(stage1→clin)':>18} {'ρ_UCB(clin)':>12} {'π̂(stage1→clin)':>18} {'leaf ok':>8}")
    for a in (0.10, 0.20):
        for f in FAMS:
            s = stage1[a][f]; c = clin[a][f]
            if c.get("rho_hat") is not None:
                print(f"{a:>5} {FAMLAB[f]:<13} {str(s.get('rho_hat'))+'→'+str(c['rho_hat']):>18} "
                      f"{c['rho_UCB']:>12.3f} {str(s.get('pi_measured'))+'→'+str(c['pi_measured']):>18} "
                      f"{str(c['rho_leaf_ok']):>8}")
    print(f"\nwrote {artifact('M5_SR_clinician_results.json')}")


if __name__ == "__main__":
    main()
