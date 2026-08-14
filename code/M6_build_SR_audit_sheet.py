#!/usr/bin/env python
"""M6 — CORRECTED, BLINDED clinician S_R audit instrument, grounded on M4's ACTUAL ref-NLI accepted set.

Why: the earlier `M3_SR_sheet_*.csv` used an INDEPENDENT one-per-source RNG draw, matching M4's ref-NLI
ρ-leaf picks for only 228/1232 questions (~chance); `M5 --dryrun` never caught it (it used stage-1 labels
and bypassed the sheet join). This rebuilds the instrument from M4's EXACT ref-NLI picks so clinicians label
the answers the reported certificate depends on, and blinds it (no model identity / scores / inclusion flags).

Pipeline (CPU, no GPU/network):
  (1) Replicate M4/certify_pool's shared-RNG path EXACTLY (seed R_SEED, FAMS order, per-family S_R + twin
      loops) → the score_sNLI (ref-NLI) one-per-source S_R picks.
  (2) Orient on the full pool (as M4), take the strictest passing τ at α=0.20, and the ACCEPTED set
      (score ≤ τ*) — the ρ-leaf audit universe. SELF-CHECK: reproduce M4 (τ*=0.412661, K=1130, S=159).
  (3) Pre-specify a STRATIFIED-by-source 400-row audit subsample (frozen AUDIT_SEED) — the minimum-sufficient
      fixed-τ clinician audit; `--full` instead builds all K rows (retires the hybrid caveat).
  (4) Emit a BLINDED, SHUFFLED (frozen BLIND_SEED) per-clinician instrument: only blind_id + Question +
      Reference + Candidate_Answer + label columns. NO answerer, scores, stage-1 label, acceptance, or audit
      flag (fixes the blinding regression). The analyst key maps blind_id → everything.

Outputs (sheets/key gitignored — answer-bearing / de-anon; meta committable):
  M6_SR_instrument_annotator1.csv, _annotator2.csv   (blinded; 400 rows default, K with --full)
  M6_SR_audit_key.csv                                (analyst: blind_id,item_id,source,qid,answerer,sNLI_r,stage1_H,accepted,in_audit400)
  M6_SR_audit_meta.json                              (committable: counts, τ*, strata, self-check — no answers)
"""
import os, sys, json, csv, glob, math, hashlib
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
import pilot_project_margin as P
from M5_certify_SR_clinician import build_byq
from M4_certify_real_arm import rho_ucb

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
R_SEED = P.R_SEED; FAMS = P.FAMS; GAMMA = P.GAMMA
ALPHA = 0.20; NLI = "score_sNLI"
N_AUDIT = 400; AUDIT_SEED = 20260714; BLIND_SEED = 20260715   # frozen before any label is seen
M4_EXPECT = dict(tau_star=0.412661, K=1130, S=159)


def reconstruct_ref_nli_accepted(byq, pool):
    rng = np.random.default_rng(R_SEED); nli = None
    for f in FAMS:
        allA = [(byq[q][i], q) for q in pool for i in range(len(byq[q]))]
        raw = np.array([a[0][1][f] for a in allA]); hlab = np.array([a[0][2] for a in allA])
        _, flip = C.orient(hlab, raw); sign = -1.0 if flip else 1.0
        picks = [(q, byq[q][int(rng.integers(len(byq[q])))]) for q in pool]
        for q in pool:
            h0 = [c for c in byq[q] if c[2] == 0]; h1 = [c for c in byq[q] if c[2] == 1]
            if h0 and h1: rng.integers(len(h0)); rng.integers(len(h1))
        if f == NLI: nli = (picks, sign)
    picks, sign = nli
    scores = np.array([sign * pk[1][NLI] for (_, pk) in picks])
    H = np.array([pk[2] for (_, pk) in picks], int)
    grid = C.candidate_grid(scores, gamma=GAMMA, n_grid=28); m = max(len(grid), 1)
    tau_star = Kc = Sc = None
    for tau in grid:
        acc = scores <= tau; K = int(acc.sum())
        if K == 0: continue
        Sacc = int(H[acc].sum())
        if float(binom.cdf(Sacc, K, ALPHA)) <= C.DELTA_GLOBAL / m:
            tau_star = float(tau); Kc, Sc = K, Sacc
    return picks, scores, H, tau_star, Kc, Sc, (scores <= tau_star), m


def load_text():
    txt = {}
    for fp in glob.glob(artifact("natural_real_*.jsonl")):
        for l in open(fp):
            r = json.loads(l)
            if r.get("model_answer") and r.get("answerer"):
                txt[(r["source"], str(r["qid"]), r["answerer"])] = r["model_answer"]
    return txt


def load_qk():
    qk = {}
    for l in open(artifact("real_sources_questions.jsonl")):
        r = json.loads(l); qk[(r["source"], str(r["qid"]))] = (r.get("Q", ""), r.get("K", ""))
    return qk


def main():
    full = "--full" in sys.argv
    byq = build_byq(); pool = list(byq.keys())
    picks, scores, H, tau_star, K, S, acc_mask, m = reconstruct_ref_nli_accepted(byq, pool)
    ok = (abs(tau_star - M4_EXPECT["tau_star"]) < 1e-5 and K == M4_EXPECT["K"] and S == M4_EXPECT["S"])
    print(f"[self-check] ref-NLI ρ leaf: τ*={tau_star:.6f} K={K} S={S} (M4 {M4_EXPECT}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("ABORT: reconstruction != M4."); sys.exit(1)
    print(f"[self-check] ρ̂={S/K:.4f}  ρ_UCB(δ={C.DELTA_GLOBAL}/{m})={rho_ucb(S, K, C.DELTA_GLOBAL / m):.4f} (M4 0.1754)")

    txt = load_text(); qk = load_qk()
    accepted = [(q, pk, float(sc)) for (q, pk), sc, a in zip(picks, scores, acc_mask) if a]  # item_id = index
    # ---- stratified-by-source 400 audit subsample (frozen) ----
    by_src = {}
    for i, (q, _, _) in enumerate(accepted): by_src.setdefault(q[0], []).append(i)
    arng = np.random.default_rng(AUDIT_SEED); audit_idx = set(); strata = {}
    for src, idxs in sorted(by_src.items()):
        n_s = int(round(N_AUDIT * len(idxs) / len(accepted)))
        sel = arng.choice(idxs, size=min(n_s, len(idxs)), replace=False)
        audit_idx.update(int(x) for x in sel); strata[src] = dict(pool=len(idxs), sampled=int(len(sel)))
    # ---- analyst key over ALL K accepted ----
    key = []
    for i, (q, pk, sc) in enumerate(accepted):
        key.append(dict(item_id=i, source=q[0], qid=str(q[1]), answerer=pk[0], sNLI_r=sc,
                        stage1_H=int(pk[2]), accepted=1, in_audit400=1 if i in audit_idx else 0))
    # ---- blinded, shuffled instrument for the chosen scope ----
    scope = [k["item_id"] for k in key] if full else sorted(audit_idx)
    brng = np.random.default_rng(BLIND_SEED); order = list(scope); brng.shuffle(order)
    blind_of = {iid: bid for bid, iid in enumerate(order)}
    for k in key: k["blind_id"] = blind_of.get(k["item_id"], "")
    # BUILD-ID: binds the blank instrument to this key's exact blind_id->item_id map (tamper/stale evidence).
    build_id = hashlib.sha256(json.dumps(sorted((bid, iid) for iid, bid in blind_of.items())).encode()).hexdigest()[:16]
    text_of = {i: (qk.get((q[0], str(q[1])), ("", ""))[0], qk.get((q[0], str(q[1])), ("", ""))[1],
                   txt.get((q[0], str(q[1]), pk[0]), "")) for i, (q, pk, sc) in enumerate(accepted)}
    inst_cols = ["build_id", "blind_id", "Question", "Reference_Knowledge", "Candidate_Answer",
                 "clinician_verdict", "severity", "confidence", "notes"]
    for a in (1, 2):
        with open(artifact(f"M6_SR_instrument_annotator{a}.csv"), "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=inst_cols); w.writeheader()
            for bid in range(len(order)):
                Q, Kref, cand = text_of[order[bid]]
                w.writerow(dict(build_id=build_id, blind_id=bid, Question=Q, Reference_Knowledge=Kref,
                                Candidate_Answer=cand, clinician_verdict="", severity="", confidence="", notes=""))
    keycols = ["build_id", "blind_id", "item_id", "source", "qid", "answerer", "sNLI_r", "stage1_H", "accepted", "in_audit400"]
    with open(artifact("M6_SR_audit_key.csv"), "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(keycols)
        for k in key:
            w.writerow([build_id, k["blind_id"], k["item_id"], k["source"], k["qid"], k["answerer"],
                        f"{k['sNLI_r']:.6f}", k["stage1_H"], k["accepted"], k["in_audit400"]])
    meta = dict(run="M6 corrected blinded S_R audit instrument (grounded on M4 ref-NLI accepted set)",
                build_id=build_id, alpha=ALPHA, tau_star=tau_star, K_accepted=K, S_accept_halluc_stage1=S,
                rho_hat_stage1=round(S / K, 4), grid_m=m, self_check_reproduces_M4=bool(ok),
                instrument_scope=("full-%d" % K if full else "audit-%d" % len(audit_idx)),
                n_audit_400=len(audit_idx), audit_seed=AUDIT_SEED, blind_seed=BLIND_SEED, strata=strata,
                note=("Blinded instrument (blind_id only; no answerer/scores/flags). Both clinicians label every "
                      "row. Default scope = pre-specified stratified 400 (fixed-τ audit, caveat retained); "
                      "--full = all K accepted (retires the hybrid caveat). Recompute with M6_certify_SR_audit.py."))
    json.dump(meta, open(artifact("M6_SR_audit_meta.json"), "w"), indent=2)
    strata_str = ", ".join("{}:{}/{}".format(s, v["sampled"], v["pool"]) for s, v in strata.items())
    print("[built] scope={} rows | key over all {} accepted | stratified 400 = {}".format(
        len(order), K, strata_str))
    print("[wrote] M6_SR_instrument_annotator{1,2}.csv (blinded), M6_SR_audit_key.csv, M6_SR_audit_meta.json")


if __name__ == "__main__":
    main()
