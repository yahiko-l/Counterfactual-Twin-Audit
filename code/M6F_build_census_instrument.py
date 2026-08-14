#!/usr/bin/env python
"""M6F — census-completion clinician instrument for the real-stream frozen-threshold audit.

Completes the M6-400 stratified accepted-set audit to the FULL realized stream, per the frozen
revision prespecification (refine-logs/analysis_plan_v2_20260717.md Section 4):

  main block (832 unique new research rows)
    * 730 accepted rows NOT in the M6 stratified-400 (completes 1130/1130 accepted labels)
    * 102 rejected rows (completes 1232/1232 whole-stream labels)
  + 48 hidden repeats of main-block rows (QC; stratified by source x stage-1 label; NOT flagged
    to clinicians)
  + 20 visible calibration rows (blind_id C01..C20; drawn from M6-400 BOTH-AGREE rows, half
    HALLUCINATED half FAITHFUL; excluded from analysis)
  + severity backfill block: M6-400 error/disagreement rows lacking a harm grade — measured
    EMPTY (both annotators graded severity on every HALLUCINATED verdict), recorded in meta.

Blinding as M6: rows carry blind ids only — no answerer, scores, stage-1 labels, acceptance
flags, tau, or beta-star. Both clinicians receive identical sheets and label independently.

Self-checks (abort on failure):
  * ref-NLI reconstruction reproduces M4 exactly (tau*=0.412661, K=1130, S=159)
  * new-accepted ∩ audit400 = ∅ and new-accepted ∪ audit400 = all 1130 accepted
  * counts: 730 + 102 = 832 main; 48 repeats; 20 calibration; per-doctor rows = 900

Outputs (instruments/key gitignored — answer-bearing; meta committable):
  M6F_census_instrument_annotator{1,2}.csv, M6F_census_key.csv, M6F_census_meta.json
"""
import os, sys, json, csv, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from M5_certify_SR_clinician import build_byq
from M6_build_SR_audit_sheet import (reconstruct_ref_nli_accepted, load_text, load_qk,
                                     M4_EXPECT, AUDIT_SEED, N_AUDIT)
import certify as C
from M4_certify_real_arm import rho_ucb

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
REPEAT_SEED = 20260719    # frozen before any census label is seen
CAL_SEED = 20260720
CENSUS_BLIND_SEED = 20260718
N_REPEATS_PER_CELL = 8    # 3 sources x 2 stage-1 labels x 8 = 48
N_CAL = 20

VERDICT_MAP = {"HALLUCINATED": 1, "FAITHFUL": 0}


def load_m6_consensus():
    """M6-400 per-blind-id verdicts from the two FILLED sheets + key -> item_id level."""
    key = {int(r["blind_id"]): r for r in csv.DictReader(open(artifact("M6_SR_audit_key.csv")))
           if r["blind_id"] != ""}
    verd = {}
    for a in (1, 2):
        fp = artifact(f"M6_SR_instrument_annotator{a}_FILLED.csv")
        for r in csv.DictReader(open(fp)):
            b = int(r["blind_id"])
            v = r["clinician_verdict"].strip().upper()
            sev = (r.get("severity") or "").strip().lower()
            verd.setdefault(b, {})[a] = (v, sev)
    rows = []
    for b, per in sorted(verd.items()):
        if len(per) != 2 or b not in key:
            continue
        v1, s1 = per[1]; v2, s2 = per[2]
        rows.append(dict(m6_blind_id=b, item_id=int(key[b]["item_id"]), source=key[b]["source"],
                         qid=key[b]["qid"], answerer=key[b]["answerer"],
                         v1=v1, v2=v2, s1=s1, s2=s2,
                         both_agree=(v1 == v2 and v1 in VERDICT_MAP),
                         any_error=(v1 == "HALLUCINATED" or v2 == "HALLUCINATED"),
                         disagree=(v1 != v2)))
    return rows


def main():
    byq = build_byq(); pool = list(byq.keys())
    picks, scores, H, tau_star, K, S, acc_mask, m = reconstruct_ref_nli_accepted(byq, pool)
    ok = (abs(tau_star - M4_EXPECT["tau_star"]) < 1e-5 and K == M4_EXPECT["K"] and S == M4_EXPECT["S"])
    print(f"[self-check] ref-NLI rho leaf: tau*={tau_star:.6f} K={K} S={S} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("ABORT: reconstruction != M4."); sys.exit(1)

    txt = load_text(); qk = load_qk()
    accepted = [(q, pk, float(sc)) for (q, pk), sc, a in zip(picks, scores, acc_mask) if a]
    rejected = [(q, pk, float(sc)) for (q, pk), sc, a in zip(picks, scores, acc_mask) if not a]

    # ---- reproduce the frozen M6 stratified-400 EXACTLY (same seed/path as M6) ----
    by_src = {}
    for i, (q, _, _) in enumerate(accepted): by_src.setdefault(q[0], []).append(i)
    arng = np.random.default_rng(AUDIT_SEED); audit_idx = set()
    for src, idxs in sorted(by_src.items()):
        n_s = int(round(N_AUDIT * len(idxs) / len(accepted)))
        sel = arng.choice(idxs, size=min(n_s, len(idxs)), replace=False)
        audit_idx.update(int(x) for x in sel)
    new_acc = [i for i in range(len(accepted)) if i not in audit_idx]
    assert len(audit_idx) == 400 and len(new_acc) == K - 400, (len(audit_idx), len(new_acc))
    assert not (set(new_acc) & audit_idx) and len(set(new_acc) | audit_idx) == K
    print(f"[self-check] audit400={len(audit_idx)}  new accepted={len(new_acc)}  rejected={len(rejected)}")

    # ---- main block: 730 new accepted + 102 rejected ----
    main_items = []
    for i in new_acc:
        q, pk, sc = accepted[i]
        main_items.append(dict(kind="main", src_list="accepted", idx=i, q=q, pk=pk, sc=sc, accepted=1))
    for j, (q, pk, sc) in enumerate(rejected):
        main_items.append(dict(kind="main", src_list="rejected", idx=j, q=q, pk=pk, sc=sc, accepted=0))
    assert len(main_items) == (K - 400) + len(rejected) == 832, len(main_items)

    # ---- 48 hidden repeats, stratified by source x stage-1 label over the main block ----
    rrng = np.random.default_rng(REPEAT_SEED)
    cells = {}
    for t, it in enumerate(main_items):
        cells.setdefault((it["q"][0], int(it["pk"][2])), []).append(t)
    repeats = []
    for cell, idxs in sorted(cells.items()):
        take = min(N_REPEATS_PER_CELL, len(idxs))
        for t in rrng.choice(idxs, size=take, replace=False):
            repeats.append(dict(kind="repeat", repeat_of=int(t), **{k: main_items[int(t)][k]
                            for k in ("src_list", "idx", "q", "pk", "sc", "accepted")}))
    print(f"[built] repeats={len(repeats)} over cells={{{', '.join(f'{k}:{min(N_REPEATS_PER_CELL,len(v))}' for k,v in sorted(cells.items()))}}}")

    # ---- 20 calibration rows from M6-400 both-agree (10 HALLUCINATED / 10 FAITHFUL) ----
    m6 = load_m6_consensus()
    both = [r for r in m6 if r["both_agree"]]
    halluc = [r for r in both if r["v1"] == "HALLUCINATED"]
    faith = [r for r in both if r["v1"] == "FAITHFUL"]
    crng = np.random.default_rng(CAL_SEED)
    cal = ([halluc[i] for i in crng.choice(len(halluc), size=min(10, len(halluc)), replace=False)] +
           [faith[i] for i in crng.choice(len(faith), size=min(10, len(faith)), replace=False)])
    assert len(cal) == N_CAL, len(cal)

    # ---- severity backfill need (measured; plan allows <=100) ----
    backfill = [r for r in m6 if (r["any_error"] or r["disagree"]) and
                ((r["v1"] == "HALLUCINATED" and not r["s1"]) or (r["v2"] == "HALLUCINATED" and not r["s2"]))]
    print(f"[measured] severity-backfill rows needed = {len(backfill)} (plan allowed <=100)")

    # ---- blind order over main+repeats; calibration block first, visibly marked ----
    formal = main_items + repeats
    brng = np.random.default_rng(CENSUS_BLIND_SEED)
    order = list(range(len(formal))); brng.shuffle(order)
    build_id = hashlib.sha256(json.dumps([int(x) for x in order]).encode()).hexdigest()[:16]

    def rowtext(it):
        q, pk = it["q"], it["pk"]
        Q, Kref = qk.get((q[0], str(q[1])), ("", ""))
        cand = txt.get((q[0], str(q[1]), pk[0]), "")
        return Q, Kref, cand

    inst_cols = ["build_id", "blind_id", "block", "Question", "Reference_Knowledge", "Candidate_Answer",
                 "clinician_verdict", "severity", "confidence", "notes"]
    acc_of_m6 = {int(r["item_id"]): r for r in m6}
    # accepted list indexed text for calibration rows
    for a in (1, 2):
        with open(artifact(f"M6F_census_instrument_annotator{a}.csv"), "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=inst_cols); w.writeheader()
            for ci, r in enumerate(cal, start=1):
                q, pk, sc = accepted[r["item_id"]]
                Q, Kref = qk.get((q[0], str(q[1])), ("", ""))
                cand = txt.get((q[0], str(q[1]), pk[0]), "")
                w.writerow(dict(build_id=build_id, blind_id=f"C{ci:02d}", block="calibration",
                                Question=Q, Reference_Knowledge=Kref, Candidate_Answer=cand,
                                clinician_verdict="", severity="", confidence="", notes=""))
            for bid, t in enumerate(order, start=1):
                Q, Kref, cand = rowtext(formal[t])
                w.writerow(dict(build_id=build_id, blind_id=f"F{bid:04d}", block="formal",
                                Question=Q, Reference_Knowledge=Kref, Candidate_Answer=cand,
                                clinician_verdict="", severity="", confidence="", notes=""))

    # ---- analyst key ----
    keycols = ["build_id", "blind_id", "block", "kind", "src_list", "idx", "repeat_of_blind",
               "m6_blind_id", "source", "qid", "answerer", "sNLI_r", "stage1_H", "accepted"]
    blind_of_formal = {t: f"F{bid:04d}" for bid, t in enumerate(order, start=1)}
    with open(artifact("M6F_census_key.csv"), "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(keycols)
        for ci, r in enumerate(cal, start=1):
            q, pk, sc = accepted[r["item_id"]]
            w.writerow([build_id, f"C{ci:02d}", "calibration", "calibration", "accepted", r["item_id"], "",
                        r["m6_blind_id"], q[0], str(q[1]), pk[0], f"{sc:.6f}", int(pk[2]), 1])
        for t, it in enumerate(formal):
            rep_of = blind_of_formal[it["repeat_of"]] if it["kind"] == "repeat" else ""
            q, pk = it["q"], it["pk"]
            w.writerow([build_id, blind_of_formal[t], "formal", it["kind"], it["src_list"], it["idx"], rep_of,
                        "", q[0], str(q[1]), pk[0], f"{it['sc']:.6f}", int(pk[2]), it["accepted"]])

    meta = dict(run="M6F census-completion instrument (whole realized stream, frozen tau)",
                prereg="refine-logs/analysis_plan_v2_20260717.md (commit 013ff04, sha256 a71f4e57...)",
                build_id=build_id, tau_star=tau_star, K_accepted=K, N_stream=K + len(rejected),
                self_check_reproduces_M4=bool(ok),
                blocks=dict(main_new_accepted=len(new_acc), main_rejected=len(rejected),
                            hidden_repeats=len(repeats), calibration=len(cal),
                            severity_backfill_needed=len(backfill)),
                rows_per_doctor=N_CAL + len(formal),
                seeds=dict(audit_seed_m6=AUDIT_SEED, repeat_seed=REPEAT_SEED, cal_seed=CAL_SEED,
                           census_blind_seed=CENSUS_BLIND_SEED),
                note=("Completes M6-400 to the full 1,232-row realized stream. Blinded (no answerer/scores/"
                      "flags/tau). Calibration block C01-C20 visible, excluded from analysis; 48 repeats "
                      "hidden inside the formal block. Severity backfill measured EMPTY: both M6 annotators "
                      "graded severity on every HALLUCINATED verdict. Consensus rules and Table 9R wording "
                      "are prespecified in analysis_plan_v2."))
    json.dump(meta, open(artifact("M6F_census_meta.json"), "w"), indent=2)
    print(f"[wrote] M6F_census_instrument_annotator{{1,2}}.csv ({N_CAL}+{len(formal)}={N_CAL+len(formal)} rows/doctor), "
          f"M6F_census_key.csv, M6F_census_meta.json  build_id={build_id}")


if __name__ == "__main__":
    main()
