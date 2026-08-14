#!/usr/bin/env python
"""
B4_extend_key_allcells.py -- additive extension of the B4 unblinding key with the ref-NLI and
LLM-judge gate outcomes, so the human-adjudication recompute can be run for those two headline
cells as well as P(True) (paper-review #4).

NON-DESTRUCTIVE: reads the released B4_key.csv (audit evidence -- never overwritten) and
scores_strong[_artificial].csv, recomputes the D_cal rejected-truthful denominators and per-pair
"fail" flags for score_sNLI and score_sjudge EXACTLY as m11_build_blind_sheet.py did for
score_sptrue (same split seed 7, same orientation, same headline thresholds), and writes a NEW
file:

  B4_key_allcells.csv  = B4_key.csv  +  {refnli_failpair, judge_failpair,
                                          sNLI_rT, sNLI_rH, sjudge_rT, sjudge_rH}

Self-validation: asserts the per-cell (m_c, s_c) reproduce the released Table-1 lower bounds
(ref-NLI 0.2132 on m_c=488 ; judge 0.4731 on m_c=347), i.e. the same DELTA normalization the
released P(True) 0.327 uses (NLOW=234). CPU only; no network.

Run:  python code/B4_extend_key_allcells.py
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
LAB = artifact("scores_strong.csv")
ART = artifact("scores_strong_artificial.csv")
KEY_IN = artifact("B4_key.csv")
KEY_OUT = artifact("B4_key_allcells.csv")

# headline (family, oriented threshold, expected D_cal m_c, released Table-1 LCB, headline alpha)
HEADLINE = {
    "score_sptrue": (0.222824, 278, 0.327, 0.10),
    "score_sNLI":   (0.518631, 488, 0.2132, 0.10),
    "score_sjudge": (0.245085, 347, 0.4731, 0.20),
}
NLOW = 234                       # |alphas| * sum_f |grid_f| -- identical global normalization as the
DELTA = C.DELTA_LOWER / NLOW     # released P(True) recompute (reproduces 0.327 / 0.2132 / 0.4731)


def load(path, src):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r.setdefault("source", src)
    return rows


def main():
    rows = load(LAB, "pqa_labeled") + load(ART, "pqa_artificial")
    qid = np.array([int(r["qid"]) for r in rows])
    H = np.array([int(r["H"]) for r in rows])
    FAMS = list(HEADLINE)
    RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
    byq = {}
    for i in range(len(rows)):
        byq.setdefault(int(qid[i]), {})[int(H[i])] = i
    QIDS = sorted(byq)

    def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
        qids = np.asarray(qids)
        perm = np.random.default_rng(seed).permutation(len(qids))
        n = len(qids)
        a = int(n * fr[0]); b = a + int(n * fr[1])
        return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())

    D_sel, D_cal, D_test = split(QIDS)
    sel_idx = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
    cal_q = [q for q in QIDS if q in D_cal]

    DEN = {}   # family -> {qid: (rT, rH, is_fail)}
    for f in FAMS:
        tau, exp_mc, exp_lcb, _ = HEADLINE[f]
        _, fl = C.orient(H[sel_idx], RAW[f][sel_idx])
        s = (-RAW[f] if fl else RAW[f])
        d = {}
        for q in cal_q:
            rT = float(s[byq[q][0]]); rH = float(s[byq[q][1]])
            if rT > tau:                                  # truthful twin rejected
                d[q] = (rT, rH, bool(rH <= tau))          # is_fail = hallucinated twin accepted
        DEN[f] = d
        s_c = sum(v[2] for v in d.values())
        lcb = round(C.cp_lower_psi(s_c, len(d), DELTA), 4)
        assert len(d) == exp_mc, f"{f}: m_c={len(d)} != expected {exp_mc}"
        assert abs(lcb - exp_lcb) < 5e-3, f"{f}: recomputed LCB {lcb} != released {exp_lcb} (DELTA drift)"
        print(f"[ok] {f:13s} tau={tau} m_c={len(d)} s_c={s_c} psi_g_LCB={lcb} (released {exp_lcb})")

    # ---- additive merge onto the released key (by qid) ----
    key = list(csv.DictReader(open(KEY_IN)))
    new_cols = ["refnli_failpair", "judge_failpair", "sNLI_rT", "sNLI_rH", "sjudge_rT", "sjudge_rH"]
    for r in key:
        q = int(r["qid"])
        dn, dj = DEN["score_sNLI"], DEN["score_sjudge"]
        r["refnli_failpair"] = str(dn[q][2]) if q in dn else ""
        r["judge_failpair"]  = str(dj[q][2]) if q in dj else ""
        r["sNLI_rT"]   = dn[q][0] if q in dn else ""
        r["sNLI_rH"]   = dn[q][1] if q in dn else ""
        r["sjudge_rT"] = dj[q][0] if q in dj else ""
        r["sjudge_rH"] = dj[q][1] if q in dj else ""
        # consistency: membership flags in the released key must match the recomputed denominators
        assert (r["in_refnli"] == "True") == (q in dn), f"in_refnli mismatch at qid {q}"
        assert (r["in_judge"] == "True") == (q in dj), f"in_judge mismatch at qid {q}"

    cols = list(key[0].keys())
    with open(KEY_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader(); w.writerows(key)
    nref = sum(r["refnli_failpair"] == "True" for r in key)
    njud = sum(r["judge_failpair"] == "True" for r in key)
    print(f"\nwrote {KEY_OUT}  (+{len(new_cols)} cols; refnli fail pairs={nref}, judge fail pairs={njud})")
    print("membership flags in released B4_key.csv are consistent with the recomputed ref-NLI/judge denominators.")


if __name__ == "__main__":
    main()
