#!/usr/bin/env python
"""
B4_recompute_cell.py -- cell-parameterized human-adjudication recompute for the THREE headline
cells (P(True) / ref-NLI / LLM-judge), the ref-NLI/judge extension of B4_human_consensus_analysis.py
(paper-review #4).

Identical logic to the released P(True) recompute (per-annotator eta + relabel LCB + Cohen kappa +
strict/lenient consensus + measured-eta band; conservative -- a doubted twin role is dropped from
numerator and denominator), but keyed on the chosen cell's denominator (in_<cell>) and gate outcome
(<cell>_failpair) from B4_key_allcells.csv. Same global DELTA (NLOW=234) as the released LCBs.

  CELL=ptrue  -> in_ptrue ,  ptrue_failpair  , alpha 0.10 , released LCB 0.327  (m_c 278, s_c 120)
  CELL=refnli -> in_refnli,  refnli_failpair , alpha 0.10 , released LCB 0.2132 (m_c 488, s_c 138)
  CELL=judge  -> in_judge ,  judge_failpair  , alpha 0.20 , released LCB 0.4731 (m_c 347, s_c 198)

Inputs:
  B4_key_allcells.csv  (run B4_extend_key_allcells.py first)
  B4_sheet_annotator1_allcells.csv / B4_sheet_annotator2_allcells.csv   (the two filled sheets;
    override with B4_D1=/B4_D2=).  Falls back to *_FILLED.csv (P(True)-only) so CELL=ptrue
    reproduces the released result with no extra files.

Output:  B4_human_adjudication_<cell>_results.json   (override with B4_OUT=)

Run:  CELL=refnli python code/B4_recompute_cell.py
"""
import os, csv, io, math, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
from collections import Counter

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
CELL = os.environ.get("CELL", "ptrue").strip().lower()
SPEC = {  # cell -> (membership_flag, failpair_col, alpha, released_s_c, released_m_c, released_lcb)
    "ptrue":  ("in_ptrue",  "ptrue_failpair",  0.10, 120, 278, 0.327),
    "refnli": ("in_refnli", "refnli_failpair", 0.10, 138, 488, 0.2132),
    "judge":  ("in_judge",  "judge_failpair",  0.20, 198, 347, 0.4731),
}
assert CELL in SPEC, f"CELL must be one of {list(SPEC)}, got {CELL!r}"
MEMB, FAILCOL, ALPHA, S0, M0, LCB0 = SPEC[CELL]

KEY_PATH = os.environ.get("B4_KEY", artifact("B4_key_allcells.csv"))
def _default_sheet(n):
    a = artifact(f"B4_sheet_annotator{n}_allcells.csv")
    return a if os.path.exists(a) else artifact(f"B4_sheet_annotator{n}_FILLED.csv")
D1_PATH = os.environ.get("B4_D1", _default_sheet(1))
D2_PATH = os.environ.get("B4_D2", _default_sheet(2))
OUT = os.environ.get("B4_OUT", artifact(f"B4_human_adjudication_{CELL}_results.json"))

NLOW = 234
DELTA = C.DELTA_LOWER / NLOW
assert abs(C.cp_lower_psi(S0, M0, DELTA) - LCB0) < 5e-3, f"NLOW normalization drifted from released {LCB0}"


def load(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return {int(r["item_id"]): r for r in csv.DictReader(io.StringIO(raw.decode(enc)))}
        except Exception:
            pass
    raise RuntimeError(f"cannot decode {path}")


def main():
    if not os.path.exists(KEY_PATH):
        sys.exit(f"missing {KEY_PATH}; run B4_extend_key_allcells.py first.")
    key = {int(r["item_id"]): r for r in csv.DictReader(open(KEY_PATH))}
    D1, D2 = load(D1_PATH), load(D2_PATH)
    head = [i for i in key if key[i][MEMB] == "True"]
    n = len(head)
    assert n == M0, f"{CELL}: head m_c={n} != expected {M0}"

    def hi(D, i): return (D[i]["hallucination_in"] or "").strip().lower()
    def flag(D, i, L): h = hi(D, i); return h == L.lower() or h == "both"
    def roles(i): h = "A" if key[i]["A_role"] == "hallucinated" else "B"; return h, ("B" if h == "A" else "A")
    def t_ok(D, i): _, t = roles(i); return not flag(D, i, t)
    def h_real(D, i): h, _ = roles(i); return flag(D, i, h)
    def confirms(D, i): return h_real(D, i) and t_ok(D, i)
    def isfail(i): return key[i][FAILCOL] == "True"

    # label coverage (a blank row is an un-adjudicated pair -> PENDING, not a real "non-confirm")
    def labeled(D, i): return hi(D, i) != ""
    cov1 = sum(labeled(D1, i) for i in head); cov2 = sum(labeled(D2, i) for i in head)
    fully = (cov1 == n and cov2 == n)

    def eta(D):
        by = {}
        for d in sorted(set(key[i]["difficulty"] for i in head)):
            ids = [i for i in head if key[i]["difficulty"] == d and labeled(D, i) and hi(D, i) != "unsure"]
            if ids:
                by[d] = [round(sum(0 if confirms(D, i) else 1 for i in ids) / len(ids), 4), len(ids)]
        ids = [i for i in head if labeled(D, i) and hi(D, i) != "unsure"]
        e = round(sum(0 if confirms(D, i) else 1 for i in ids) / len(ids), 4) if ids else None
        return e, len(ids), by

    def relabel(D):
        m = s = 0
        for i in head:
            if not labeled(D, i) or hi(D, i) == "unsure":   # conservative: trust the MedHallu twin role
                m += 1; s += isfail(i); continue
            if t_ok(D, i):
                m += 1
                if isfail(i) and h_real(D, i): s += 1
        return s, m, round(C.cp_lower_psi(s, m, DELTA), 4)

    def band(e):
        if e is None: return None
        swc = int(math.floor(S0 * (1 - e)))
        return round(C.cp_lower_psi(swc, M0, DELTA), 4)

    def consensus(mode):
        m = s = nonconf = nlab = 0
        for i in head:
            if not (labeled(D1, i) and labeled(D2, i)):     # un-adjudicated -> conservative (trust role)
                m += 1; s += isfail(i); continue
            nlab += 1
            to = (t_ok(D1, i) and t_ok(D2, i)) if mode == "strict" else (t_ok(D1, i) or t_ok(D2, i))
            hr = (h_real(D1, i) and h_real(D2, i)) if mode == "strict" else (h_real(D1, i) or h_real(D2, i))
            if to:
                m += 1
                if isfail(i) and hr: s += 1
            cf = (confirms(D1, i) and confirms(D2, i)) if mode == "strict" else (confirms(D1, i) or confirms(D2, i))
            if not cf: nonconf += 1
        e = round(nonconf / nlab, 4) if nlab else None
        return dict(eta=e, s_c=s, m_c=m, relabel_psi_lcb=round(C.cp_lower_psi(s, m, DELTA), 4), band_psi_lcb=band(e))

    # Cohen kappa over jointly-labeled rows only
    both = [i for i in head if labeled(D1, i) and labeled(D2, i)]
    a = [hi(D1, i) for i in both]; b = [hi(D2, i) for i in both]
    if both:
        po = sum(x == y for x, y in zip(a, b)) / len(both)
        ca, cb = Counter(a), Counter(b)
        pe = sum((ca[l] / len(both)) * (cb[l] / len(both)) for l in set(a + b))
        kappa = round((po - pe) / (1 - pe), 4) if pe != 1 else None
    else:
        po = kappa = None

    res = {}
    for nm, D in (("doctor_1", D1), ("doctor_2", D2)):
        e, ne, by = eta(D); s, m, l = relabel(D)
        res[nm] = dict(eta_overall=e, eta_n=ne, eta_by_difficulty=by,
                       relabel=dict(s_c=s, m_c=m, psi_lcb=l), band_psi_lcb=band(e), still_fails=bool(l > ALPHA))
    res["consensus_strict"] = consensus("strict")
    res["consensus_lenient"] = consensus("lenient")
    minlcb = min(res["doctor_1"]["relabel"]["psi_lcb"], res["doctor_2"]["relabel"]["psi_lcb"],
                 res["consensus_strict"]["relabel_psi_lcb"], res["consensus_lenient"]["relabel_psi_lcb"])

    status = "COMPLETE" if fully else "PENDING_LABELS"
    out = dict(
        run=f"B4 human adjudication recompute -- {CELL} cell",
        cell={"ptrue": "P(True)-8B, pi=0.10, alpha=0.10", "refnli": "ref-NLI, pi=0.10, alpha=0.10",
              "judge": "LLM-judge-8B, pi=0.10, alpha=0.20"}[CELL],
        status=status,
        label_coverage=dict(m_c=n, labeled_doctor_1=cov1, labeled_doctor_2=cov2, fully_adjudicated=fully,
                            note=("blank rows are treated CONSERVATIVELY (MedHallu twin role trusted) until "
                                  "adjudicated, so a PENDING result is a valid lower bound but not the final "
                                  "human-relabeled number")),
        original=dict(s_c=S0, m_c=M0, psi_lcb=LCB0, alpha=ALPHA),
        inter_annotator=dict(jointly_labeled=len(both), raw_agreement=(round(po, 4) if po is not None else None),
                             cohen_kappa=kappa,
                             raw_label_disagreements=sum(x != y for x, y in zip(a, b)) if both else 0),
        n_annotators=2, third_adjudicator=False,
        results=res, min_relabel_psi_lcb=minlcb, alpha=ALPHA,
        verdict=(("WITHIN-QUESTION FAILURE SURVIVES HUMAN ADJUDICATION (min psi_g LCB %.4f > alpha %.2f)" % (minlcb, ALPHA))
                 if (fully and minlcb > ALPHA) else
                 ("PENDING: %d/%d (D1) and %d/%d (D2) pairs adjudicated; current conservative min LCB %.4f"
                  % (cov1, n, cov2, n, minlcb)) if not fully else
                 "DOES NOT survive"),
        note=("Same conservative recompute and global DELTA (NLOW=234) as the released P(True) cell; "
              "reproduces P(True) 0.327 exactly when CELL=ptrue. Gate outcome (<cell>_failpair) from "
              "B4_key_allcells.csv; human labels only adjust the rejected-truthful denominator and the "
              "accepted-hallucination numerator."))
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("wrote", OUT)
    print("[%s] status=%s  coverage D1/D2=%d/%d, %d/%d  kappa=%s  min relabel psi_g LCB=%.4f (alpha=%.2f) -> %s"
          % (CELL, status, cov1, n, cov2, n, kappa, minlcb, ALPHA,
             ("SURVIVES" if (fully and minlcb > ALPHA) else "PENDING" if not fully else "does NOT survive")))


if __name__ == "__main__":
    main()
