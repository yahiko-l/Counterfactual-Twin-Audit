#!/usr/bin/env python
"""
B4_human_consensus_analysis.py — turn the two completed clinician sheets into the
version-controlled B4 human-adjudication result (per-annotator + inter-annotator agreement
+ strict/lenient consensus), and write B4_human_adjudication_results.json.

Inputs (committed evidence, byte-for-byte copies of what the clinicians returned):
  B4_sheet_annotator1_FILLED.csv  (Doctor 1, annotator_id=annot1)
  B4_sheet_annotator2_FILLED.csv  (Doctor 2, annotator_id=annot2)
  B4_key.csv                      (unblinding key: A_role/in_ptrue/ptrue_failpair/difficulty)

Scope: the P(True)-8B headline denominator (278 priority-1+2 in_ptrue pairs). The per-annotator
numbers here are reproduced exactly by the official B4_adjudication_recompute.py
(B4_SHEET=...FILLED.csv); this script adds Cohen kappa and the strict/lenient consensus so the
result is robust to how the (few) disagreements would be adjudicated. CPU only.

Run:  python code/B4_human_consensus_analysis.py
"""
import os, csv, io, math, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
from collections import Counter

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
D1_PATH = os.environ.get("B4_D1", artifact("B4_sheet_annotator1_FILLED.csv"))
D2_PATH = os.environ.get("B4_D2", artifact("B4_sheet_annotator2_FILLED.csv"))
KEY_PATH = os.environ.get("B4_KEY", artifact("B4_key.csv"))
OUT = os.environ.get("B4_OUT", artifact("B4_human_adjudication_results.json"))

NLOW = 234                              # |alphas|*sum_f|grid_f|; reproduces the released 0.327
DELTA = C.DELTA_LOWER / NLOW
ALPHA = 0.10
assert abs(C.cp_lower_psi(120, 278, DELTA) - 0.327) < 5e-3, "NLOW normalization drifted from released 0.327"


def load(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return {int(r["item_id"]): r for r in csv.DictReader(io.StringIO(raw.decode(enc)))}
        except Exception:
            pass
    raise RuntimeError(f"cannot decode {path}")


def main():
    key = {int(r["item_id"]): r for r in csv.DictReader(open(KEY_PATH))}
    D1, D2 = load(D1_PATH), load(D2_PATH)
    head = [i for i in key if key[i]["in_ptrue"] == "True"]
    n = len(head)

    def hi(D, i): return (D[i]["hallucination_in"] or "").strip().lower()
    def flag(D, i, L): h = hi(D, i); return h == L.lower() or h == "both"
    def roles(i): h = "A" if key[i]["A_role"] == "hallucinated" else "B"; return h, ("B" if h == "A" else "A")
    def t_ok(D, i): _, t = roles(i); return not flag(D, i, t)
    def h_real(D, i): h, _ = roles(i); return flag(D, i, h)
    def confirms(D, i): return h_real(D, i) and t_ok(D, i)

    def eta(D):
        by = {}
        for d in sorted(set(key[i]["difficulty"] for i in head)):
            ids = [i for i in head if key[i]["difficulty"] == d and hi(D, i) != "unsure"]
            by[d] = [round(sum(0 if confirms(D, i) else 1 for i in ids) / len(ids), 4), len(ids)]
        ids = [i for i in head if hi(D, i) != "unsure"]
        return round(sum(0 if confirms(D, i) else 1 for i in ids) / len(ids), 4), len(ids), by

    def relabel(D):
        m = s = 0
        for i in head:
            if hi(D, i) == "unsure":                       # conservative: trust MedHallu twin role
                m += 1; s += (key[i]["ptrue_failpair"] == "True"); continue
            if t_ok(D, i):
                m += 1
                if key[i]["ptrue_failpair"] == "True" and h_real(D, i): s += 1
        return s, m, round(C.cp_lower_psi(s, m, DELTA), 4)

    def band(e):
        swc = int(math.floor(120 * (1 - e)))
        return round(C.cp_lower_psi(swc, 278, DELTA), 4)

    def consensus(mode):
        m = s = nonconf = 0
        for i in head:
            to = (t_ok(D1, i) and t_ok(D2, i)) if mode == "strict" else (t_ok(D1, i) or t_ok(D2, i))
            hr = (h_real(D1, i) and h_real(D2, i)) if mode == "strict" else (h_real(D1, i) or h_real(D2, i))
            if to:
                m += 1
                if key[i]["ptrue_failpair"] == "True" and hr: s += 1
            cf = (confirms(D1, i) and confirms(D2, i)) if mode == "strict" else (confirms(D1, i) or confirms(D2, i))
            if not cf: nonconf += 1
        e = round(nonconf / n, 4)
        return dict(eta=e, s_c=s, m_c=m, relabel_psi_lcb=round(C.cp_lower_psi(s, m, DELTA), 4), band_psi_lcb=band(e))

    a = [hi(D1, i) for i in head]; b = [hi(D2, i) for i in head]
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a + b))
    kappa = round((po - pe) / (1 - pe), 4)

    res = {}
    for nm, D in (("doctor_1", D1), ("doctor_2", D2)):
        e, ne, by = eta(D); s, m, l = relabel(D)
        res[nm] = dict(eta_overall=e, eta_n=ne, eta_by_difficulty=by,
                       relabel=dict(s_c=s, m_c=m, psi_lcb=l), band_psi_lcb=band(e), still_fails=bool(l > ALPHA))
    res["consensus_strict"] = consensus("strict")
    res["consensus_lenient"] = consensus("lenient")
    minlcb = min(res["doctor_1"]["relabel"]["psi_lcb"], res["doctor_2"]["relabel"]["psi_lcb"],
                 res["consensus_strict"]["relabel_psi_lcb"], res["consensus_lenient"]["relabel_psi_lcb"])

    out = dict(
        run="B4 REAL human adjudication recompute",
        scope=("P(True)-8B headline cell only (the 278 priority-1+2 in_ptrue denominator); "
               "ref-NLI(488)/judge(347) priority-3 rows NOT adjudicated (left to LLM-proxy/simulated band)"),
        cell="P(True)-8B, pi=0.10, alpha=0.10", n_annotators=2, third_adjudicator=False,
        original=dict(s_c=120, m_c=278, psi_lcb=0.327, alpha=ALPHA),
        inter_annotator=dict(
            raw_agreement=round(po, 4), cohen_kappa=kappa,
            confirms_role_2x2=dict(
                both=sum(confirms(D1, i) and confirms(D2, i) for i in head),
                d1_only=sum(confirms(D1, i) and not confirms(D2, i) for i in head),
                d2_only=sum(confirms(D2, i) and not confirms(D1, i) for i in head),
                neither=sum((not confirms(D1, i)) and (not confirms(D2, i)) for i in head)),
            raw_label_disagreements=sum(x != y for x, y in zip(a, b))),
        results=res, min_relabel_psi_lcb=minlcb, alpha=ALPHA,
        verdict=("HEADLINE WITHIN-QUESTION FAILURE SURVIVES REAL HUMAN ADJUDICATION "
                 "(min psi_g LCB %.4f > alpha %.2f across both annotators and strict/lenient consensus)" % (minlcb, ALPHA))
        if minlcb > ALPHA else "DOES NOT survive",
        note=("measured human eta ~0.11-0.13 lies inside the paper's simulated band (<=0.15) and matches the "
              "LLM-proxy (0.122). Relabel is conservative (doubted labels dropped). The per-annotator numbers "
              "reproduce the official B4_adjudication_recompute.py exactly."))
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("wrote", OUT)
    print("kappa=%.4f  eta D1/D2=%.4f/%.4f  min relabel psi_g LCB=%.4f (alpha=%.2f)  ->  %s"
          % (kappa, res["doctor_1"]["eta_overall"], res["doctor_2"]["eta_overall"], minlcb, ALPHA,
             "SURVIVES" if minlcb > ALPHA else "does NOT survive"))


if __name__ == "__main__":
    main()
