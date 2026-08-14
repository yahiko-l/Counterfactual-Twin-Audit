#!/usr/bin/env python
"""
B4 LLM-PROXY (full-965) recompute across ALL THREE headline cells (P(True)/ref-NLI/judge) — extends
B4_llm_recompute.py (P(True) only) to the ref-NLI and LLM-judge headline cells. LLM CROSS-CHECK, NOT
human clinical adjudication (still pending).

The B4 key only stores `ptrue_failpair`, so we recompute each family's rejected-truthful failure flags
DIRECTLY from the score CSVs using the SAME DEN logic as m11_build_blind_sheet.py (split seed 7, orient
on D_sel, headline thresholds), then map qid -> item_id via B4_key.csv. Integrity guards: (i) each
family's recomputed m_c/s_c matches the released cell (278/120, 488/138, 347/198); (ii) the recomputed
P(True) failpair flags match the key's stored ptrue_failpair exactly; (iii) cp_lower_psi reproduces each
released psi_lcb (0.327 / 0.2132 / 0.4731) at Holm level DELTA_LOWER/NLOW.

Per cell reports: measured proxy eta (strict: an item counts as confirmed only if the LLM flags the
hallucinated twin and NOT the truthful twin), the measured-eta worst-case band, and the direct
LLM-relabeled CP lower bound, plus the verdict (does the failure still certify, lcb > alpha?).
"""
import os, sys, csv, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

SHEET = "results/B4_sheet_gpt55.csv"; KEY = "results/B4_key.csv"
LAB = "data/scores/scores_strong.csv"; ART = "data/scores/scores_strong_artificial.csv"
NLOW = 234; delta_eff = C.DELTA_LOWER / NLOW

# cell -> (family score col, in_flag, oriented tau, m_c0, s_c0, alpha, released psi_lcb)
CELLS = {
    "P(True)": ("score_sptrue", "in_ptrue",  0.222824, 278, 120, 0.10, 0.327),
    "ref-NLI": ("score_sNLI",   "in_refnli", 0.518631, 488, 138, 0.10, 0.2132),
    "judge":   ("score_sjudge", "in_judge",  0.245085, 347, 198, 0.20, 0.4731),
}
for nm, (col, flag, tau, mc, sc, a, rel) in CELLS.items():
    got = C.cp_lower_psi(sc, mc, delta_eff)
    assert abs(got - rel) < 5e-3, f"{nm} NLOW normalization drift: cp_lower_psi({sc},{mc})={got} != {rel}"

# ---- recompute per-family rejected-truthful failure flags from scores (mirror m11 exactly) ----
FAMS = [c[0] for c in CELLS.values()]
def load(path, src):
    rows = list(csv.DictReader(open(path)))
    for r in rows: r.setdefault("source", src)
    return rows
rows = load(LAB, "pqa_labeled") + load(ART, "pqa_artificial")
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
byq = {}
for i in range(len(rows)): byq.setdefault(int(qid[i]), {})[int(H[i])] = i
QIDS = sorted(byq)
def split(qids, seed=7, fr=(0.40, 0.40, 0.20)):
    qids = np.asarray(qids); perm = np.random.default_rng(seed).permutation(len(qids)); n = len(qids)
    a = int(n*fr[0]); b = a + int(n*fr[1])
    return set(qids[perm[:a]].tolist()), set(qids[perm[a:b]].tolist()), set(qids[perm[b:]].tolist())
D_sel, D_cal, D_test = split(QIDS)
sel_idx = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
cal_q = [q for q in QIDS if q in D_cal]
DEN = {}   # family -> {qid: is_fail (hallucinated accepted among rejected-truthful)}
for nm, (f, flag, tau, mc, sc, a, rel) in CELLS.items():
    _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); s = (-RAW[f] if fl else RAW[f])
    d = {}
    for q in cal_q:
        rT = float(s[byq[q][0]]); rH = float(s[byq[q][1]])
        if rT > tau: d[q] = bool(rH <= tau)
    assert len(d) == mc, f"{nm}: recomputed m_c={len(d)} != {mc}"
    assert sum(d.values()) == sc, f"{nm}: recomputed s_c={sum(d.values())} != {sc}"
    DEN[f] = d

# ---- load LLM verdicts + key ----
sheet = {int(r["item_id"]): r for r in csv.DictReader(open(SHEET))}
key = {int(r["item_id"]): r for r in csv.DictReader(open(KEY))}

def llm_halluc_on(letter, i):
    hi = (sheet[i]["hallucination_in"] or "").strip().lower()
    return hi == letter.lower() or hi == "both"
def h_t_letters(i):
    h = "A" if key[i]["A_role"] == "hallucinated" else "B"; return h, ("B" if h == "A" else "A")
def confirms_role(i):
    h, t = h_t_letters(i); return llm_halluc_on(h, i) and not llm_halluc_on(t, i)

# integrity (ii): recomputed P(True) failpair == key ptrue_failpair
for i in key:
    if key[i]["in_ptrue"] == "True":
        assert DEN["score_sptrue"][int(key[i]["qid"])] == (key[i]["ptrue_failpair"] == "True"), \
            f"ptrue failpair mismatch at item {i}"

report = {"run": "B4 LLM-proxy (gpt-5.5-codex) recompute across all 3 headline cells -- cross-check, NOT human",
          "NLOW": NLOW, "delta_eff": delta_eff, "cells": {}}
print("=" * 96)
print("B4 LLM-PROXY (gpt-5.5-codex) — per-cell recompute [NOT human clinical adjudication]")
print("=" * 96)
for nm, (f, flag, tau, mc0, sc0, alpha, rel) in CELLS.items():
    ids = [i for i in key if key[i][flag] == "True"]
    isfail = {i: DEN[f][int(key[i]["qid"])] for i in ids}
    assert sum(isfail.values()) == sc0
    # measured proxy eta (strict)
    adj = [i for i in ids if (sheet[i]["hallucination_in"] or "").strip().lower() != "unsure"]
    eta = sum(0 if confirms_role(i) else 1 for i in adj) / len(adj)
    # by difficulty
    strata = {}
    for d in sorted(set(key[i]["difficulty"] for i in ids)):
        a_d = [i for i in adj if key[i]["difficulty"] == d]
        strata[d] = (round(sum(0 if confirms_role(i) else 1 for i in a_d) / len(a_d), 4), len(a_d)) if a_d else (None, 0)
    # (a) measured-eta worst-case band
    s_wc = int(math.floor(sc0 * (1 - eta))); band = round(C.cp_lower_psi(s_wc, mc0, delta_eff), 4)
    # (b) direct LLM-relabeled
    m_c = s_c = 0
    for i in ids:
        if (sheet[i]["hallucination_in"] or "").strip().lower() == "unsure":
            m_c += 1; s_c += isfail[i]; continue
        h, t = h_t_letters(i)
        if not llm_halluc_on(t, i):                       # truthful twin not flagged -> keep in denom
            m_c += 1
            if isfail[i] and llm_halluc_on(h, i): s_c += 1
    lcb = round(C.cp_lower_psi(s_c, m_c, delta_eff), 4)
    still = bool(lcb > alpha)
    report["cells"][nm] = dict(family=f, alpha=alpha, m_c0=mc0, s_c0=sc0, released_lcb=rel,
                               eta=round(eta, 4), eta_by_difficulty=strata,
                               band_lcb=band, relabel=dict(s_c=s_c, m_c=m_c, lcb=lcb), still_fails=still)
    print(f"\n[{nm}  alpha={alpha}  m_c={mc0} s_c={sc0}  released psi_lcb={rel}]")
    print(f"   measured proxy eta = {eta:.4f}  (" + ", ".join(f"{d}:{v[0]}(n={v[1]})" for d, v in strata.items()) + ")")
    print(f"   (a) measured-eta band   : psi_lcb = {band}  (s_wc={s_wc}/{mc0})")
    print(f"   (b) LLM-relabeled       : s_c {sc0}->{s_c}, m_c {mc0}->{m_c}, psi_lcb = {lcb}")
    print(f"   VERDICT: {'STILL FAILS' if still else 'NO LONGER certified-fails'} (psi_lcb {lcb} {'>' if still else '<='} alpha {alpha})")

json.dump(report, open("results/B4_llm_recompute_allcells_results.json", "w"), indent=2)
allfail = all(report["cells"][nm]["still_fails"] for nm in CELLS)
print("\n" + "=" * 96)
print(f"SUMMARY: all 3 headline cells {'STILL certify failure' if allfail else 'do NOT all certify'} under LLM-relabeling.")
print("  wrote results/B4_llm_recompute_allcells_results.json")
print("=" * 96)
