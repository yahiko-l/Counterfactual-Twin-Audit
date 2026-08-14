#!/usr/bin/env python
"""
M11 — Build the REAL blinded human-adjudication sheet (replaces the placeholder B4_blind_sheet.csv).

Joins MedHallu raw text (Question / Knowledge / Ground Truth=truthful / Hallucinated Answer) onto the
ACTUAL certified rejected-truthful denominators of the three headline cells (computed on D_cal, the
calibration fold, with the released split seed 7 and the headline thresholds), blinds the
truthful/hallucinated order per item, and writes:

  B4_blind_sheet.csv  — clinician-facing, FULLY BLIND (no qid / no H / no scores / no twin role).
  B4_key.csv          — unblinding key (item_id -> qid, which letter is the hallucinated twin, scores,
                         cell-membership flags). Kept SEPARATE so the sheet stays blind.
  B4_README.md        — adjudication protocol (2 blinded annotators + 3rd-adjudicator, severity,
                         how a human-derived eta is computed and fed to the label-noise band).

Headline cells (D_cal, seed 7):  P(True)-8B tau=0.222824 (m_c=278) ; ref-NLI tau=0.518631 (m_c=488) ;
LLM-judge-8B tau=0.245085 (m_c=347).  The sheet is the DEDUP UNION of all three denominators, with a
priority column so partial completion is still useful (priority 1 = the 120 P(True) headline FAILURE
pairs = accepted-hallucination-while-truthful-rejected; 2 = rest of the 278; 3 = ref-NLI/judge-only).
CPU only. Does NOT adjudicate — produces the sheet a human fills in.
"""
import os, sys, csv, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

# MedHallu snapshot actually used. Override with MEDHALLU_SNAPSHOT; the default resolves
# the same revision under the local HF cache (HF_HOME, else ~/.cache/huggingface).
SNAP = os.environ.get("MEDHALLU_SNAPSHOT", os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "hub", "datasets--UTAustin-AIHealth--MedHallu", "snapshots",
    "515060458a945c633debc6fd5baac7764416b724"))
LAB = "data/scores/scores_strong.csv"; ART = "data/scores/scores_strong_artificial.csv"
OUT_SHEET = "results/B4_blind_sheet.csv"; OUT_KEY = "results/B4_key.csv"
ART_OFFSET = 1000000
# headline (family, oriented threshold, expected D_cal m_c) — from m3_combined_results.json
HEADLINE = {"score_sptrue": (0.222824, 278), "score_sNLI": (0.518631, 488), "score_sjudge": (0.245085, 347)}
BLIND_SEED = 20240619

def as_text(x):
    if isinstance(x, (list, tuple, np.ndarray)): return "\n".join(str(t) for t in x)
    return str(x)

# ---- MedHallu raw text by qid ----
lab = pd.read_parquet(f"{SNAP}/pqa_labeled/train-00000-of-00001.parquet")
art = pd.read_parquet(f"{SNAP}/pqa_artificial/train-00000-of-00001.parquet")
TXT = {}
for i, r in lab.iterrows():
    TXT[int(i)] = dict(Q=str(r["Question"]).strip(), K=as_text(r["Knowledge"]).strip(),
                       truthful=str(r["Ground Truth"]).strip(), halluc=str(r["Hallucinated Answer"]).strip(),
                       difficulty=str(r["Difficulty Level"]).strip(), category=str(r["Category of Hallucination"]).strip(),
                       source="pqa_labeled")
for i, r in art.iterrows():
    TXT[ART_OFFSET + int(i)] = dict(Q=str(r["Question"]).strip(), K=as_text(r["Knowledge"]).strip(),
                       truthful=str(r["Ground Truth"]).strip(), halluc=str(r["Hallucinated Answer"]).strip(),
                       difficulty=str(r["Difficulty Level"]).strip(), category=str(r["Category of Hallucination"]).strip(),
                       source="pqa_artificial")

# ---- scores + split (mirror m3_combined exactly) ----
def load(path, src):
    rows = list(csv.DictReader(open(path)))
    for r in rows: r.setdefault("source", src)
    return rows
rows = load(LAB, "pqa_labeled") + load(ART, "pqa_artificial")
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
FAMS = list(HEADLINE)
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

# ---- per-family oriented twin scores on D_cal; rejected-truthful denominators ----
DEN = {}   # family -> dict(qid -> (rT, rH, is_fail))
for f in FAMS:
    tau, exp_mc = HEADLINE[f]
    _, fl = C.orient(H[sel_idx], RAW[f][sel_idx]); s = (-RAW[f] if fl else RAW[f])
    d = {}
    for q in cal_q:
        rT = float(s[byq[q][0]]); rH = float(s[byq[q][1]])
        if rT > tau:                       # truthful rejected
            d[q] = (rT, rH, bool(rH <= tau))   # is_fail = hallucinated accepted among them
    DEN[f] = d
    assert len(d) == exp_mc, f"{f}: D_cal rejected-truthful m_c={len(d)} != expected {exp_mc}"
    print(f"[ok] {f} tau={tau} m_c={len(d)} (fail pairs s_c={sum(v[2] for v in d.values())})")

# ---- assemble dedup union with priority + membership ----
union = sorted(set().union(*[set(DEN[f]) for f in FAMS]))
items = []
for q in union:
    in_p = q in DEN["score_sptrue"]; in_n = q in DEN["score_sNLI"]; in_j = q in DEN["score_sjudge"]
    p_fail = in_p and DEN["score_sptrue"][q][2]
    priority = 1 if p_fail else (2 if in_p else 3)
    items.append(dict(qid=q, in_ptrue=in_p, in_refnli=in_n, in_judge=in_j, ptrue_failpair=p_fail, priority=priority))
items.sort(key=lambda x: (x["priority"], x["qid"]))

# ---- blind: randomize truthful/hallucinated -> A/B ----
rng = np.random.default_rng(BLIND_SEED)
sheet_rows, key_rows = [], []
for k, it in enumerate(items):
    q = it["qid"]; t = TXT[q]
    a_is_halluc = bool(rng.random() < 0.5)
    ansA = t["halluc"] if a_is_halluc else t["truthful"]
    ansB = t["truthful"] if a_is_halluc else t["halluc"]
    # blind sheet deliberately OMITS difficulty/category (priming risk) and qid/H/scores (leakage);
    # those live in B4_key.csv for post-hoc stratification.
    sheet_rows.append(dict(item_id=k, priority=it["priority"],
        Question=t["Q"], Reference_Knowledge=t["K"], Answer_A=ansA, Answer_B=ansB,
        hallucination_in="", A_supported_by_reference="", B_supported_by_reference="",
        severity="", confidence="", annotator_id="", notes=""))
    key_rows.append(dict(item_id=k, qid=q, source=t["source"], priority=it["priority"],
        A_role=("hallucinated" if a_is_halluc else "truthful"), B_role=("truthful" if a_is_halluc else "hallucinated"),
        A_twin_H=int(a_is_halluc), B_twin_H=int(not a_is_halluc),
        in_ptrue=it["in_ptrue"], in_refnli=it["in_refnli"], in_judge=it["in_judge"], ptrue_failpair=it["ptrue_failpair"],
        sptrue_rT=DEN["score_sptrue"][q][0] if it["in_ptrue"] else "", sptrue_rH=DEN["score_sptrue"][q][1] if it["in_ptrue"] else "",
        difficulty=t["difficulty"], category=t["category"]))

SHEET_COLS = ["item_id", "priority", "Question", "Reference_Knowledge",
              "Answer_A", "Answer_B", "hallucination_in", "A_supported_by_reference",
              "B_supported_by_reference", "severity", "confidence", "annotator_id", "notes"]
with open(OUT_SHEET, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=SHEET_COLS, quoting=csv.QUOTE_ALL); w.writeheader(); w.writerows(sheet_rows)
KEY_COLS = list(key_rows[0])
with open(OUT_KEY, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=KEY_COLS, quoting=csv.QUOTE_ALL); w.writeheader(); w.writerows(key_rows)

nfail = sum(r["ptrue_failpair"] for r in key_rows)
byp = {p: sum(1 for r in key_rows if r["priority"] == p) for p in (1, 2, 3)}
print(f"\nUNION items={len(items)}  (P(True) 278 ∪ ref-NLI 488 ∪ judge 347)")
print(f"  priority 1 (P(True) headline FAILURE pairs)         = {byp[1]}")
print(f"  priority 2 (rest of the 278 P(True) denominator)    = {byp[2]}")
print(f"  priority 3 (ref-NLI / judge denominator, not above) = {byp[3]}")
print(f"  P(True) headline cell fully covered by priority 1+2 = {byp[1]+byp[2]} (=278, fail s_c={nfail})")
print(f"\nwrote {OUT_SHEET}  ({len(sheet_rows)} blinded items, {len(SHEET_COLS)} cols)")
print(f"wrote {OUT_KEY}    (unblinding key — keep separate from annotators)")
