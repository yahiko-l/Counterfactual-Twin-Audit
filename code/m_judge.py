#!/usr/bin/env python
"""
General-judge STRENGTH/RECENCY robustness — generalization of m_medical.py.
Certifies the ψ_g audit with an arbitrary (typically stronger/newer) general judge family and
compares head-to-head with BOTH the general Llama-3-8B judge (m3_combined) and the OpenBioLLM-8B
medical judge (m_medical_results), if present.

Answers the reviewer question "would a STRONGER / NEWER judge still leak?" — same prevalence-
conditioned joint-cell protocol as m3_combined.py / m_medical.py (deployment stream one-answer-
per-question; cluster-valid ρ; exact conditional-binomial ψ_g; Holm-pooled; union ≤ δ_total).

Honest caveat this experiment probes (K2): a stronger judge rejects fewer truthful answers ⇒ the
ψ_g denominator m_c shrinks; if m_c < min-mass floor, ψ_g becomes UNCERTIFIABLE (not absent).
The min-mass family is certified jointly, so a denominator collapse shows up as fewer joint cells.

Inputs (env): JUDGE_TAG (label/output suffix), JUDGE_LAB=scores_<tag>.csv (1k+SelfCheck),
JUDGE_ART=scores_<tag>_artificial.csv (9k).
"""
import os, sys, json, csv
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
TAG = os.environ.get("JUDGE_TAG", "judge")
LAB = os.environ.get("JUDGE_LAB", f"data/scores/scores_{TAG}.csv")
ART = os.environ.get("JUDGE_ART", f"data/scores/scores_{TAG}_artificial.csv")
OUTP = f"results/m_{TAG}_results.json"
FAMS = ["score_sjudge", "score_sptrue", "score_sNLI"]   # score_sjudge here = this judge's P(hallucination)
ALPHAS = [0.05, 0.10, 0.20]; GAMMA = C.GAMMA_DEFAULT; PIS = [0.50, 0.20, 0.10, 0.05]

def load(p, s):
    r = list(csv.DictReader(open(p)))
    for x in r: x.setdefault("source", s)
    return r
rows = load(LAB, "pqa_labeled") + (load(ART, "pqa_artificial") if os.path.exists(ART) else [])
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
byq = {}
for i in range(len(rows)): byq.setdefault(int(qid[i]), {})[int(H[i])] = i
QIDS = np.array(sorted(byq))
perm = np.random.default_rng(7).permutation(len(QIDS)); n = len(QIDS)
Dsel = set(QIDS[perm[:int(.4*n)]].tolist()); Dcal = set(QIDS[perm[int(.4*n):int(.8*n)]].tolist())
sel = np.array([byq[q][h] for q in QIDS if q in Dsel for h in (0, 1)])
ORI = {}; FLIP = {}
for f in FAMS:
    _, fl = C.orient(H[sel], RAW[f][sel]); ORI[f] = (-RAW[f] if fl else RAW[f]); FLIP[f] = bool(fl)
def tw(f, qs):
    s = ORI[f]; q = [x for x in QIDS if x in qs]
    return np.array([s[byq[x][0]] for x in q]), np.array([s[byq[x][1]] for x in q])

from sklearn.metrics import roc_auc_score
def joint_at_pi(pi):
    rho_p, low_p, mass_p = [], [], []
    META = {}
    for f in FAMS:
        rTs, rHs = tw(f, Dsel); grid = C.candidate_grid(np.r_[rTs, rHs], gamma=GAMMA, n_grid=28)
        rTc, rHc = tw(f, Dcal); ncal = len(rTc)
        rng = np.random.default_rng(2024); takeH = rng.random(ncal) < pi
        ss = np.where(takeH, rHc, rTc); hs = takeH.astype(int); META[f] = []
        for ti, t in enumerate(grid):
            acc = ss <= t; K = int(acc.sum()); S = int(hs[acc].sum()); m_c, sf = C.psi_counts(rTc, rHc, t)
            mass_p.append((f, ti, C.minmass_p(m_c, ncal, GAMMA)))
            META[f].append(dict(tau=round(float(t), 5), rho_hat=(round(S/K, 4) if K else None), m_c=m_c,
                                psi_hat=(round(sf/m_c, 4) if m_c else None)))
            for a in ALPHAS:
                rho_p.append((f, a, ti, float(binom.cdf(S, K, a)) if K else 1.0))
                low_p.append((f, a, ti, C.lower_p_psi(sf, m_c, a)))
    rr = C.holm_reject([x[3] for x in rho_p], C.DELTA_GLOBAL); lr = C.holm_reject([x[3] for x in low_p], C.DELTA_LOWER)
    mr = C.holm_reject([x[2] for x in mass_p], C.DELTA_MASS)
    rok = {(x[0], x[1], x[2]): bool(rr[i]) for i, x in enumerate(rho_p)}
    lok = {(x[0], x[1], x[2]): bool(lr[i]) for i, x in enumerate(low_p)}
    mok = {(x[0], x[1]): bool(mr[i]) for i, x in enumerate(mass_p)}
    out = {}
    for f in FAMS:
        for a in ALPHAS:
            j = [ti for ti in range(len(META[f])) if rok[(f, a, ti)] and lok[(f, a, ti)] and mok[(f, ti)]]
            if j: out[(f, a)] = META[f][max(j, key=lambda ti: META[f][ti]["psi_hat"] or 0)]
    return out

# K4 for this judge
rT, rH = tw("score_sjudge", set(QIDS.tolist()))
pa = float(np.mean((rH > rT) + 0.5*(rH == rT)))
thisjudge = {"risk_auc": round(float(roc_auc_score(H, ORI["score_sjudge"])), 4), "paired_acc": round(pa, 4)}
res = {"judge_tag": TAG, "n_twins": len(QIDS), "this_judge": thisjudge,
       "orientation": {f: {"flipped": FLIP[f], "dsel_raw_auc": (round(float(roc_auc_score(H[sel], RAW[f][sel])), 4)
                          if len(set(H[sel].tolist())) > 1 else None)} for f in FAMS},
       "joint_by_pi": {}}
for pi in PIS:
    cells = joint_at_pi(pi); res["joint_by_pi"][f"pi={pi}"] = {"n": len(cells),
        "cells": {f"{f}|{a}": v for (f, a), v in cells.items()}}
this_by_pi = {k: v["n"] for k, v in res["joint_by_pi"].items()}

# compare to general (m3_combined) and medical (m_medical_results), if present
cmp = {}
gp = "results/m3_combined_results.json"
if os.path.exists(gp):
    g = json.load(open(gp))
    cmp["general_Llama3-8B"] = {"paired_acc": g["K4_combined_judge"]["paired_acc"],
                               "joint_by_pi": {k: v["joint_cells"] for k, v in g["pooled"].items()}}
mp = "results/m_medical_results.json"
if os.path.exists(mp):
    m = json.load(open(mp))
    cmp["medical_OpenBioLLM-8B"] = {"paired_acc": m["medical_judge"]["paired_acc"],
                                    "joint_by_pi": {k: v["n"] for k, v in m["joint_by_pi"].items()}}
for tag_, path_ in [("qwen3_32b", "results/m_qwen3_32b_results.json"),
                    ("qwen36_27b", "results/m_qwen36_27b_results.json")]:
    if os.path.exists(path_) and TAG != tag_:
        x = json.load(open(path_))
        cmp[tag_] = {"paired_acc": x["this_judge"]["paired_acc"],
                     "joint_by_pi": {k: v["n"] for k, v in x["joint_by_pi"].items()}}
res["comparison"] = cmp
n10 = res["joint_by_pi"]["pi=0.1"]["n"]
res["verdict"] = (f"C1 REPRODUCES with the {TAG} judge (not a weak/old-judge artifact)" if n10 > 0
                  else f"C1 does NOT reproduce with the {TAG} judge at π=0.10 — likely a ψ_g-denominator "
                       f"collapse (stronger judge rejects fewer truthful ⇒ m_c<γ); inspect min-mass certs")
json.dump(res, open(OUTP, "w"), indent=2, default=str)
print("=" * 78)
print(f"{TAG.upper()} JUDGE STRENGTH/RECENCY ROBUSTNESS — {len(QIDS)} twins")
print(f"this judge ({TAG}): risk-AUROC={thisjudge['risk_auc']} paired-acc={thisjudge['paired_acc']} (K4: ≥0.5 ⇒ no inversion)")
print("joint cells (ρ≤α ∧ ψ_g>α) by π:", this_by_pi)
for name, c in cmp.items():
    print(f"  vs {name} (paired-acc {c['paired_acc']}):", c["joint_by_pi"])
print("VERDICT:", res["verdict"])
print(f"  wrote {OUTP}"); print("=" * 78)
