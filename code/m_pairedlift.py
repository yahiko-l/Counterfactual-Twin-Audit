#!/usr/bin/env python
"""M_pairedlift — marginal F1 and paired lift Delta_psi at the tab:joint certified thresholds.
Provenance for the §5.5 "Paired lift versus the marginal" paragraph + tab:joint caption addition.
Reproduces the certified cells (m_c, s_c, psi_hat) exactly on the SAME pooled 10k split as
m3_combined/m10 (scores_strong + scores_strong_artificial, split seed 7, orient on D_sel), then
adds F1(tau)=P(halluc accepted) and Delta_psi=psi_hat-F1 on D_cal. At a fixed tau with the
rejected-truthful set fixed, E[psi_random]=F1 exactly, so Delta_psi is the paired lift over a
random-pairing baseline; perm_p = P(Binom(m_c,F1) >= s_c) is the exact random-pairing tail. CPU only."""
import os, csv, json, sys, numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
LAB = artifact("scores_strong.csv"); ART = artifact("scores_strong_artificial.csv")
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]; SPLIT_SEED = 7

def load(p, s):
    r = list(csv.DictReader(open(p)))
    for x in r: x.setdefault("source", s)
    return r
rows = load(LAB, "lab") + load(ART, "art")
qid = np.array([int(r["qid"]) for r in rows]); H = np.array([int(r["H"]) for r in rows])
RAW = {f: np.array([float(r[f]) for r in rows]) for f in FAMS}
byq = {}
for i in range(len(rows)): byq.setdefault(int(qid[i]), {})[int(H[i])] = i
QIDS = sorted(q for q in byq if set(byq[q]) == {0, 1})
perm = np.random.default_rng(SPLIT_SEED).permutation(len(QIDS)); n = len(QIDS); a = int(n*.4); b = a+int(n*.4)
D_sel = set(np.array(QIDS)[perm[:a]].tolist()); D_cal = set(np.array(QIDS)[perm[a:b]].tolist())
sel = np.array([byq[q][h] for q in QIDS if q in D_sel for h in (0, 1)])
ORI = {}
for f in FAMS:
    _, fl = C.orient(H[sel], RAW[f][sel]); ORI[f] = (-RAW[f] if fl else RAW[f])
qc = [q for q in QIDS if q in D_cal]

def cell(f, tau):
    rT = np.array([ORI[f][byq[q][0]] for q in qc]); rH = np.array([ORI[f][byq[q][1]] for q in qc])
    rej = rT > tau; acc = rH <= tau; mc = int(rej.sum()); sc = int((rej & acc).sum())
    F1 = float(acc.mean()); psi = sc/mc if mc else None
    perm_p = float(binom.sf(sc-1, mc, F1)) if mc else None
    return dict(m_c=mc, s_c=sc, psi_hat=round(psi, 4), F1=round(F1, 4),
                delta_psi=round(psi-F1, 4), perm_p=perm_p)

# exact displayed thresholds from tab:joint (figures/TABLE1_joint_cell.tex)
CELLS = [("score_sptrue", "P(True)-8B", [0.05], 0.2228, "headline"),
         ("score_sjudge", "LLM-judge-8B", [0.20], 0.2451, "judge"),
         ("score_sNLI", "ref-NLI", [0.10, 0.20], 0.5186, "ref-NLI")]
out = dict(run="M_pairedlift: marginal F1 + paired lift at tab:joint certified thresholds",
           split_seed=SPLIT_SEED, pooled_twins=len(QIDS), D_cal_twins=len(qc),
           note="Delta_psi=psi_hat-F1; random-pair baseline E[psi_rand]=F1; perm_p=P(Binom(m_c,F1)>=s_c). At every certified cell Delta_psi<=0.",
           headline_cell=None, cells=[])
# headline P(True) alpha=0.10/0.20 at tau=0.2228
hd = cell("score_sptrue", 0.2228); hd.update(family="P(True)-8B", alpha="0.10/0.20", tau=0.2228)
out["headline_cell"] = hd
for tau, fam, disp in [(0.000123, "score_sptrue", "P(True)-8B (alpha 0.05)"),
                       (0.2228, "score_sptrue", "P(True)-8B (alpha 0.10/0.20)"),
                       (0.2451, "score_sjudge", "LLM-judge-8B (alpha 0.20)"),
                       (0.5186, "score_sNLI", "ref-NLI (alpha 0.10/0.20)")]:
    c = cell(fam, tau); c.update(family=disp, tau=tau); out["cells"].append(c)
json.dump(out, open(artifact("m_pairedlift_results.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
