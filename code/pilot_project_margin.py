#!/usr/bin/env python
"""R010c (part 2b) — projected within-question margin for the pilot (power projection, NOT a certificate).

Joins detector scores (scores_pilot.csv from score_generic on twins_pilot_scoring_input.jsonl) with the
stage-1 CONSENSUS natural-H labels, then per detector family × source/pool × α:
  1. orient the risk score (AUROC flip) on natural H;
  2. realize S_R (frozen R, one answer per source question) and find the strictest τ on the D_sel-frozen
     candidate grid whose one-per-source ρ(τ)=P(H=1|accept) clears the exact-binomial UCB ≤ α (cluster-valid
     because S_R is one answer per source);
  3. on the design-B within-question twins (per T-question: seeded natural-H=0 answer as truthful twin,
     seeded natural-H=1 answer as hallucinated twin), at that τ compute m_c=#(rT>τ), s=#(rT>τ ∧ rH≤τ),
     ψ̂=s/m_c and a nominal ψ_g LCB (Clopper-Pearson, δ=DELTA_LOWER single-cell OPTIMISTIC + a Holm-scaled
     conservative variant).
This uses stage-1 consensus H (pre-screen) — clinicians certify at M3/M4. δtotal=0.05 Holm/union-floor is an
M4 step; here we PROJECT whether a cell can plausibly reach ψ_g,LCB ≥ α+0.05 with m_c ≥ target. CPU only.

Usage: python pilot_project_margin.py [--consensus strict|lenient] [scores_csv]
Output: results/pilot_projection_results.json + printed table.
"""
import os, sys, json, csv, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
from scipy.stats import binom
from collections import defaultdict

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
R_SEED = 20260710
FAMS = ["score_sptrue", "score_sjudge", "score_sNLI"]
ALPHAS = [0.10, 0.20]
GAMMA = C.GAMMA_DEFAULT

def cons_mode():
    if "--consensus" in sys.argv:
        return sys.argv[sys.argv.index("--consensus")+1]
    return "strict"

EXCLUDE = [s for s in os.environ.get("PILOT_EXCLUDE_ANSWERERS", "AntAngel,HuatuoGPT-3-32B").split(",") if s.strip()]
def _excluded(a): return any(x.lower() in a.lower() for x in EXCLUDE)

def load_labels(mode):
    """natural consensus H per (source, orig_qid, answerer)."""
    files = sorted(glob.glob(artifact("pilot_labeled_*.jsonl")))
    judges = defaultdict(dict)
    for f in files:
        t = os.path.basename(f)[len("pilot_labeled_"):-6]
        for l in open(f):
            r = json.loads(l)
            if r.get("H") is None: continue
            if _excluded(r["answerer"]): continue
            judges[(r["source"], r["qid"], r["answerer"])][t] = int(r["H"])
    H = {}
    for k, hs in judges.items():
        present = list(hs.values())
        if not present: continue
        H[k] = (1 if all(v == 1 for v in present) else 0) if mode == "strict" else (1 if any(v == 1 for v in present) else 0)
    return H

def load_scores(csvp):
    """answer detector score per (source, orig_qid, answerer): the H=1 (halluc=panel answer) rows.
    qid is an integer index; recover (source, orig_qid, answerer) via the sidecar map."""
    mapp = artifact("twins_pilot_scoring_map.json")
    idmap = json.load(open(mapp)) if os.path.exists(mapp) else {}
    sc = {}
    fams_present = None
    for r in csv.DictReader(open(csvp)):
        if int(r["H"]) != 1:  # H=1 row carries the panel answer's score; H=0 row is the reference (ignored)
            continue
        meta = idmap.get(str(r["qid"])) or idmap.get(r["qid"])
        if not meta:
            continue
        key = (meta["source"], meta["orig_qid"], meta["answerer"])
        if fams_present is None:
            fams_present = [f for f in FAMS if f in r]
        sc[key] = {f: float(r[f]) for f in fams_present}
    return sc, (fams_present or FAMS)

def main():
    mode = cons_mode()
    csvp = ([a for a in sys.argv[1:] if a.endswith(".csv")] or [artifact("scores_pilot.csv")])[0]
    if not os.path.exists(csvp):
        print(f"scores CSV not found: {csvp} — run score_generic on twins_pilot_scoring_input.jsonl first."); return
    H = load_labels(mode)
    sc, fams = load_scores(csvp)
    keys = [k for k in sc if k in H]
    print(f"consensus={mode}  scored answers={len(sc)}  labeled={len(H)}  joined={len(keys)}  fams={fams}")
    # group by question
    byq = defaultdict(list)  # (source,orig) -> [(answerer, scoredict, H)]
    for (source, orig, ans) in keys:
        byq[(source, orig)].append((ans, sc[(source, orig, ans)], H[(source, orig, ans)]))
    sources = sorted({s for (s, _) in byq})

    def project(pool_keys, label, alpha):
        rng = np.random.default_rng(R_SEED)
        res = {}
        for f in fams:
            # arrays over all answers in pool (for orientation)
            allA = [(byq[q][i], q) for q in pool_keys for i in range(len(byq[q]))]
            raw = np.array([a[0][1][f] for a in allA]); hlab = np.array([a[0][2] for a in allA])
            oriented, flip = C.orient(hlab, raw)
            sign = -1.0 if flip else 1.0
            def sval(d): return sign * d[f]
            # ---- S_R: one answer per source question (frozen R) ----
            S_scores, S_H = [], []
            for q in pool_keys:
                cand = byq[q]
                pick = cand[int(rng.integers(len(cand)))]
                S_scores.append(sval(pick[1])); S_H.append(pick[2])
            S_scores = np.array(S_scores); S_H = np.array(S_H, int)
            if S_H.sum() == 0 or len(S_scores) < 10:
                res[f] = dict(note="degenerate S_R", n=len(S_scores), pi_SR=float(S_H.mean()) if len(S_H) else None); continue
            grid = C.candidate_grid(S_scores, gamma=GAMMA, n_grid=28)
            # strictest tau (max coverage of the low-risk accept set) with one-per-source rho UCB<=alpha
            m = max(len(grid), 1)
            tau_star = None; rho_at = None
            for tau in grid:  # ascending tau = looser; we want a tau that passes rho<=alpha, prefer larger accept
                acc = S_scores <= tau; K = int(acc.sum())
                if K == 0: continue
                Sacc = int(S_H[acc].sum())
                p = float(binom.cdf(Sacc, K, alpha))
                if p <= C.DELTA_GLOBAL / m:   # certify rho<=alpha at this tau (preview budget)
                    tau_star = float(tau); rho_at = Sacc / K   # keep the LAST (largest) passing tau
            if tau_star is None:
                res[f] = dict(note="no tau with rho_UCB<=alpha", pi_SR=float(S_H.mean()),
                              min_rho=float(min((int(S_H[S_scores<=t].sum())/max(int((S_scores<=t).sum()),1)) for t in grid))); continue
            # ---- design-B within-question twins at tau_star ----
            m_c = s_c = nT = 0
            for q in pool_keys:
                cand = byq[q]
                h0 = [c for c in cand if c[2] == 0]; h1 = [c for c in cand if c[2] == 1]
                if not h0 or not h1: continue
                nT += 1
                tw_t = h0[int(rng.integers(len(h0)))]; tw_h = h1[int(rng.integers(len(h1)))]
                rT = sval(tw_t[1]); rH = sval(tw_h[1])
                if rT > tau_star:
                    m_c += 1
                    if rH <= tau_star: s_c += 1
            psi_hat = (s_c / m_c) if m_c > 0 else None
            # nominal LCBs: optimistic single-cell delta, and a Holm-scaled conservative (fams*alphas*|grid|*|sources|)
            nlow_proj = max(len(fams) * len(ALPHAS) * len(grid) * max(len(sources), 1), 1)
            lcb_opt = round(C.cp_lower_psi(s_c, m_c, C.DELTA_LOWER), 4) if m_c > 0 else None
            lcb_cons = round(C.cp_lower_psi(s_c, m_c, C.DELTA_LOWER / nlow_proj), 4) if m_c > 0 else None
            res[f] = dict(flip=bool(flip), tau_star=round(tau_star, 6), rho_hat=round(float(rho_at), 4),
                          n_SR=len(S_scores), pi_SR=round(float(S_H.mean()), 4), twin_questions=nT,
                          m_c=m_c, s_c=s_c, psi_hat=(round(psi_hat, 4) if psi_hat is not None else None),
                          psi_lcb_optimistic=lcb_opt, psi_lcb_holmscaled=lcb_cons, nlow_proj=nlow_proj)
        return res

    out = {"consensus": mode, "fams": fams, "alphas": ALPHAS, "results": {}}
    pools = {"pooled": list(byq.keys())}
    for s in sources:
        pools[s] = [q for q in byq if q[0] == s]
    for alpha in ALPHAS:
        for label, pk in pools.items():
            out["results"][f"{label}@a{alpha}"] = project(pk, label, alpha)
    json.dump(out, open(artifact("pilot_projection_results.json"), "w"), indent=2)

    print(f"\n{'cell':<20}{'fam':<14}{'tau*':>9}{'rho':>7}{'pi_SR':>7}{'m_c':>6}{'psi_hat':>9}{'LCB_opt':>9}{'LCB_holm':>9}")
    for cell, r in out["results"].items():
        for f, d in r.items():
            if "psi_hat" in d and d["psi_hat"] is not None:
                print(f"{cell:<20}{f.replace('score_s',''):<14}{d['tau_star']:>9.4f}{d['rho_hat']:>7.3f}"
                      f"{d['pi_SR']:>7.3f}{d['m_c']:>6}{d['psi_hat']:>9.4f}"
                      f"{str(d['psi_lcb_optimistic']):>9}{str(d['psi_lcb_holmscaled']):>9}")
            else:
                print(f"{cell:<20}{f.replace('score_s',''):<14}  {d.get('note','')}")
    print(f"\nwrote {artifact('pilot_projection_results.json')}")

if __name__ == "__main__":
    main()
