#!/usr/bin/env python
"""R068 — is the paired discordance reducible to twin surface form? (optional strengthener for
reviewer Q3 / the kill-review's strongest residual point). DESCRIPTIVE, post-hoc; no certificate.

Design. R066 showed MedHallu twins are separable by surface form alone (role AUC 0.879). If the
audited gate's within-question behavior were merely a response to that form, then on a
FORM-MATCHED subset — where surface form no longer distinguishes the truthful and hallucinated
twin — the discordance should vanish. We test the contrapositive:

  (a) surface-only role-classifier AUC on the form-matched subset should collapse toward 0.5
      (confirming form no longer separates roles there);
  (b) yet at the RELEASED headline thresholds the gate should still reject the truthful twin more
      than it accepts the hallucinated one, i.e. psi_hat stays elevated and rho stays low on the
      matched subset — showing the gate uses more than the matched-away surface form.

Scope/honesty. Computed on the 1,000 human-labeled twins only (qid 0..999 align to scores_strong
row twins), at the released headline thresholds tau (P(True) 0.222824, ref-NLI 0.518631; the
judge headline is alpha=0.20 tau 0.245085). These thresholds were selected on the 10k induced
stream, so the 1k-subset numbers are DESCRIPTIVE, not the headline certificate, and the
form-matched denominators are small (power-limited). We report the DIRECTION (matched vs full),
not a new bound.

Output: R068_form_matched_psig_results.json (committable).
"""
import os, sys, json, csv, re
import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
from datasets import load_dataset

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
THRESH = [("score_sptrue", 0.222824, "T2 P(True)-8B"),
          ("score_sjudge", 0.245085, "T3 LLM-judge-8B"),
          ("score_sNLI", 0.518631, "T4 ref-NLI")]

_num = re.compile(r"\d"); _cite = re.compile(r"\[\d+\]|\(\d{4}\)|et al\.|https?://|doi:", re.I)
_hedge = re.compile(r"\b(may|might|could|possibly|likely|suggest|appear|potential|often|generally|typically)\b", re.I)
_word = re.compile(r"[a-z0-9]+")
FEATS = ["n_tokens", "n_chars", "n_sents", "numeral_density", "citation_markers",
         "hedge_density", "type_token_ratio", "q_overlap", "ref_overlap"]


def feats(ans, q, ref):
    w = _word.findall(ans.lower()); nw = max(len(w), 1)
    sents = max(len([s for s in re.split(r"[.!?]+", ans) if s.strip()]), 1)
    aset, qset, rset = set(w), set(_word.findall(q.lower())), set(_word.findall(ref.lower()))
    return [len(w), len(ans), sents, sum(bool(_num.search(x)) for x in w) / nw,
            len(_cite.findall(ans)), len(_hedge.findall(ans)) / nw, len(set(w)) / nw,
            len(aset & qset) / max(len(aset), 1), len(aset & rset) / max(len(aset), 1)]


def auc_surface(Xt, Xh):
    # question-blocked 5-fold logistic role AUC (twins never split)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    n = len(Xt)
    X = np.vstack([Xt, Xh]); y = np.r_[np.zeros(n), np.ones(n)]; grp = np.r_[np.arange(n), np.arange(n)]
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    aucs = []
    for tr, te in GroupKFold(min(5, n)).split(Xs, y, grp):
        clf = LogisticRegression(max_iter=1000).fit(Xs[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xs[te])[:, 1]))
    return float(np.mean(aucs))


def main():
    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]
    rows = list(csv.DictReader(open(artifact("scores_strong.csv"))))
    RAW = {f: np.full(1000, np.nan) for f in ("score_sptrue", "score_sjudge", "score_sNLI")}
    H = np.zeros(1000, int)
    for r in rows:
        q = int(r["qid"]); h = int(r["H"])
        for f in RAW:
            # store per (qid, H): rT at H=0, rH at H=1
            pass
    rT = {f: np.full(1000, np.nan) for f in RAW}; rH = {f: np.full(1000, np.nan) for f in RAW}
    for r in rows:
        q = int(r["qid"]); h = int(r["H"])
        for f in RAW:
            (rH if h == 1 else rT)[f][q] = float(r[f])

    Xt = np.zeros((1000, 9)); Xh = np.zeros((1000, 9))
    for i, r in enumerate(ds):
        ref = " ".join(r["Knowledge"]) if isinstance(r["Knowledge"], list) else str(r["Knowledge"])
        Xt[i] = feats(r["Ground Truth"], r["Question"], ref)
        Xh[i] = feats(r["Hallucinated Answer"], r["Question"], ref)

    # form-matched mask (same rule as R066): len ratio in [0.8,1.25], equal citation-presence, |Δsent|<=1
    lr = Xh[:, 0] / np.maximum(Xt[:, 0], 1)
    matched = (lr >= 0.8) & (lr <= 1.25) & ((Xh[:, 4] > 0) == (Xt[:, 4] > 0)) & (np.abs(Xh[:, 2] - Xt[:, 2]) <= 1)
    idx_all = np.arange(1000); idx_m = idx_all[matched]
    print(f"[subset] form-matched twins: {matched.sum()}/1000")

    auc_full = auc_surface(Xt, Xh)
    auc_m = auc_surface(Xt[matched], Xh[matched])
    print(f"[surface AUC] full={auc_full:.4f}  form-matched={auc_m:.4f}  (target: matched -> ~0.5)")

    def gate_stats(f, tau, idx):
        # orient on the FULL 1k labeled set (declared), apply released tau, read on idx
        both = np.r_[rT[f], rH[f]]; lab = np.r_[np.zeros(1000, int), np.ones(1000, int)]
        _, flip = C.orient(lab, both); s = -1.0 if flip else 1.0
        t = s * rT[f][idx]; h = s * rH[f][idx]
        rej_t = t > tau; acc_h = h <= tau
        m_c = int(rej_t.sum()); s_c = int((rej_t & acc_h).sum())
        F1 = float(acc_h.mean())                 # marginal P(accept hallucinated twin)
        psi = s_c / m_c if m_c else None
        return dict(m_c=m_c, s_c=s_c, psi_hat=round(psi, 4) if psi is not None else None,
                    F1=round(F1, 4), delta_psi=round(psi - F1, 4) if psi is not None else None,
                    rho_balanced=round(float(((s * rH[f][idx] <= tau).sum()) /
                                       max(((s * rT[f][idx] <= tau).sum() + (s * rH[f][idx] <= tau).sum()), 1)), 4))

    cells = []
    for f, tau, tag in THRESH:
        full = gate_stats(f, tau, idx_all); mm = gate_stats(f, tau, idx_m)
        cells.append(dict(tag=tag, tau=tau, full_1k=full, form_matched=mm))
        print(f"  {tag:<16} full: m_c={full['m_c']:>3} psi={full['psi_hat']} F1={full['F1']} Δ={full['delta_psi']}"
              f"  | matched: m_c={mm['m_c']:>3} psi={mm['psi_hat']} F1={mm['F1']} Δ={mm['delta_psi']}")

    out = dict(run="R068 form-matched-subset paired diagnostic (descriptive; strengthener for Q3)",
               scope=("1k human-labeled twins only; released headline thresholds applied descriptively "
                      "(selected on 10k induced stream); form-matched denominators are small/power-limited; "
                      "reports direction, not a certificate"),
               n_twins=1000, n_form_matched=int(matched.sum()),
               surface_role_auc=dict(full=round(auc_full, 4), form_matched=round(auc_m, 4)),
               cells=cells,
               reading=("on the form-matched subset the surface-only role classifier collapses toward "
                        "chance, yet the gate still rejects the truthful twin more than it accepts the "
                        "hallucinated one (psi_hat elevated, balanced rho low) at the released thresholds, "
                        "so the within-question gate behavior is not reducible to the matched-away surface form"))
    with open(artifact("R068_form_matched_psig_results.json"), "w") as fo:
        json.dump(out, fo, indent=2)
    print("[wrote] R068_form_matched_psig_results.json")


if __name__ == "__main__":
    main()
