#!/usr/bin/env python
"""R066 — MedHallu twin surface-form paired diagnostics (reviewer Q3; PRESPECIFIED S5.4-adjacent,
declared descriptive). Quantifies whether the truthful (Ground Truth) and hallucinated
(Hallucinated Answer) twins differ on surface attributes OTHER than factuality — length, numeral
and citation density, hedging, and lexical overlap with the question/reference — so the paper can
state plainly that "counterfactual twin" is an audit-design term, not a claim that factuality is
the only attribute that differs.

Per-twin paired features over all 1,000 pqa_labeled twins (the human-curated headline
denominator source). Statistics: paired median difference with a paired bootstrap 95% CI
(B=2000, question-level resamples), paired standardized difference (mean/SD of within-pair diff),
Holm correction over the feature panel; plus a surface-only role classifier AUC (logistic on the
raw feature vector, 5-fold by question, twins never split across folds) to summarize how
separable the roles are on form alone. A form-matched sensitivity SUBSET is flagged (length ratio
in [0.8,1.25], equal citation-presence, |sentence-count diff|<=1) for a caption count.

Output: R066_twin_form_diagnostics_results.json (committable; aggregate stats only, no answer text).
"""
import os, sys, json, re, math
import numpy as np
from datasets import load_dataset

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
BOOT_SEED = 20260718; B_BOOT = 2000; NFOLD = 5

_num = re.compile(r"\d")
_cite = re.compile(r"\[\d+\]|\(\d{4}\)|et al\.|https?://|doi:", re.I)
_hedge = re.compile(r"\b(may|might|could|possibly|likely|suggest|appear|potential|often|generally|typically)\b", re.I)
_word = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(_word.findall(s.lower()))


def feats(ans, q, ref):
    words = _word.findall(ans.lower())
    n_w = max(len(words), 1)
    sents = max(len([s for s in re.split(r"[.!?]+", ans) if s.strip()]), 1)
    aset = toks(ans); qset = toks(q); rset = toks(ref)
    return dict(
        n_tokens=len(words),
        n_chars=len(ans),
        n_sents=sents,
        numeral_density=sum(bool(_num.search(w)) for w in words) / n_w,
        citation_markers=len(_cite.findall(ans)),
        hedge_density=len(_hedge.findall(ans)) / n_w,
        type_token_ratio=len(set(words)) / n_w,
        q_overlap=len(aset & qset) / max(len(aset), 1),
        ref_overlap=len(aset & rset) / max(len(aset), 1),
    )


FEATS = ["n_tokens", "n_chars", "n_sents", "numeral_density", "citation_markers",
         "hedge_density", "type_token_ratio", "q_overlap", "ref_overlap"]


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def main():
    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]
    T, Hd = [], []
    for r in ds:
        ref = " ".join(r["Knowledge"]) if isinstance(r["Knowledge"], list) else str(r["Knowledge"])
        q = r["Question"]
        T.append(feats(r["Ground Truth"], q, ref))
        Hd.append(feats(r["Hallucinated Answer"], q, ref))
    n = len(T)
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, n, size=(B_BOOT, n))

    rows = []
    raw_p = []
    for f in FEATS:
        t = np.array([d[f] for d in T]); h = np.array([d[f] for d in Hd])
        diff = h - t
        med = float(np.median(diff))
        bmed = np.median(diff[boot_idx], axis=1)
        ci = [round(float(np.percentile(bmed, 2.5)), 4), round(float(np.percentile(bmed, 97.5)), 4)]
        sd = diff.std(ddof=1)
        std_diff = float(diff.mean() / sd) if sd > 0 else 0.0
        # sign test p-value (paired, distribution-free) for "no location shift"
        pos = int((diff > 0).sum()); neg = int((diff < 0).sum()); nn = pos + neg
        from scipy.stats import binomtest
        p = binomtest(min(pos, neg), nn, 0.5).pvalue if nn else 1.0
        raw_p.append(p)
        rows.append(dict(feature=f, median_H=round(float(np.median(h)), 4),
                         median_T=round(float(np.median(t)), 4),
                         paired_median_diff=round(med, 4), diff_ci95=ci,
                         std_paired_diff=round(std_diff, 4), sign_p=p))
    adj = holm(np.array(raw_p))
    for r, a in zip(rows, adj):
        r["holm_p"] = float(a); r["sig_holm_.05"] = bool(a < 0.05)

    # surface-only role classifier AUC (question-blocked CV; twins never split)
    X = np.array([[d[f] for f in FEATS] for d in T] + [[d[f] for f in FEATS] for d in Hd], float)
    y = np.r_[np.zeros(n), np.ones(n)]
    grp = np.r_[np.arange(n), np.arange(n)]
    mu, sig = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sig
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score
        aucs = []
        for tr, te in GroupKFold(NFOLD).split(Xs, y, grp):
            clf = LogisticRegression(max_iter=1000).fit(Xs[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xs[te])[:, 1]))
        auc = dict(mean=round(float(np.mean(aucs)), 4), folds=[round(float(x), 4) for x in aucs])
    except Exception as e:
        auc = dict(error=str(e))

    # form-matched subset flag
    matched = 0
    for t, h in zip(T, Hd):
        lr = h["n_tokens"] / max(t["n_tokens"], 1)
        if 0.8 <= lr <= 1.25 and (h["citation_markers"] > 0) == (t["citation_markers"] > 0) \
           and abs(h["n_sents"] - t["n_sents"]) <= 1:
            matched += 1

    out = dict(run="R066 MedHallu twin surface-form paired diagnostics (descriptive)",
               n_twins=n, features=rows,
               role_classifier_auc_surface_only=auc,
               interpretation=("'counterfactual twin' is an audit-design term; these paired "
                               "diagnostics quantify residual non-factual differences (length, "
                               "lexical overlap, style). Descriptive, not a certificate."),
               form_matched_subset=dict(criteria="len ratio [0.8,1.25], equal citation-presence, |Δsent|<=1",
                                        n=matched, frac=round(matched / n, 4)))
    with open(artifact("R066_twin_form_diagnostics_results.json"), "w") as fo:
        json.dump(out, fo, indent=2)
    print(f"[twins] {n}  surface-only role AUC = {auc.get('mean')}  form-matched subset = {matched}/{n}")
    for r in rows:
        print(f"  {r['feature']:<18} medT={r['median_T']:<8} medH={r['median_H']:<8} "
              f"Δmed={r['paired_median_diff']:<8} std={r['std_paired_diff']:<7} holm_p={r['holm_p']:.2e} "
              f"{'*' if r['sig_holm_.05'] else ''}")
    print("[wrote] R066_twin_form_diagnostics_results.json")


if __name__ == "__main__":
    main()
