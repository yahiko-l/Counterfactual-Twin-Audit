#!/usr/bin/env python
"""Recompute per-family detector AUROC for the M12 second-dataset scores CSVs, so the AUROC ranges cited
in the paper (Table tab:second + sec:res-second) are verifiable against result files. Writes one meta
JSON per dataset (scores_*_meta.json) + a combined m12_auroc_summary.json. CPU."""
import csv, json
import numpy as np
from sklearn.metrics import roc_auc_score

FAMS = ["score_sjudge", "score_sptrue", "score_sNLI"]
DATASETS = {
    "halueval":        "data/scores/scores_halueval.csv",
    "medquad_subtle":  "data/scores/scores_medquad_subtle.csv",
    "medquad_blatant": "data/scores/scores_medquad.csv",   # blatant style
    "medquad_subtle_medgemma": "data/scores/scores_medquad_subtle_medgemma.csv",  # unconfounded judge (Gemma family)
}
# MedHallu pooled (two-file: 1k labeled + 9k artificial) for a consistent headline-family AUROC range
MEDHALLU = ["data/scores/scores_strong.csv", "data/scores/scores_strong_artificial.csv"]
summary = {}
mh_rows = [r for p in MEDHALLU for r in csv.DictReader(open(p))]
mhH = np.array([int(r["H"]) for r in mh_rows])
mh_auroc = {f: round(float(roc_auc_score(mhH, np.array([float(r[f]) for r in mh_rows]))), 4) for f in FAMS}
summary["medhallu_pooled"] = dict(n_rows=len(mh_rows), auroc=mh_auroc,
                                  auroc_min=min(mh_auroc.values()), auroc_max=max(mh_auroc.values()))
print(f"{'medhallu_pooled':16s} AUROC " + " ".join(f"{f.replace('score_s',''):>6s}={mh_auroc[f]}" for f in FAMS) +
      f"  -> range [{min(mh_auroc.values())}, {max(mh_auroc.values())}]")
for name, path in DATASETS.items():
    rows = list(csv.DictReader(open(path)))
    H = np.array([int(r["H"]) for r in rows])
    auroc = {}
    for f in FAMS:
        s = np.array([float(r[f]) for r in rows])
        auroc[f] = round(float(roc_auc_score(H, s)), 4)
    lo, hi = min(auroc.values()), max(auroc.values())
    summary[name] = dict(n_rows=len(rows), auroc=auroc, auroc_min=lo, auroc_max=hi)
    json.dump(dict(dataset=name, n_rows=len(rows), auroc=auroc, auroc_range=[lo, hi]),
              open(path.replace(".csv", "_meta.json"), "w"), indent=2)
    print(f"{name:16s} AUROC " + " ".join(f"{f.replace('score_s',''):>6s}={auroc[f]}" for f in FAMS) +
          f"  -> range [{lo}, {hi}]")
json.dump(summary, open("results/m12_auroc_summary.json", "w"), indent=2)
print("wrote results/m12_auroc_summary.json + per-dataset *_meta.json")
