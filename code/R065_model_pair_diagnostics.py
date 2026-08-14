#!/usr/bin/env python
"""R065 — ordered model-pair composition + leave-one-model-out for the real-stream paired cell
(reviewer Q4). DESCRIPTIVE / POST-HOC diagnostic per the analysis-plan deviation policy: labeled
as such, no new certificates, no model-specific attribution (the Scope paragraph's disclaimer
stands; this quantifies the confound instead of leaving it undocumented).

Basis: the frozen M3 twin pairing's ref-NLI rejected-truthful set at the fixed tau* = 0.412661
(analyst key M3_twin_key.csv; in_mc_sNLI / failpair_sNLI are the stage-1 proxy outcomes on the
frozen pairs; the clinician-relabeled aggregate for the same leaf is m_c = 57, s_c = 34 in
M4_real_arm_certificate.json — clinician relabels are per-pair blinded and reported only in
aggregate here). Outputs:
  * ordered matrix N_ab = #{M_T = a, M_H = b} and S_ab = #{... and F = 1} over the in_mc set;
  * per-model role marginals (truthful-role count / hallucinated-role count / F = 1 count);
  * leave-one-model-out psi_hat_(-m) with simultaneous (Bonferroni over models) two-sided 95%
    Clopper–Pearson intervals, labeled sensitivity.

Output: R065_model_pair_diagnostics_results.json (committable; model names + counts only).
"""
import os, sys, json, csv
import numpy as np
from scipy.stats import beta as sbeta

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
KEY = artifact("M3_twin_key.csv")
M4 = artifact("M4_real_arm_certificate.json")


def cp2(s, m, conf):
    lo = 0.0 if s == 0 else float(sbeta.ppf((1 - conf) / 2, s, m - s + 1))
    hi = 1.0 if s == m else float(sbeta.ppf(1 - (1 - conf) / 2, s + 1, m - s))
    return [round(lo, 4), round(hi, 4)]


def main():
    rows = list(csv.DictReader(open(KEY)))
    mc_rows = []
    for r in rows:
        if r["in_mc_sNLI"] != "True":
            continue
        # truthful-role / hallucinated-role answerer from the A/B role columns
        if r["A_twin_H"] == "0":
            mt, mh = r["A_answerer"], r["B_answerer"]
        else:
            mt, mh = r["B_answerer"], r["A_answerer"]
        mc_rows.append(dict(mt=mt, mh=mh, F=(r["failpair_sNLI"] == "True"), src=r["source"]))
    m_c = len(mc_rows); s_c = sum(r["F"] for r in mc_rows)
    models = sorted({r["mt"] for r in mc_rows} | {r["mh"] for r in mc_rows})
    print(f"[basis] frozen ref-NLI in_mc set: m_c={m_c}, s_c={s_c} (stage-1 proxy outcomes); "
          f"clinician-relabeled aggregate for the same leaf: 57/34 (M4)")

    N = {a: {b: 0 for b in models} for a in models}
    S = {a: {b: 0 for b in models} for a in models}
    role = {m: dict(truthful=0, halluc=0, F1=0) for m in models}
    for r in mc_rows:
        N[r["mt"]][r["mh"]] += 1
        S[r["mt"]][r["mh"]] += int(r["F"])
        role[r["mt"]]["truthful"] += 1
        role[r["mh"]]["halluc"] += 1
        role[r["mh"]]["F1"] += int(r["F"])

    lomo = []
    conf_simul = 1 - 0.05 / len(models)   # Bonferroni over the model panel, two-sided
    for m in models:
        keep = [r for r in mc_rows if r["mt"] != m and r["mh"] != m]
        mm, sm = len(keep), sum(r["F"] for r in keep)
        lomo.append(dict(model=m, m_c=mm, s_c=sm,
                         psi_hat=round(sm / mm, 4) if mm else None,
                         cp95_simultaneous=cp2(sm, mm, conf_simul) if mm else None))
        print(f"  LOMO -{m:<28} m={mm:>3} s={sm:>3} psi={sm/mm if mm else float('nan'):.3f} "
              f"CI={lomo[-1]['cp95_simultaneous']}")

    m4 = json.load(open(M4))["cells"]["score_sNLI"]["psi_leaf"]
    out = dict(run="R065 ordered model-pair composition + LOMO (descriptive/post-hoc diagnostic)",
               label=("descriptive confounding diagnostic; every retained pair is cross-model, so "
                      "model identity is confounded with twin role by design; no adjustment can "
                      "identify a single-model effect without same-model repeated generations"),
               basis=dict(set="frozen M3 pairing, ref-NLI rejected-truthful at tau*=0.412661",
                          outcome_labels="stage-1 proxy (failpair_sNLI)", m_c=m_c, s_c=s_c,
                          clinician_relabeled_aggregate=dict(m_c=m4["m_c"], s_c=m4["s_c"],
                                                             psi_hat=m4["psi_hat"])),
               models=models, ordered_matrix_N=N, ordered_matrix_S=S, role_marginals=role,
               lomo=dict(note="simultaneous two-sided CP at 0.05/|models| (Bonferroni); sensitivity, not certificates",
                         cells=lomo))
    with open(artifact("R065_model_pair_diagnostics_results.json"), "w") as fo:
        json.dump(out, fo, indent=2)
    psis = [c["psi_hat"] for c in lomo if c["psi_hat"] is not None]
    print(f"[lomo] psi_hat range without any single model: [{min(psis):.3f}, {max(psis):.3f}] "
          f"(pooled {s_c/m_c:.3f})")
    print("[wrote] R065_model_pair_diagnostics_results.json")


if __name__ == "__main__":
    main()
