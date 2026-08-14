#!/usr/bin/env python
"""Worst-case lower-envelope floor for the clinician-adjudicated cells (answers the reviewer charge that
'dropping a doubted pair can only weaken the failure' is not monotone: removing a NON-failure from the
denominator raises s/m).

The genuine WORST case for the paired lower bound keeps the FULL stage-1 rejected-truthful denominator (drop
nothing) and counts ONLY doubly-confirmed failures (strict two-clinician consensus) in the numerator -> the
smallest s over the largest m -> the LCB's infimum over any relabel that respects the confirmed failures.
We report the heterogeneity-robust HB psibar_g LCB (operative) and the exact CP psi_g LCB at each cell's level.

Cells:
  - Real Clinical Arm ref-NLI (@alpha=0.20): full stage-1 m_c=64, strict s_c=34; level=DELTA_LOWER/468.
  - MedHallu 3 headline cells (§5.13): full stage-1 m_c and strict consensus s_c from the B4 artifacts;
    level recovered from each cell's reported original exact psi_lcb.
CPU only. Output: worstcase_relabel_results.json
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
from scipy.optimize import brentq

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
E = math.e


def hb_lcb(s, m, level):
    if m <= 0 or s <= 0:
        return 0.0
    sh = s / m
    lcbH = sh - math.sqrt(math.log(1.0 / level) / (2.0 * m))
    lcbB = C.cp_lower_psi(s, m, level / E)
    return float(max(lcbH, lcbB, 0.0))


def worst(full_m, strict_s, level):
    return dict(m_c=full_m, s_c=strict_s, psi_hat=round(strict_s / full_m, 4),
                psibar_lcb_robust=round(hb_lcb(strict_s, full_m, level), 4),
                psi_lcb_exact=round(C.cp_lower_psi(strict_s, full_m, level), 4))


def main():
    out = {"run": "worst-case lower-envelope floor (full stage-1 denominator, doubly-confirmed numerator)"}

    # ---- Real Clinical Arm ref-NLI ----
    m3 = json.load(open(artifact("M3_human_adjudication_results.json")))
    nli = m3["cells"]["sNLI"]
    full_m = nli["stage1"]["m_c"]                       # 64: full stage-1 rejected-truthful denominator
    strict_s = nli["human"]["consensus_strict"]["s_c"]  # 34: doubly-confirmed failures
    NLOW = int(m3.get("NLOW", 468))
    out["real_arm_refNLI"] = {
        "alpha": 0.20,
        "budget_0.10": worst(full_m, strict_s, C.DELTA_LOWER / NLOW),
        "budget_0.05": worst(full_m, strict_s, (C.DELTA_LOWER / 2) / NLOW),
    }

    # ---- MedHallu 3 headline cells ----
    med = {}
    for tag, order in (("ptrue", "P(True)"), ("refnli", "ref-NLI"), ("judge", "judge")):
        p = artifact(f"B4_human_adjudication_{tag}_results.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        o = d["original"]
        strict = d["results"]["consensus_strict"]["s_c"]
        # recover the exact-leaf level from the reported original exact psi_lcb
        lvl = brentq(lambda L: C.cp_lower_psi(o["s_c"], o["m_c"], L) - o["psi_lcb"], 1e-12, 0.5)
        w = worst(o["m_c"], strict, lvl)
        w["alpha"] = d["alpha"]; w["level"] = float(lvl)
        w["conservative_min_lcb"] = d.get("min_relabel_psi_lcb")
        w["clears_alpha"] = bool(w["psibar_lcb_robust"] > d["alpha"])
        med[order] = w
    out["medhallu_3cell"] = med

    json.dump(out, open(artifact("worstcase_relabel_results.json"), "w"), indent=2, ensure_ascii=False)
    ra = out["real_arm_refNLI"]
    print("[Real Clinical Arm ref-NLI @α=0.20]  worst-case (full m=%d, strict s=%d)" %
          (ra["budget_0.10"]["m_c"], ra["budget_0.10"]["s_c"]))
    print("   robust ψ̄_g LCB: δtot0.10=%.4f  δtot0.05=%.4f  (exact %.4f/%.4f)" %
          (ra["budget_0.10"]["psibar_lcb_robust"], ra["budget_0.05"]["psibar_lcb_robust"],
           ra["budget_0.10"]["psi_lcb_exact"], ra["budget_0.05"]["psi_lcb_exact"]))
    print("[MedHallu 3-cell]  worst-case robust ψ̄_g LCB (order P(True)/ref-NLI/judge):")
    for k in ("P(True)", "ref-NLI", "judge"):
        if k in med:
            w = med[k]
            print("   %-9s α=%.2f  full m=%d strict s=%d  robust=%.4f exact=%.4f  %s"
                  % (k, w["alpha"], w["m_c"], w["s_c"], w["psibar_lcb_robust"], w["psi_lcb_exact"],
                     "✓>α" if w["clears_alpha"] else "✗"))
    print("wrote", artifact("worstcase_relabel_results.json"))


if __name__ == "__main__":
    main()
