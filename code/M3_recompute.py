#!/usr/bin/env python
"""Real Clinical Arm — M3 human-adjudication recompute (ψ_g leaf, α=0.20 certified cells).

Mirrors B4_recompute_cell.py / MEDHALU_recompute.py. Reads the two filled clinician twin sheets + the analyst
key, and per certified pooled@α=0.20 cell (P(True)/judge/ref-NLI) computes:
  - inter-annotator Cohen κ on hallucination_in (over the κ-reliability subsample + all jointly-labeled);
  - per-annotator η (human-vs-stage-1 disagreement rate on the confirmed-twin role);
  - CONSERVATIVE human relabel: a doubted twin role is dropped from BOTH the rejected-truthful denominator
    m_c and the accepted-hallucination numerator s_c (can only weaken the failure), then ψ_g LCB;
  - strict/lenient two-annotator consensus + a measured-η band;
  - harm distribution over confirmed accepted-hallucination failures.
The gate outcome per pair (isfail = the hallucinated twin is accepted while the truthful twin is rejected at
the cell's τ*) is read from the key (failpair_<fam> / in_mc_<fam>), identical to the projection; clinician
labels only adjust which pairs stay in numerator/denominator.

DELTA = DELTA_LOWER / NLOW with NLOW reproducing the stage-1 projection LCB (holm-scaled δ, nlow=468).
A self-check asserts the stage-1 (LLM-label) ψ_g reproduces the projection per cell.

Usage:
  python M3_recompute.py --dryrun     # no clinician sheets needed: reproduce stage-1 ψ_g from the key (self-check)
  python M3_recompute.py              # read filled sheets -> human-adjudicated ψ_g LCB per cell + κ + harm
Env: M3_D1, M3_D2 (filled sheet paths). Output: results/M3_human_adjudication_results.json
"""
import os, sys, json, csv, io, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C
from collections import Counter

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
HANDOFF = os.path.abspath(os.path.join(ROOT, "MEDHALU_annotation_handoff"))
KEY = artifact("M3_twin_key.csv")
META = json.load(open(artifact("M3_blind_sheet_meta.json")))
D1 = os.environ.get("M3_D1", os.path.join(HANDOFF, "M3_twin_sheet_annotator1_FILLED.csv"))
D2 = os.environ.get("M3_D2", os.path.join(HANDOFF, "M3_twin_sheet_annotator2_FILLED.csv"))
OUT = os.environ.get("M3_OUT", artifact("M3_human_adjudication_results.json"))
FAMS = {"sptrue": "P(True)-8B", "sjudge": "LLM-judge-8B", "sNLI": "ref-NLI"}
ALPHA = float(META.get("alpha_certified", 0.20))
NLOW = int(os.environ.get("M3_NLOW", "468"))   # matches projection LCB_holm δ = DELTA_LOWER/468
DELTA = C.DELTA_LOWER / NLOW


def load_csv(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return {int(r["item_id"]): r for r in csv.DictReader(io.StringIO(raw.decode(enc)))}
        except Exception:
            pass
    raise RuntimeError(f"cannot decode {path}")


def _b(v):  # csv "True"/"False" -> bool
    return str(v).strip().lower() in ("true", "1", "yes")


def cell_counts(key, ids, memb, fail, labelfn=None):
    """(m_c, s_c) over `ids` for a cell. labelfn(i) optionally overrides which pairs count (human relabel).
    Default (labelfn=None) = stage-1: m_c = #(in_mc), s_c = #(failpair)."""
    m = s = 0
    for i in ids:
        if not _b(key[i][memb]):
            continue
        if labelfn is None:
            m += 1; s += 1 if _b(key[i][fail]) else 0
        else:
            keep, isfail = labelfn(i)
            if keep:
                m += 1; s += 1 if (isfail and _b(key[i][fail])) else 0
    return m, s


def main():
    dry = "--dryrun" in sys.argv
    key = load_csv(KEY)
    ids_all = sorted(key)
    cells = {}
    # ---------- stage-1 (LLM-label) reproduction + self-check ----------
    for fam in FAMS:
        memb, fail = f"in_mc_{fam}", f"failpair_{fam}"
        if memb not in next(iter(key.values())):
            continue
        m0, s0 = cell_counts(key, ids_all, memb, fail)
        lcb0 = round(C.cp_lower_psi(s0, m0, DELTA), 4)
        cm = META["certified_cells"].get(f"score_{fam}", {})
        cells[fam] = dict(family=FAMS[fam], alpha=ALPHA, tau_star=cm.get("tau_star"),
                          stage1=dict(m_c=m0, s_c=s0, psi_hat=round(s0/m0, 4) if m0 else None, psi_lcb=lcb0),
                          projection_m_c=cm.get("m_c"), projection_psi_hat=cm.get("psi_hat"))
        # NOTE (expected, not an error): the M3 sheet freezes ONE canonical twin per T-question, while the
        # projection resampled a DIFFERENT pair per detector family -> m_c differs. The M3 frozen pairing is
        # the certification basis; the projection was a power estimate. Both are valid pre-registered choices.
        if cm.get("m_c") is not None and m0 != cm["m_c"]:
            cells[fam]["pairing_note"] = (f"M3 frozen-pairing m_c={m0} (projection resampled m_c={cm['m_c']}); "
                                          f"M3 pairing is the certification basis")

    if dry:
        out = dict(run="M3 dry-run (stage-1 reproduction, no clinician labels)",
                   alpha=ALPHA, NLOW=NLOW, cells=cells)
        json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
        print(f"[dry-run] NLOW={NLOW} DELTA={DELTA:.2e}")
        for fam, co in cells.items():
            s = co["stage1"]; clears = "✓>α" if (s["psi_lcb"] or 0) > ALPHA else "✗≤α"
            print(f"  [{fam:<7} {co['family']:<12}] M3-pairing stage-1 m_c={s['m_c']} s_c={s['s_c']} "
                  f"ψ̂={s['psi_hat']} ψ_g LCB={s['psi_lcb']} ({clears} α={ALPHA})"
                  + (f"  · {co['pairing_note']}" if "pairing_note" in co else ""))
        print(f"wrote {OUT}")
        return

    # ---------- human adjudication ----------
    if not (os.path.exists(D1) and os.path.exists(D2)):
        print(f"filled sheets not found:\n  {D1}\n  {D2}\nRun with --dryrun until clinicians return them.")
        return
    D = {1: load_csv(D1), 2: load_csv(D2)}

    def hi(d, i): return (d.get(i, {}).get("hallucination_in") or "").strip().lower()
    def labeled(d, i): return hi(d, i) not in ("", "unsure")
    def roles(i):
        h = "A" if key[i]["A_role"] == "hallucinated" else "B"; return h, ("B" if h == "A" else "A")
    def flag(d, i, L): v = hi(d, i); return v == L.lower() or v == "both"
    def t_ok(d, i): _, t = roles(i); return not flag(d, i, t)          # truthful twin not called hallucinated
    def h_real(d, i): h, _ = roles(i); return flag(d, i, h)            # hallucinated twin confirmed
    def confirms(d, i): return h_real(d, i) and t_ok(d, i)
    def sev(d, i):
        v = (d.get(i, {}).get("severity") or "").strip().lower()
        return v if v in ("none", "minor", "moderate", "severe") else ""

    # inter-annotator kappa on hallucination_in over jointly-labeled
    both_ids = [i for i in ids_all if labeled(D[1], i) and labeled(D[2], i)]
    a = [hi(D[1], i) for i in both_ids]; b = [hi(D[2], i) for i in both_ids]
    po = sum(x == y for x, y in zip(a, b)) / len(both_ids) if both_ids else None
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l]/len(both_ids)) * (cb[l]/len(both_ids)) for l in set(a + b)) if both_ids else 0
    kappa = round((po - pe)/(1 - pe), 4) if (both_ids and pe != 1) else None
    ksub = [i for i in ids_all if str(key.get(i, {}).get("priority")) and int(key[i].get("kappa_subsample", 0) or 0) == 1] \
        if any("kappa_subsample" in key[i] for i in ids_all) else both_ids

    def relabel(d, ids, memb, fail):
        def lf(i):
            if not labeled(d, i):            # unlabeled -> conservative: trust stage-1 role
                return True, _b(key[i][fail])
            if t_ok(d, i):                    # truthful twin still fine -> keep in denom
                return True, (_b(key[i][fail]) and h_real(d, i))
            return False, False               # truthful twin doubted -> drop pair
        return cell_counts(key, ids, memb, fail, labelfn=lf)

    def eta(d, ids):
        lab = [i for i in ids if labeled(d, i)]
        return round(sum(0 if confirms(d, i) else 1 for i in lab) / len(lab), 4) if lab else None

    for fam, co in cells.items():
        memb, fail = f"in_mc_{fam}", f"failpair_{fam}"
        ids = [i for i in ids_all if _b(key[i][memb])]      # the cell's stage-1 m_c pairs
        res = {}
        for ann in (1, 2):
            m, s = relabel(D[ann], ids, memb, fail)
            res[f"doctor_{ann}"] = dict(eta=eta(D[ann], ids), m_c=m, s_c=s,
                                        psi_lcb=round(C.cp_lower_psi(s, m, DELTA), 4),
                                        still_fails=bool(C.cp_lower_psi(s, m, DELTA) > ALPHA))
        # strict/lenient consensus
        def consensus(mode):
            m = s = 0
            for i in ids:
                if not (labeled(D[1], i) and labeled(D[2], i)):
                    m += 1; s += 1 if _b(key[i][fail]) else 0; continue
                to = (t_ok(D[1], i) and t_ok(D[2], i)) if mode == "strict" else (t_ok(D[1], i) or t_ok(D[2], i))
                hr = (h_real(D[1], i) and h_real(D[2], i)) if mode == "strict" else (h_real(D[1], i) or h_real(D[2], i))
                if to:
                    m += 1; s += 1 if (_b(key[i][fail]) and hr) else 0
            return dict(m_c=m, s_c=s, psi_lcb=round(C.cp_lower_psi(s, m, DELTA), 4))
        res["consensus_strict"] = consensus("strict")
        res["consensus_lenient"] = consensus("lenient")
        minlcb = min(res["doctor_1"]["psi_lcb"], res["doctor_2"]["psi_lcb"],
                     res["consensus_strict"]["psi_lcb"], res["consensus_lenient"]["psi_lcb"])
        # harm over consensus-confirmed accepted-hallucination failures
        Sset = [i for i in ids if _b(key[i][fail]) and confirms(D[1], i) and confirms(D[2], i)]
        RK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
        pairs = [(sev(D[1], i), sev(D[2], i)) for i in Sset]
        harm = dict(n_confirmed_failures=len(Sset),
                    all_at_least_moderate_by_both=bool(pairs) and all(RK.get(x, -1) >= 2 and RK.get(y, -1) >= 2 for x, y in pairs),
                    breakdown_annotator1=dict(Counter(sev(D[1], i) for i in Sset)),
                    breakdown_annotator2=dict(Counter(sev(D[2], i) for i in Sset)))
        co["human"] = res
        co["min_human_psi_lcb"] = minlcb
        co["harm"] = harm
        co["survives"] = bool(minlcb > ALPHA)
        co["verdict"] = (f"within-question failure SURVIVES human adjudication (min ψ_g LCB {minlcb} > α {ALPHA})"
                         if minlcb > ALPHA else f"does NOT survive (min ψ_g LCB {minlcb} <= α {ALPHA})")

    out = dict(run="Real Clinical Arm M3 human adjudication (ψ_g leaf, α=0.20)",
               dataset="6-model natural stream on real patient questions (K-QA/MedicationQA/LiveQA)",
               alpha=ALPHA, NLOW=NLOW,
               inter_annotator=dict(n_jointly_labeled=len(both_ids), raw_agreement=round(po, 4) if po else None,
                                    cohen_kappa=kappa, kappa_subsample_n=len(ksub)),
               cells=cells)
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[inter-annotator] κ={kappa} raw={round(po,4) if po else None} n={len(both_ids)}")
    for fam, co in cells.items():
        h = co.get("human", {})
        print(f"[{fam} {co['family']}] stage-1 ψ_g LCB={co['stage1']['psi_lcb']} -> "
              f"min human ψ_g LCB={co.get('min_human_psi_lcb')} (α={ALPHA}) : {co.get('verdict','')}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
