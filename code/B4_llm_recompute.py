#!/usr/bin/env python
"""
B4 RECOMPUTE — turn a COMPLETED blinded adjudication sheet into a human-derived label-noise band
and a human-relabeled headline psi_g lower bound. Run AFTER clinicians fill B4_sheet_gpt55.csv.

Replaces the SIMULATED eta-band (paper Table 4(c) / m4_validity.py: s_wc=floor(s_c*(1-eta))) with a
MEASURED eta, and additionally reports the direct human-relabeled CP lower bound on the headline cell.

Pipeline:
  1. join completed B4_sheet_gpt55.csv (human `hallucination_in` ∈ {A,B,both,neither,unsure}) with
     B4_key.csv (unblinding: which letter is the MedHallu-hallucinated twin, cell membership).
  2. measure eta = P(human does NOT confirm the MedHallu-hallucinated answer is the hallucination),
     overall and per difficulty stratum, on the headline P(True) denominator (priority 1+2).
  3. (a) measured-eta worst-case band  : s_wc=floor(s_c*(1-eta)) -> cp_lower_psi  [comparable to sim band]
     (b) direct human-relabeled recompute: rebuild s_c,m_c from human labels -> cp_lower_psi  [gold]
  4. verdict: does the human-validated headline lower bound still exceed alpha=0.10?

If the sheet is not yet adjudicated, prints a PENDING summary and exits 0 (safe to run now).
CPU only.
"""
import os, sys, csv, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

SHEET = "results/B4_sheet_gpt55.csv"; KEY = "results/B4_key.csv"
ALPHA_HEAD = 0.10                      # headline P(True) cell alpha
NLOW = 234                             # |ALPHAS| * sum_f|grid_f| (pooled-10k low-family size; reproduces 0.327)
# integrity: the released headline ψ_lcb=0.327 at (s_c=120, m_c=278)
assert abs(C.cp_lower_psi(120, 278, C.DELTA_LOWER / NLOW) - 0.327) < 5e-3, "NLOW normalization drifted from released 0.327"

sheet = {int(r["item_id"]): r for r in csv.DictReader(open(SHEET))}
key = {int(r["item_id"]): r for r in csv.DictReader(open(KEY))}

def filled(r): return (r.get("hallucination_in", "") or "").strip().lower() in {"a", "b", "both", "neither", "unsure"}
done = [i for i in sheet if filled(sheet[i])]
head_ids = [i for i in key if key[i]["in_ptrue"] == "True"]                 # the 278 P(True) denominator
head_done = [i for i in head_ids if i in done]

if len(head_done) < len(head_ids):
    print("=" * 80)
    print("B4 LLM-PROXY — PENDING (sheet not yet fully adjudicated)")
    print("=" * 80)
    print(f"headline P(True) denominator items (priority 1+2)  : {len(head_ids)} (=278)")
    print(f"  of which priority-1 FAILURE pairs (most important): {sum(key[i]['ptrue_failpair']=='True' for i in head_ids)} (=120)")
    print(f"adjudicated so far                                  : {len(head_done)}/{len(head_ids)}")
    print(f"total sheet items (incl. ref-NLI/judge, priority 3) : {len(sheet)} (=965)")
    print("\nTO COMPLETE: have 2 blinded clinicians independently fill `hallucination_in` (A/B/both/")
    print("neither/unsure) + `severity` for at least priority 1+2 (the 278), adjudicate disagreements,")
    print("then re-run this script. It will emit the measured-eta band + human-relabeled headline LCB.")
    sys.exit(0)

# ---------- adjudicated: measure eta + recompute ----------
def human_calls_halluc_on(letter, r):
    """did the human flag `letter` (A/B) as containing the hallucination?"""
    hi = (r["hallucination_in"] or "").strip().lower()
    return hi == letter.lower() or hi == "both"

def confirms_role(i):
    """True if human CONFIRMS the MedHallu twin roles (hallucinated answer flagged, truthful not)."""
    r = sheet[i]; k = key[i]
    h_letter = "A" if k["A_role"] == "hallucinated" else "B"
    t_letter = "B" if h_letter == "A" else "A"
    return human_calls_halluc_on(h_letter, r) and not human_calls_halluc_on(t_letter, r)

# eta on the headline denominator (and per difficulty)
def eta_over(ids):
    adj = [i for i in ids if i in done and (sheet[i]["hallucination_in"] or "").strip().lower() != "unsure"]
    if not adj: return None, 0
    bad = sum(0 if confirms_role(i) else 1 for i in adj)
    return bad / len(adj), len(adj)

eta_all, n_all = eta_over(head_ids)
strata = {}
for d in sorted(set(key[i]["difficulty"] for i in head_ids)):
    strata[d] = eta_over([i for i in head_ids if key[i]["difficulty"] == d])

# headline original counts
s_c0 = sum(key[i]["ptrue_failpair"] == "True" for i in head_ids)            # 120
m_c0 = len(head_ids)                                                        # 278
delta_eff = C.DELTA_LOWER / NLOW

# (a) measured-eta worst-case band (paper's s_wc=floor(s_c*(1-eta)) formula)
def band_at(eta):
    s_wc = int(math.floor(s_c0 * (1 - eta))); return round(C.cp_lower_psi(s_wc, m_c0, delta_eff), 4), s_wc
band = {f"{e}": band_at(e) for e in (0.0, 0.05, 0.10, 0.15)}
band["measured"] = band_at(eta_all) + (round(eta_all, 4),)

# (b) direct human-relabeled recompute: a pair stays in m_c only if human confirms the truthful twin is
# supported (not flagged halluc); stays in s_c only if it is a failure pair AND human confirms the accepted
# (hallucinated) twin really hallucinates.
def relabeled():
    m_c = s_c = 0
    for i in head_ids:
        r = sheet[i]; k = key[i]
        h_letter = "A" if k["A_role"] == "hallucinated" else "B"; t_letter = "B" if h_letter == "A" else "A"
        if (r["hallucination_in"] or "").strip().lower() == "unsure":  # keep as-is (conservative: trust MedHallu)
            m_c += 1; s_c += (k["ptrue_failpair"] == "True"); continue
        truthful_ok = not human_calls_halluc_on(t_letter, r)
        halluc_real = human_calls_halluc_on(h_letter, r)
        if truthful_ok:
            m_c += 1
            if k["ptrue_failpair"] == "True" and halluc_real: s_c += 1
    return s_c, m_c, round(C.cp_lower_psi(s_c, m_c, delta_eff), 4)
s_h, m_h, lcb_h = relabeled()

report = dict(run="B4 LLM-proxy (gpt-5.5-codex) recompute -- cross-check, not human clinical", n_adjudicated=len(done), headline_done=len(head_done),
              eta_measured=dict(overall=round(eta_all, 4) if eta_all is not None else None, n=n_all,
                                by_difficulty={d: (round(v[0], 4) if v[0] is not None else None, v[1]) for d, v in strata.items()}),
              headline_original=dict(s_c=s_c0, m_c=m_c0, psi_lcb=0.327, alpha=ALPHA_HEAD),
              measured_eta_band=band,
              human_relabeled=dict(s_c=s_h, m_c=m_h, psi_lcb=lcb_h, alpha=ALPHA_HEAD,
                                   still_fails=bool(lcb_h > ALPHA_HEAD)))
json.dump(report, open("results/B4_llm_recompute_results.json", "w"), indent=2)
print("=" * 80); print("B4 LLM-PROXY (gpt-5.5-codex) RECOMPUTE [NOT human clinical adjudication] — headline P(True)-8B pi=0.10 alpha=0.10")
print("=" * 80)
print(f"measured eta (overall) = {eta_all:.4f} on n={n_all}; by difficulty: " +
      ", ".join(f"{d}:{v[0]:.3f}(n={v[1]})" for d, v in strata.items() if v[0] is not None))
print(f"\n(a) measured-eta worst-case band : psi_lcb={band['measured'][0]} at eta={band['measured'][2]} "
      f"(sim band was 0.327@0 -> {band['0.05'][0]}@.05 -> {band['0.1'][0]}@.10)")
print(f"(b) human-relabeled headline     : s_c {s_c0}->{s_h}, m_c {m_c0}->{m_h}, psi_lcb {lcb_h}")
print(f"\nVERDICT: human-validated headline {'STILL FAILS' if lcb_h > ALPHA_HEAD else 'NO LONGER certified-fails'} "
      f"(psi_lcb {lcb_h} {'>' if lcb_h > ALPHA_HEAD else '<='} alpha {ALPHA_HEAD})")
print("  wrote results/B4_llm_recompute_results.json")
