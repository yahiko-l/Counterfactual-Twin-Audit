#!/usr/bin/env python
"""M6 — FIXED-τ clinician S_R audit recompute (grounds the ρ leaf on clinician labels; no threshold reselection).

Pairs with M6_build_SR_audit_sheet.py. Reads clinician verdicts (blinded, by blind_id) on M4's ACTUAL ref-NLI
accepted set and re-certifies the ρ leaf at the PUBLISHED τ*=0.412661 WITHOUT reselecting τ (so the paired ψ_g
leaf stays coherent). Two fixed-τ reports:

  * FULL-K clinician ρ leaf (needs the --full instrument; retires the hybrid caveat): ρ=#{clinician-halluc}/K
    at fixed τ*, exact-binomial one-sided UCB at M4's grid level δ_global/m (m=26).
  * 400-ROW STRATIFIED fixed-τ audit (minimum sufficient; keeps the nominal caveat): the sample uses fixed
    per-source quotas, so inference is STRATIFIED — invert each source's hypergeometric at δ/(#strata) and sum
    the per-source count upper bounds (a valid simultaneous finite-population bound on the realized K-row rate).

Robustness (per adversarial review):
  * BLINDED instrument: clinicians see only blind_id + text; the analyst key maps blind_id → everything.
  * FAIL-CLOSED join: the filled instrument's blind_id set must EXACTLY equal the key's (for the scope), no
    duplicates — a stale/permuted/mis-keyed sheet aborts instead of silently misattaching.
  * PASS uses WORST-CASE (disagreement/'unsure' = hallucinated) on ALL rows, or a third-clinician adjudicator
    file if present — never the agreement-selected subset (which is not the randomized audit sample). The
    strict-consensus point estimate (disagreements dropped) is reported separately as descriptive only.

`--dryrun` uses the key's stage-1 H (no clinician files) and MUST reproduce M4 (K=1130,S=159,ρ̂=0.1407,
ρ_UCB=0.1754) through the REAL blind_id→item_id join. CPU only.

Usage:
  python M6_certify_SR_audit.py --dryrun
  python M6_certify_SR_audit.py            # reads M6_SR_instrument_annotator{1,2}_FILLED.csv (+ _adjudicator_FILLED.csv opt.)
Output: M6_SR_audit_results.json
"""
import os, sys, json, csv, math
import numpy as np
from scipy.stats import beta as _beta, hypergeom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as C

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
ALPHA = 0.20; TAU_STAR = 0.412661
GRID_M = 26; LEVEL_FULL = C.DELTA_GLOBAL / GRID_M     # δ/26: matches M4's released ρ_UCB
LEVEL_AUDIT = 0.025                                    # one pre-specified fixed-τ audit (m=1)
KEY = artifact("M6_SR_audit_key.csv")


def cp_upper(k, n, level):
    if n == 0: return 1.0
    return 1.0 if k == n else float(_beta.ppf(1 - level, k + 1, n - k))


def hyper_count_ub(s_obs, n_draw, N_pop, level):
    """Largest S with P[Hypergeom(N_pop,S,n_draw) <= s_obs] > level (one-sided UCB on the population count)."""
    for S in range(s_obs, N_pop + 1):
        if float(hypergeom.cdf(s_obs, N_pop, S, n_draw)) <= level:
            return S - 1
    return N_pop


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0: return None
    po = float((a == b).mean())
    pe = sum(((a == k).mean()) * ((b == k).mean()) for k in (0, 1))
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def load_key():
    rows = list(csv.DictReader(open(KEY)))
    by_item, by_blind = {}, {}
    build_id = rows[0].get("build_id", "") if rows else ""
    for r in rows:
        d = dict(item_id=int(r["item_id"]), source=r["source"], qid=r["qid"], answerer=r["answerer"],
                 stage1_H=int(r["stage1_H"]), accepted=int(r["accepted"]), in_audit400=int(r["in_audit400"]),
                 blind_id=(int(r["blind_id"]) if str(r["blind_id"]).strip() != "" else None))
        by_item[d["item_id"]] = d
        if d["blind_id"] is not None: by_blind[d["blind_id"]] = d
    return by_item, by_blind, build_id


def read_instrument(path, key_build_id=None):
    if not os.path.exists(path): return None
    m = {}; seen_bid = None
    for r in csv.DictReader(open(path)):
        bid = int(r["blind_id"]); v = str(r.get("clinician_verdict", "")).strip().upper()
        b = r.get("build_id", "")
        if key_build_id is not None and b != key_build_id:
            raise SystemExit(f"FAIL-CLOSED: {os.path.basename(path)} build_id {b!r} != key {key_build_id!r} "
                             f"(stale instrument from a different key/shuffle).")
        if bid in m: raise SystemExit(f"FAIL-CLOSED: duplicate blind_id {bid} in {os.path.basename(path)}")
        m[bid] = 1 if v.startswith("HALL") else 0 if v.startswith("FAITH") else -1
    return m


def main():
    dry = "--dryrun" in sys.argv
    by_item, by_blind, build_id = load_key()
    acc_ids = [i for i, d in by_item.items() if d["accepted"] == 1]
    audit_ids = [i for i in acc_ids if by_item[i]["in_audit400"] == 1]
    scope_blind = set(by_blind.keys())           # the instrument scope (400 or full-K)
    N = len(acc_ids)

    if dry:
        # full ρ-leaf self-check needs all K accepted; the audit routes through the blind_id -> item_id join
        Hfull = {i: by_item[i]["stage1_H"] for i in acc_ids}
        H_audit_via_blind = {by_blind[b]["item_id"]: by_blind[b]["stage1_H"]
                             for b in scope_blind if by_blind[b]["in_audit400"] == 1}
        src = "stage-1 (self-check; audit via blind_id join)"
    else:
        a1 = read_instrument(artifact("M6_SR_instrument_annotator1_FILLED.csv"), build_id)
        a2 = read_instrument(artifact("M6_SR_instrument_annotator2_FILLED.csv"), build_id)
        adj = read_instrument(artifact("M6_SR_instrument_adjudicator_FILLED.csv"), build_id)
        if a1 is None or a2 is None:
            print("real mode needs M6_SR_instrument_annotator{1,2}_FILLED.csv"); sys.exit(1)
        # FAIL-CLOSED: both filled instruments' blind_id set must equal the key's scope exactly
        for nm, mp in (("annotator1", a1), ("annotator2", a2)):
            if set(mp.keys()) != scope_blind:
                miss = scope_blind - set(mp.keys()); extra = set(mp.keys()) - scope_blind
                raise SystemExit(f"FAIL-CLOSED: {nm} blind_id set != key scope "
                                 f"(missing {len(miss)}, extra {len(extra)}); stale/mis-keyed sheet.")
        src = "clinician (blinded, fail-closed join)"

    def cons(b, mode):
        """Consensus label for blind_id b: 'strict' (disagreement->None), 'wc' (disagreement/unsure->1)."""
        v1, v2 = a1[b], a2[b]
        if v1 == 1 and v2 == 1: return 1
        if v1 == 0 and v2 == 0: return 0
        if adj and adj.get(b, -1) in (0, 1): return adj[b]      # adjudicated
        return None if mode == "strict" else 1                  # drop (strict) / worst-case

    def rho_at(item_ids, labels, level):
        s = int(sum(labels[i] for i in item_ids if labels.get(i) is not None))
        n = sum(1 for i in item_ids if labels.get(i) is not None)
        return dict(K=n, S=s, rho_hat=round(s / n, 4) if n else None,
                    rho_UCB=round(cp_upper(s, n, level), 4), certifies=bool(cp_upper(s, n, level) <= ALPHA))

    def stratified_audit(item_ids, labels):
        """Stratified finite-population UCB: per source invert Hypergeom at LEVEL_AUDIT/#strata, sum count UBs."""
        strata = sorted(set(by_item[i]["source"] for i in acc_ids))
        lvl = LEVEL_AUDIT / len(strata); tot_ub = 0; per = {}
        for src_ in strata:
            Ns = sum(1 for i in acc_ids if by_item[i]["source"] == src_)                 # realized accepted in source
            samp = [i for i in item_ids if by_item[i]["source"] == src_ and labels.get(i) is not None]
            ns = len(samp); ss = int(sum(labels[i] for i in samp))
            ub = hyper_count_ub(ss, ns, Ns, lvl) if ns else Ns
            tot_ub += ub; per[src_] = dict(N=Ns, n=ns, s=ss, count_UB=ub)
        return dict(strata=per, total_count_UB=tot_ub, rate_UB=round(tot_ub / N, 4),
                    certifies=bool(tot_ub / N <= ALPHA), level=LEVEL_AUDIT, bonferroni_over=len(strata))

    out = dict(run="M6 fixed-τ clinician S_R audit (M4 ref-NLI accepted set; stratified, worst-case, fail-closed)",
               label_source=src, tau_star=TAU_STAR, alpha=ALPHA, N_accepted=N,
               level_full=LEVEL_FULL, level_audit=LEVEL_AUDIT)

    if dry:
        # full-K ρ leaf on stage-1 → must reproduce M4
        S = sum(Hfull.values()); K = len(Hfull)
        ucb = cp_upper(S, K, LEVEL_FULL)
        okrepro = (K == 1130 and S == 159 and abs(S / K - 0.1407) < 5e-4 and abs(ucb - 0.1754) < 5e-4)
        out["self_check"] = "PASS" if okrepro else "FAIL"
        out["full_1130"] = dict(K=K, S=S, rho_hat=round(S / K, 4), rho_UCB=round(ucb, 4),
                                certifies=bool(ucb <= ALPHA))
        # 400 stratified audit on stage-1, routed through the blind_id -> item_id join (proves the mapping)
        out["audit_400_stratified"] = stratified_audit(audit_ids, H_audit_via_blind)
        print(f"[dryrun] full ρ leaf K={K} S={S} ρ̂={S/K:.4f} ρ_UCB={ucb:.4f} (M4 1130/159/0.1407/0.1754) "
              f"-> {out['self_check']}")
        print(f"[dryrun] 400 stratified audit rate_UB={out['audit_400_stratified']['rate_UB']} "
              f"certifies ρ≤{ALPHA}: {out['audit_400_stratified']['certifies']}")
    else:
        strict = {by_blind[b]["item_id"]: cons(b, "strict") for b in scope_blind}
        wc = {by_blind[b]["item_id"]: cons(b, "wc") for b in scope_blind}
        both = [b for b in scope_blind if a1[b] in (0, 1) and a2[b] in (0, 1)]
        out["kappa"] = cohen_kappa([a1[b] for b in both], [a2[b] for b in both])
        out["n_both_labeled"] = len(both); out["n_scope"] = len(scope_blind)
        out["n_unsure_either"] = sum(1 for b in scope_blind if a1[b] == -1 or a2[b] == -1)
        out["n_binary_disagree"] = sum(1 for b in both if a1[b] != a2[b])
        out["n_non_consensus"] = out["n_unsure_either"] + out["n_binary_disagree"]   # worst-cased or adjudicated
        out["adjudicator_present"] = bool(adj)
        labeled_full = [by_blind[b]["item_id"] for b in scope_blind]
        is_full = set(labeled_full) == set(acc_ids)
        out["scope"] = "full-%d" % N if is_full else "audit-%d" % len(scope_blind)
        # descriptive point estimate (strict, dropped) + certificate (worst-case / adjudicated, all rows)
        out["point_estimate_strict"] = rho_at(labeled_full, strict, LEVEL_FULL if is_full else LEVEL_AUDIT)
        if is_full:
            out["full_certificate_worstcase"] = rho_at(labeled_full, wc, LEVEL_FULL)
        aud_ids_labeled = [by_blind[b]["item_id"] for b in scope_blind if by_blind[b]["in_audit400"] == 1]
        out["audit_400_stratified_worstcase"] = stratified_audit(aud_ids_labeled, wc)
        print(f"[real] κ={out['kappa']} both-labeled={len(both)}/{len(scope_blind)} "
              f"non-consensus(worst-cased/adjudicated)={out['n_non_consensus']} "
              f"(binary-disagree={out['n_binary_disagree']}, unsure={out['n_unsure_either']})")
        if is_full:
            fc = out["full_certificate_worstcase"]
            print(f"[real] full-{N} worst-case: S={fc['S']} ρ_UCB={fc['rho_UCB']} certifies ρ≤{ALPHA}: {fc['certifies']}")
        aw = out["audit_400_stratified_worstcase"]
        print(f"[real] 400 stratified worst-case: rate_UB={aw['rate_UB']} certifies ρ≤{ALPHA}: {aw['certifies']}")

    json.dump(out, open(artifact("M6_SR_audit_results.json"), "w"), indent=2)
    print(f"wrote {artifact('M6_SR_audit_results.json')}")


if __name__ == "__main__":
    main()
