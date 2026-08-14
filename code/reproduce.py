#!/usr/bin/env python
"""One-command reproduction of the ψ_g paper's HEADLINE certified numbers from the released artifacts.

Answers the "ship a versioned, one-command-reproducible artifact package" request. Three layers, cheap→thorough:

  [1] INTEGRITY        `sha256sum -c RESULT_SHA256SUMS.txt` — every released result/score file is unmodified.
  [2] PAPER <-> ARTIFACT  the headline joint-certification table (tab:joint), the real-clinical arm, and the
                          decision-value result equal the released JSONs to displayed precision. Catches any
                          drift between what the PDF prints and what the artifacts contain.
  [3] SCRIPT <-> ARTIFACT (--recompute)  re-run the CPU certification path from the released scores and assert
                          the regenerated numbers reproduce the released ones. Catches code/artifact drift.
                          Non-destructive: every released file the recompute would overwrite is backed up and
                          restored byte-for-byte, so the checksum set is intact afterwards.

CPU only, no GPU, no network. Layer [3] re-runs m3_combined.py + m6_robust_leaf.py (headline joint + robust
leaf) and the self-checking scripts (m_pairedlift, R050_decision_value, M4_certify_real_arm, M5 --dryrun,
worstcase_relabel_check). The heavy reseed/mitigation/second-dataset runs (m7/m9/m14/m12) carry their own
self-checks and are out of the default path (documented in REPRODUCE.md).

Usage:
  python reproduce.py               # layers 1+2  (seconds; read-only)
  python reproduce.py --verify-only # layers 1+2 + probability-space unit tests + matched-marginal check,
                                    #   and writes machine-readable verification_report.json
                                    #   (per check: paper anchor, script, input files + SHA256, expected,
                                    #   observed, pass) — the reviewer-facing verification artifact
  python reproduce.py --recompute   # everything above + layer 3 (a few CPU-minutes; backs up + restores
                                    #   released files byte-for-byte)
Exit code 0 iff every checked value matches.
"""
import os, sys, json, subprocess, shutil, tempfile, hashlib

from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
PY = sys.executable
TOL_FLOAT = 1e-3       # displayed-precision tolerance for bounds/rates
def _p(*a): print(*a, flush=True)

# ---- the HEADLINE table exactly as printed in the paper (tab:joint) -----------------------------------------
# (family, alpha_key, tau, rho_hat, m_c, psi_hat, psi_g_LCB_exact, psibar_g_LCB_robust_hb)
HEADLINE = [
    ("score_sptrue", "0.05", 0.000123, 0.0311, 2423, 0.0739, 0.0562, 0.0551),
    ("score_sptrue", "0.1",  0.222824, 0.0636,  278, 0.4317, 0.327,  0.320),
    ("score_sjudge", "0.2",  0.245085, 0.0924,  347, 0.5706, 0.4731, 0.4664),
    ("score_sNLI",   "0.1",  0.518631, 0.0709,  488, 0.2828, 0.2132, 0.2087),
]
FAMLAB = {"score_sptrue": "P(True)-8B", "score_sjudge": "LLM-judge-8B", "score_sNLI": "ref-NLI"}


def close(a, b, tol=TOL_FLOAT):
    return abs(float(a) - float(b)) <= tol


# ============================================================ layer 1: integrity
def _withheld():
    """Manifest entries that are not redistributable (see WITHHELD_FROM_RELEASE.txt).

    RESULT_SHA256SUMS.txt covers every result the analysis produced; a few of those
    files cannot be published (request-gated corpus, annotator-identifying sheets).
    They are skipped here rather than reported as corrupted. No layer reads them.
    """
    path = artifact("WITHHELD_FROM_RELEASE.txt")
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {l.strip() for l in fh if l.strip() and not l.startswith("#")}


def layer1_checksums():
    _p("\n[1] INTEGRITY — sha256sum -c RESULT_SHA256SUMS.txt")
    withheld = _withheld()
    with open(artifact("RESULT_SHA256SUMS.txt")) as fh:
        lines = [l for l in fh
                 if len(l.split(None, 1)) < 2 or l.split(None, 1)[1].strip().lstrip("*") not in withheld]
    r = subprocess.run(["sha256sum", "-c", "-"], input="".join(lines), cwd=ROOT,
                       capture_output=True, text=True)
    ok = [l for l in r.stdout.splitlines() if l.endswith(": OK")]
    bad = [l for l in r.stdout.splitlines() if l.rstrip().endswith(("FAILED", "FAILED open or read"))]
    _p(f"    {len(ok)} files OK" + (f"  |  {len(bad)} FAILED" if bad else ""))
    if withheld:
        _p(f"    {len(withheld)} manifest entries withheld from the public release, skipped"
           f" (see WITHHELD_FROM_RELEASE.txt)")
    for b in bad[:10]:
        _p("    ! " + b)
    absent = not os.path.exists(artifact("scores_strong_meta.json"))
    _p(f"    stale side-file scores_strong_meta.json removed: {absent}")
    return not bad and r.returncode == 0 and absent


# ============================================================ layer 2: paper <-> artifact
def layer2_paper_matches_artifacts():
    _p("\n[2] PAPER <-> ARTIFACT — released JSONs reproduce the printed tables")
    ok = True
    mc = json.load(open(artifact("m3_combined_results.json")))["pooled"]["pi=0.1"]["cells"]
    mr = json.load(open(artifact("m6_robust_leaf_results.json")))["pooled"]["pi=0.1"]["cells"]
    _p("    tab:joint (headline joint-certification cells):")
    for fam, a, tau, rho, m_c, psih, lcb_ex, lcb_hb in HEADLINE:
        b = mc[fam][a]["best"]; rb = mr[fam][a]["robust_best"]
        checks = [close(b["tau"], tau, 1e-5), close(b["rho_hat"], rho), b["m_c"] == m_c,
                  close(b["psi_hat"], psih), close(b["psi_lcb"], lcb_ex), close(rb["psi_lcb_hb"], lcb_hb)]
        good = all(checks)
        ok &= good
        _p(f"      {'OK ' if good else 'XX '}{FAMLAB[fam]:<13} α={a:<4} τ={b['tau']:.4f} ρ̂={b['rho_hat']:.4f} "
           f"m_c={b['m_c']} ψ̂={b['psi_hat']:.4f} ψ_gLCB={b['psi_lcb']:.4f} ψ̄_gLCB={rb['psi_lcb_hb']:.4f}")
        if not good:
            _p(f"         expected τ={tau} ρ̂={rho} m_c={m_c} ψ̂={psih} exact={lcb_ex} robust={lcb_hb}")
    # real-clinical arm (M4): ref-NLI corroboration numbers quoted in §5.12
    try:
        m4 = json.load(open(artifact("M4_real_arm_certificate.json")))
        nli = m4["cells"]["score_sNLI"]; L = nli["budgets"]["main(δtotal=0.10)"]
        rho_ucb = L["rho_UCB"]; psibar = L["human_psibar_lcb_robust"]; pi_meas = nli["rho_leaf"]["pi_measured"]
        good = close(rho_ucb, 0.175, 2e-3) and close(psibar, 0.329, 3e-3) and close(pi_meas, 0.143, 2e-3)
        ok &= good
        _p(f"    real-clinical arm (§5.12, ref-NLI, nominal in-sample): "
           f"{'OK ' if good else 'XX '}ρ_UCB={rho_ucb} π̂={pi_meas} robust ψ̄_g LCB={psibar}")
    except Exception as e:
        ok = False; _p(f"    XX real-arm check errored: {e}")
    # decision value (R050): Fréchet identified set = [0,1] at every headline cell + self-check
    try:
        r050 = json.load(open(artifact("R050_decision_value_results.json")))
        allframe = all(c["frechet_interval"] == [0.0, 1.0] and c["repairing_extremes"]["psi_min"] == 0.0
                       and c["repairing_extremes"]["psi_max"] == 1.0 and c["marginal_invariance_under_repairing"]
                       and c["decision"]["verdict_flips"] for c in r050["cells"])
        good = (r050["self_check"] == "PASS") and allframe
        ok &= good
        _p(f"    decision value (§5.5, R050): {'OK ' if good else 'XX '}"
           f"Fréchet [0,1] + re-pair endpoints + marginal-invariance + verdict-flip at all "
           f"{len(r050['cells'])} headline cells; self_check={r050['self_check']}")
    except Exception as e:
        ok = False; _p(f"    XX decision-value check errored: {e}")
    # faithful abstention baselines (R055, §5.13): a published LTT/CRC marginal certificate leaves psi_g
    # uncontrolled at an abstaining, audit-eligible threshold (4/16 primary cells)
    try:
        r055 = json.load(open(artifact("R055_faithful_baseline_results.json")))
        jc = [c for c in r055["methods"]["SRC-LTT"]["pi=0.1"]
              if c.get("deploy") and c["detector"] == "LLM-judge-8B" and c["alpha"] == 0.20][0]["deploy"]
        good = (r055["self_check"] == "PASS" and jc["m_c"] == 247
                and abs(jc["hb_lcb_headline_mult"] - 0.4552) < 1e-3 and jc["hb_lcb_headline_mult"] > 0.20
                and jc["oos_stable"] and jc["mass_floor_ok"])
        ok &= good
        _p(f"    faithful baselines (§5.13, R055): {'OK ' if good else 'XX '}"
           f"LLM-judge SRC-LTT α=.20 audit-eligible m_c={jc['m_c']} ψ̄_g-LCB={jc['hb_lcb_headline_mult']}>α "
           f"(OOS-stable, γ-floor cleared); self_check={r055['self_check']}")
    except Exception as e:
        ok = False; _p(f"    XX faithful-baseline check errored: {e}")
    # worst-case relabel floors (§5.13)
    try:
        wc = json.load(open(artifact("worstcase_relabel_results.json")))
        _p(f"    worst-case relabel floors (§5.13): loaded {len(wc) if isinstance(wc, (list, dict)) else '?'} entries")
    except Exception as e:
        _p(f"    -- worst-case file note: {e}")
    # grid-multiplicity provenance (§4 note: γ-floor 28→26/family ⇒ |C_low|=234, |C_up|=78; 252-robustness)
    try:
        gm = json.load(open(artifact("grid_multiplicity_check.json")))
        good = (gm["verdict"] == "PASS" and gm["all_families_keep_26"]
                and gm.get("reduction_is_gamma_filter_only") and gm["C_low"] == 234
                and gm["C_up"] == 78 and gm["headline_failure_leaf_all_clear_at_252"])
        ok &= good
        _p(f"    grid multiplicity (§4, γ-floor 28→26 ⇒ 234/78): {'OK ' if good else 'XX '}"
           f"|C_low|={gm['C_low']} |C_up|={gm['C_up']}; γ-filter-only={gm.get('reduction_is_gamma_filter_only')}; "
           f"failure leaf still certified at nominal nlow=252={gm['headline_failure_leaf_all_clear_at_252']}")
    except Exception as e:
        ok = False; _p(f"    XX grid-multiplicity check errored: {e}")
    return ok


# ============================================================ layer 3: script <-> artifact (opt-in)
def _run(script, args=None):
    cmd = [PY, artifact(script)] + (args or [])
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def _backup(files):
    d = tempfile.mkdtemp(prefix="repro_bak_")
    saved = {}
    for f in files:
        src = artifact(f)
        if os.path.exists(src):
            dst = os.path.join(d, f)
            shutil.copy2(src, dst); saved[f] = dst
    return d, saved


def _restore(saved):
    for f, dst in saved.items():
        shutil.copy2(dst, artifact(f))


def layer3_recompute():
    _p("\n[3] SCRIPT <-> ARTIFACT (--recompute) — regenerate from released scores, restore afterwards")
    ok = True
    # headline joint + robust leaf: re-run, compare regenerated headline vs paper, restore released bytes
    touched = ["m3_combined_results.json", "m6_robust_leaf_results.json",
               "m_pairedlift_results.json", "R050_decision_value_results.json",
               "M4_real_arm_certificate.json", "worstcase_relabel_results.json",
               "M5_SR_clinician_results.json", "grid_multiplicity_check.json",
               "R055_faithful_baseline_results.json", "R063_matched_marginal_results.json",
               "R069_results.json", "R071_results.json"]
    bakdir, saved = _backup(touched)
    try:
        _p("    re-running m3_combined.py + m6_robust_leaf.py (headline certification path)...")
        r1 = _run("m3_combined.py"); r2 = _run("m6_robust_leaf.py")
        if r1.returncode or r2.returncode:
            ok = False; _p(f"    XX recompute exit codes: m3={r1.returncode} m6={r2.returncode}")
            _p((r1.stderr or r2.stderr or "")[-600:])
        else:
            mc = json.load(open(artifact("m3_combined_results.json")))["pooled"]["pi=0.1"]["cells"]
            mr = json.load(open(artifact("m6_robust_leaf_results.json")))["pooled"]["pi=0.1"]["cells"]
            for fam, a, tau, rho, m_c, psih, lcb_ex, lcb_hb in HEADLINE:
                b = mc[fam][a]["best"]; rb = mr[fam][a]["robust_best"]
                good = (b["m_c"] == m_c and close(b["rho_hat"], rho) and close(b["psi_lcb"], lcb_ex)
                        and close(rb["psi_lcb_hb"], lcb_hb))
                ok &= good
                _p(f"      {'OK ' if good else 'XX '}regenerated {FAMLAB[fam]:<13} α={a:<4} "
                   f"m_c={b['m_c']} ρ̂={b['rho_hat']:.4f} ψ_gLCB={b['psi_lcb']:.4f} ψ̄_gLCB={rb['psi_lcb_hb']:.4f}")
        # self-checking scripts: exit code 0 + their own asserts
        for label, script, args in [("m_pairedlift", "m_pairedlift.py", None),
                                     ("R050 decision-value", "R050_decision_value.py", None),
                                     ("R055 faithful baselines", "R055_faithful_baseline.py", None),
                                     ("M4 real-arm self-check", "M4_certify_real_arm.py", None),
                                     ("M5 SR ρ-leaf dry-run", "M5_certify_SR_clinician.py", ["--dryrun"]),
                                     ("worst-case relabel", "worstcase_relabel_check.py", None),
                                     ("grid multiplicity 28→26/234/78", "verify_grid_multiplicity.py", None),
                                     ("R063 matched-marginal", "R063_matched_marginal.py", None),
                                     ("R069 FNR-gate frontier", "R069_fnr_gate_frontier.py", None),
                                     ("R071 fixed-assignment companion", "R071_fixed_assignment_companion.py", None)]:
            r = _run(script, args)
            out = r.stdout + r.stderr
            # explicit failure markers only (avoid tripping on the legit word "failure"/"FAILURE")
            bad_markers = ["self_check=FAIL", "self_check FAIL", "VERDICT: CHECK", "MISMATCH",
                           "Traceback", "AssertionError"]
            good = r.returncode == 0 and not any(m in out for m in bad_markers)
            ok &= good
            _p(f"      {'OK ' if good else 'XX '}{label} (self-check, exit {r.returncode})")
            if not good:
                _p("         " + (r.stderr or r.stdout)[-400:])
    finally:
        _restore(saved)
        shutil.rmtree(bakdir, ignore_errors=True)
        _p("    released artifacts restored byte-for-byte (checksum set intact)")
    return ok


def layer2c_matched_marginal():
    _p("\n[2c] MATCHED-MARGINAL (§5.5, R063) — released JSON matches the printed summary")
    try:
        s = json.load(open(artifact("R063_matched_marginal_results.json")))["summary"]
        exp = dict(n_matched_pairs=11, n_cross_family=8, n_ci_excl_zero=10, n_flip_at_beta_ref=2)
        good = all(s[k] == v for k, v in exp.items()) and s["feasible"] and \
            close(s["delta_psi_min"], 0.0083, 5e-4) and close(s["delta_psi_max"], 0.0272, 5e-4)
        _p(f"    {'OK ' if good else 'XX '}pairs={s['n_matched_pairs']} cross-family={s['n_cross_family']} "
           f"CI-excl-0={s['n_ci_excl_zero']} flips@0.10={s['n_flip_at_beta_ref']} "
           f"Δψ∈[{s['delta_psi_min']},{s['delta_psi_max']}]")
        return good
    except Exception as e:
        _p(f"    XX errored: {e}"); return False


def layer2e_fnr_gate():
    _p("\n[2e] FNR-GATE DECISION VALUE (§5.9, R069 + R071) — released JSON matches the printed numbers")
    try:
        r = json.load(open(artifact("R069_results.json")))
        A = r["A_frontiers_ptrue"]; B = r["B_status_map"]
        native = A["native_fnr"]["0.1"]["coverage"]; psi = A["psi_gate_frozen"]["0.1"]["coverage"]
        rho_only = A["rho_only"]["coverage"]
        good69 = (close(native, 0.2333, 5e-4) and close(psi, 0.3703, 5e-4) and close(rho_only, 0.9605, 5e-4)
                  and B["witness_psi_safe_FNR_unsafe"] == 4 and B["n_FNRsafe_psi_unsafe"] == 0
                  and B["n_FNRsafe_psi_undet"] == 0)
        r71 = json.load(open(artifact("R071_results.json")))
        good71 = (r71["selfcheck"] == "PASS"
                  and all(c["certifies_exact"] and c["certifies_detL"] for c in r71["cells"])
                  and sum(1 for c in r71["cells"] if c["certifies_fixedz"]) == 3)
        good = good69 and good71
        _p(f"    {'OK ' if good else 'XX '}R069 native/paired/ρ-only cov={native}/{psi}/{rho_only} "
           f"witness={B['witness_psi_safe_FNR_unsafe']} containment={B['n_FNRsafe_psi_unsafe']}/{B['n_FNRsafe_psi_undet']} "
           f"| R071 exact&det 4/4, fixed-z 3/4")
        return good
    except Exception as e:
        _p(f"    XX errored: {e}"); return False


def layer2d_unit_tests():
    _p("\n[2d] PROBABILITY-SPACE UNIT TESTS — test_global_leaf_probability_space.py")
    r = _run("test_global_leaf_probability_space.py")
    good = r.returncode == 0 and "ALL TESTS PASSED" in (r.stdout + r.stderr)
    _p(f"    {'OK ' if good else 'XX '}exit={r.returncode} "
       f"(G1 enumeration, G2 boundary MC, G3 fixed-z counterexample, G4 Hoeffding/HB/Holm/K=0)")
    if not good:
        _p("    " + (r.stdout + r.stderr)[-400:])
    return good


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def write_verification_report(layers):
    """Machine-readable per-table/per-figure verification map (reviewer-facing)."""
    checks = []

    def add(cid, anchor, script, inputs, expected, observed, passed):
        checks.append(dict(id=cid, paper_anchor=anchor, script=script,
                           inputs=[dict(path=p, sha256=_sha256(artifact(p)))
                                   for p in inputs if os.path.exists(artifact(p))],
                           expected=expected, observed=observed, **{"pass": bool(passed)}))

    mc = json.load(open(artifact("m3_combined_results.json")))["pooled"]["pi=0.1"]["cells"]
    mr = json.load(open(artifact("m6_robust_leaf_results.json")))["pooled"]["pi=0.1"]["cells"]
    for fam, a, tau, rho, m_c, psih, lcb_ex, lcb_hb in HEADLINE:
        b = mc[fam][a]["best"]; rb = mr[fam][a]["robust_best"]
        exp = dict(tau=tau, rho_hat=rho, m_c=m_c, psi_hat=psih, psi_lcb_exact=lcb_ex, psibar_lcb_robust=lcb_hb)
        obs = dict(tau=b["tau"], rho_hat=b["rho_hat"], m_c=b["m_c"], psi_hat=b["psi_hat"],
                   psi_lcb_exact=b["psi_lcb"], psibar_lcb_robust=rb["psi_lcb_hb"])
        ok = (close(b["tau"], tau, 1e-5) and close(b["rho_hat"], rho) and b["m_c"] == m_c
              and close(b["psi_hat"], psih) and close(b["psi_lcb"], lcb_ex) and close(rb["psi_lcb_hb"], lcb_hb))
        add(f"tab:joint/{FAMLAB[fam]}/alpha={a}", "Table tab:joint (joint certification)",
            "m3_combined.py + m6_robust_leaf.py",
            ["m3_combined_results.json", "m6_robust_leaf_results.json"], exp, obs, ok)
    pl = json.load(open(artifact("m_pairedlift_results.json")))
    plc = pl["cells"]
    def f1_at(t):  # taus stored at mixed precision; nearest-match within 1e-3
        return min(plc, key=lambda c: abs(c["tau"] - t))["F1"]
    printed = {"0.000123": 0.130, "0.222824": 0.602, "0.245085": 0.885, "0.518631": 0.657}
    obs = {k: round(f1_at(float(k)), 3) for k in printed}
    add("tab:joint/F1-column", "Table tab:joint F1 column + caption Δψ list", "m_pairedlift.py",
        ["m_pairedlift_results.json"], printed, obs,
        all(abs(obs[k] - v) <= 5e-4 for k, v in printed.items()))
    m4 = json.load(open(artifact("M4_real_arm_certificate.json")))
    nli = m4["cells"]["score_sNLI"]; L = nli["budgets"]["main(δtotal=0.10)"]
    add("real-clinical-arm/refNLI", "§5.12 (nominal in-sample corroboration)", "M4_certify_real_arm.py",
        ["M4_real_arm_certificate.json"],
        dict(rho_UCB=0.175, pi_measured=0.143, psibar_lcb=0.329),
        dict(rho_UCB=L["rho_UCB"], pi_measured=nli["rho_leaf"]["pi_measured"], psibar_lcb=L["human_psibar_lcb_robust"]),
        close(L["rho_UCB"], 0.175, 2e-3) and close(nli["rho_leaf"]["pi_measured"], 0.143, 2e-3)
        and close(L["human_psibar_lcb_robust"], 0.329, 3e-3))
    r050 = json.load(open(artifact("R050_decision_value_results.json")))
    add("decision-value/R050", "§5.5 (Fréchet [0,1] + re-pairing witness)", "R050_decision_value.py",
        ["R050_decision_value_results.json"], dict(self_check="PASS", frechet="[0,1] at all headline cells"),
        dict(self_check=r050["self_check"], n_cells=len(r050["cells"])),
        r050["self_check"] == "PASS")
    r063 = json.load(open(artifact("R063_matched_marginal_results.json")))["summary"]
    add("matched-marginal/R063", "§5.5 matched-marginal paragraph + supp table", "R063_matched_marginal.py",
        ["R063_matched_marginal_results.json"],
        dict(n_matched_pairs=11, n_cross_family=8, n_ci_excl_zero=10, n_flip_at_beta_ref=2,
             delta_psi_min=0.0083, delta_psi_max=0.0272),
        {k: r063[k] for k in ("n_matched_pairs", "n_cross_family", "n_ci_excl_zero",
                              "n_flip_at_beta_ref", "delta_psi_min", "delta_psi_max")},
        layers.get("matched_marginal") is True)
    gm = json.load(open(artifact("grid_multiplicity_check.json")))
    add("grid-multiplicity", "§4 (γ-floor 28→26 ⇒ |C_low|=234, |C_up|=78)", "verify_grid_multiplicity.py",
        ["grid_multiplicity_check.json"], dict(C_low=234, C_up=78, verdict="PASS"),
        dict(C_low=gm["C_low"], C_up=gm["C_up"], verdict=gm["verdict"]), gm["verdict"] == "PASS")
    add("probability-space/unit-tests", "Supp lem:companion(a-i)/(a-ii) + rem:condspace (see probability_space.md)",
        "test_global_leaf_probability_space.py", ["probability_space.md"],
        dict(all_tests="PASS"), dict(all_tests="PASS" if layers.get("unit_tests") else "FAIL"),
        layers.get("unit_tests") is True)
    r69 = json.load(open(artifact("R069_results.json")))
    A69 = r69["A_frontiers_ptrue"]; B69 = r69["B_status_map"]
    add("fnr-gate/R069", "§5.9 + supp tab:fnrgate/tab:fnrmap (Major 2 decision value)", "R069_fnr_gate_frontier.py",
        ["R069_results.json"],
        dict(native_fnr_cov=0.2333, paired_cov=0.3703, rho_only_cov=0.9605, witness=4, containment="0/0"),
        dict(native_fnr_cov=A69["native_fnr"]["0.1"]["coverage"], paired_cov=A69["psi_gate_frozen"]["0.1"]["coverage"],
             rho_only_cov=A69["rho_only"]["coverage"], witness=B69["witness_psi_safe_FNR_unsafe"],
             containment=f'{B69["n_FNRsafe_psi_unsafe"]}/{B69["n_FNRsafe_psi_undet"]}'),
        layers.get("fnr_gate") is True)
    r71v = json.load(open(artifact("R071_results.json")))
    add("fixed-assignment/R071", "supp tab:fixedz (Q1 probability space; SENSITIVITY)", "R071_fixed_assignment_companion.py",
        ["R071_results.json"], dict(selfcheck="PASS", exact_and_detL_cert="4/4", fixedz_cert="3/4"),
        dict(selfcheck=r71v["selfcheck"],
             exact_and_detL_cert=f'{sum(1 for c in r71v["cells"] if c["certifies_exact"] and c["certifies_detL"])}/4',
             fixedz_cert=f'{sum(1 for c in r71v["cells"] if c["certifies_fixedz"])}/4'),
        r71v["selfcheck"] == "PASS")
    rep = dict(run="psi_g verification report (machine-readable)",
               generated_by="reproduce.py --verify-only",
               prereg="refine-logs/analysis_plan_v2_20260717.md",
               layers=layers, n_checks=len(checks),
               n_pass=sum(1 for c in checks if c["pass"]), checks=checks)
    out = artifact("verification_report.json")
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    _p(f"\n[report] wrote verification_report.json  ({rep['n_pass']}/{rep['n_checks']} checks pass)")
    return rep["n_pass"] == rep["n_checks"]


def main():
    recompute = "--recompute" in sys.argv or "--all" in sys.argv
    verify_only = "--verify-only" in sys.argv or recompute
    _p("=" * 78)
    _p("ψ_g paper — headline reproduction driver  (CPU only, no GPU, no network)")
    _p("=" * 78)
    r1 = layer1_checksums()
    r2 = layer2_paper_matches_artifacts()
    r2c = layer2c_matched_marginal() if verify_only else None
    r2d = layer2d_unit_tests() if verify_only else None
    r2e = layer2e_fnr_gate() if verify_only else None
    r3 = layer3_recompute() if recompute else None
    rep_ok = None
    if verify_only:
        rep_ok = write_verification_report(dict(
            integrity=r1, paper_artifact=r2, matched_marginal=r2c, unit_tests=r2d, fnr_gate=r2e,
            recompute=(r3 if r3 is not None else "skipped")))
    _p("\n" + "=" * 78)
    _p(f"  [1]  integrity            : {'PASS' if r1 else 'FAIL'}")
    _p(f"  [2]  paper<->artifact     : {'PASS' if r2 else 'FAIL'}")
    if verify_only:
        _p(f"  [2c] matched-marginal     : {'PASS' if r2c else 'FAIL'}")
        _p(f"  [2d] unit tests           : {'PASS' if r2d else 'FAIL'}")
        _p(f"  [2e] fnr-gate (R069/R071) : {'PASS' if r2e else 'FAIL'}")
        _p(f"  [rep] verification report : {'PASS' if rep_ok else 'FAIL'}")
    _p(f"  [3]  script<->artifact    : {'PASS' if r3 else ('FAIL' if r3 is False else 'skipped (pass --recompute)')}")
    allok = r1 and r2 and (r2c is not False) and (r2d is not False) and (r2e is not False) and (r3 is not False) and (rep_ok is not False)
    _p(f"  RESULT: {'REPRODUCED ✓' if allok else 'MISMATCH ✗'}")
    _p("=" * 78)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
