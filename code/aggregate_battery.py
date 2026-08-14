#!/usr/bin/env python
"""
Aggregate the FULL ψ_g judge-robustness battery into one table:
  5 existing judges (general-8B / OpenBio-8B / DeepSeek-V4 / Qwen3-32B / Qwen3.6-27B)
  + 7 new DEPLOYED MEDICAL LLMs (R057+).
Normalizes the 3 result-JSON schemas (general=m3_combined, medical=m_medical, this=m_judge) and
prints a markdown table sorted by paired-acc + a JSON summary for the figure. Missing judges skipped.
"""
import os, json, sys
from _paths import artifact                                # released layout, docs/LAYOUT.md

HERE = os.path.dirname(os.path.abspath(__file__))          # <package>/code
ROOT = os.path.dirname(HERE)                               # <package>
PIS = ["pi=0.5", "pi=0.2", "pi=0.1", "pi=0.05"]

# tag -> (display, family, params, deployment, is_medical, json, schema)
REG = [
    ("general_8b",       "Llama-3-8B",        "Llama",      "8B",        "local", False, "m3_combined_results.json", "general"),
    ("openbio_8b",       "OpenBioLLM-8B",     "Llama",      "8B",        "local", True,  "m_medical_results.json",   "medical"),
    ("deepseek_v4",      "DeepSeek-V4-Flash", "DeepSeek",   "284B-A13B", "API",   False, "m_deepseek_v4_results.json", "this"),
    ("qwen3_32b",        "Qwen3-32B",         "Qwen",       "32B",       "local", False, "m_qwen3_32b_results.json", "this"),
    ("qwen36_27b",       "Qwen3.6-27B",       "Qwen",       "27B",       "local", False, "m_qwen36_27b_results.json", "this"),
    # ---- new deployed MEDICAL judges (R057+) ----
    ("medgpt_oss_20b",   "MedGPT-oss-20B",    "GPT-OSS",    "20B-MoE",   "local", True,  "m_medgpt_oss_20b_results.json", "this"),
    ("medgemma_27b",     "MedGemma-27B",      "Gemma-3",    "27B",       "local", True,  "m_medgemma_27b_results.json", "this"),
    ("baichuan_m2_32b",  "Baichuan-M2-32B",   "Qwen2",      "32B",       "local", True,  "m_baichuan_m2_32b_results.json", "this"),
    ("openbio_70b",      "OpenBioLLM-70B",    "Llama",      "70B",       "local", True,  "m_openbio_70b_results.json", "this"),
    ("huatuo_o1_72b",    "HuatuoGPT-o1-72B",  "Qwen2",      "72B",       "local", True,  "m_huatuo_o1_72b_results.json", "this"),
    ("antangel_103b",    "AntAngelMed-103B",  "Ling/Bailing", "103B-MoE", "local", True, "m_antangel_103b_results.json", "this"),
    ("baichuan_m3_235b", "Baichuan-M3-235B",  "Qwen3-MoE",  "235B-MoE",  "local", True,  "m_baichuan_m3_235b_results.json", "this"),
]


def extract(path, schema):
    d = json.load(open(path))
    if schema == "general":
        pa = d["K4_combined_judge"]["paired_acc"]; au = d["K4_combined_judge"].get("risk_auc")
        cells = {pi: d["pooled"][pi]["joint_cells"] for pi in PIS}
    elif schema == "medical":
        pa = d["medical_judge"]["paired_acc"]; au = d["medical_judge"].get("risk_auc")
        cells = {pi: d["joint_by_pi"][pi]["n"] for pi in PIS}
    else:  # this (m_judge)
        pa = d["this_judge"]["paired_acc"]; au = d["this_judge"].get("risk_auc")
        cells = {pi: d["joint_by_pi"][pi]["n"] for pi in PIS}
    orient = d.get("orientation")  # only present on new m_judge runs
    return pa, au, cells, orient


def main():
    rows = []
    for tag, disp, fam, params, dep, med, jf, schema in REG:
        p = artifact(jf)
        if not os.path.exists(p):
            print(f"  [skip] {disp}: {jf} not found yet", file=sys.stderr); continue
        try:
            pa, au, cells, orient = extract(p, schema)
        except Exception as e:
            print(f"  [skip] {disp}: parse error {e}", file=sys.stderr); continue
        rows.append(dict(tag=tag, disp=disp, fam=fam, params=params, dep=dep, med=med,
                         paired_acc=pa, risk_auc=au, cells=cells, orient=orient))
    rows.sort(key=lambda r: (r["paired_acc"] if r["paired_acc"] is not None else -1))

    # ---- markdown table ----
    out = []
    out.append("| Judge | Family | Params | Med? | risk-AUROC | paired-acc (K4) | π=.50 | π=.20 | π=.10 | π=.05 |")
    out.append("|---|---|---|:---:|---:|---:|:---:|:---:|:---:|:---:|")
    for r in rows:
        c = r["cells"]
        out.append(f"| {r['disp']} | {r['fam']} | {r['params']} | {'✅' if r['med'] else '—'} | "
                   f"{r['risk_auc']} | {r['paired_acc']} | {c['pi=0.5']} | {c['pi=0.2']} | {c['pi=0.1']} | {c['pi=0.05']} |")
    table = "\n".join(out)
    print(table)

    nmed = sum(r["med"] for r in rows)
    span = (min(r["paired_acc"] for r in rows), max(r["paired_acc"] for r in rows)) if rows else (None, None)
    holds10 = all(r["cells"]["pi=0.1"] > 0 for r in rows) if rows else False
    neg50 = all(r["cells"]["pi=0.5"] == 0 for r in rows) if rows else False
    summary = {"n_judges": len(rows), "n_medical": nmed, "paired_acc_span": span,
               "C1_holds_all_at_pi=0.10": holds10, "honest_negative_all_at_pi=0.50": neg50,
               "rows": rows}
    json.dump(summary, open(artifact("battery_summary.json"), "w"), indent=2)
    print(f"\n# judges={len(rows)} ({nmed} medical) | paired-acc span {span[0]}–{span[1]} | "
          f"C1 holds@π=.10 for all: {holds10} | honest negative@π=.50 for all: {neg50}")
    print(f"# wrote battery_summary.json")


if __name__ == "__main__":
    main()
