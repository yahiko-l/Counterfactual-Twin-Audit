#!/usr/bin/env python
"""#2a — download HaluEval QA and emit twins JSONL for score_generic.py.
HaluEval QA: knowledge / question / right_answer / hallucinated_answer  -> exact psi_g twin structure.
truthful = right_answer (H=0), hallucinated = hallucinated_answer (H=1), premise=knowledge (ref-NLI).
General-domain (HotpotQA-derived): a CROSS-DOMAIN stress test of the psi_g pathology."""
import os, json
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset
OUT = "data/twins/twins_halueval.jsonl"
ds = load_dataset("pminervini/HaluEval", "qa", split="data")
print(f"HaluEval qa rows={len(ds)}; cols={ds.column_names}")
n = 0
with open(OUT, "w") as f:
    for i, r in enumerate(ds):
        t = str(r["right_answer"]).strip(); h = str(r["hallucinated_answer"]).strip()
        if not t or not h or t == h:        # need a genuine distinct twin pair
            continue
        f.write(json.dumps(dict(qid=i, Q=str(r["question"]).strip(), K=str(r["knowledge"]).strip(),
                                truthful=t, halluc=h, difficulty="", category="", source="halueval_qa")) + "\n")
        n += 1
print(f"wrote {OUT}  ({n} twins)")
