#!/usr/bin/env python
"""#2b — construct a MEDICAL counterfactual-twin set from MedQuAD (NIH/NLM consumer-health QA).
MedQuAD gives (question, reference answer) but NO hallucinated twin, so we GENERATE one per question
(mirroring MedHallu's construction): given Q + the correct answer, an LLM writes a plausible-but-
medically-incorrect answer of similar length/style. truthful = concise reference extract (H=0);
halluc = generated wrong answer (H=1); K (ref-NLI premise + judge context) = full reference answer.

This is a SEMI-external medical test: the source QA + reference answers are external (NIH), the
hallucinations are ours. Caveat (documented): K contains the truthful answer's source, so ref-NLI is a
near-oracle detector here — which only makes the psi_g failure HARDER to certify, not easier.

Generation on GEN_GPUS (default '2,3', parallel to the HaluEval scoring on 0/1). vLLM spawn-guarded.
Writes twins_medquad.jsonl for score_generic.py. Env: N_Q (target #questions), GEN_GPUS, GEN_MODEL."""
import os, sys, json, re, time
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

if __name__ == "__main__":
    N_Q = int(os.environ.get("N_Q", "5000"))
    GEN_GPUS = os.environ.get("GEN_GPUS", "2,3"); TP = len([g for g in GEN_GPUS.split(",") if g != ""])
    MODEL = os.environ.get("GEN_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    OUT = os.environ.get("OUT", "data/twins/twins_medquad.jsonl")
    SEED = int(os.environ.get("SEED", "20260619"))
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ["CUDA_VISIBLE_DEVICES"] = GEN_GPUS
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    from datasets import load_dataset
    log(f"loading MedQuAD (lavita/MedQuAD); target N_Q={N_Q}")
    ds = load_dataset("lavita/MedQuAD", split="train")

    def first_sents(txt, k=3, cap=480):
        s = re.split(r'(?<=[.!?])\s+', txt.strip()); out = " ".join(s[:k]).strip()
        return out[:cap].rsplit(" ", 1)[0] if len(out) > cap else out

    # filter: non-empty Q + answer; concise truthful extract in [40,480] chars; dedupe by question
    seen = set(); cand = []
    for r in ds:
        Q = (r.get("question") or "").strip(); A = (r.get("answer") or "").strip()
        if not Q or not A or len(A) < 40: continue
        key = Q.lower()
        if key in seen: continue
        tr = first_sents(A)
        if len(tr) < 40: continue
        seen.add(key)
        cand.append(dict(Q=Q, truthful=tr, K=A[:2000], qtype=str(r.get("question_type", "")),
                         focus=str(r.get("question_focus", "")), sem=str(r.get("umls_semantic_group", ""))))
    log(f"  {len(cand)} unique usable MedQuAD QA after filtering")
    rng = np.random.default_rng(SEED); rng.shuffle(cand)
    cand = cand[:N_Q]
    log(f"  sampled {len(cand)} questions for twin construction")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"loading vLLM {MODEL} on GPU(s) {GEN_GPUS} (TP={TP})")
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
              max_model_len=2048, tensor_parallel_size=TP, seed=SEED, disable_log_stats=True)
    SYS = "You are helping build a benchmark for medical hallucination detection."
    STYLE = os.environ.get("HALLUC_STYLE", "subtle").strip().lower()   # subtle (default) | blatant
    _SUBTLE = ("Write a SUBTLY INCORRECT version of the answer. Keep MOST of the answer accurate and copy its "
               "structure and length, but introduce ONE small, hard-to-detect medical error: change a single "
               "dose/number, swap one gene/drug/organism/anatomical name for a closely related one, reverse one "
               "direction or contraindication, or alter one mechanism detail. It must read as fluent and "
               "authoritative so the error is hard to spot without domain expertise.")
    _BLATANT = ("Write an INCORRECT answer to the same question. It must sound clinically plausible and be of "
                "similar length and style, but contain a clear medical error (a wrong fact, mechanism, dosage, "
                "or recommendation) that makes it a hallucination.")
    def prompt(x):
        u = (f"Question: {x['Q']}\n\nA correct answer is:\n{x['truthful']}\n\n"
             + (_SUBTLE if STYLE == "subtle" else _BLATANT)
             + " Do NOT say it is incorrect, do NOT add disclaimers, and do NOT restate the correct answer. "
               "Output ONLY the incorrect answer.")
        return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                                       add_generation_prompt=True, tokenize=False)
    prompts = [prompt(x) for x in cand]
    log(f"generating {len(prompts)} hallucinated twins (temp=0.7)")
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256, seed=SEED))
    REFUSE = re.compile(r"\b(i cannot|i can't|i'm sorry|as an ai|cannot fulfill|i am unable|i won't)\b", re.I)
    n = 0
    with open(OUT, "w") as f:
        for i, (x, o) in enumerate(zip(cand, outs)):
            h = o.outputs[0].text.strip().strip('"').strip()
            h = re.sub(r"^(incorrect answer|answer)\s*[:：]\s*", "", h, flags=re.I).strip()
            if len(h) < 20 or h.lower() == x["truthful"].lower() or REFUSE.search(h): continue
            f.write(json.dumps(dict(qid=i, Q=x["Q"], K=x["K"], truthful=x["truthful"], halluc=h,
                                    difficulty="", category=x["sem"], source="medquad")) + "\n")
            n += 1
    log(f"wrote {OUT}  ({n}/{len(cand)} valid twins; {len(cand)-n} dropped as refusal/degenerate)")
