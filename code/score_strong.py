#!/usr/bin/env python
"""
M3 — STRONG-JUDGE rescoring of MedHallu twins (GPU 0–3). Replaces the pilot 1.5B proxy.
Reuses idea-stage/pilot/pilot_scores.py machinery (vLLM yes/no logprobs + roberta NLI).

Risk score r(Q,A): higher ⇒ more likely hallucinated; gate accepts iff r ≤ τ. Score families:
  score_sjudge    : P(Yes) "does the answer contain a hallucination?"   [STRONG LLM-judge, headline]
  score_sptrue    : 1 − P(Yes) "is this answer factually correct?"      [strong self-report / P(True)]
  score_selfcheck : mean_k P(contradiction | premise=sample_k, hyp=A)   [answer-conditioned SelfCheckGPT-NLI;
                    REPLACES the pilot's degenerate q-level semantic-entropy (AUROC 0.500)]
  score_sNLI      : P(contradiction | premise=Knowledge, hyp=A)         [reference NLI, answer-conditioned]

CRITICAL data-leakage line: the judge / P(True) / SelfCheck scores are SCORES UNDER AUDIT.
The H label (truthful vs hallucinated) comes ONLY from MedHallu and is written through unchanged
from the dataset; no score is ever used as a label. qid/H ordering reproduces the pilot exactly
(ds.select(range(N_Q)), H=0 = Ground Truth row, H=1 = Hallucinated Answer row).

NOTE: the executable body is under `if __name__ == "__main__":` — vLLM ≥0.22 (V1 engine) uses the
`spawn` start method for its engine-core process, which re-imports this module; without the guard the
child would re-run LLM() and multiprocessing aborts ("start a new process before bootstrapping").
Harmless on the old vLLM 0.11.2 too.
"""
import os, sys, json, time, math, gc, re
import numpy as np
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

if __name__ == "__main__":
    N_Q   = int(os.environ.get("N_Q", "1000"))
    K_SE  = int(os.environ.get("K_SE", "10"))
    DS_CONFIG  = os.environ.get("DS_CONFIG", "pqa_labeled")     # pqa_labeled (1k) | pqa_artificial (9k denom-expand)
    QID_OFFSET = int(os.environ.get("QID_OFFSET", "0"))         # offset to avoid qid collision when combining
    SKIP_SE    = os.environ.get("SKIP_SE", "0") == "1"          # skip expensive SE sampling + SelfCheck (fast pass)
    GEN_GPU = os.environ.get("GEN_GPU", "0"); NLI_GPU = os.environ.get("NLI_GPU", "1")
    GEN_GPUS = os.environ.get("GEN_GPUS", GEN_GPU)             # comma list for tensor-parallel (e.g. "0,1")
    TP = len([g for g in GEN_GPUS.split(",") if g != ""])      # tensor_parallel_size = #generation GPUs
    MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    NLI_MODEL = os.environ.get("NLI_MODEL", "roberta-large-mnli")
    OUT = os.environ.get("OUT", "data/scores/scores_strong.csv")
    SEED = int(os.environ.get("SEED", "20260609"))
    # ---- new-judge-family knobs (all OPT-IN; defaults reproduce the original 5-judge runs verbatim) ----
    TRUST = os.environ.get("TRUST_REMOTE_CODE", "0") == "1"      # bailing_moe_v2 (AntAngelMed) custom code
    QUANT = (os.environ.get("QUANT", "").strip() or None)        # e.g. "fp8" -> fit Baichuan-M3-235B on 4xH100
    MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
    NO_SYSTEM = os.environ.get("NO_SYSTEM", "0") == "1"          # force-fold system into the user turn (Gemma-3)
    JUDGE_MODE = os.environ.get("JUDGE_MODE", "logprob").strip().lower()  # logprob (default) | freegen (reasoning judges)
    JUDGE_MAXTOK = int(os.environ.get("JUDGE_MAXTOK", "512"))    # freegen budget for the reason-then-verdict pass
    SHARE_NLI = os.environ.get("SHARE_NLI", "0") == "1"          # TP fills all visible GPUs -> NLI on cuda:0 post-vLLM
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["CUDA_VISIBLE_DEVICES"] = GEN_GPUS if SHARE_NLI else f"{GEN_GPUS},{NLI_GPU}"

    from datasets import load_dataset
    log(f"loading MedHallu {DS_CONFIG} (N_Q={N_Q}); model={MODEL}; K_SE={K_SE}; SKIP_SE={SKIP_SE}")
    ds = load_dataset("UTAustin-AIHealth/MedHallu", DS_CONFIG, split="train")
    if N_Q < len(ds): ds = ds.select(range(N_Q))
    def as_text(x): return "\n".join(str(t) for t in x) if isinstance(x, list) else str(x)
    rows = []
    for qi, r in enumerate(ds):
        Q = str(r["Question"]).strip(); K = as_text(r.get("Knowledge", "")).strip()
        diff = str(r.get("Difficulty Level", "")).strip().lower(); cat = str(r.get("Category of Hallucination", "")).strip()
        q = qi + QID_OFFSET
        rows.append(dict(idx=len(rows), qid=q, H=0, difficulty=diff, category="",  source=DS_CONFIG, answer=str(r["Ground Truth"]).strip(), Q=Q, K=K))
        rows.append(dict(idx=len(rows), qid=q, H=1, difficulty=diff, category=cat, source=DS_CONFIG, answer=str(r["Hallucinated Answer"]).strip(), Q=Q, K=K))
    log(f"  built {len(rows)} (Q,A) examples ({sum(x['H'] for x in rows)} hallucinated)")
    qid2Q = {}; [qid2Q.setdefault(x["qid"], x["Q"]) for x in rows]; uq = sorted(qid2Q)
    scorecols = {}

    # ---- PHASE A: ALL vLLM generation FIRST (keep parent CUDA-clean so the v1 engine-core spawns).
    # Loading any torch model onto CUDA before vLLM() initializes CUDA in the parent and breaks spawn.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"loading vLLM {MODEL} on physical GPU(s) {GEN_GPUS} (TP={TP})")
    gtok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=TRUST)
    # qwen3_5 (Qwen3.5/3.6) on Hopper(SM90) defaults to the FlashInfer GDN-prefill kernel, whose
    # first-run JIT can HANG here; GDN_PREFILL_BACKEND=triton forces the stable Triton GDN path
    # (vllm _resolve_gdn_prefill_backend). Omitted → no additional_config → no-op for every other
    # judge (Llama / Qwen2.5 / Qwen3-dense).
    _gdn = os.environ.get("GDN_PREFILL_BACKEND", "").strip()
    _extra = {"additional_config": {"gdn_prefill_backend": _gdn}} if _gdn else {}
    if QUANT: _extra["quantization"] = QUANT
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
              max_model_len=MAX_MODEL_LEN, tensor_parallel_size=TP, enforce_eager=(os.environ.get("EAGER", "0") == "1"),
              trust_remote_code=TRUST, disable_log_stats=True, seed=SEED, **_extra)
    LLAMA3_CHAT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
                   "<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
                   "<|start_header_id|>assistant<|end_header_id|>\n\n")
    # enable_thinking=False suppresses Qwen3's <think> block; reasoning_effort (gpt-oss harmony) caps the
    # analysis channel so the model reaches a Yes/No verdict. Both are passed as template kwargs and are
    # harmlessly ignored by templates that don't read them (Llama-3, Qwen2.5, Gemma-3).
    _tmpl_kw = {"enable_thinking": False}
    if os.environ.get("REASONING_EFFORT", "").strip():
        _tmpl_kw["reasoning_effort"] = os.environ["REASONING_EFFORT"].strip()
    def _apply(msgs):
        try:
            return gtok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **_tmpl_kw)
        except TypeError:
            return gtok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    _warned_sys = []
    def render(sysmsg, user):
        # OpenBioLLM / some base-tuned medical models ship NO chat_template -> fall back to the canonical
        # Llama-3 instruct format (same special tokens, identical string to the general judge's template).
        if getattr(gtok, "chat_template", None):
            if not NO_SYSTEM:
                try:
                    return _apply([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}])
                except Exception as e:  # template rejects a system role (e.g. Gemma-3) -> fold into user turn
                    if not _warned_sys:
                        log(f"  [render] system role rejected by template ({type(e).__name__}: {str(e)[:120]}); folding into user turn")
                        _warned_sys.append(1)
            return _apply([{"role": "user", "content": (sysmsg + "\n\n" + user) if sysmsg else user}])
        return LLAMA3_CHAT.format(s=sysmsg, u=user)
    _EOT = gtok.convert_tokens_to_ids("<|eot_id|>")
    # Use <|eot_id|> as an extra stop ONLY if it genuinely round-trips (Llama family). Qwen3 etc. lack it
    # (turn ends on <|im_end|>) and stop correctly on their own config eos, so STOP_IDS stays None there.
    STOP_IDS = [_EOT] if isinstance(_EOT, int) and _EOT >= 0 and gtok.convert_ids_to_tokens(_EOT) == "<|eot_id|>" else None
    _MARK = re.compile(r'(?:answer|final answer|verdict|conclusion)\s*[:：]\s*\*{0,2}\s*(yes|no)\b', re.I)
    _BARE = re.compile(r'^\*{0,2}\s*(yes|no)\b', re.I)
    def _parse_verdict(txt):  # TEXT fallback only (used if the forced one-word pass yields neither yes nor no)
        m = list(_MARK.finditer(txt))
        if m: return 1.0 if m[-1].group(1).lower() == "yes" else 0.0
        lines = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
        if lines:
            bm = _BARE.match(lines[-1])
            if bm: return 1.0 if bm.group(1).lower() == "yes" else 0.0
        return 0.5
    def _yn_from_lp(o):  # P(yes) from first-token top-k logprobs; None if neither yes nor no present
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        y = n = -1e9
        for tid, info in lp.items():
            t = info.decoded_token.strip().lower()
            if t in ("yes", "y", "true"): y = max(y, info.logprob)
            elif t in ("no", "n", "false"): n = max(n, info.logprob)
        return None if (y == -1e9 and n == -1e9) else math.exp(y) / (math.exp(y) + math.exp(n))
    def yesno(prompts):
        if JUDGE_MODE == "freegen":
            # REASON-then-EXTRACT: reasoning judges (HuatuoGPT-o1, Baichuan-M2/M3, AntAngel) don't emit Yes/No
            # as the first token and conclude in varied styles/LANGUAGES (AntAngel reasons in Chinese). Let them
            # reason freely, then APPEND a forced one-word lead-in and read the Yes/No logprob -> robust +
            # continuous, language-agnostic. (Pure text-parsing the verdict gave AntAngel only ~7% parse.)
            r1 = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=JUDGE_MAXTOK, stop_token_ids=STOP_IDS))
            txts = [o.outputs[0].text for o in r1]
            p2 = [prompts[i] + txts[i] + "\n\nBased on the reasoning above, the final answer (one word, Yes or No) is:"
                  for i in range(len(prompts))]
            r2 = llm.generate(p2, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
            res = []; nfb = 0
            for i in range(len(prompts)):
                v = _yn_from_lp(r2[i])
                if v is None: v = _parse_verdict(txts[i]); nfb += 1
                res.append(v)
            log(f"  freegen reason->verdict: {len(res)-nfb}/{len(res)} via forced-logprob, {nfb} text-fallback")
            return res
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)); res = []
        for o in outs:
            v = _yn_from_lp(o); res.append(0.5 if v is None else v)
        return res
    SYSJ = "You are a meticulous medical fact-checker."
    ASK = ("Answer exactly one word: Yes or No." if JUDGE_MODE != "freegen"
           else "Reason briefly, then end your reply with a final line exactly 'ANSWER: Yes' or 'ANSWER: No'.")
    jp = [render(SYSJ, f"Question: {x['Q']}\n\nReference knowledge:\n{x['K'][:3000]}\n\nProposed answer: {x['answer']}\n\nDoes the proposed answer contain a hallucination (a statement medically incorrect or not supported by the reference)? {ASK}") for x in rows]
    log("scoring: STRONG LLM-judge P(Yes hallucination)"); scorecols["score_sjudge"] = yesno(jp)
    pp = [render(SYSJ, f"Question: {x['Q']}\n\nAnswer: {x['answer']}\n\nIs this answer factually correct? {ASK}") for x in rows]
    log("scoring: strong P(True) -> risk=1-P(yes)"); scorecols["score_sptrue"] = [1.0 - p for p in yesno(pp)]
    qid2samples = {}
    if not SKIP_SE:
        log(f"sampling {K_SE} stochastic answers/question for SelfCheckGPT ({len(uq)} q)")
        SYSA = "You are a helpful medical assistant. Answer concisely."
        sep = [render(SYSA, f"Answer this medical question in 1-3 sentences.\n\nQuestion: {qid2Q[q]}") for q in uq]
        seo = llm.generate(sep, SamplingParams(temperature=1.0, top_p=0.95, max_tokens=128, n=K_SE, seed=SEED, stop_token_ids=STOP_IDS))
        qid2samples = {q: [c.text.strip() for c in seo[i].outputs if c.text.strip()] for i, q in enumerate(uq)}
    else:
        log("SKIP_SE=1 -> skipping SelfCheck sampling (judge+ptrue+refNLI only)")
    del llm; gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- PHASE B: now load NLI (CUDA in parent is fine; vLLM workers already done) and score.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer as HFTok
    # SHARE_NLI (TP fills every GPU, e.g. 235B at TP=8) -> NLI on CPU: vLLM's spawned TP workers do NOT
    # free GPU memory synchronously on `del llm`, so loading NLI on a gen GPU OOMs. Otherwise NLI goes to
    # a DEDICATED GPU (NLI_GPU, never a gen GPU) so the lingering workers are irrelevant.
    nli_dev = "cpu" if SHARE_NLI else (f"cuda:{TP}" if torch.cuda.device_count() > TP else "cuda:0")
    ntok = HFTok.from_pretrained(NLI_MODEL)
    try:
        log(f"loading NLI {NLI_MODEL} on {nli_dev}")
        _ndt = torch.float32 if nli_dev == "cpu" else torch.float16
        nmod = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=_ndt).to(nli_dev).eval()
    except Exception as e:  # any GPU OOM/alloc failure (incl. torch.OutOfMemoryError) -> robust CPU fp32
        log(f"  NLI load on {nli_dev} failed ({type(e).__name__}); falling back to CPU fp32"); nli_dev = "cpu"
        nmod = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=torch.float32).to("cpu").eval()
    id2lab = {int(k): v.lower() for k, v in nmod.config.id2label.items()}; lab2id = {v: k for k, v in id2lab.items()}
    CON = lab2id.get("contradiction"); ENT = lab2id.get("entailment")
    log(f"  NLI labels={id2lab} CON={CON} ENT={ENT}")
    @torch.no_grad()
    def nli_probs(prem, hyp, bs=128, max_len=512):
        out = []
        for i in range(0, len(prem), bs):
            enc = ntok(prem[i:i+bs], hyp[i:i+bs], truncation="only_first", max_length=max_len, padding=True, return_tensors="pt").to(nli_dev)
            out.append(torch.softmax(nmod(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out, 0)
    # reference NLI (answer-conditioned, premise=Knowledge)
    log("scoring: reference NLI contradiction (premise=Knowledge, hyp=Answer)")
    scorecols["score_sNLI"] = nli_probs([x["K"][:2000] for x in rows], [x["answer"] for x in rows])[:, CON].tolist()
    # answer-conditioned SelfCheckGPT-NLI = mean_k P(contradiction | sample_k -> answer); answer-specific
    if not SKIP_SE:
        log("scoring: answer-conditioned SelfCheckGPT-NLI = mean_k P(contradiction | sample_k -> answer)")
        prem, hyp, owner = [], [], []
        for xi, x in enumerate(rows):
            for s in qid2samples.get(x["qid"], []):
                prem.append(s[:2000]); hyp.append(x["answer"]); owner.append(xi)
        con = nli_probs(prem, hyp)[:, CON] if prem else np.zeros(0)
        from collections import defaultdict
        agg = defaultdict(list)
        for o, c in zip(owner, con): agg[o].append(float(c))
        sc = np.full(len(rows), np.nan)
        for xi in range(len(rows)):
            sc[xi] = float(np.mean(agg[xi])) if agg[xi] else 0.5     # no samples -> uninformative 0.5
        scorecols["score_selfcheck"] = sc.tolist()

    # ----------------------------------------------------------------------------- save
    import csv
    names = [c for c in ["score_sjudge", "score_sptrue", "score_selfcheck", "score_sNLI"] if c in scorecols]
    log(f"writing {OUT} with {names}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "qid", "H", "difficulty", "category", "source"] + names)
        for i, x in enumerate(rows):
            w.writerow([x["idx"], x["qid"], x["H"], x["difficulty"], x["category"], x["source"]] + [f"{scorecols[c][i]:.6f}" for c in names])
    from sklearn.metrics import roc_auc_score
    H = np.array([x["H"] for x in rows]); summ = {}
    for c in names:
        s = np.array(scorecols[c])
        try: a = roc_auc_score(H, s)
        except Exception: a = float("nan")
        summ[c] = dict(auroc=round(float(a), 4), mean_H1=round(float(np.nanmean(s[H == 1])), 4), mean_H0=round(float(np.nanmean(s[H == 0])), 4))
    log("AUROC per strong score (higher should => hallucinated):")
    for k, v in summ.items(): log(f"   {k:15s} {v}")
    # (metadata side-file scores_strong_meta.json intentionally NOT written: it is not consumed by the
    #  certification pipeline and was removed from the release; the AUROC summary above is logged instead.)
    log(f"DONE in {time.time()-T0:.1f}s -> {OUT}")
