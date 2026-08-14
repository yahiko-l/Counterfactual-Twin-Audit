# Counterfactual-Twin Audit

Reference implementation for:

> **Finite-Sample Certification of Within-Question Hallucination Risk: A Counterfactual-Twin
> Audit of Selective Medical Question Answering with Large Language Models**

The paper defines ψ_g, a paired, selection-conditioned hallucination rate: given that a gate
rejects the truthful answer to a question, how often does it accept the hallucinated twin?
A global accepted-hallucination certificate cannot bound that quantity, and on MedHallu the
same threshold is certified safe under the global criterion and certified to exceed the target
under the paired one.

## Status: code-only release, manuscript under review

This repository currently publishes the **code**. The released score files, the generated
results, and the de-identified clinician-adjudication tables are held back while the paper is
under review, and are added here in full on acceptance.

`RESULT_SHA256SUMS.txt` is published now even though the files it covers are not. It is a
commitment: it fixes, at submission time, the exact bytes of all 131 result and score files
that will be added later, so the eventual release can be checked against what was submitted.

Until then the reproduction driver cannot run, because the data it reads is not here:

```bash
python code/reproduce.py        # needs data/ and results/, added on acceptance
```

## Layout

| Path | Contents |
|---|---|
| `code/` | Every script. `certify.py` is the shared CPU certification library, `reproduce.py` is the reproduction driver, `_paths.py` maps a filename to the tree it lives in |
| `docs/README.md` | Full guide: what each reproduction layer proves, seeds, environment |
| `docs/REPRODUCE.md` | Step-by-step reproduction, including the heavy runs kept off the default path |
| `docs/ARTIFACT_MANIFEST.md` | Every result file, the script that regenerates it, and its role in the manuscript |
| `docs/LAYOUT.md` | The full package layout and how the scripts resolve their inputs |
| `docs/WITHHELD_FROM_RELEASE.txt` | The entries that stay unpublished after acceptance too, with the reason for each |
| `docs/probability_space.md` | Note on the global leaf's probability space, with unit tests |
| `RESULT_SHA256SUMS.txt` | Per-file checksums of the results and scores that follow on acceptance |

The documentation describes the complete artifact, so it refers to `data/`, `results/` and
`adjudication/` throughout. Those directories arrive with the full release.

## What arrives on acceptance

- `data/scores/`, `data/twins/`, `data/streams/`: the released detector scores, the
  counterfactual twin corpora, and the natural-stream generations the certification path reads;
- `results/`: every generated result, certificate and summary, matching the checksums above;
- `adjudication/`: de-identified per-row clinician tables, derived labels only, with a
  self-contained recompute script.

Ten entries of the checksum manifest stay unpublished even then: the MedHalu score and result
files, because that corpus is request-gated, and the clinician blind, filled and key sheets,
because they are answer-bearing and carry an annotator identifier.
`docs/WITHHELD_FROM_RELEASE.txt` gives the reason for each.

The MedHallu, HaluEval, MedQuAD, K-QA, MedicationQA, and LiveQA-Med source corpora are publicly
available from their original distributors and are not redistributed here.

## Environment

```bash
pip install -r requirements.txt   # Python 3.12; certification is CPU-only and network-free
```

Only the scoring scripts need a GPU (vLLM). `environment.lock.yml` is the full pin.

## Hygiene note

A few scripts record the absolute model and cache paths of the machine that produced the
scores. They affect no reported number.
