# Counterfactual-Twin Audit: artifact package

> **Code-only release: the manuscript is under review.** The score files, the generated results
> and the de-identified clinician tables this document refers to are added on acceptance, so
> `data/`, `results/` and `adjudication/` are not in this repository yet. `RESULT_SHA256SUMS.txt`
> commits now to the bytes that will arrive. Editors and reviewers hold the complete artifact.

Code, released results, and certification scripts for **"Finite-Sample Certification of
Within-Question Hallucination Risk: A Counterfactual-Twin Audit of Selective Medical Question
Answering with Large Language Models."**

A selective question-answering gate is normally trusted once its *global* accepted-hallucination
rate is certified below a target. This package certifies a different quantity, the paired,
selection-conditioned rate psi_g: given that the gate **rejects** the truthful answer to a
question, how often does it **accept** the hallucinated twin of that same question? The
certification path runs on CPU from released score files under declared seeds, and
`reproduce.py` checks the headline results against a per-file SHA256 manifest.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt   # Python 3.12
python code/reproduce.py               # layers 1+2: checksums + paper-vs-artifact (seconds, read-only)
python code/reproduce.py --verify-only # + unit tests, writes verification_report.json
python code/reproduce.py --recompute   # + layer 3: regenerate from the released scores (CPU-minutes)
```

Exit code `0` iff every checked value matches. Certification is CPU-only and network-free;
only the scoring scripts need a GPU (vLLM).

## What `reproduce.py` proves

| Layer | Claim it checks | How |
|---|---|---|
| 1. Integrity | Every published result and score file is unmodified | `sha256sum -c RESULT_SHA256SUMS.txt` (121 published entries; 10 withheld, see below) |
| 2. Paper vs artifact | The headline joint-certification table, the real-clinical arm, the decision-value result, the faithful-baseline cells, and the grid-multiplicity note equal the released JSONs to displayed precision | direct comparison against the paper's printed values |
| 3. Script vs artifact | Those released numbers are what the code actually produces | re-runs the headline certification path from the released scores and asserts the regenerated values match |

`reproduce.py` covers the headline certification chain. The heavier analyses (the 4000-reseed
grid, the mitigation frontier, the second-benchmark reproductions) carry their own self-checks
and are run separately; `docs/REPRODUCE.md` says how.

`--verify-only` additionally writes `verification_report.json`, mapping every checked manuscript
table and figure element to its input files (with SHA256), generating script, expected value,
observed value, and pass flag. Layer 3 is non-destructive with respect to the release: every
checksummed file it regenerates is backed up and restored byte for byte, so the checksum set stays
intact. `verification_report.json` is the one file it rewrites in place, and it is not checksummed.

## Layout

The full directory map is `docs/LAYOUT.md`. The entry points:

| Path | Contents |
|---|---|
| `docs/ARTIFACT_MANIFEST.md` | File-level map: every released result file, the script that regenerates it, and its role in the manuscript |
| `docs/REPRODUCE.md` | Step-by-step reproduction guide, including the heavy runs kept out of the default path |
| `RESULT_SHA256SUMS.txt` | Per-file checksums over every result the analysis produced |
| `docs/WITHHELD_FROM_RELEASE.txt` | The 10 manifest entries that are not redistributable, with the reason for each; `reproduce.py` skips them |
| `adjudication/` | De-identified per-row clinician-adjudication tables plus a self-contained `recompute_adjudication.py` |
| `docs/probability_space.md` | Note on the global leaf's probability space, with unit tests |
| `requirements.txt`, `environment.lock.yml` | Environment pins |

## Clinician adjudication data

`adjudication/` reproduces the headline clinician-derived counts and carries **derived
labels only**: no question or answer text, no risk scores, and no unblinding key. Row identifiers
are non-invertible HMAC codes. It covers the accepted-set audit, the MedHallu headline human
relabel, the real-clinical paired leaf, and the MedHalu clinical reproduction with its
confirmed-failure harm grades.

The MedHallu, HaluEval, MedQuAD, K-QA, MedicationQA, and LiveQA-Med source corpora are publicly
available from their original distributors and are not redistributed here.

## What is withheld, and why it does not affect reproduction

`RESULT_SHA256SUMS.txt` is an integrity manifest over every result the analysis produced, which
includes ten files that cannot be published: the MedHalu score and result files, because that
corpus is request-gated and its raw data must not be redistributed, and the B4 clinician blind,
filled and key sheets, because they are answer-bearing and carry an `annotator_id` column. The
latter are available from the authors under a data-use agreement; their de-identified, key-free
reproduction is `adjudication/`, above.

`docs/WITHHELD_FROM_RELEASE.txt` lists all ten with the reason for each, and `reproduce.py` skips
them at layer 1 rather than reporting them as corrupted. No layer reads them: every headline
number is reproduced from the published files alone.

## Seeds and environment

Calibration split `7`, prevalence draw `2024`, score generation `20260609`. Headline scores were
generated under vLLM 0.11.2; `environment.lock.yml` pins the environment the optional raw-score
regeneration path expects (vLLM 0.22.1, Python 3.12, PyTorch 2.11.0, transformers 4.57.1,
NumPy 1.26.4, SciPy 1.16.3).

## Known artifact-hygiene notes

Three optional helper functions sit off the headline code path; the cluster-valid global rate is
computed inline. Some scripts and two score-metadata files record the absolute model and cache
paths of the machine that produced them. These affect no reported number, and the metadata files
are left byte-identical so the checksum manifest stays valid.
