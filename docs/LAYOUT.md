# Package layout

The working tree that produced this artifact keeps every file in one directory.
The release does not: code, input data, generated results and documentation are
separate trees, so what a file is can be read off where it lives.

| Directory | Files | Contents |
|---|---|---|
| `code/` | 60 | Every script, in one directory so that the local imports (`certify.py`, `_paths.py`) resolve. |
| `docs/` | 6 | Guides, the file-level artifact manifest, and the withheld-file list. |
| (package root) | 4 | Entry README, environment pins, and the checksum manifest. |

This is the code-only release published while the manuscript is under review.
`data/`, `results/` and `adjudication/` are described below and throughout the
documentation, and are added on acceptance; the table above is what is here now.

## Running from here

Run everything from the package root:

```bash
python code/reproduce.py               # checksums + paper-vs-artifact
python code/reproduce.py --recompute    # + regenerate the headline cells
```

Scripts resolve their inputs by filename through `code/_paths.py`, which applies the
same rule this package was laid out with, so a script finds `scores_strong.csv` in
`data/scores/` without being told where it is. The few scripts that address inputs
relative to the working directory expect that directory to be the package root.

## Relation to the working tree

The released scripts differ from the working-tree originals in one respect: input and
output paths are resolved through `_paths.py` instead of being addressed next to the
script. Nothing else is changed, no computation, no constant, no seed. Layer 3 of
`code/reproduce.py` regenerates the headline certificates with these scripts from the
released scores, which is what establishes that the rewrite preserved them.

`RESULT_SHA256SUMS.txt` checksums the bytes this package ships, so the entries for the
rewritten scripts are their released hashes. Every result, score and data file is
byte-identical to the working tree, which is why a few of them still record the
directory they were produced under in a provenance field: `results/m3_results.json`,
`results/m4_results.json` and `data/scores/scores_confabst_meta.json` name their inputs
as `experiments/psi_g/...`. Those strings are history, not paths this package resolves;
editing them would break the checksum they are covered by.
