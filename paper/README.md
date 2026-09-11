# paper/

The Technion 00970215 Track 2 write-up (Benchmark Track — curate new text-tabular datasets that
pass the MulTaBench curation pipeline).

## Contents

- **`main.tex`** — the submission. *Two New Text-Tabular Datasets for MulTaBench*: five sections
  followed by nine appendices carrying every gridded cell, the cross-environment reproduction,
  the two invalid shortcuts and the compute cost. It compiles on a plain TeX Live, MiKTeX or
  Overleaf install and loads no conference style file.
- **`source/instructions.pdf`** — the assignment brief, committed here so it travels with the
  write-up it governs.
- **`assets/`** — tables and figures generated from `results/curation/`. Nothing in this
  directory is hand-typed: every table and figure is produced by a script or a direct read of a
  committed CSV, so every number traces back to `results/curation/`.

## Deviation disclosed in the paper

E5 fine-tuning (the TAR / target-aware condition) ran **10 epochs**, not the `E5TrainArgs`
default of 50 (patience 3). This was a compute tradeoff. It is conservative in the direction that
matters: Delta_Awareness grew monotonically with the epoch budget in every measurement here (e.g.
LightGBM fold 0 on Udemy went +0.0099 at 2 epochs to +0.0322 at 10), so the full budget would be
expected to widen the accepted margins rather than narrow them. See
`docs/findings/04-environment-and-performance.md`.

TabPFN-2.5 was blocked for part of the project by a Prior Labs API-key requirement, misdiagnosed
for weeks as Hugging Face gating (same findings document). The blocker is resolved, and **every
accepted grid includes TabPFN-2.5 as a measured model, not an absent one.** It is one of Udemy's
three passing learners, with that dataset's largest Delta_Awareness (+0.016), and it posts the
largest Delta_Joint on Vietnam housing (+0.324). It appears in the paper as a methodological
anecdote, not as a limitation on the results.

## Where the evidence lives

- `docs/findings/01-criterion-and-pipeline.md` .. `04-environment-and-performance.md` — the four
  canonical findings documents.
- `results/curation/` — every measured grid, log and screen, indexed by
  `results/curation/INDEX.md`.
