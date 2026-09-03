# paper/

The Technion 097215 Track 2 write-up (Benchmark Track — curate new text-tabular datasets
that pass the MulTaBench curation pipeline). Deadline **2026-10-26**.

## Contents

- **`source/instructions.pdf`** — the assignment brief. Was untracked in the repo root; now
  committed here so it travels with the write-up it governs.
- **`report.md`** — the report **skeleton, not a draft**. Each section is a heading plus a
  note naming the evidence folder its numbers must come from. Drafting the actual prose is
  separate work, deliberately scoped out here.
- **`assets/`** — tables and figures generated from `results/curation/`, once the report is
  drafted. Nothing in this directory is hand-typed: every table and figure must be produced
  by a script or a direct read of a committed CSV, so every number in the report traces back
  to `results/curation/`.

## Deviation to disclose in any submission

E5 fine-tuning (the TAR / target-aware condition) ran **10 epochs**, not the `E5TrainArgs`
default of 50 (patience 3). This was a CPU-feasibility tradeoff. It is conservative in the
direction that matters: Delta_Awareness grew monotonically with the epoch budget in every
measurement here (e.g. LightGBM fold 0 on Udemy went +0.0099 at 2 epochs to +0.0322 at 10),
so the paper's full budget would be expected to widen the accepted margins, not narrow them.
See `docs/findings/04-environment-and-performance.md`.

One further note for the writeup: TabPFN-2.5 was unavailable (a Prior Labs licensing block,
misdiagnosed for weeks as a Hugging Face gating issue — see the same findings doc) for the
`REG_TEXT_EDU_UDEMY_ACADEMY` grid, where it is recorded as a non-pass. It is available now, so
a re-run would only strengthen that result, never weaken it.

## Where the evidence lives

- `docs/findings/01-criterion-and-pipeline.md` .. `04-environment-and-performance.md` — the
  four canonical findings documents.
- `results/curation/` — every measured grid, log and screen, indexed by
  `results/curation/INDEX.md`.
- `docs/status/STATE.md` — the live handoff, if the write-up needs the current state rather
  than a snapshot.
