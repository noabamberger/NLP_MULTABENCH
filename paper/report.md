# [Report title]

**Skeleton -- not a draft.** Each section below is a heading plus a note naming the evidence
folder its numbers and claims must come from. No prose, no findings, no tables are written
here yet; see `paper/README.md` for why that is deliberate. When drafting, every number that
appears in a section must trace to a file inside its named evidence folder -- do not
transcribe from memory or from this skeleton's notes.

## 1. Introduction and track

Technion NLP course 097215, Track 2 (Benchmark Track): curate new text-tabular dataset(s)
that pass the MulTaBench curation criterion, rather than proposing a new algorithm.

Evidence: `paper/source/instructions.pdf`.

## 2. The curation criterion

What "passes" means, the four conditions, the CLI-flag-to-paper-condition mapping, and why
this project's numbers can be trusted (validation against the paper's own shipped artifacts
and anchor dataset).

Evidence: `docs/findings/01-criterion-and-pipeline.md`.

## 3. Method: an automated mining pipeline

The T0 -> T3 funnel (discovery, typing profile, Delta_Joint screen, adversarial validation,
full grid, TAR), its measured yield at each stage, and the rules learned along the way.

Evidence: `docs/findings/02-mining-method-rules.md`, `results/curation/screening/`.

## 4. Result 1: `REG_TEXT_EDU_UDEMY_ACADEMY`

Evidence: `results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/`.

## 5. Result 2: `REG_TEXT_HOUSES_VIETNAM_2024`

Evidence: `results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/`.

## 6. In progress: `REG_TEXT_GAMES_MTG_CARD_PRICES`

Evidence: `results/curation/in_progress/`.

## 7. Negative results and what they taught

Board games, anime, metacritic, and the screen-time rejections that never reached a full
grid.

Evidence: `results/curation/rejected/`, `docs/findings/03-methodological-findings.md`.

## 8. Methodological contribution

The cheap-screen principle -- a cheap screen is only valid where cheapness does not change
the quantity being screened -- and its two measured instances (the TAR epoch budget, and the
TAR fold count).

Evidence: `docs/findings/03-methodological-findings.md`.

## 9. Reproducibility and deviations

Environment constraints, performance engineering, and the runner-fidelity checks against the
paper's own anchor dataset.

Evidence: `docs/findings/04-environment-and-performance.md`, `results/curation/validation/`.

## 10. Limitations

- E5 fine-tuning ran 10 epochs, not the `E5TrainArgs` default of 50.
- Two float knife-edge cells exist where a `>` comparison is decided by float64
  representation rather than by evidence (Vietnam housing's TabPFNv2, board games' TabM) --
  neither verdict depends on the affected cell.
