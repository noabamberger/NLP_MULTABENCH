# REJECTED — anime popularity (`REG_TEXT_MEDIA_ANIME_POPULARITY`)

**Rejected: 0 of 5 on Delta_Awareness against a quorum of 3.**

Delta_Joint is healthy. Delta_Awareness is not merely small, it is *negative on three of the four
measured models* — a LoRA-tuned encoder does worse here than a frozen one.

## Result

`grid.csv` — 4 models x 4 states x 5 folds = 80 cells, no gaps, one T4, `ft` at 10 epochs.
TabPFN-2.5 was not run, so its cell is absent and counts as a non-pass; the quorum denominator
stays at 5.

| model | Delta_Joint | Delta_Awareness | verdict |
|---|---|---|---|
| TabM | +0.037 | -0.002 | fail |
| TabPFNv2 | +0.033 | -0.002 | fail |
| CatBoost | +0.031 | -0.001 | fail |
| LightGBM | +0.031 | 0.000 | fail |
| TabPFN-2.5 | not run | not run | non-pass |

**0 of 5. REJECTED.**

## The fold-0 screen had looked positive

This dataset was promoted to a full grid on a fold-0 Delta_Awareness screen of **+0.003 for
LightGBM and +0.002 for CatBoost**. The full grid reversed it: every model landed at or below
zero.

Anime is the cleaner-cut of the two reversals in that round — board games at least kept one model
above the line, whereas here nothing survived. Both cases are the same error: a single fold cannot
resolve a criterion whose threshold (0.001) sits far inside the per-(model, fold) noise band
(board games' sigma = 0.0063, anime's own 0.0026, all cells pooled 0.0047 — the threshold sits
inside all three). See
[`docs/findings/03-methodological-findings.md`](../../../docs/findings/03-methodological-findings.md).

## Files in this folder

| file | schema | rows | what it is |
|---|---|---|---|
| `grid.csv` | kaggle | 80 | the full T4 grid, 4 models x 4 states x 5 folds |

The fold-0 screen that promoted it lives in
`../../screening/t2_joint/screen_wave2_fold0.csv`.
