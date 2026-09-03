# REJECTED — Metacritic PC games (`REG_TEXT_MEDIA_METACRITIC_SCORED`)

**Rejected at the target, before any verdict was worth computing: the `Metacritic` target is
82% sentinel zeros, so the "regression" is mostly a has-a-score indicator.**

## The sentinel target

The source file (`thedevastator/get-your-game-on-metacritic-recommendations-and`) holds 12,624
rows, of which **10,357 carry `Metacritic == 0`** — not a score of zero but a marker for "never
reviewed". Regressing on the raw column would mean fitting a bimodal mixture of a real 20-96
score band and a spike at zero, and R^2 would mostly be measuring whether a title was reviewed at
all, not how well it scored.

`derived_input.csv` is the **repaired** file the grid was built from: the sentinel rows are
dropped, leaving **2,267 scored rows** (`Metacritic` 20-96, 71 distinct, \|z\|max 4.67). It is
kept here because the grid cannot be reproduced without it — the repair happens in
`curation_lab/screen/media_specs.py::metacritic_repaired`, not in the raw Kaggle file.

## Why the repaired dataset was still abandoned

Even repaired, the candidate is weak:

- Only one text column survives (`ResponseName`, the game's title — a proper noun, not free text),
  against 15 structured columns.
- The structured block is dominated by `RecommendationCount` (feature importance 0.62) and
  `PriceInitial` (0.16); the remainder are genre one-hots.
- Its fold-0 Delta_Awareness screen was **-0.014 for LightGBM and -0.006 for CatBoost** — the
  wrong sign on both models. (Computed from `screening/t2_joint/screen4_fold0.csv`, dataset
  `REG_TEXT_GAMES_METACRITIC_PC`.)

The frozen grid in `grid.csv` was therefore stopped part-way rather than completed: **52 rows, 3
frozen states and no `ft` state at all.** All five models have the full 5 folds of `no_text` and
`text_only`; what is missing is the `all` state, which only LightGBM and CatBoost reach and only
for a single fold each. Delta_Joint is therefore a one-fold quantity here, and what it measured is
unpersuasive on its own terms — +0.040 for LightGBM and **-0.003** for CatBoost.

**Do not lift either number into a table as if it measured the dataset.** `all` is one fold while
the baselines are five, and the baselines swing hard across folds: `no_text` runs 0.118-0.279 for
CatBoost and 0.039-0.248 for LightGBM, a span four to five times the deltas being reported. Fold 0
happens to sit below both 5-fold means, so switching the subtrahend from the fold-0 baseline to the
5-fold mean drags each delta down — LightGBM from +0.048 to +0.040, and CatBoost from **+0.031 to
-0.003**. CatBoost's negative sign is produced by the fold mismatch, not by the dataset, with
`no_text` fold 2 at 0.279 doing most of the work on its own.

This is the same principle stated in
[`accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md`](../../accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md)
— a delta is a difference of two means, so its halves must share the conditions they are measured
under or the drift between them lands inside the delta. There the halves had to share a machine;
here they would have to share a fold set, and they do not. The rejection does not rest on either
number: it rests on the target being 82% sentinel zeros.

No verdict is claimed from this file, and none should be: it is an incomplete grid on a candidate
rejected on target quality.

## Files in this folder

| file | schema | rows | what it is |
|---|---|---|---|
| `derived_input.csv` | source table | 2,267 | the scored file the grid was built from (sentinel rows dropped) |
| `grid.csv` | cpu | 52 | the partial frozen grid, `REG_TEXT_MEDIA_METACRITIC_SCORED` |
| `logs/dj_media_metacritic.log` | — | — | run log, including the `10357/12624` sentinel-drop line |

The fold-0 screen is in `../../screening/t2_joint/screen4_fold0.csv` under
`REG_TEXT_GAMES_METACRITIC_PC`.
