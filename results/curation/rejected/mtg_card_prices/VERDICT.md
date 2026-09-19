# BORDERLINE — MTG card prices (`REG_TEXT_GAMES_MTG_CARD_PRICES`)

**Verdict (final): BORDERLINE. Not counted as an accepted dataset.**

`pass_matrix.passes()` returns **4 of 5**, but two of those passes are float knife-edges. With
them counted as fails, the result is **2 of 5**, below the quorum of 3. This is the only gridded
candidate where the verdict depends on how knife-edges are counted, so we claim neither an
acceptance nor a clean rejection. We report it among the negative results.

## The dataset

| | |
|---|---|
| Source | Kaggle `douglascampospires/mtg-all-cards` |
| Task | regression, target `price_usd_log10` (log10 of the USD market price) |
| Text columns | `CARD_TEXT`, `TYPE` |
| Structured | `CMC`, `NUMBER_OF_EDITIONS`, `power`, `toughness`, `first_edition_year` (numeric); `RARITY`, `COLOR_PIE` (categorical) |
| Rows | 28,507 |
| Preparation | `curation_lab/prep/mtg_cards.py` (parses the packed `PRICES` string, log-transforms the target, splits `POWER_TOUGHNESS` and `FIRST_EDITION` into numerics) |

The target keeps `|z|max = 5.36` after the log transform. The repo warns about this and never
clips it.

## Result

The grid is complete: 100 of 100 cells (5 models x 4 states x 5 folds). It ran on Kaggle T4s in
two kernels split by model, with `ft` at 10 epochs.

| model | no_text | text_only | all | ft | Delta_Joint | Delta_Awareness | passes() | knife-edges as fails |
|---|---|---|---|---|---|---|---|---|
| TabPFNv2 | 0.520 | 0.269 | 0.573 | 0.579 | +0.053 | +0.006 | PASS | pass |
| TabM | 0.504 | 0.305 | 0.578 | 0.580 | +0.074 | +0.002 | PASS | pass |
| CatBoost | 0.525 | 0.292 | 0.586 | 0.587 | +0.061 | +0.001 | PASS (knife-edge) | fail |
| LightGBM | 0.518 | 0.271 | 0.572 | 0.573 | +0.054 | +0.001 | PASS (knife-edge) | fail |
| TabPFN-2.5 | 0.523 | 0.310 | 0.592 | 0.591 | +0.069 | -0.001 | fail | fail |

**4 of 5 by `passes()`, 2 of 5 with knife-edges as fails. BORDERLINE.**

For CatBoost and LightGBM the difference of rounded means is `0.0010000000000000009`. It clears
the strict `> 0.001` only through float64 representation. Unrounded, `ft - all` is +0.0009 for
CatBoost and +0.0011 for LightGBM.

Pooled over the 25 (learner, fold) cells (unrounded, t = mean / (sd / sqrt(n))):

| | mean | sd | positive | t | range |
|---|---|---|---|---|---|
| Delta_Joint | 0.0620 | 0.0108 | 25/25 | 28.85 | [+0.0403, +0.0820] |
| Delta_Awareness | 0.0018 | 0.0049 | 18/25 | 1.87 | [-0.0122, +0.0096] |

The joint term is solid. The awareness term is not: it is small and changes sign within
CatBoost, LightGBM and TabM across folds, and it is negative on four of TabPFN-2.5's five folds.
This is the same pattern as board games and anime, where a real joint gain was not matched by
target-awareness.

## Files in this folder

| file | schema | rows | what it is |
|---|---|---|---|
| `grid_gpu_light_cat_tabm.csv` | kaggle | 60 | T4, `light,cat,tabm` x 4 states x folds 0-4 (kernel `multabench-mtg-lct`) |
| `grid_gpu_tabpfn.csv` | kaggle | 40 | T4, `tabpfnv2,tabpfnv2p5` x 4 states x folds 0-4 (kernel `multabench-mtg-pfn`) |

To reproduce the `passes()` count, run the command below. It prints `4 of 5` and
`VERDICT: ACCEPTED`, because the function counts the knife-edges as passes. The borderline call is
our reading of that output, not something the function returns.

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m curation_lab.kaggle.verdict_from_runs \
  results/curation/rejected/mtg_card_prices/grid_gpu_*.csv
```

This grid replaces the earlier CPU frozen-only grid (75 cells, no `ft` state), which has been
removed.
