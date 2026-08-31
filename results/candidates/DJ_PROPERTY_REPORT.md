# Delta_Joint — `nguyentiennhan/vietnam-housing-dataset-2024`

Registered as `REG_TEXT_HOUSES_VIETNAM_2024`. Full 75-cell frozen grid, CPU, 5 models x 3
frozen states x 5 folds. **No fine-tuning was run.**

## Spec (auto-derived, human-checked)

Target `Price` (687 distinct, |z|max 2.54, no clipping needed). Text = `Address` — genuine
free text (street / ward / district / city) and **no structured column duplicates it**, which
is what keeps `no_text` from trivially matching `all`. 6 numeric + 4 categorical survive.
No leakage columns detected.

## Result: PASSES for all 5 models

| model | all | no_text | text_only | Delta_Joint |
|---|---|---|---|---|
| TabPFN-2.5 | 0.680 | 0.356 | 0.346 | **+0.324** |
| TabPFNv2 | 0.645 | 0.342 | 0.340 | **+0.303** |
| TabM | 0.633 | 0.313 | 0.334 | **+0.299** |
| CatBoost | 0.631 | 0.341 | 0.325 | **+0.290** |
| LightGBM | 0.591 | 0.342 | 0.312 | **+0.249** |

Per-(model,fold) deltas over all 25 cells:

```
mean = 0.2872   std = 0.0247   positive = 25/25   t = 58.03
```

Fold-level noise on Delta_Joint was calibrated earlier at about +/-0.015, so a mean of 0.287
is ~19x the noise band. This is not a marginal pass.

Both unimodal baselines are non-degenerate (~0.31-0.36), so the joint gain is genuine
complementarity between address text and structured attributes, not an artifact of an empty
or saturated condition.

## Novelty caveat (for a human to judge)

MulTaBench already contains housing datasets: `REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020`,
`REG_TEXT_HOUSES_SAN_FRANCISCO_PERMITS_APPLICATIONS`, `REG_TEXT_HOUSES_AIRBNB_SEATTLE`,
`MUL_TEXT_HOUSES_MELBOURNE_AIRBNB`. This is a different market (Vietnam) and a different task
(listing address -> sale price, vs. permits or nightly rates), but the *domain* overlaps.

## Still required

Delta_Awareness (TAR) has NOT been measured. Delta_Joint alone does not accept a dataset.
