# ACCEPTED — `REG_TEXT_HOUSES_VIETNAM_2024`

**Verdict: ACCEPTED, 5 of 5 learners (quorum 3).**
The complete committee, measured: 5 models x 4 states x 5 folds = **100 cells, no gaps and
nothing counted as absent.** The verdict is computed by
`multabench.leaderboard.analysis.pass_matrix.passes()`.

## The dataset

| | |
|---|---|
| Source | Kaggle `nguyentiennhan/vietnam-housing-dataset-2024` |
| Registered as | `REG_TEXT_HOUSES_VIETNAM_2024` |
| Task | regression |
| Target | `Price` (687 distinct, \|z\|max 2.54, no clipping needed) |
| Text column | `Address` |
| Structured | 6 numeric + 4 categorical survive |
| Rows | 30,229 |
| Leakage columns dropped | none detected |

`Address` is genuine free text — street / ward / district / city — and **no structured column
duplicates it**, which is what keeps `no_text` from trivially matching `all`.

## Result

Per-state means over folds 0-4, rounded to 3 decimals before differencing, delta = 0.001:

| model | no_text | text_only | all | ft | Delta_Joint | Delta_Awareness |
|---|---|---|---|---|---|---|
| LightGBM | 0.342 | 0.310 | 0.592 | 0.607 | +0.250 | +0.015 |
| TabPFN-2.5 | 0.356 | 0.346 | 0.680 | 0.685 | **+0.324** | +0.005 |
| TabM | 0.312 | 0.336 | 0.633 | 0.638 | +0.297 | +0.005 |
| CatBoost | 0.341 | 0.322 | 0.632 | 0.636 | +0.291 | +0.004 |
| TabPFNv2 | 0.342 | 0.340 | 0.646 | 0.647 | +0.304 | +0.001 (knife-edge, see below) |

All five models clear both criteria, well past the `RHO = 3/5` quorum. Four clear
Delta_Awareness with a real margin; TabPFNv2 sits exactly on the threshold.

Evidence: `grid_frozen_gpu.csv` + `grid_tar.csv` + `grid_tar_pfn25.csv`, 100 rows in total,
all kaggle schema.

## All four states were measured on one machine

Delta_Joint originally came from CPU (`grid_frozen_cpu.csv`) and Delta_Awareness would have come
from GPU. Each delta is a difference of two means, so its two halves must share an environment or
the drift between them lands inside the delta. **The GPU `ft` half was deliberately not
differenced against the CPU `all` half.** All four states were re-run on the same Kaggle T4.

The cross-machine gap is small — over the 25 overlapping `all` cells CPU and GPU agree to mean
+0.0003, worst case 0.0062 — so mixing the halves would not have been catastrophic for Delta_Joint (+0.25 to
+0.30, ~40x that noise). It would have been fatal for Delta_Awareness, where CatBoost (+0.004)
and TabM (+0.005) are *smaller* than the 0.0062 cross-machine noise. That is the real reason the
halves were not mixed.

The control is decisive: for LightGBM and CatBoost, `no_text` — the one state that touches no
encoder — differs by exactly 0.0000 across machines. Loading, curation, the 90/10 split, the
seeding and the tree learners are bit-identical; divergence enters only where text does.

## The caveat: TabPFNv2's Delta_Awareness is a float knife-edge

`verdict_from_runs.py` reports TabPFNv2 as a pass, and that is what the criterion's own
implementation returns, but the margin is not real:

```
mean(ft) - mean(all) = 0.647 - 0.646 = 0.0010000000000000009  > 0.001  ->  True
```

The difference of the rounded means is **exactly delta**. It clears a strict `>` only because
float64 cannot represent `0.647 - 0.646` as exactly `0.001`. Rounding the difference to 4 decimals
first flips it to a fail. This cell is decided by floating-point representation, not by evidence
about the dataset.

**The verdict does not depend on it.** Excluding that cell leaves **4 of 5**, still above the
quorum of 3.

## TabPFN-2.5 is measured, not absent

Earlier grids had no TabPFN-2.5 cells at all: every one failed with `TabPFNLicenseError`. The
cause was not Hugging Face gating — `tabpfn/model_loading.py` lists 2.5/2.6/3 in `_HF_REPOS` and
calls `browser_auth.ensure_license_accepted()`, which wants a **Prior Labs** API key and raises
without one. `HF_TOKEN` never could have satisfied it, which is why the blocker survived so long.
TabPFN v2 is absent from `_HF_REPOS`, which is why it always ran.

With `TABPFN_TOKEN` supplied, the model runs clean: **20 of 20 cells, no failures**
(`grid_tar_pfn25.csv`). Its Delta_Joint of **+0.324 is the largest of the five**, and its
Delta_Awareness of +0.005 matches TabM and CatBoost.

This dataset therefore no longer rests on an empty cell. Any earlier statement describing
TabPFN-2.5 here as failing with `TabPFNLicenseError`, or this dataset as passing 3 of 5, is
**stale** and must not be carried forward.

## The CPU frozen grid stands on its own

`grid_frozen_cpu.csv` is a full 75-cell frozen grid (5 models x 3 frozen states x 5 folds) run
entirely on CPU, before any fine-tuning. Over its 25 per-(model, fold) Delta_Joint values:

```
mean = 0.2872   std = 0.0247   positive = 25/25   t = 58.03
```

Fold-level noise on Delta_Joint was calibrated earlier at about **+/-0.015**, so a mean of 0.287
is ~19x the noise band. Both unimodal baselines are non-degenerate (~0.31-0.36), so the joint gain
is genuine complementarity between address text and structured attributes, not an artifact of an
empty or saturated condition.

## Domain-novelty caveat

MulTaBench already contains four housing datasets — `REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020`,
`REG_TEXT_HOUSES_SAN_FRANCISCO_PERMITS_APPLICATIONS`, `REG_TEXT_HOUSES_AIRBNB_SEATTLE`,
`MUL_TEXT_HOUSES_MELBOURNE_AIRBNB`. This is a different market (Vietnam) and a different task
(listing address -> sale price, versus permits or nightly rates), but the *domain* overlaps.

Novelty matters for the writeup, not for whether the criterion is met. It is a human call, not a
measurement.

## Two CSV schemas in this folder

Nothing was reformatted. `grid_frozen_cpu.csv` uses the **cpu** columns
(`model, dataset, fold, multimodal_state, test_score, ...`); `grid_frozen_gpu.csv`, `grid_tar.csv`
and `grid_tar_pfn25.csv` use the **kaggle** columns (`state, score, secs, epochs, dataset, model,
fold`). Neither loader reads both: `cpu` files go through
`curation_lab.criterion.deltas.normalize`, `kaggle` files through
`curation_lab.kaggle.verdict_from_runs.load` (only the latter renames `score` to `test_score`).
Both canonicalize to `[model, dataset, state, fold, test_score]`, so the two frames merge cleanly
afterwards — which is how the cross-machine comparison above was computed.

## Files in this folder

| file | schema | rows | what it is |
|---|---|---|---|
| `grid_frozen_cpu.csv` | cpu | 75 | CPU frozen grid, 5 models x 3 states x 5 folds; the standalone Delta_Joint evidence |
| `grid_frozen_gpu.csv` | kaggle | 40 | T4 `no_text` + `text_only`, 4 models x 5 folds |
| `grid_tar.csv` | kaggle | 40 | T4 `all` + `ft` at 10 epochs, 4 models x 5 folds |
| `grid_tar_pfn25.csv` | kaggle | 20 | T4 TabPFN-2.5, all four states x 5 folds |
| `logs/dj_property.log` | — | — | CPU frozen run log |

The accepted verdict is computed over `grid_frozen_gpu.csv` + `grid_tar.csv` +
`grid_tar_pfn25.csv` — the 100 same-machine cells.

## Reproduce

```bash
# all vs ft (the expensive half)
python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full \
  --dataset-ref nguyentiennhan/vietnam-housing-dataset-2024 \
  --dataset-name REG_TEXT_HOUSES_VIETNAM_2024 \
  --folds 0,1,2,3,4 --models light,cat,tabm,tabpfnv2 --states all,ft --timeout 21000

# the frozen unimodal halves, same machine
python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full \
  ... --states no_text,text_only

# verdict over whatever has been collected
python -m curation_lab.kaggle.verdict_from_runs \
  results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/grid_frozen_gpu.csv \
  results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/grid_tar.csv \
  results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/grid_tar_pfn25.csv
```

`machine_shape=NvidiaTeslaT4` is load-bearing: Kaggle's default accelerator is a P100 (sm_60),
which the image's torch cannot launch kernels on even though `torch.cuda.is_available()` returns
True.
