# Delta_Awareness (TAR) — `nguyentiennhan/vietnam-housing-dataset-2024`

Registered as `REG_TEXT_HOUSES_VIETNAM_2024`. Companion to `DJ_PROPERTY_REPORT.md`,
which measured Delta_Joint on CPU and explicitly left Delta_Awareness unmeasured.

Run on a **Kaggle T4 GPU** notebook (`talkraicer/multabench-tar-gpu`), driven by
`curation_lab.kaggle.push`. Spec is auto-derived by `screen/auto_spec.py` and matches
the CPU grid exactly: target `Price`, text `['Address']`, 6 numeric + 4 categorical,
no leakage columns.

## The whole grid was re-measured on one machine

Delta_Joint originally came from CPU and Delta_Awareness from GPU. Each delta is a
difference of two means, so its two halves must share an environment or the drift
between them lands in the delta. All four states were therefore re-run on the same
T4: **4 models x 4 states x 5 folds = 80 cells, no gaps.**

The frozen unimodal states reproduced the CPU numbers to 3 decimals (LightGBM
`no_text` 0.342 both), which is expected — `no_text` never touches E5 and `text_only`
uses a frozen encoder. The `all` state did move (LightGBM 0.591 CPU -> 0.592 GPU
over 5 folds, but 0.591 -> 0.625 on fold 0 alone), which is exactly why the halves
were not mixed.

## Result: quorum met, dataset ACCEPTED

Per-state means over folds 0-4, rounded to 3 decimals before differencing
(`pass_matrix.passes()`, delta = 0.001):

| model | no_text | text_only | all | ft | Delta_Joint | Delta_Awareness |
|---|---|---|---|---|---|---|
| LightGBM | 0.342 | 0.310 | 0.592 | 0.607 | **+0.250** | **+0.015** |
| TabM | 0.312 | 0.336 | 0.633 | 0.638 | **+0.297** | **+0.005** |
| CatBoost | 0.341 | 0.322 | 0.632 | 0.636 | **+0.291** | **+0.004** |
| TabPFNv2 | 0.342 | 0.340 | 0.646 | 0.647 | **+0.304** | +0.001 (see below) |
| TabPFN-2.5 | — | — | — | — | — | NOT MEASURED |

**Three models pass both criteria with a real margin**, which meets `RHO = 3/5`.
The verdict does not depend on either of the two doubtful rows.

### TabPFNv2 is exactly on the threshold — do not count it

`verdict_from_runs.py` reports TabPFNv2 as a pass, and that is what the criterion's
own implementation returns, but the margin is not real:

```
mean(ft) - mean(all) = 0.647 - 0.646 = 0.0010000000000000009  > 0.001  -> True
```

The difference of the rounded means is exactly delta. It clears a strict `>` only
because float64 cannot represent `0.647 - 0.646` as exactly `0.001`. Rounding the
difference to 4 decimals first (as the notebook does) flips it to a fail. This cell
is decided by floating-point representation, not by evidence about the dataset —
treat it as a non-pass and rely on the three models above.

### TabPFN-2.5 could not run

`TabPFNLicenseError: TabPFN requires a one-time license acceptance to download`,
on all 10 of its cells. This is the blocker already recorded in `RESUME.md`: the
weights sit behind a gated HuggingFace repo. It needs a human to accept the
Prior-Labs licence once. Counted as a non-pass here, which is how the paper treats
an empty (model, dataset) cell and is the conservative direction — it can only make
acceptance harder.

## Encoder sharing: 25 ft runs, 10 fine-tunings

Fine-tuning happens only in the embedding step, never end to end, so the tuned
encoder is a function of the training texts, labels, device and hyperparameters —
not of the tabular learner. Per fold the committee differs only in `USE_VAL_SPLIT`
(True for LightGBM/CatBoost/TabM, False for the TabPFNs), giving 2 distinct
fine-tunings per fold rather than 5.

`runner/tar_cache.py` was wired in to exploit that. Measured over the sweep:

```
[tar_cache] {'hits': 15, 'misses': 10, 'corrupt': 0}
```

10 misses = 2 fine-tunings x 5 folds, exactly as predicted.

The cache keys on a hash of the actual fine-tuning arguments, **not** on the
tree/TFM grouping, so a wrong grouping would cost a wasted miss rather than serve a
model an encoder it would never have trained. The grouping was also verified
empirically and independently: `test_encoder_sharing_groups_are_real`
(`MULTABENCH_TAR_SLOW=1`) confirms LightGBM and CatBoost tune identical encoders to
within 1e-5 while TabPFNv2 does not.

## Reproducing

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
python -m curation_lab.kaggle.verdict_from_runs results/candidates/dj_property_tar_*.csv
```

`machine_shape=NvidiaTeslaT4` is load-bearing: Kaggle's default accelerator is a
P100 (sm_60), which the image's torch cannot launch kernels on even though
`torch.cuda.is_available()` returns True.

Raw rows: `dj_property_tar_all_ft.csv`, `dj_property_tar_frozen.csv`.

## Still open

1. **TabPFN-2.5 licence** — one human step, and the only thing standing between this
   and a complete five-model grid.
2. **Domain novelty** — unchanged from `DJ_PROPERTY_REPORT.md`: MulTaBench already
   contains four housing datasets, though none is Vietnamese listing-address -> price.
   A human call, not a measurement.
