# Pipeline validation — runs on the paper anchor

The four CSVs here are **not candidate screening.** Every row is a run on
`MUL_TEXT_PRODUCT_SENTIMENT`, one of the paper's own accepted datasets (N = 5,091, 4 classes,
1 text + 1 structured column), used as an anchor to prove that this runner reproduces MulTaBench
before any of its numbers about a *new* dataset are worth reading.

They are filed separately from the funnel for exactly that reason: putting them under
`screening/` would have misrepresented what they measure.

All four use the **cpu** schema
(`model, dataset, fold, multimodal_state, test_score, runtime, n_train, n_test, m_features,
task_type, tune_e5`).

## 1. The runner reproduces the paper exactly for `no_text`

`no_text` is the encoder-free condition: no E5, no PCA over embeddings, nothing but the structured
column. It is the clean test of whether the data loading, curation, the 90/10 split, the seeding
and the learners themselves are faithful.

Across the 36-run grid in `phase1_grid.csv`, **`no_text` reproduces the paper's value exactly for
every model** — mean absolute difference 0.0000 (TabPFNv2 0.0003). Anchor values, LightGBM fold 0:

```
no_text     0.83454303717305      (identical to the paper's value)
text_only   0.7658517388790819
all         0.8618478948517495
```

Each model has its own `no_text` value and reproduces its own: CatBoost 0.8344916001979384,
TabM 0.8165508512942514, TabPFNv2 0.8337871980150887. `0.83454303717305` is LightGBM's.

The encoder-dependent states do not reproduce bit-exactly, and are not expected to: `text_only`
has a mean *signed* difference of -0.0000 across the grid but a range of -0.046 to +0.038 —
symmetric scatter, not bias — because it reduces one text column to 30 PCA components to predict
4 classes, a weak high-variance signal. `all` carries a small genuine positive bias (mean +0.0058,
max +0.0235), real but far smaller than a single fold suggested.

## 2. Delta_Joint — the quantity the criterion consumes — agrees within 0.012

| model | Delta_Joint (ours) | Delta_Joint (paper) | sign agrees |
|---|---|---|---|
| CatBoost | 0.075 | 0.073 | yes |
| LightGBM | 0.058 | 0.046 | yes |
| TabM | 0.085 | 0.086 | yes |
| TabPFNv2 | 0.076 | 0.079 | yes |

**Within 0.012 across four models, all four signs matching**, three of four within 0.003. That is
the calibration behind the +/-0.015 fold-noise band quoted throughout the candidate reports, and
it is why a candidate whose Delta_Joint is below about 0.02 is re-measured on more folds rather
than accepted or rejected on one reading.

## 3. `cache_check.csv` — the frozen embedding cache is bit-exact

Encoding ~5k rows with a frozen E5 on CPU costs ~10 minutes per run, so the frozen states were
cached. A wrong cache would silently corrupt every downstream number, so it had to be proven
rather than assumed.

`cache_check.csv` is the end-to-end proof: LightGBM, `all`, fold 0, run **through** the cache.

```
uncached (phase1_lgbm.csv)   0.8618478948517495
cached   (cache_check.csv)   0.8618478948517495
IDENTICAL                    True
```

The cache keys **per string**, not per text list, so all 5 folds and all 5 learners share one
encode of each unique text. It is guarded three ways: `enable_cache()` refuses without an explicit
`frozen_only=True`; `run_one()` never enables it when `tune_e5` is set (a LoRA-tuned E5 returns
different vectors for the same text under the same base-model name); and `test_cache.py` (4 tests)
asserts bit-exactness, correct ordering under a re-ordered subset request, and no collision
between the same text under different column names.

**On the speedup figures.** The runtime recorded *in this CSV* is 438.9 s, which is the cold,
cache-**populating** run — down from ~600 s uncached even on the first pass, because the fit and
transform passes within one run now share encodes. The ~40x / "600 s cold to 11 s warm" figure is
the warm-cache measurement recorded in the Phase 1 findings; it is not what this file shows. What
this file shows is the bit-exactness.

## 4. `tabpfn25_retry.csv` — a historical record, not a live limitation

One row: TabPFN-2.5, `no_text`, fold 0, score 0.835752. It records an early retry from the period
when TabPFN-2.5 could not be loaded at all and contributed no cells to any grid.

The blocker was misdiagnosed for a long time as Hugging Face gating. It was not:
`tabpfn/model_loading.py` lists 2.5/2.6/3 in `_HF_REPOS` and calls
`browser_auth.ensure_license_accepted()`, which wants a **Prior Labs** API key and raises without
one. `HF_TOKEN` never could have satisfied it. TabPFN v2 is absent from `_HF_REPOS`, which is why
it always ran.

**That blocker is resolved.** With `TABPFN_TOKEN` supplied, TabPFN-2.5 runs clean — 20 of 20 cells
on Vietnam housing, where it posts the *largest* Delta_Joint of the five committee models. This
file is kept as the record of the failure mode and of how long it took to diagnose, not as a
statement about what the pipeline can do now. See
`accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md`.

## Files

| file | rows | what it is |
|---|---|---|
| `phase1_lgbm.csv` | 3 | LightGBM fold 0, three frozen states — the first fidelity check |
| `phase1_grid.csv` | 36 | the five-model frozen grid: LightGBM + CatBoost x 3 states x 5 folds, TabM + TabPFNv2 on fold 0 |
| `cache_check.csv` | 1 | LightGBM `all` fold 0 through the embedding cache; proves bit-exactness |
| `tabpfn25_retry.csv` | 1 | TabPFN-2.5 `no_text` fold 0, from the period when the model could not be loaded |
