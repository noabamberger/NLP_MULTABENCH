# Environment and performance

## Environment

- **`.venv/Scripts/python.exe` only** — never the system Python. The system interpreter hosts an
  unrelated project (`tap-text-tabular`) pinned to pandas 3.0.3 / numpy 2.4.0, incompatible with
  this repo.
- **pandas must be 2.3.3** (the repo's pin). Under pandas 3.x, string columns get the new `str`
  dtype and `tabstar.preprocessing.feat_types.is_numerical_feature` raises `ValueError:
  Unsupported dtype str for series <col>`, which takes down all feature detection — and therefore
  the `no_text` / `text_only` states.
- **`PYTHONIOENCODING=utf-8` always** — the console codepage here is cp1255, and simply printing an
  emoji model name raises `UnicodeEncodeError`. Pass `encoding="utf-8"` to every `read_csv`/
  `to_csv` that touches model names.
- **No CUDA on this machine**: `DEVICE` is `None` and anything run locally runs on CPU. Frozen E5
  embedding of ~5k rows costs ~10 minutes per run; LoRA fine-tuning (`ft`) is far more expensive.
  **This is why the pipeline runs on Kaggle GPU**: every curation grid in `results/curation/` that
  contains an `ft` state was measured on a Kaggle T4, and a full 5 models x 4 states x 5 folds grid
  costs under an hour there (~0.65 GPU-h for Udemy's 100 cells, 2-3% of the ~30 h weekly quota).
  Local CPU is for frozen-only (Delta_Joint) work and for reproducing historical grids; a full CPU
  grid costs days of serial wall-clock on the single local machine.

  The one CPU grid with an `ft` state is `REG_TEXT_EDU_UDEMY_ACADEMY/grid.csv`, kept deliberately
  as the cross-environment comparison against that dataset's T4 grid — see its `VERDICT.md` for
  what the two lanes agree and disagree about.

## Performance economics

| optimization | speedup | status |
|---|---|---|
| frozen embedding cache | ~40x | done, bit-exact (4 tests) |
| `max_length` cap, frozen encode | ~7x | off by default |
| `max_length` cap, TAR training loop | 333 s -> ~8 s per step | done |
| TAR encoder sharing (25 runs -> 10 fine-tunings) | 2.5x | done; measured 2184 s cold then 1082 s on a cache hit |

**The `max_length` cap is not bit-exact.** Masked padding is algebraically inert, but changing the
padded length reassociates float32 matmuls and moves embeddings ~1e-7. Compare with `atol=1e-5`,
never `array_equal`. Fine for screening; never for numbers compared against the paper.

**Why encoder sharing is correct.** Fine-tuning happens only in the embedding step, so the tuned
encoder is a function of `(x_train, y_train, e5_train_kwargs)` alone. `USE_VAL_SPLIT` is True for
LightGBM/CatBoost/TabM and False for both TabPFNs, giving 2 distinct fine-tunings per fold rather
than 5. The cache is keyed on argument content, not on model grouping, so it stays correct if
upstream ever threads a real fold into `split_to_val`. Verified empirically, not just argued: on
`REG_TEXT_HOUSES_VIETNAM_2024`, `test_encoder_sharing_groups_are_real`
(`MULTABENCH_TAR_SLOW=1`) confirms LightGBM and CatBoost tune identical encoders to within 1e-5
while TabPFNv2 does not, and the sweep log recorded `{'hits': 15, 'misses': 10, 'corrupt': 0}` —
exactly the predicted 2 fine-tunings x 5 folds.

## The TabPFN-2.5 blocker, and why it survived so long

It is now **RESOLVED**, and the reason it persisted is itself a finding worth recording: the
blocker was mis-attributed to Hugging Face gating for weeks. `tabpfn/model_loading.py` lists
2.5/2.6/3 in `_HF_REPOS` and calls `browser_auth.ensure_license_accepted()`, which wants a
**Prior Labs API key**, so `HF_TOKEN` could never have satisfied it — every attempt to fix it by
accepting an HF licence was aimed at the wrong door. v2 is absent from `_HF_REPOS`, which is why it
always ran. The Windows `select.select` crash (`OSError: WinError 10038`) was a real symptom of the
interactive fallback, not the cause.

**Lesson: a blocker diagnosed from its symptom rather than its source can outlive several attempts
to clear it.** With `TABPFN_TOKEN` supplied (read first by `browser_auth.get_cached_token()`), the
model runs clean — 20 of 20 cells on Vietnam housing, where it posts the largest Delta_Joint
(+0.324) of the five committee models. The full five-model committee is now available for every
future grid.

Remaining blocker: `multabench/e5/e5_finetune.py:245` asserts `CUDA_VISIBLE_DEVICES`, bypassed by
`run_one(..., cpu_ft=True)`.

## The known test failure

`test_training_passage_matches_what_the_dataset_tokenizes` compares detokenized text (which
reinserts spaces around punctuation) against the raw string. A flaw in the test, not the pipeline.
Baseline is 1 failed.
