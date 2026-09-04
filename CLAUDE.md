# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is (and what it is being used for)

This is the official **MulTaBench** source code (paper: arXiv 2605.10616) — a benchmark and curation
pipeline for **multimodal tabular learning** (tabular + text, tabular + image).

`instructions.pdf` (untracked, in repo root) is the assignment brief for the **Technion NLP course
097215, Spring 2026 final project**, deadline **2026-10-26**. This checkout exists to serve
**Track 2 (Benchmark Track)**: curate *new* text-tabular dataset(s) and prove they pass the
MulTaBench curation pipeline. Practical consequences for any work here:

- The deliverable is **evidence about a dataset**, not a new algorithm. Default to *using* the
  existing 5 learners and 4 conditions rather than modifying them.
- **Grading is driven by the magnitude of the performance deltas** (Δ_Joint and Δ_Awareness), so
  anything that changes measured scores (splits, seeds, PCA dim, fold count, encoder choice) is
  load-bearing and must not drift silently from the paper defaults.
- Standard scope = 1 new passing dataset. Outstanding scope = a *systematic method* that yields
  ≥5 passing datasets (mining / LLM synthesis / web enrichment / tri-modal).

### The curation criteria (the thing being reproduced)

Evaluate 5 learners — **TabM, CatBoost, LightGBM, TabPFN v2, TabPFN v2.5** — under 4 conditions.
CLI `--multimodal_state` flag → paper condition → results-CSV label:

| Paper condition | CLI flag | CSV `multimodal_state` |
|---|---|---|
| Unimodal Structured (tabular only) | `no_text` | `no_text` |
| Unimodal Unstructured (text only) | `txt` or `text_only` | `text_only` |
| Joint Frozen (all feats, frozen E5) | `all` | `all` |
| Joint Target-Aware / TAR (LoRA-tuned E5) | `ft` or `ft-txt` | `ft` |

A dataset **passes** if, for **≥3 of the 5 learners** (`RHO = 3/5`):

- `Δ_Joint     = mean(all) - max(mean(no_text), mean(text_only)) > δ`
- `Δ_Awareness = mean(ft)  - mean(all)                          > δ`

with paper default `δ = 0.001`, per-state means over 5 folds **rounded to 3 decimals before
differencing**. This rule is implemented once, in
`multabench/leaderboard/analysis/pass_matrix.py::passes()` — reuse it, do not reimplement it.

## Commands

Setup (`init.sh` is bash-only: pyenv/Homebrew + `uv` venv; on Windows use WSL or create the venv
manually and `pip install -r requirements.txt`):

```bash
source init.sh
source .venv/bin/activate
cp .env.example .env      # WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY
```

Repo root must be importable (`init.sh` writes a `.pth` into site-packages and appends
`PYTHONPATH` to the activate script). `GPU=<idx>` in `.env` pins CUDA device; unset → CPU.

One benchmark run = one (model, dataset, fold, condition):

```bash
python benchmark.py --model light --dataset_name MUL_TEXT_MICHELIN_RESTAURANTS --fold 0 --multimodal_state all
```

A full curation sweep for one dataset is **5 models × 4 states × 5 folds = 100 runs**. `benchmark.py`
has no built-in sweep driver; in practice sweeps are not driven locally at all — they run on Kaggle
GPU via `curation_lab.kaggle.push` (see Environment below), which owns the fold/model/state loop and
writes a `kaggle`-schema results CSV.

Model keys (`--model`, from each class's `SHORT_NAME`): `light` `cat` `xgb` `rf` `real` `tabm`
`iclv2` `dpt` `tabpfnv2` `tabpfnv2p5` `tabstar` `agmm` `ctx`. The 5 curation models are
`light cat tabm tabpfnv2 tabpfnv2p5`.

Other useful entry points:

```bash
python multabench/scripts/do_tagging.py --dataset <NAME>          # print schema/description to write an annotated/ module
streamlit run multabench/scripts/do_leaderboard.py                # results dashboard (9 tabs)
python -m multabench.leaderboard.analysis.committee_pool          # rebuild pool_scores_long.csv
python -m multabench.leaderboard.analysis.pass_matrix             # rebuild pass_matrix.csv
python -m multabench.leaderboard.analysis.committee_delta_sweep   # δ sensitivity sweep
python multabench/scripts/do_multabench_audit.py                  # assert every dataset has ≤5 multimodal cols
python multabench/scripts/do_kaggle_prepare.py --dataset_name <NAME> --slug multabench-<x>
python multabench/scripts/do_kaggle_upload.py  --dataset_dir kaggle_uploads/multabench-<x>
```

`pytest` is in `requirements.in` but **there is no test suite and no lint config** — verification
here means running the benchmark/analysis modules and inspecting output.

## Architecture

### Dataset registration (the path to add a new dataset)

A dataset needs **two** things whose names must match exactly:

1. An enum member in `multabench/datasets/all_datasets.py`, in one of `OpenMLDatasetID`
   (value = OpenML id), `KaggleDatasetID` (value = `owner/slug[/file.csv]`), `UrlDatasetID`
   (value = URL; if not downloadable at runtime, the CSV is vendored under
   `multabench/datasets/url_datasets/` and mapped in `downloading.get_csv_local_path`), or
   `MulTaBenchDatasetID` (value = Kaggle slug of an *already curated* MulTaBench release).
   Name **must** be `{BIN|MUL|REG}_{TEXT|IMAGE}_{REST}` — module-load asserts enforce the prefix,
   and reject duplicate names/values across enums.
2. A module `multabench/datasets/annotated/<SAME_NAME>.py` exporting module-level `TARGET`
   (`CuratedTarget`), plus optional `FEATURES` (`list[CuratedFeature]`), `COLS_TO_DROP`, `CONTEXT`,
   `IMAGE_FOLDER`, `LOADING_FUNC`, `PROCESSING_FUNC`. `curation_mapping.py` auto-imports every
   module in that package at import time and keys it by filename — **a syntax/import error in any
   annotated module breaks the whole package**. Copy an existing file (e.g.
   `REG_TEXT_FOOD_WINE_VIVINO_SPAIN.py`) as the template; the docstring holds the `do_tagging.py`
   schema dump.

Load path: `downloading.download_dataset` dispatches by enum type → `curation.curate_dataset`
applies drop/target/value/name curation → drops null-target and missing-image rows →
`multimodal.filter_by_multimodality` slices columns for the requested state → returns
`MultimodalDataset(x, y, task_type, dataset_id, image_folder)`.

`MulTaBenchDatasetID` bypasses all of the above: `benchmark/load.py` pulls the pre-curated
`data.csv` + `metadata.json` + `images/` from Kaggle (`chico89/<slug>`) and applies its own state
filter. Curated datasets are produced by `benchmark/curation/prepare.py` (+ `upload.py`).

### Feature typing is inferred, not declared

Nothing marks a column "text" in the enum. `preprocessing/feat_types.py::_is_text_feature` calls a
non-numeric column **text** iff it has ≥100 distinct values **or** ≥80% unique ratio; otherwise
**categorical**. Image columns are detected by filename suffix in
`baselines/preprocessing/feature_types.py::is_image_feature`. This means the `no_text`/`text_only`
split — and therefore pass/fail — is sensitive to column cardinality: a short free-text field with
few distinct values silently becomes categorical and lands in the "structured" condition.

### Model pipeline

`baselines/abstract_model.py::TabularModel` is a template-method base; each baseline sets five
class flags (`USE_VAL_SPLIT`, `USE_MEDIAN_FILLING`, `USE_CATEGORICAL_ENCODING`,
`USE_TEXT_EMBEDDINGS`, `USE_TARGET_ENCODER`) and implements `initialize_model` / `fit_model`.
`fit()` → `fit_preprocessor` → `transform_preprocessor` → `fit_model`. Shared preprocessing does
dates → image encode+PCA → numeric/categorical → text encode+PCA, producing columns named
`{col}_img_pca_{i}` / `{col}_txt_pca_{i}`.

Note the ordering trap documented in `do_model_agnostic_preprocessing`: `categorical_indices` is
computed **before** the PCA expansion, so models needing categorical columns at fit time must
resolve **by name** (`[c for c in x.columns if c in self.categorical_features]`), as LightGBM does.

- Text: `intfloat/e5-small-v2`, one embedding per column formatted `"passage: {col}: {val}"`, then
  per-column PCA to 30 dims (`--pca_components`, or `--no_pca yes` which bails out via
  `MultimodalError` if >5 multimodal columns). `--e5_model tf-idf` swaps in skrub `StringEncoder`.
- Image: DINOv3 ViT-S/16 CLS token → per-column PCA 30.
- TAR (`ft`): one shared E5 LoRA-tuned on the target — regression targets are discretized into 20
  bins first, and each (row, text column) becomes a separate training example.

### Evaluation and metrics

`baselines/benchmarks/evaluate.py` subsamples to ≤10k train (`DOWNSTREAM_EXAMPLES`), splits 90/10
with test capped at 2000 (`preprocessing/splits.py`), seeds with `SEED + fold`, fits, scores, and
returns a flat dict logged to W&B. `test_score` is **R² for regression, ROC-AUC for binary,
macro OvR AUC for multiclass** (`baselines/training/metrics.py`) — higher is always better, which is
what makes the Δ comparisons directional.

### Results and analysis layer

`multabench/leaderboard/results/**.csv` are **W&B run exports** (columns `Name, test_score, fold,
multimodal_state, model, dataset, ...`), one directory per experiment family (`text/`, `images/`,
`text_source/`, `more_baselines/`, `sensitivity/`, `tabstar_corpus/`, `analysis_*`). Dataset names
are **not** consistent across families — the 20 accepted datasets appear under short pre-rename
names in some dirs and long post-rename names in others; `committee_pool.py` owns the
reconciliation (token-subset matching + 2 manual overrides). Do not add a fresh ad-hoc mapping.

Analysis chain (each runnable as `python -m ...`, each writes into
`results/analysis_curation_sensitivity/`):
`committee_pool.py` → `pool_scores_long.csv` (canonical long-format scores) →
`pass_matrix.py` → `pass_matrix.csv` (dataset × model booleans) →
`committee_panel_pass_rates.py` / `committee_delta_sweep.py` (committee-membership, quorum-size and
δ sensitivity). `passes()` **asserts** row completeness (5 folds × 4 states) rather than averaging
over gaps — one legitimately-missing row is hardcoded in `_KNOWN_MISSING_ROWS`; any new assertion
failure is a real data gap to investigate, not something to append to that set.

## Environment (this machine)

Use the project venv for everything: **`.venv/Scripts/python.exe`**, never the system Python.
The system interpreter hosts an unrelated project (`tap-text-tabular`) pinned to pandas 3.0.3 /
numpy 2.4.0, which is incompatible with this repo; installing this repo's pins there breaks it.

- **pandas must be 2.3.3** (the repo's pin). Under pandas 3.x, string columns get the new `str`
  dtype and `tabstar.preprocessing.feat_types.is_numerical_feature` raises
  `ValueError: Unsupported dtype str for series <col>`, which takes down all feature detection —
  and therefore the `no_text` / `text_only` states.
- Beyond `requirements.txt`, the import chain needs `tabstar`, `kagglehub`, `datasets`, `openml`,
  `pytabkit` and `wandb` present. `init.sh` is bash-only, so on Windows build the venv manually.
- Always run Python with `PYTHONIOENCODING=utf-8`. The console codepage here is cp1255, and simply
  printing an emoji `MODEL_NAME` raises `UnicodeEncodeError`. Pass `encoding="utf-8"` to every
  `read_csv`/`to_csv` that touches model names.
- No CUDA on this machine: `DEVICE` is `None` and anything run locally runs on CPU. Frozen E5
  embedding of ~5k rows costs ~10 minutes per run; LoRA fine-tuning (`ft`) is far more expensive.
- **Curation grids run on Kaggle GPU, not locally.** `curation_lab/kaggle/` is the pipeline:
  `push_code.py` (re-version the code dataset first, or the run silently uses stale code) ->
  `push.py --machine-shape NvidiaTeslaT4` -> `verdict_from_runs.py`. Split pushes by model so all
  four states for a learner share one session. A full 5 x 4 x 5 = 100-cell grid costs well under an
  hour of T4 time. Local CPU is for frozen-only (Delta_Joint) work and reproduction only.

## Gotchas

- **W&B is mandatory for `benchmark.py`.** `utils/logging.py::wandb_run` raises if
  `WANDB_API_KEY`/`WANDB_ENTITY` are missing. To evaluate without W&B, call
  `evaluate_on_dataset()` / `evaluate_on_loaded_dataset()` directly instead of patching the logger.
  Note the `wandb` *package* is still required regardless: `evaluate.py:17` imports
  `multabench.utils.logging`, which does `import wandb` at module level. Only `wandb_run()` needs
  credentials.
- **The README's `--multimodal_state` table is stale.** Actual accepted values (see `benchmark.py`):
  `all, non, ft, img, text_only, no_text, txt, non_txt, ft-txt, ft-img-ft-txt`. The README's
  `no_img` / `all 🔥` do not exist as CLI values.
- **`all` and `ft` both map to `MultimodalState.ALL`.** The conditions differ only by the derived
  `tune_dino`/`tune_e5` flags. `ft` means "fine-tune whichever encoder matches the dataset's
  modality" (image datasets → DINO, text datasets → E5); for a text dataset `ft` and `ft-txt` are
  equivalent. Never infer the condition from `MultimodalState` alone.
- **`FOLDS = 10` in `evaluate.py` is unused**; the paper protocol and every analysis module assume
  **folds 0–4**.
- TabPFN v2/v2.5 are skipped on highly-multiclass datasets (`is_invalid_model_dataset_pair`, e.g.
  Wine Review), and `exit()` silently — such (model, dataset) cells legitimately have no data, which
  `build_pass_matrix` leaves as `NaN` for the caller to decide on (the paper treats it as non-pass).
- `MultimodalError` is a *soft skip* meaning "this state is not applicable to this dataset"
  (e.g. `text_only` on a dataset with no detected text column); `benchmark.py` catches it and exits
  cleanly, so a missing run may mean "not applicable", not "crashed".
- Regression targets are only *warned* about for |z| > 5 outliers (`check_extreme_outliers`), never
  clipped — outlier-heavy targets can dominate R² and wreck the deltas.
