# Phase 1: Grounding, Baseline Validation & Pipeline Lock — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove we understand the MulTaBench curation criterion exactly, and that a W&B-free runner reproduces the paper's protocol — before any candidate dataset work begins.

**Architecture:** A new `curation_lab/` package sits beside the read-only `multabench/` package. It enters the existing pipeline at `evaluate_on_loaded_dataset()` to inherit the paper's protocol unchanged, writes results as CSV rows in the schema the existing analysis layer already consumes, and delegates all pass/fail logic to `pass_matrix.passes()`.

**Tech Stack:** Python 3.12, pandas, pytest, LightGBM/CatBoost/TabM/TabPFN (via the existing `multabench` baselines), E5-small-v2 via `transformers`, `kagglehub` for dataset download.

**Spec:** `docs/superpowers/specs/2026-08-28-curation-lab-design.md` (Phase 1 = spec section 7, validation steps 1–4).

## Global Constraints

Every task's requirements implicitly include this section.

- **`multabench/` is READ-ONLY.** Never edit, never add files to it — including `multabench/datasets/annotated/` (auto-imported by `curation_mapping.py:11-19`, which re-raises on failure). Enter at `evaluate_on_loaded_dataset()` (`multabench/baselines/benchmarks/evaluate.py:25`).
- **Never modify `benchmark.py`.** It is referenced only as the behavioural spec our runner must match.
- **No W&B.** `multabench/utils/logging.py::wandb_run` raises without credentials. The runner must never import or call it.
- **Windows console is cp1255.** Printing the emoji model names (`LightGBM 💡`) raises `UnicodeEncodeError`. Every Python invocation in this plan sets `PYTHONIOENCODING=utf-8`, and every `read_csv`/`to_csv` touching model names passes `encoding="utf-8"`.
- **CSV schema is load-bearing.** `multimodal_state` must hold the CLI-style label (`no_text`/`text_only`/`all`/`ft`) — **not** the `MultimodalState` enum value (`"all 🔥"` etc.), which `pass_matrix._STATES` will not match. `model` must hold the emoji `MODEL_NAME` so `committee_pool._MODEL_LABELS` maps it.
- **Do NOT call `build_pass_matrix()` on fewer than 10 models.** It ends with `matrix[CURATION_MODELS + EXTRA_MODELS]` and raises `KeyError: "['RandomForest', 'RealMLP', 'TabDPT', 'TabICLv2', 'XGBoost'] not in index"`. Candidate verdicts call `passes()` per model and aggregate.
- **Local machine is Windows, CPU-only** (no CUDA, no `nvidia-smi`). `multabench.constants.DEVICE` is `None` unless `GPU` is set in `.env`; leave it unset.
- **Asymmetric tolerance.** Frozen states (`no_text`/`text_only`/`all`) are seeded and must agree closely. `ft` is stochastic LoRA fine-tuning — directional agreement only. Never plan or claim bit-exact `ft` reproduction.
- **Curation rule is never reimplemented.** All pass/fail logic delegates to `multabench.leaderboard.analysis.pass_matrix.passes()`.
- Run pytest as `python -m pytest` (not bare `pytest`) so the repo root lands on `sys.path`.

---

## File Structure

| File | Responsibility |
|---|---|
| `curation_lab/__init__.py` | package marker (empty) |
| `curation_lab/criterion/__init__.py` | package marker (empty) |
| `curation_lab/criterion/deltas.py` | canonicalize result frames; compute Δ_Joint/Δ_Awareness; grid-completeness reporting; verdicts via `passes()` |
| `curation_lab/criterion/fidelity.py` | compare our runner's scores against shipped paper CSVs |
| `curation_lab/runner/__init__.py` | package marker (empty) |
| `curation_lab/runner/results.py` | the load-bearing CSV row schema; append rows as UTF-8 |
| `curation_lab/runner/paper.py` | load a MulTaBench paper dataset at a given condition |
| `curation_lab/runner/run.py` | execute one `(dataset, model, state, fold)` run; CLI |
| `tests/curation_lab/test_deltas.py` | criterion tests (offline, pandas only) |
| `tests/curation_lab/test_results.py` | schema-contract tests (offline) |
| `docs/superpowers/plans/phase1-findings.md` | recorded outputs of the fidelity checks |

**Anchor dataset (decided, with evidence):** `MUL_TEXT_PRODUCT_SENTIMENT`.
Selection criteria: text modality, complete shipped 5×4×5 grid in `results/text/`, smallest CPU cost, unambiguous ground-truth verdict. It is the cheapest of the 20 (median LightGBM runtime 49s vs 784s for the most expensive), the smallest text surface (N=5091, **1 text column**, 1 structured column, 4 classes), and is a clean **5-of-5 accept** in the shipped `pass_matrix.csv`. Its pool-CSV long name is `MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT`; its `MulTaBenchDatasetID` name is `MUL_TEXT_PRODUCT_SENTIMENT`.

---

### Task 1: Criterion harness

Offline — needs only pandas. This is validation step 1 and can run before any environment work.

**Files:**
- Create: `curation_lab/__init__.py`
- Create: `curation_lab/criterion/__init__.py`
- Create: `curation_lab/criterion/deltas.py`
- Test: `tests/curation_lab/test_deltas.py`

**Interfaces:**
- Consumes: `multabench.leaderboard.analysis.pass_matrix.passes`, `multabench.leaderboard.analysis.committee_pool._MODEL_LABELS`
- Produces:
  - `STATES: tuple[str, ...]` = `("no_text", "text_only", "all", "ft")`
  - `FOLDS: tuple[int, ...]` = `(0, 1, 2, 3, 4)`
  - `CURATION_MODELS: tuple[str, ...]` = `("LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5")`
  - `QUORUM: int = 3`, `DELTA: float = 0.001`
  - `normalize(df: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame`
  - `screen_deltas(scores: pd.DataFrame) -> pd.DataFrame`
  - `missing_cells(scores: pd.DataFrame, models=CURATION_MODELS) -> list[tuple[str, str, int]]`
  - `verdict(scores: pd.DataFrame, delta: float = DELTA) -> dict`

- [ ] **Step 1: Create the package markers**

```bash
mkdir -p curation_lab/criterion curation_lab/runner tests/curation_lab
touch curation_lab/__init__.py curation_lab/criterion/__init__.py curation_lab/runner/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/curation_lab/test_deltas.py`:

```python
"""Criterion tests. Offline: reads only shipped CSVs, fits no models."""
import pandas as pd
import pytest

from curation_lab.criterion.deltas import (
    CURATION_MODELS,
    missing_cells,
    normalize,
    screen_deltas,
    verdict,
)

POOL_CSV = "multabench/leaderboard/results/analysis_curation_sensitivity/pool_scores_long.csv"
MATRIX_CSV = "multabench/leaderboard/results/analysis_curation_sensitivity/pass_matrix.csv"

KNOWN_ACCEPT = "MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT"   # 5 of 5 in shipped matrix
KNOWN_REJECT = "REG_TEXT_FOOD_RAMEN_RATINGS_2022"      # 0 of 5 in shipped matrix


@pytest.fixture(scope="module")
def pool():
    return pd.read_csv(POOL_CSV, encoding="utf-8")


def _one(pool, dataset):
    sub = pool[(pool["dataset"] == dataset) & (pool["model"].isin(CURATION_MODELS))]
    return normalize(sub)


def test_normalize_maps_emoji_model_names_and_state_column():
    raw = pd.DataFrame({
        "model": ["LightGBM 💡", "CatBoost 😸"],
        "multimodal_state": ["all", "ft"],
        "fold": [0, 0],
        "test_score": [0.5, 0.6],
    })
    out = normalize(raw, dataset="D")
    assert list(out.columns) == ["model", "dataset", "state", "fold", "test_score"]
    assert set(out["model"]) == {"LightGBM", "CatBoost"}
    assert set(out["state"]) == {"all", "ft"}
    assert set(out["dataset"]) == {"D"}


def test_verdict_accepts_known_positive(pool):
    got = verdict(_one(pool, KNOWN_ACCEPT))
    assert got["accepted"] is True
    assert sum(got["per_model"].values()) == 5


def test_verdict_rejects_known_negative(pool):
    got = verdict(_one(pool, KNOWN_REJECT))
    assert got["accepted"] is False
    assert sum(got["per_model"].values()) == 0


def test_verdict_reports_missing_cells_instead_of_asserting(pool):
    scores = _one(pool, KNOWN_ACCEPT)
    trimmed = scores[~((scores["model"] == "TabM") & (scores["state"] == "ft") & (scores["fold"] == 3))]
    with pytest.raises(ValueError) as err:
        verdict(trimmed)
    assert "TabM" in str(err.value) and "ft" in str(err.value)


def test_missing_cells_empty_on_complete_grid(pool):
    assert missing_cells(_one(pool, KNOWN_ACCEPT)) == []


def test_screen_deltas_gives_no_verdict_without_ft(pool):
    scores = _one(pool, KNOWN_ACCEPT)
    frozen = scores[scores["state"] != "ft"]
    out = screen_deltas(frozen)
    assert "accepted" not in out.columns
    assert out["delta_joint"].notna().all()
    assert out["delta_awareness"].isna().all()


def test_reproduces_shipped_pass_matrix_exactly():
    """The full 10-model matrix must re-derive with zero mismatched cells."""
    from multabench.leaderboard.analysis.pass_matrix import build_pass_matrix

    df = pd.read_csv(POOL_CSV, encoding="utf-8")
    expected = pd.read_csv(MATRIX_CSV, index_col="dataset", encoding="utf-8")
    got = build_pass_matrix(df).reindex(index=expected.index, columns=expected.columns)
    mismatches = (got.fillna("NA").astype(str) != expected.fillna("NA").astype(str)).values.sum()
    assert int(mismatches) == 0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/curation_lab/test_deltas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'curation_lab.criterion.deltas'`

- [ ] **Step 4: Implement `curation_lab/criterion/deltas.py`**

```python
"""Delta computation and pass/fail verdicts.

The curation rule itself is NEVER reimplemented here -- verdict() delegates to
multabench.leaderboard.analysis.pass_matrix.passes(). Two entry points are kept
deliberately separate so a cheap partial screen can never emit something that
looks like a verdict:

    screen_deltas()  any fold count, raw deltas, NO verdict     (T2 screen)
    verdict()        full 5-fold x 4-state grid, real pass/fail (T3)
"""
from __future__ import annotations

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import _MODEL_LABELS
from multabench.leaderboard.analysis.pass_matrix import passes

STATES: tuple[str, ...] = ("no_text", "text_only", "all", "ft")
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
CURATION_MODELS: tuple[str, ...] = ("LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5")
QUORUM: int = 3
DELTA: float = 0.001

_CANONICAL = ["model", "dataset", "state", "fold", "test_score"]


def normalize(df: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame:
    """Canonicalize any results frame to [model, dataset, state, fold, test_score].

    Accepts either emoji MODEL_NAMEs (as our runner and the shipped W&B exports
    write) or the short labels used in pool_scores_long.csv, and either a
    `multimodal_state` or `state` column.
    """
    out = df.copy()
    out["model"] = out["model"].astype(str).str.strip().map(lambda m: _MODEL_LABELS.get(m, m))
    if "multimodal_state" in out.columns and "state" not in out.columns:
        out = out.rename(columns={"multimodal_state": "state"})
    if dataset is not None:
        out["dataset"] = dataset
    if "dataset" not in out.columns:
        raise ValueError("No `dataset` column and no `dataset=` argument given.")
    out["fold"] = out["fold"].astype(int)
    out["test_score"] = out["test_score"].astype(float)
    return out[_CANONICAL].reset_index(drop=True)


def screen_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    """Raw Delta_Joint / Delta_Awareness per model. No completeness check, NO verdict.

    Means are rounded to 3 decimals before differencing, matching passes().
    Delta_Awareness is NaN when the `ft` state is absent (the frozen-only screen).
    """
    rows = []
    for model, sub in scores.groupby("model"):
        means = sub.groupby("state")["test_score"].mean().round(3)
        unimodal = [means[s] for s in ("no_text", "text_only") if s in means.index]
        if "all" in means.index and unimodal:
            delta_joint = float(means["all"] - max(unimodal))
        else:
            delta_joint = float("nan")
        if "ft" in means.index and "all" in means.index:
            delta_awareness = float(means["ft"] - means["all"])
        else:
            delta_awareness = float("nan")
        rows.append({
            "model": model,
            "n_rows": len(sub),
            "delta_joint": delta_joint,
            "delta_awareness": delta_awareness,
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def missing_cells(scores: pd.DataFrame, models: tuple[str, ...] = CURATION_MODELS) -> list[tuple[str, str, int]]:
    """(model, state, fold) triples absent from the full grid, sorted."""
    have = set(zip(scores["model"], scores["state"], scores["fold"]))
    want = {(m, s, f) for m in models for s in STATES for f in FOLDS}
    return sorted(want - have)


def verdict(scores: pd.DataFrame, delta: float = DELTA) -> dict:
    """Real pass/fail. Requires the complete 5-model x 4-state x 5-fold grid.

    Raises ValueError naming the absent cells rather than letting passes() fail
    with an opaque AssertionError -- the message is a re-queue list.
    """
    datasets = scores["dataset"].unique()
    if len(datasets) != 1:
        raise ValueError(f"verdict() takes exactly one dataset, got {list(datasets)}")
    absent = missing_cells(scores)
    if absent:
        raise ValueError(
            f"Incomplete grid for {datasets[0]}: {len(absent)} missing (model, state, fold) "
            f"cells -- re-queue these runs: {absent}"
        )
    per_model = {
        model: bool(passes(sub, delta=delta))
        for model, sub in scores[scores["model"].isin(CURATION_MODELS)].groupby("model")
    }
    n_pass = sum(per_model.values())
    return {
        "dataset": str(datasets[0]),
        "per_model": per_model,
        "n_pass": n_pass,
        "quorum": QUORUM,
        "accepted": n_pass >= QUORUM,
        "deltas": screen_deltas(scores),
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/curation_lab/test_deltas.py -v`
Expected: PASS — 7 passed.

If `test_verdict_rejects_known_negative` errors with an incomplete-grid `ValueError`, that dataset has gaps in the pool CSV. Substitute another confirmed 0-of-5 reject and re-run: `MUL_TEXT_FOOD_YELP_REVIEWS`, `REG_TEXT_CONSUMER_LAPTOP_INDIAN_PRICES`, `REG_TEXT_FOOD_BEER_RATINGS`, `REG_TEXT_FOOD_COFFEE_REVIEW`.

**Do not substitute `REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO`.** It is also a 0-of-5 reject, but it is the one dataset with a genuinely missing row in the source data — `pass_matrix._KNOWN_MISSING_ROWS` hardcodes `("TabPFNv2", "REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO", "ft", 4)`. Our `missing_cells()` does not carry that paper-specific exemption (it is irrelevant for new candidates), so `verdict()` would correctly but unhelpfully raise on it.

- [ ] **Step 6: Commit**

```bash
git add curation_lab/__init__.py curation_lab/criterion/ curation_lab/runner/__init__.py tests/curation_lab/test_deltas.py
git commit -m "feat: criterion harness delegating to pass_matrix.passes()"
```

---

### Task 2: Environment bring-up

`torch`, `lightgbm`, `catboost`, `transformers`, `sentence_transformers`, `peft` and `dotenv` are already importable on the system Python 3.12.10. Only `tabstar` and `kagglehub` are missing. Reinstalling the multi-GB torch stack into a fresh venv is not worth it, so install the two gaps into the current interpreter.

**Files:**
- Create: `.env` (gitignored — never commit it)

**Interfaces:**
- Produces: a Python environment where `import multabench.baselines.benchmarks.evaluate` succeeds and the anchor dataset downloads.

- [ ] **Step 1: Confirm the exact gap**

Run: `python -c "import tabstar" ; python -c "import kagglehub"`
Expected: both raise `ModuleNotFoundError`. Everything else in the stack already imports.

- [ ] **Step 2: Install the two missing packages**

```bash
python -m pip install "tabstar>=1.1.15" "kagglehub>=1.0.0"
```

The repo targets Python 3.11 (`init.sh`) but this machine has only 3.12.10 and no `py` launcher. If `tabstar` refuses to install on 3.12, stop and create a 3.11 environment instead — do not work around it by vendoring or stubbing `tabstar`, which supplies the seed and split functions the protocol depends on.

- [ ] **Step 3: Verify the runner import path opens**

Run: `PYTHONIOENCODING=utf-8 python -c "import multabench.baselines.benchmarks.evaluate as e; print('import ok', e.DOWNSTREAM_EXAMPLES, e.MEMORY)"`
Expected: `import ok 10000 32G`

- [ ] **Step 4: Create `.env` with Kaggle credentials**

```bash
cp .env.example .env
```

Then edit `.env` and fill `KAGGLE_USERNAME` and `KAGGLE_KEY` (from kaggle.com → Settings → Create New Token). `HF_TOKEN` may stay as the placeholder; `multabench/constants.py` only re-exports it. **Leave `WANDB_*` untouched — our runner never calls W&B. Leave `GPU` commented out — this machine has no CUDA.**

- [ ] **Step 5: Verify the anchor dataset downloads and matches its published shape**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.benchmark.load import load_multabench_dataset
d = load_multabench_dataset(MulTaBenchDatasetID.MUL_TEXT_PRODUCT_SENTIMENT)
print('rows', len(d.x), 'cols', list(d.x.columns), 'task', d.task_type, 'classes', d.y.nunique())
"
```
Expected: `rows 5091`, 2 feature columns, task `SupervisedTask.MULTICLASS`, `classes 4`. This matches the published summary row (Product Sentiment, N=5091, 1 structured + 1 text column, 4 classes). A row count far from 5091 means the wrong dataset or a stale Kaggle cache — stop and investigate before proceeding.

- [ ] **Step 6: Commit the gitignore guard (no code yet)**

```bash
printf '\n# curation_lab working data\ndata/candidates/\nresults/candidates/\n' >> .gitignore
git add .gitignore
git commit -m "chore: ignore curation_lab working data"
```

---

### Task 3: Results writer

The CSV schema is the contract that lets the existing analysis layer consume our output unmodified. It gets its own test cycle because the two easy mistakes here (writing the emoji enum value into `multimodal_state`, or short model names into `model`) silently produce a frame that `passes()` will not match.

**Files:**
- Create: `curation_lab/runner/results.py`
- Test: `tests/curation_lab/test_results.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `COLUMNS: list[str]`
  - `row_from_summary(summary: dict, state_flag: str) -> dict`
  - `append_row(csv_path: str, row: dict) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/curation_lab/test_results.py`:

```python
import pandas as pd

from curation_lab.runner.results import COLUMNS, append_row, row_from_summary

SUMMARY = {
    "model": "LightGBM 💡",
    "dataset": "MUL_TEXT_PRODUCT_SENTIMENT",
    "fold": 0,
    "test_score": 0.9123,
    "multimodal_state": "all 🔥",     # the enum value -- must be overridden
    "runtime": 12.5,
    "n_train": 4582,
    "n_test": 509,
    "m_features": 31,
    "task_type": "MUL",
    "tune_e5": False,
}


def test_state_flag_overrides_the_enum_value():
    row = row_from_summary(SUMMARY, state_flag="all")
    assert row["multimodal_state"] == "all"      # NOT "all 🔥"


def test_model_keeps_the_emoji_name():
    row = row_from_summary(SUMMARY, state_flag="all")
    assert row["model"] == "LightGBM 💡"


def test_row_has_exactly_the_declared_columns():
    assert list(row_from_summary(SUMMARY, state_flag="ft").keys()) == COLUMNS


def test_append_writes_utf8_and_roundtrips(tmp_path):
    path = tmp_path / "out.csv"
    append_row(str(path), row_from_summary(SUMMARY, state_flag="no_text"))
    append_row(str(path), row_from_summary(SUMMARY, state_flag="text_only"))
    back = pd.read_csv(path, encoding="utf-8")
    assert len(back) == 2
    assert list(back.columns) == COLUMNS
    assert back["model"].iloc[0] == "LightGBM 💡"
    assert set(back["multimodal_state"]) == {"no_text", "text_only"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/curation_lab/test_results.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'curation_lab.runner.results'`

- [ ] **Step 3: Implement `curation_lab/runner/results.py`**

```python
"""The run-row CSV schema.

Two fields are load-bearing for the existing analysis layer to consume our output
unmodified:

  multimodal_state  the CLI-style flag (no_text/text_only/all/ft), NOT the
                    MultimodalState enum value ("all 🔥"), which
                    pass_matrix._STATES will not match.
  model             the emoji MODEL_NAME, so committee_pool._MODEL_LABELS maps it.

Everything is written UTF-8: the model names contain emoji and the Windows
console/default codepage here is cp1255.
"""
from __future__ import annotations

import os

import pandas as pd

COLUMNS: list[str] = [
    "model",
    "dataset",
    "fold",
    "multimodal_state",
    "test_score",
    "runtime",
    "n_train",
    "n_test",
    "m_features",
    "task_type",
    "tune_e5",
]

VALID_STATES = {"no_text", "text_only", "all", "ft"}


def row_from_summary(summary: dict, state_flag: str) -> dict:
    """Project an evaluate_on_loaded_dataset() summary onto the schema.

    `state_flag` overrides whatever `multimodal_state` the summary carries,
    because evaluate_on_loaded_dataset() logs the MultimodalState enum and both
    `all` and `ft` map to MultimodalState.ALL -- the enum alone cannot tell them
    apart.
    """
    if state_flag not in VALID_STATES:
        raise ValueError(f"state_flag must be one of {sorted(VALID_STATES)}, got {state_flag!r}")
    row = {c: summary.get(c) for c in COLUMNS}
    row["multimodal_state"] = state_flag
    return row


def append_row(csv_path: str, row: dict) -> None:
    """Append one row, writing the header only when creating the file."""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        csv_path, mode="a", header=write_header, index=False, encoding="utf-8"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/curation_lab/test_results.py -v`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
git add curation_lab/runner/results.py tests/curation_lab/test_results.py
git commit -m "feat: run-row CSV schema with state-flag override"
```

---

### Task 4: Paper dataset loader

**Files:**
- Create: `curation_lab/runner/paper.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `STATE_BY_FLAG: dict[str, MultimodalState]`
  - `load_paper_dataset(name: str, state_flag: str) -> MultimodalDataset`

- [ ] **Step 1: Implement `curation_lab/runner/paper.py`**

```python
"""Load a MulTaBench paper dataset at one of the four curation conditions.

Phase 1 only needs the already-curated Kaggle-hosted datasets; spec-driven
ingestion of new candidates is Phase 2.
"""
from __future__ import annotations

from multabench.benchmark.load import load_multabench_dataset
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.curation import MultimodalDataset
from multabench.datasets.multimodal import MultimodalState

# `all` and `ft` deliberately share MultimodalState.ALL: they use identical
# features and differ only in whether the text encoder is fine-tuned (tune_e5).
STATE_BY_FLAG: dict[str, MultimodalState] = {
    "no_text": MultimodalState.NO_TEXT,
    "text_only": MultimodalState.TEXT_ONLY,
    "all": MultimodalState.ALL,
    "ft": MultimodalState.ALL,
}


def load_paper_dataset(name: str, state_flag: str) -> MultimodalDataset:
    if state_flag not in STATE_BY_FLAG:
        raise ValueError(f"Unknown state flag {state_flag!r}; expected one of {sorted(STATE_BY_FLAG)}")
    try:
        dataset_id = MulTaBenchDatasetID[name]
    except KeyError:
        raise ValueError(f"{name!r} is not a MulTaBenchDatasetID member") from None
    return load_multabench_dataset(dataset_id, multimodal_state=STATE_BY_FLAG[state_flag])
```

- [ ] **Step 2: Verify each state slices the columns as expected**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "
from curation_lab.runner.paper import load_paper_dataset
for flag in ['no_text','text_only','all','ft']:
    d = load_paper_dataset('MUL_TEXT_PRODUCT_SENTIMENT', flag)
    print(f'{flag:10s} rows={len(d.x)} cols={list(d.x.columns)}')
"
```
Expected: `no_text` has strictly fewer columns than `all`; `text_only` has exactly the text column(s); `all` and `ft` show identical column lists. All four report `rows=5091`.

- [ ] **Step 3: Commit**

```bash
git add curation_lab/runner/paper.py
git commit -m "feat: paper dataset loader with condition mapping"
```

---

### Task 5: The runner

**Files:**
- Create: `curation_lab/runner/run.py`

**Interfaces:**
- Consumes: `curation_lab.runner.results.row_from_summary`, `append_row`; `curation_lab.runner.paper.load_paper_dataset`, `STATE_BY_FLAG`
- Produces:
  - `MODELS: dict[str, type]` keyed by `SHORT_NAME`
  - `run_one(dataset: str, model_key: str, state_flag: str, fold: int, out_csv: str, e5_overrides: dict | None = None) -> dict`

- [ ] **Step 1: Implement `curation_lab/runner/run.py`**

```python
"""Execute one (dataset, model, state, fold) run and append a result row.

Enters the existing pipeline at evaluate_on_loaded_dataset() so the paper's
protocol -- seeds, 90/10 split, 2000-row test cap, metric selection -- is
inherited unchanged. Deliberately never imports multabench.utils.logging, whose
wandb_run() raises without credentials.
"""
from __future__ import annotations

import argparse

from tabstar.training.devices import get_device

from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
from multabench.baselines.catboost import CatBoost
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.constants import DEVICE
from multabench.finetune.train_args import E5TrainArgs

from curation_lab.runner.paper import STATE_BY_FLAG, load_paper_dataset
from curation_lab.runner.results import append_row, row_from_summary

MODELS: dict[str, type] = {
    LightGBM.SHORT_NAME: LightGBM,        # "light"
    CatBoost.SHORT_NAME: CatBoost,        # "cat"
    TabM.SHORT_NAME: TabM,                # "tabm"
    TabPFNv2.SHORT_NAME: TabPFNv2,        # "tabpfnv2"
    TabPFNv2p5.SHORT_NAME: TabPFNv2p5,    # "tabpfnv2p5"
}


def run_one(dataset: str, model_key: str, state_flag: str, fold: int, out_csv: str,
            e5_overrides: dict | None = None) -> dict:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model {model_key!r}; expected one of {sorted(MODELS)}")
    tune_e5 = state_flag == "ft"
    e5_train_kwargs = None
    if tune_e5:
        e5_train_kwargs = E5TrainArgs().to_dict()
        e5_train_kwargs.update(e5_overrides or {})

    loaded = load_paper_dataset(dataset, state_flag)
    summary = evaluate_on_loaded_dataset(
        model_cls=MODELS[model_key],
        dataset=loaded,
        fold=fold,
        device=get_device(device=DEVICE),
        train_examples=DOWNSTREAM_EXAMPLES,
        multimodal_state=STATE_BY_FLAG[state_flag],
        tune_e5=tune_e5,
        e5_train_kwargs=e5_train_kwargs,
    )
    append_row(out_csv, row_from_summary(summary, state_flag=state_flag))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run one MulTaBench evaluation without W&B.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--state", required=True, choices=sorted(STATE_BY_FLAG))
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--out", default="results/candidates/phase1.csv")
    p.add_argument("--e5-epochs", type=int, default=None,
                   help="Override E5 fine-tuning epochs (ft only); use to timebox CPU runs.")
    args = p.parse_args()
    overrides = {"epochs": args.e5_epochs} if args.e5_epochs is not None else None
    summary = run_one(args.dataset, args.model, args.state, args.fold, args.out, overrides)
    print(f"{args.model} {args.state} fold={args.fold} score={summary['test_score']:.4f} "
          f"runtime={summary['runtime']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the cheapest possible run**

Run:
```bash
PYTHONIOENCODING=utf-8 python -m curation_lab.runner.run \
  --dataset MUL_TEXT_PRODUCT_SENTIMENT --model light --state no_text --fold 0 \
  --out results/candidates/phase1_lgbm.csv
```
Expected: a line ending `-> results/candidates/phase1_lgbm.csv`, and the CSV exists with 1 data row. `no_text` needs no text encoder, so this should finish in well under a minute and proves the whole path works before any E5 download.

- [ ] **Step 3: Run the two frozen states that need E5**

```bash
for S in text_only all; do
  PYTHONIOENCODING=utf-8 python -m curation_lab.runner.run \
    --dataset MUL_TEXT_PRODUCT_SENTIMENT --model light --state $S --fold 0 \
    --out results/candidates/phase1_lgbm.csv
done
```
Expected: two more rows. The first downloads `intfloat/e5-small-v2`. With one text column over ~5k rows on CPU, expect single-digit-to-low-double-digit minutes per run — note the actual `runtime` values, since Task 9's cache decision depends on them.

- [ ] **Step 4: Verify the written schema is exactly right**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "
import pandas as pd
d = pd.read_csv('results/candidates/phase1_lgbm.csv', encoding='utf-8')
print(d[['model','multimodal_state','fold','test_score']].to_string(index=False))
assert set(d['multimodal_state']) <= {'no_text','text_only','all','ft'}, d['multimodal_state'].unique()
assert d['model'].str.contains('LightGBM').all()
print('schema ok')
"
```
Expected: three rows and `schema ok`. Any emoji (`all 🔥`) in `multimodal_state` means the override in `row_from_summary` regressed.

- [ ] **Step 5: Commit**

```bash
git add curation_lab/runner/run.py
git commit -m "feat: W&B-free single-run runner"
```

---

### Task 6: Fidelity comparison

Validation step 2. Compares our three frozen scores against the shipped paper CSV for the identical `(model, state, fold)` keys.

**Files:**
- Create: `curation_lab/criterion/fidelity.py`
- Create: `docs/superpowers/plans/phase1-findings.md`

**Interfaces:**
- Consumes: `curation_lab.criterion.deltas.normalize`
- Produces: `compare_to_shipped(ours_csv: str, dataset: str, frozen_tol: float = 0.02) -> pd.DataFrame`

- [ ] **Step 1: Implement `curation_lab/criterion/fidelity.py`**

```python
"""Compare our runner's scores against the shipped paper results.

Tolerance is asymmetric by design. The frozen states are seeded and must agree
closely. `ft` is stochastic LoRA fine-tuning and the shipped numbers came from
different hardware and library versions, so it is reported but never asserted.
"""
from __future__ import annotations

import os

import pandas as pd

from curation_lab.criterion.deltas import normalize

SHIPPED_DIR = "multabench/leaderboard/results/text"
FROZEN_STATES = ("no_text", "text_only", "all")


def compare_to_shipped(ours_csv: str, dataset: str, frozen_tol: float = 0.02) -> pd.DataFrame:
    """One row per (model, state, fold) we ran, with the shipped score alongside."""
    shipped_path = os.path.join(SHIPPED_DIR, f"{dataset}.csv")
    shipped = normalize(pd.read_csv(shipped_path, encoding="utf-8"), dataset=dataset)
    ours = normalize(pd.read_csv(ours_csv, encoding="utf-8"))
    ours = ours[ours["dataset"] == dataset]

    merged = ours.merge(
        shipped, on=["model", "dataset", "state", "fold"],
        how="left", suffixes=("_ours", "_paper"),
    )
    merged["abs_diff"] = (merged["test_score_ours"] - merged["test_score_paper"]).abs()
    merged["frozen"] = merged["state"].isin(FROZEN_STATES)
    merged["within_tol"] = merged["frozen"] & (merged["abs_diff"] <= frozen_tol)
    return merged.sort_values(["model", "state", "fold"]).reset_index(drop=True)
```

- [ ] **Step 2: Run the comparison**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "
from curation_lab.criterion.fidelity import compare_to_shipped
df = compare_to_shipped('results/candidates/phase1_lgbm.csv', 'MUL_TEXT_PRODUCT_SENTIMENT')
print(df[['model','state','fold','test_score_ours','test_score_paper','abs_diff','within_tol']].to_string(index=False))
frozen = df[df['frozen']]
print('frozen rows:', len(frozen), 'within tol:', int(frozen['within_tol'].sum()))
"
```
Expected: 3 frozen rows, all `within_tol == True`.

**If any frozen row exceeds tolerance, stop — do not proceed to Task 7.** Diagnose in this order: (a) `test_score_paper` is NaN → the merge key is wrong, most likely `model` not normalizing to the short label; (b) all three are off by a similar large amount → wrong dataset variant or a stale Kaggle cache; (c) only `text_only`/`all` are off → the E5 model or PCA settings differ from the defaults (`pca_components=30`, `intfloat/e5-small-v2`).

- [ ] **Step 3: Record the findings**

Create `docs/superpowers/plans/phase1-findings.md` with the printed comparison table pasted verbatim under a `## Task 6 — LightGBM frozen fidelity` heading, plus a one-line verdict stating whether the frozen states reproduced within tolerance and the observed per-run `runtime` values.

- [ ] **Step 4: Commit**

```bash
git add curation_lab/criterion/fidelity.py docs/superpowers/plans/phase1-findings.md
git commit -m "feat: fidelity comparison against shipped paper results"
```

---

### Task 7: Full frozen grid across all five models

Validation step 3. Proportionate to the goal — this proves every model class runs end-to-end and agrees, not that we can regenerate the paper.

Cost is asymmetric, so fold coverage is too: LightGBM and CatBoost are cheap enough for all five folds; TabM, TabPFNv2 and TabPFN-2.5 run fold 0 only.

**Files:**
- Modify: `docs/superpowers/plans/phase1-findings.md`

**Interfaces:**
- Consumes: `curation_lab.runner.run.run_one`, `curation_lab.criterion.fidelity.compare_to_shipped`
- Produces: `results/candidates/phase1_grid.csv`

- [ ] **Step 1: Run the cheap models across all five folds**

```bash
for M in light cat; do for S in no_text text_only all; do for F in 0 1 2 3 4; do
  PYTHONIOENCODING=utf-8 python -m curation_lab.runner.run \
    --dataset MUL_TEXT_PRODUCT_SENTIMENT --model $M --state $S --fold $F \
    --out results/candidates/phase1_grid.csv
done; done; done
```
Expected: 30 rows. Re-running a `(model, state, fold)` triple appends a duplicate rather than replacing it — if you need to redo one, delete the CSV and rerun the loop.

- [ ] **Step 2: Run the expensive models on fold 0 only**

```bash
for M in tabm tabpfnv2 tabpfnv2p5; do for S in no_text text_only all; do
  PYTHONIOENCODING=utf-8 python -m curation_lab.runner.run \
    --dataset MUL_TEXT_PRODUCT_SENTIMENT --model $M --state $S --fold 0 \
    --out results/candidates/phase1_grid.csv
done; done
```
Expected: 9 more rows, 39 total. TabPFN on CPU is slow; if a run exceeds roughly an hour, note it and continue — the anchor has only 4 classes and ~4.6k training rows, which is inside TabPFN's `MAX_ROWS = 10000`.

- [ ] **Step 3: Compare the whole grid**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "
from curation_lab.criterion.fidelity import compare_to_shipped
df = compare_to_shipped('results/candidates/phase1_grid.csv', 'MUL_TEXT_PRODUCT_SENTIMENT')
frozen = df[df['frozen']]
print(frozen.groupby('model')[['abs_diff']].agg(['count','max']).to_string())
print('all within tol:', bool(frozen['within_tol'].all()))
"
```
Expected: `all within tol: True`, with five model groups listed.

- [ ] **Step 4: Confirm the frozen screen reproduces the published Δ_Joint sign**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "
import pandas as pd
from curation_lab.criterion.deltas import normalize, screen_deltas
ours = normalize(pd.read_csv('results/candidates/phase1_grid.csv', encoding='utf-8'))
print(screen_deltas(ours).to_string(index=False))
"
```
Expected: `delta_joint` positive for every model (the anchor is a 5-of-5 accept), `delta_awareness` NaN throughout since no `ft` runs exist yet.

- [ ] **Step 5: Append findings and commit**

Add a `## Task 7 — five-model frozen grid` section to `docs/superpowers/plans/phase1-findings.md` with both tables pasted verbatim.

```bash
git add docs/superpowers/plans/phase1-findings.md
git commit -m "docs: record five-model frozen fidelity results"
```

---

### Task 8: Single timeboxed TAR (`ft`) run

Validation step 4a. One run, directional only. Its purpose is to prove the fine-tuning path executes end-to-end on CPU and moves the score in the right direction — **not** to reproduce the paper's `ft` number.

**Files:**
- Modify: `docs/superpowers/plans/phase1-findings.md`

- [ ] **Step 1: Run LightGBM `ft` on fold 0 with reduced epochs**

```bash
PYTHONIOENCODING=utf-8 python -m curation_lab.runner.run \
  --dataset MUL_TEXT_PRODUCT_SENTIMENT --model light --state ft --fold 0 \
  --e5-epochs 3 --out results/candidates/phase1_ft.csv
```

`E5TrainArgs` defaults to 50 epochs with patience 3, which is impractical on CPU; `--e5-epochs 3` timeboxes it. Expected: one row with `multimodal_state == "ft"` and `tune_e5 == True`. Budget up to a couple of hours; if it has not finished by then, stop it and retry with `--e5-epochs 1`.

- [ ] **Step 2: Check direction, not equality**

Run:
```bash
PYTHONIOENCODING=utf-8 python - <<'PY'
import pandas as pd

def lgbm(df, state, fold=0):
    df = df.copy()
    df["model"] = df["model"].astype(str).str.strip()
    sel = df[(df["multimodal_state"] == state) & (df["fold"] == fold)
             & df["model"].str.contains("LightGBM")]
    return sel["test_score"].mean()

ft = pd.read_csv("results/candidates/phase1_ft.csv", encoding="utf-8")
grid = pd.read_csv("results/candidates/phase1_grid.csv", encoding="utf-8")
paper = pd.read_csv("multabench/leaderboard/results/text/MUL_TEXT_PRODUCT_SENTIMENT.csv", encoding="utf-8")

print("ours all(fold0) = %.4f" % lgbm(grid, "all"))
print("ours ft (fold0) = %.4f" % lgbm(ft, "ft"))
print("paper ft(fold0) = %.4f" % lgbm(paper, "ft"))
print("paper all(fold0)= %.4f" % lgbm(paper, "all"))
PY
```
Expected: our `ft` score lands in the same broad range as the paper's `ft` score. Because epochs were cut from 50 to 3, our `ft` may sit below the paper's and may not exceed our own `all` — that is an accepted outcome for a timeboxed CPU run and does **not** block Phase 1. Record it and move on.

- [ ] **Step 3: Append findings and commit**

Add a `## Task 8 — TAR directional check` section recording both scores, the epoch count used, the wall-clock time, and an explicit note that bit-exact `ft` reproduction was neither attempted nor expected.

```bash
git add docs/superpowers/plans/phase1-findings.md
git commit -m "docs: record timeboxed TAR directional check"
```

---

### Task 9: Embedding cache — prove bit-exact or drop

Validation step 4b. **Only do this task if the Task 5 Step 3 runtimes make the T2 screen impractical** (as a rule of thumb: a frozen `all` run taking more than ~10 minutes). Otherwise skip it and record the decision — YAGNI.

**Files:**
- Create (conditionally): `curation_lab/runner/cache.py`
- Modify: `docs/superpowers/plans/phase1-findings.md`

**Interfaces:**
- Produces: `enable_cache(cache_dir: str) -> None`, `cached_encode_texts(...)`

- [ ] **Step 1: Decide from measured runtimes**

Re-read the `runtime` values recorded in Task 6 Step 3. If frozen runs are fast enough, append to the findings doc:

> **Task 9 — cache dropped.** Frozen `all` runs completed in _N_s, fast enough for the T2 screen. The embedding cache is unnecessary; spec section 4's optional cache is not implemented.

Commit that and **stop — the task is complete.**

- [ ] **Step 2 (only if the cache is needed): Implement it as a monkeypatch**

Create `curation_lab/runner/cache.py`:

```python
"""Optional frozen-embedding cache.

Frozen E5 output for a given (model, column, text) is identical across every
model and fold, so caching it is result-preserving -- PCA still refits per fold.
Valid ONLY when tune_e5=False; a fine-tuned encoder produces different vectors
for the same text.

Ships DISABLED. Enable only after test_cache_is_bit_exact passes.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from multabench.baselines.preprocessing import text_embeddings as _te

_original_encode = _te.encode_texts_with_e5


def _key(texts: list[str], model_name: str, col_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(col_name.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def enable_cache(cache_dir: str = ".emb_cache") -> None:
    os.makedirs(cache_dir, exist_ok=True)

    def cached_encode_texts(texts, model, tokenizer, device, col_name):
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown")
        path = os.path.join(cache_dir, f"{_key(list(texts), str(model_name), str(col_name))}.npy")
        if os.path.exists(path):
            return np.load(path)
        out = _original_encode(texts=texts, model=model, tokenizer=tokenizer,
                               device=device, col_name=col_name)
        np.save(path, out)
        return out

    _te.encode_texts_with_e5 = cached_encode_texts


def disable_cache() -> None:
    _te.encode_texts_with_e5 = _original_encode
```

- [ ] **Step 3: Write and run the bit-exactness test**

Create `tests/curation_lab/test_cache.py`:

```python
import numpy as np
import torch

from multabench.baselines.preprocessing.text_embeddings import encode_texts_with_e5
from multabench.e5.constants import E5_SMALL_V2
from multabench.e5.e5_finetune import get_vanilla_e5
from curation_lab.runner import cache as cache_mod

TEXTS = ["great product, arrived fast", "terrible, broke in a day", "ok for the price"]


def test_cache_is_bit_exact(tmp_path):
    device = torch.device("cpu")
    model, tokenizer = get_vanilla_e5(device, model_name=E5_SMALL_V2)
    uncached = encode_texts_with_e5(texts=TEXTS, model=model, tokenizer=tokenizer,
                                    device=device, col_name="review")
    cache_mod.enable_cache(str(tmp_path / "emb"))
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        miss = te.encode_texts_with_e5(texts=TEXTS, model=model, tokenizer=tokenizer,
                                       device=device, col_name="review")
        hit = te.encode_texts_with_e5(texts=TEXTS, model=model, tokenizer=tokenizer,
                                      device=device, col_name="review")
    finally:
        cache_mod.disable_cache()
    assert np.array_equal(uncached, miss)
    assert np.array_equal(uncached, hit)
```

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/curation_lab/test_cache.py -v`
Expected: PASS. **If it fails, delete `curation_lab/runner/cache.py` and record the cache as dropped** — a cache that changes results is worse than no cache.

- [ ] **Step 4: Commit**

```bash
git add curation_lab/runner/cache.py tests/curation_lab/test_cache.py docs/superpowers/plans/phase1-findings.md
git commit -m "feat: bit-exact frozen embedding cache (opt-in)"
```

---

## Phase 1 Exit Criteria

Phase 1 is complete when all of the following hold, evidenced in `docs/superpowers/plans/phase1-findings.md`:

1. `python -m pytest tests/curation_lab/ -v` passes.
2. The shipped 56×10 `pass_matrix.csv` re-derives with **0 mismatched cells**.
3. `verdict()` returns `accepted=True` for the known 5-of-5 dataset and `accepted=False` for the known 0-of-5 dataset.
4. Our runner reproduces the anchor's frozen scores for all five curation models within tolerance.
5. `screen_deltas()` on our own frozen runs yields a positive Δ_Joint for every model.
6. The `ft` path executed at least once end-to-end, with its non-exactness explicitly recorded.
7. The embedding cache is either proven bit-exact or explicitly dropped.

Only then does Phase 2 (candidate discovery) begin. **Phase 2 needs its own plan** — do not start it from this document.

## Spec Correction

Spec section 5 says `verdict()` "delegates to `pass_matrix.passes()` and `build_pass_matrix()`". `build_pass_matrix()` cannot be used for candidate datasets: it ends with `matrix[CURATION_MODELS + EXTRA_MODELS]` and raises `KeyError` unless all 10 models are present. `verdict()` calls `passes()` per model and aggregates. Update the spec when this plan is executed.
