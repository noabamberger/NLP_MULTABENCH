# Curation Lab — Design

**Date:** 2026-08-28
**Context:** Technion NLP 097215 final project, Track 2 (Benchmark). Deadline 2026-10-26.
**Status:** Approved

## 1. Goal and scope

Curate new text-tabular dataset(s) and demonstrate, using this repository's own pipeline, that
they satisfy the MulTaBench curation criteria.

**Committed target:** one dataset with a genuine `passes()` verdict and the largest achievable
delta magnitudes (the Standard project; grading is driven by delta magnitude, not dataset count).

**Gated stretch target:** the Outstanding project — a systematic discovery method yielding ≥5
passing datasets. Designed here, built only if Phases 1–2 finish early.

### The criteria being reproduced

Five learners (TabM, CatBoost, LightGBM, TabPFN v2, TabPFN v2.5) × four conditions:

| Paper condition | CLI flag | CSV `multimodal_state` | Encoder |
|---|---|---|---|
| Unimodal Structured | `no_text` | `no_text` | — |
| Unimodal Unstructured | `txt` | `text_only` | frozen E5 |
| Joint Frozen | `all` | `all` | frozen E5 |
| Joint Target-Aware (TAR) | `ft` | `ft` | LoRA-tuned E5 |

A dataset passes if, for **≥3 of the 5 learners**:

- `Δ_Joint     = mean(all) − max(mean(no_text), mean(text_only)) > δ`
- `Δ_Awareness = mean(ft)  − mean(all)                          > δ`

with δ = 0.001 and per-state means over folds 0–4, rounded to 3 decimals before differencing.
Implemented once, in `multabench/leaderboard/analysis/pass_matrix.py::passes()`.

### Environment

- **Local:** Windows, CPU only (no CUDA; `nvidia-smi` absent).
- **Remote:** university/lab GPU cluster, SLURM-style, SSH access.
- Consequence: Δ_Joint (three frozen states) is CPU-computable locally; Δ_Awareness (LoRA
  fine-tuning) requires the cluster. The criteria decompose along the hardware boundary, and the
  funnel is built on that fact.

## 2. Architecture

**Governing constraint: `multabench/` is read-only.** The grade depends on faithfully reproducing
the paper's protocol; every edit there risks silently shifting scores. New code lives in a sibling
package and calls into the repo.

The entry seam is `evaluate_on_loaded_dataset()` (`multabench/baselines/benchmarks/evaluate.py:25`).
It accepts a constructed `MultimodalDataset` and performs subsampling, splitting, fitting and
scoring — everything `benchmark.py` does except the W&B requirement and the download path.
Entering there inherits the paper protocol (seeds, 90/10 split, 2000-row test cap, metric
selection) unchanged.

```
curation_lab/
  ingest/     spec.py loader.py parsers.py     # candidate → MultimodalDataset
  screen/     preflight.py typing_probe.py     # T1 gates, cheap and local
  runner/     run.py cache.py results.py       # T2/T3 execution → CSV rows
  criterion/  deltas.py report.py              # wraps pass_matrix.passes()
  discover/   openml.py kaggle.py rules.py     # T0 (Phase 3)
  package/    slurm.py                         # T3 cluster bundles
data/candidates/<NAME>/          # raw downloads + spec.yaml (gitignored)
results/candidates/<NAME>.csv    # run rows, existing schema
RESEARCH_NOTES.md                # Phase 2 research log
```

`benchmark.py` is used only in Phase 1, to prove our runner agrees with it. The analysis chain
(`committee_pool` → `pass_matrix` → `passes()`) is consumed unmodified, so our results are judged
by the same code that produced the paper's.

## 3. Candidate ingestion and registration

The repo requires an enum member plus a module in `multabench/datasets/annotated/`.
`curation_mapping.py:11-19` auto-imports every module in that package and **re-raises on failure**,
so a malformed generated file breaks the package for every dataset. We therefore add no files there.

Each candidate gets a **`spec.yaml`** in our tree. `ingest/loader.py` builds a `CuratedDataset` in
memory and injects it into the `CURATIONS` dict at runtime, reusing the repo's full curation logic
(`curate_dataset`: column drops, target extraction, value/type curation, null-target and
missing-image row removal, then `filter_by_multimodality`) with zero files added to `multabench/`.

Candidate IDs follow `{BIN|MUL|REG}_TEXT_{NAME}`. The prefix is load-bearing: `evaluate.py:63`
derives `task_type` from `name[:3]`.

```yaml
name: REG_TEXT_<THING>
source:   {kind: openml|kaggle|url|local, ref: ...}
loader:   {file: data.csv, sep: ",", encoding: utf-8, parser: default}
target:   {raw_name: price, task_type: REGRESSION}
cols_to_drop: [id, url]
features: [{raw_name: desc, feat_type: TEXT}]     # type overrides only
```

**Target choice is a search dimension, not a fixed field.** Which column is predicted dominates
delta magnitude. The repo supports `target_override` (including a `_discrete` suffix that bins a
numeric column into multiclass). The screen sweeps a shortlist of plausible targets per candidate
rather than committing to one. This is the highest-leverage knob for the grade.

`ingest/parsers.py` holds fallback readers — encoding failures, bad delimiters, ragged rows,
multi-file joins — each selected by name from the spec, so the failure mode stays recorded rather
than patched ad hoc.

For the **final submitted dataset only**, we additionally emit a real annotated module and enum
entry, matching the upstream contribution format the brief's co-authorship offer implies.

## 4. The funnel

Each tier is strictly cheaper than the one it protects.

| Tier | Where | Cost/candidate | Gate |
|---|---|---|---|
| **T0** Metadata | API | free | ≥2k rows; ≥1 string column; ≥3 non-text columns; usable license |
| **T1** Typing probe | local | seconds | four assertions below |
| **T2** Frozen screen | local CPU | ~12 runs | `Δ_Joint > 0` |
| **T3** Full sweep | cluster | ~100 runs | `passes()` at δ = 0.001 |

### T1 — where most candidates should die

Runs the repo's own detectors on a sample (`detect_numerical_features`,
`classify_semantic_features`, `detect_image_features`) rather than reimplementing heuristics, and
asserts:

1. ≥1 column is **detected as TEXT** (≥100 distinct values, or ≥80% unique ratio). A short
   free-text field silently becomes categorical and the dataset collapses into the structured
   condition.
2. ≥1 column **survives as structured** after text removal; otherwise `no_text` raises
   `MultimodalError` and there is no Δ_Joint to measure.
3. ≤5 multimodal columns (matches `MAX_MULTIMODAL` and the `no_pca` guard).
4. Target is viable: numeric and outlier-sane for regression; ≥2 classes with enough members to
   stratify for classification.

### T2 — CPU frozen screen

LightGBM and CatBoost only, over the three frozen states, folds 0–1. Computes Δ_Joint.

The gate is deliberately looser than the real criterion (`> 0`, not `> 0.001`): two folds are a
noisy estimate, and false negatives are the expensive mistake. A discarded good dataset is
unrecoverable; a passed-through bad one costs only cluster time.

### T3 — full sweep

Five models × four states × five folds on the cluster, minus legitimate TabPFN skips.

### Execution

`runner/run.py` takes `(candidate, target, model, state, fold)`, calls
`curate_dataset(multimodal_state=...)` then `evaluate_on_loaded_dataset()`, and appends one CSV row.
No W&B (`utils/logging.py::wandb_run` raises without credentials, which is hostile to unattended
batch jobs).

Two schema details are load-bearing for the analysis chain to consume our output unmodified:

- `multimodal_state` holds the CLI-style label (`no_text`/`text_only`/`all`/`ft`), **not** the
  `MultimodalState` enum's emoji string, which `pass_matrix._STATES` will not match.
- `model` holds the emoji `MODEL_NAME` that `committee_pool._MODEL_LABELS` maps.

**The embedding cache ships disabled.** Frozen E5 output is identical across all models and folds,
so caching raw embeddings per unique string is result-preserving (PCA still refits per fold). It is
implemented as a monkeypatch over `encode_texts_with_e5`, valid only when `tune_e5=False`, and is
enabled only after Phase 1 proves bit-exact agreement. If T2 is fast enough without it, it is
dropped.

`package/slurm.py` emits an sbatch **array job** — one index per `(candidate, model, state, fold)`
task — plus a fetch step. Requires the cluster's scheduler, partition name, and time limits;
resolved at Phase 3, not a blocker earlier.

## 5. Criterion evaluation

`criterion/deltas.py` reshapes runner output into the frame `passes()` expects and delegates. The
rule is never reimplemented. Two deliberately separate entry points:

- `screen_deltas()` — any fold count, returns raw Δ_Joint / Δ_Awareness, **no verdict**. Used by T2.
- `verdict()` — requires the full 5-fold × 4-state grid, delegates to `pass_matrix.passes()` and
  `build_pass_matrix()`. Accept(D) = ≥3 of 5 models.

The separation prevents a 2-fold screen from emitting anything resembling a verdict.

`passes()` asserts grid completeness (hardcoding the paper's single known gap in
`_KNOWN_MISSING_ROWS`), so any missing run of ours trips an opaque `AssertionError`. `verdict()`
therefore checks completeness first and reports which `(model, state, fold)` cells are absent,
turning a cryptic failure into a re-queue list.

`criterion/report.py` produces the per-dataset table of Δ_Joint and Δ_Awareness per model, the
3-of-5 verdict, and the magnitude figures that drive the grade.

## 6. Failure handling

The funnel must never confuse "rejected" with "broken".

| Signal | Meaning | Action |
|---|---|---|
| `MultimodalError` | state is degenerate for this dataset | T1/T2 rejection reason, not a crash |
| Parse/encoding failure | malformed source | route to `parsers.py` fallback; log the pattern |
| `is_invalid_model_dataset_pair` | TabPFN on high-multiclass | legitimate NaN cell; Accept treats as non-pass |
| Missing row after cluster run | job died | completeness check → re-queue that array index |

## 7. Validation

There is no test suite in this repository, so "verified" is defined concretely. Four steps,
cheapest first:

1. **Re-derive a published verdict** from the shipped `results/text/*.csv` using `passes()`,
   touching no models. Confirms the criterion is understood. Nearly free; done first.
2. **Protocol-faithfulness check:** run our runner on that same paper dataset (LightGBM, all four
   states, fold 0) and compare `test_score` against the shipped row for the identical key.
3. Extend to the full five-model grid if step 2 agrees.
4. Prove the embedding cache bit-exact, or drop it.

**Expectation on step 2, stated honestly:** the shipped CSVs are W&B exports from different
hardware and library versions. Frozen states are seeded and should agree tightly; `ft` involves
stochastic LoRA fine-tuning and may not reproduce exactly. Tolerance is therefore asymmetric —
tight for `no_text`/`text_only`/`all`, directional-only for `ft`. If the frozen states match and
`ft` lands in the same direction and rough magnitude, the pipeline is locked. Claiming bit-exact
`ft` reproduction would be overclaiming.

## 8. Phase plan

Every phase follows a two-stage loop: a detailed plan proposal (inputs, outputs, failure modes,
validation steps) reviewed before implementation, then implementation with checks run and results
logged.

**Phase 1 — Grounding, baseline validation, pipeline lock.**
Validation steps 1–3 above. Exit criterion: our runner reproduces a paper dataset's frozen scores
and its published verdict.

**Phase 2 — Exploration, discovery, research logging.**
T0/T1 applied by hand over 3–5 candidates. `RESEARCH_NOTES.md` records which queries and metadata
filters were predictive versus noisy, ingestion pitfalls per platform, and quantitative dataset
profiles. Then T2 → T3 on the strongest candidate. Exit criterion: **one dataset with a real
`passes()` verdict** — the Standard deliverable.

**Phase 3 — Automation (gated on Phase 2 finishing early).**
Codify the Phase 2 findings into `discover/rules.py` as deterministic filters, automate T0–T2,
build the SLURM packager, and push for ≥5 passing datasets.

## 9. Open questions

- Cluster specifics (scheduler, partition, time limits, module environment) — needed before
  `package/slurm.py`; resolved at Phase 3.
- Which paper dataset anchors Phase 1 — chosen during Phase 1 planning; should be text-based and
  small enough for CPU.
- Whether the TF-IDF proxy screen (`--e5_model tf-idf`) earns a place as a T1.5 accelerator. The
  repo ships paired TF-IDF and E5 results in `results/analysis_tfidf/`, so its rank correlation
  against ground truth can be measured before it is trusted. Deferred; adopted only if it holds.
