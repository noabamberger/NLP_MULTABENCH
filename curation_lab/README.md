# `curation_lab/`

This project's own code. Everything under `multabench/` is the upstream benchmark and is
read-only here; everything in this package is written for Technion 00970215 Track 2 — find
text-tabular datasets, screen them cheaply, measure the full grid on a GPU, and let the
benchmark's own criterion return the verdict.

One rule governs the whole package: **the curation criterion is never reimplemented.**
`criterion/deltas.py::verdict` delegates to
`multabench.leaderboard.analysis.pass_matrix.passes()`. Every path that emits a pass or a fail
goes through that one call, so a verdict here and a verdict in the upstream analysis layer
cannot disagree.

## The funnel

A candidate is cheap to reject early and expensive to reject late, so the stages are ordered by
cost. Measured yields from the 232-candidate search are in
[`docs/findings/02-mining-method-rules.md`](../docs/findings/02-mining-method-rules.md).

| tier | question | cost | modules |
|---|---|---|---|
| T0 | does this Kaggle dataset look text-tabular at all? | free, metadata only | `discover/kaggle_search.py`, `discover/novelty.py` |
| T1 | do its columns actually *type* as text under the benchmark's own detector? | seconds | `screen/profile.py`, `screen/batch_profile.py` |
| T1.5 | is the derived spec sane — real target, no leak, structured block intact? | seconds | `screen/auto_spec.py`, `screen/audit_specs.py` |
| T2 | is Δ_Joint positive? | minutes, CPU | `screen/hunt.py`, `screen/t2_screen.py` |
| T3 | is Δ_Awareness positive? | hours, GPU | `screen/t3_tar.py`, `kaggle/` |
| verdict | does the committee reach quorum? | instant | `criterion/deltas.py` |

T2 exploits the shape of the criterion. Δ_Joint is `mean(all) − max(mean(no_text),
mean(text_only))`, so if `all` loses to `no_text` the candidate is dead whatever `text_only`
does. `hunt.py` runs `no_text` first, then `all`, and aborts before spending the second encode.

## Package map

### `criterion/` — the verdict

| module | role |
|---|---|
| `deltas.py` | the only verdict surface. `screen_deltas()` returns raw deltas from any fold count and refuses to call them a verdict; `verdict()` demands a complete 5-fold × 4-state grid and delegates to `passes()`. |
| `significance.py` | a paired fold-level view alongside the official statistic, so a marginal +0.002 and a decisive +0.2 stop looking alike. Additive; it never replaces the official number. |
| `fidelity.py` | compares this runner's scores against the shipped paper results. Frozen states are asserted, `ft` is reported but never asserted — it is stochastic LoRA on different hardware. |

### `discover/` + `ingest/` — getting a candidate in

`discover/kaggle_search.py` ranks search results on the encoding-cost model
(`rows × text_columns × tokens`). `discover/novelty.py` filters out domains MulTaBench already
covers, matched on domain words rather than slugs, because the benchmark holds wine,
restaurants, used cars and more under slugs a naive exclusion list never sees.

`ingest/candidate.py` is the reason `multabench/` stays untouched. The normal registration path
needs an enum member plus a module in `multabench/datasets/annotated/`, and one malformed
generated file there breaks dataset loading for the entire repo. This builds a `CuratedDataset`
in memory and injects it into `CURATIONS` at runtime instead — still running the repo's full
curation path, just without writing into it.

### `screen/` — the cheap tiers

`auto_spec.py` derives a spec from a raw CSV by deterministic rule, which is what lets one
codebase screen fifty candidates instead of one. Two hand-built registries exist where a rule
cannot reach: `media_specs.py` (a sentinel target needing a *row* filter, which `CandidateSpec`
cannot express) and `games_specs.py` (parsed targets, plus structured columns the junk regex
would wrongly delete). `verify_spec.py` and `verify_games.py` are thin drivers over those two
registries; both delegate the grid loop to `verify.py`, so there is one implementation of the
loop, the cache wiring and the resumable skip.

`verify.py` is the full local grid: 5 models × 4 states × 5 folds, frozen states first so a
candidate that loses Δ_Joint costs no `ft` time.

### `runner/` — executing one cell

`run.py` enters the pipeline at `evaluate_on_loaded_dataset()`, so the paper's protocol — seeds,
90/10 split, 2000-row test cap, metric choice — is inherited rather than restated. It never
calls `wandb_run()`, which is what would demand credentials; the `wandb` package is still
imported transitively and must be installed.

`cache.py` and `tar_cache.py` are the two speedups that made CPU work feasible, both installed
by monkeypatching a single upstream function and both result-preserving by construction:

- `cache.py` caches frozen E5 output per `(model, column, text)`. **Valid only when `tune_e5`
  is False** — a tuned encoder returns different vectors while reporting the same model name.
  PCA is still refit per fold from the cached vectors.
- `tar_cache.py` caches the fine-tuned encoder across processes, keyed on a hash of every
  hyperparameter including `epochs`. That key is what makes a 50-epoch run unable to be served a
  10-epoch adapter.

`results.py` owns the row schema. Two fields are load-bearing for the upstream analysis layer to
read this output unmodified: `multimodal_state` carries the CLI flag (`no_text` / `text_only` /
`all` / `ft`), not the enum value, and `model` carries the emoji `MODEL_NAME`.

### `kaggle/` — the GPU lane, and the pipeline of record

Every grid behind an accepted dataset was measured here. `build_notebook.py` generates the
notebook rather than checking in an `.ipynb`, because the notebook was iterated against real
Kaggle runs and hand-editing JSON with embedded source strings is where mistakes hide.

`push_code.py` republishes the Python tree as a new version of the `noabamberger/multabench-code`
dataset. The notebook imports from that dataset, not from a clone, so **a local change is
invisible until this runs, and the run silently uses stale code.**

`verdict_from_runs.py` stitches the per-push CSVs into one grid and applies the criterion.
`compare_environments.py` answers cross-machine questions by breaking results down by state:
`no_text` touches no encoder, so if it matches exactly, any divergence elsewhere has to come from
the encoder.

### `prep/` + `tools/`

`prep/mtg_cards.py` repairs the MTG source: a packed price string, a log-normal target, and two
columns that pack numerics and a date into text. `tools/manifest.py` is the content-addressed
custody check over the evidence tree — it proves nothing was destroyed, not that anything is
where it belongs.

## Running it

The pipeline runs on Kaggle GPU. Re-version the code dataset first, split the push by model so
all four states for a learner share one session, and never go below 10 epochs.

```bash
python -m curation_lab.kaggle.push_code -m "why this push"

python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full --full-epochs 10 \
  --candidate "<owner/slug>=REG_TEXT_<NAME>" --folds 0,1,2,3,4 \
  --models light,cat,tabm --states no_text,text_only,all,ft \
  --kernel-id noabamberger/multabench-<name>-lct

python -m curation_lab.kaggle.verdict_from_runs results/curation/<path>/grid_gpu_*.csv
```

Local CPU is for Δ_Joint-only work and reproduction:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m curation_lab.screen.verify \
  --ref <owner/slug> --name REG_TEXT_<NAME> \
  --out results/curation/<path>.csv --folds 0,1,2,3,4 --epochs 10
```

Environment rules are in [`CLAUDE.md`](../CLAUDE.md): `.venv/Scripts/python.exe` only, pandas
2.3.3, `PYTHONIOENCODING=utf-8` always.

## Tests

`pytest tests` covers the places where a silent error would corrupt a measured number: the
criterion's delta arithmetic, the result schema, both caches' bit-exactness and key sensitivity,
and the manifest.

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests -q
```

The suite passes clean.
