# ACCEPTED — `REG_TEXT_EDU_UDEMY_ACADEMY`

**Verdict: ACCEPTED, 3 of 5 learners (quorum 3).** Measured twice, on two independent machines,
and accepted at 3 of 5 in both.

The primary evidence is the **Kaggle T4 grid** (`grid_gpu_*.csv`) — a complete 5 models x 4 states
x 5 folds = 100-cell grid in which every state for a given learner was measured in one session. The
earlier CPU grid (`grid.csv`) is retained as the cross-environment comparison, not as a superseded
file: the two lanes agree on the verdict but disagree about *which* three learners pass, and that
disagreement is itself a result (see "What the two lanes disagree about").

Both verdicts are computed by `multabench.leaderboard.analysis.pass_matrix.passes()` — the repo's
own implementation of the criterion, never reimplemented here.

## The dataset

| | |
|---|---|
| Source | Kaggle `mariahalshiekh/udemy-course-academy-teaching` |
| Registered as | `REG_TEXT_EDU_UDEMY_ACADEMY` |
| Task | regression |
| Target | `price` (28 distinct, \|z\|max 3.00, no extreme outliers) |
| Text columns | `course_name`, `course_instr` |
| Structured | course statistics, 6 numeric columns |
| Rows | 2,999 |
| Leakage columns dropped | none detected |

Domain: online course listings. No course/MOOC dataset exists in MulTaBench, so this is a genuinely
new domain rather than a re-slug of an existing one.

The curation spec was derived automatically by `curation_lab/screen/auto_spec.py`; there is no
hand-authored `annotated/` module. Leakage columns are dropped by a Spearman >= 0.95 rule against
the target.

## Result — Kaggle T4 lane (primary)

Per-state means over folds 0-4, rounded to 3 decimals before differencing, delta = 0.001:

| model | no_text | text_only | all | ft | Delta_Joint | Delta_Awareness | verdict |
|---|---|---|---|---|---|---|---|
| CatBoost | 0.297 | 0.210 | 0.480 | 0.501 | +0.183 | +0.021 | **PASS** |
| LightGBM | 0.261 | 0.212 | 0.479 | 0.474 | +0.218 | -0.005 | fail |
| TabM | 0.298 | 0.221 | 0.504 | 0.492 | +0.206 | -0.012 | fail |
| TabPFN-2.5 | 0.311 | 0.211 | 0.452 | 0.464 | +0.141 | +0.012 | **PASS** |
| TabPFNv2 | 0.299 | 0.162 | 0.435 | 0.437 | +0.136 | +0.002 | **PASS** |

3 of 5 pass. Quorum is 3. **ACCEPTED.**

100 cells, 5 in every (model, state) bucket, no gaps — `passes()` asserts completeness rather than
averaging over it, and did not raise. Both kernels ran on one Tesla T4 (compute capability 7.5),
`ft` at 10 epochs, zero failed cells and zero `MultimodalError` skips.

## What the two lanes disagree about

| model | Delta_Joint CPU | Delta_Joint T4 | Delta_Awareness CPU | Delta_Awareness T4 | flip |
|---|---|---|---|---|---|
| CatBoost | +0.194 | +0.183 | +0.010 | +0.021 | no (pass -> pass) |
| LightGBM | +0.209 | +0.218 | +0.006 | **-0.005** | **pass -> fail** |
| TabM | +0.208 | +0.206 | -0.007 | -0.012 | no (fail -> fail) |
| TabPFN-2.5 | +0.140 | +0.141 | +0.016 | +0.012 | no (pass -> pass) |
| TabPFNv2 | +0.136 | +0.136 | -0.001 | **+0.002** | **fail -> pass** |

**Delta_Joint is stable across environments** — maximum drift 0.011, and TabPFNv2 reproduces
identically to three decimals. **Delta_Awareness is not.** Two learners flipped, in opposite
directions, and they cancel: CPU passed on CatBoost/LightGBM/TabPFN-2.5, the T4 passes on
CatBoost/TabPFNv2/TabPFN-2.5. Same verdict, different membership.

Neither flip is a measurement of anything: both are sub-0.011 moves against a threshold of 0.001.
In the T4 lane only CatBoost (+0.021) and TabPFN-2.5 (+0.012) sit clearly above environment noise.
**The honest reading is that this dataset passes 3 of 5 in both lanes, and that no individual
learner's TAR verdict on it is stable.** Cite the dataset-level verdict; do not cite a per-learner
Delta_Awareness as though it were reproducible.

**Provenance control.** `no_text` means are identical to three decimals across the two lanes for
all five learners (0.297 / 0.261 / 0.298 / 0.311 / 0.299). That state never touches E5, so the
match shows loading, curation, the 90/10 split, the seeding and the learners themselves are
reproducing exactly across machines; divergence enters only where text embedding does, and it is
largest in `all` (CatBoost 0.491 -> 0.480). This is the same signature the Vietnam housing grid
showed, and it is why both halves of a delta must be measured in one environment.

## Why this is not a marginal pass

**Delta_Joint is enormous on all five learners** — +0.136 to +0.218 against a delta of 0.001, i.e.
**136x to 218x the threshold**, in both lanes. It is a real complementarity result rather than an
artifact of a degenerate condition: both unimodal baselines are non-degenerate (`no_text` ~0.30,
`text_only` ~0.21), so `all` genuinely beats *each* modality alone rather than winning because one
of them collapsed.

Delta_Awareness is the narrow margin, as it is throughout this project — TAR gain is the harder of
the two criteria and is what rejected most of the paper's 56-dataset pool.

## Deviation to disclose

E5 fine-tuning ran **10 epochs** rather than the `E5TrainArgs` default of 50 (patience 3). This was
originally a CPU-feasibility compromise; the T4 grid kept 10 epochs so the two lanes stay
comparable.

The deviation is conservative in the direction that matters. Delta_Awareness grew monotonically
with the epoch budget in our measurements — LightGBM fold 0 went **+0.0099 at 2 epochs to +0.0322
at 10** — so the paper's full budget would be expected to *widen* the margin, not narrow it. Now
that the Kaggle lane is established and a full grid costs under an hour of T4 time, a 50-epoch
re-run is affordable and would sharpen the per-learner picture the two lanes currently disagree on.

## About `grid_epochs2_superseded.csv`

That file is the **epochs=2** sweep, kept deliberately. It rejected this dataset at **1 of 5**,
with Delta_Awareness between -0.010 and +0.004 — and that rejection was an artifact of the epoch
budget, not a property of the dataset. An under-trained LoRA adapter barely moves the encoder, so
`ft` ~= `all` and Delta_Awareness collapses to noise around zero by construction. A PASS at 2
epochs would have survived at 50; a FAIL at 2 epochs proves nothing.

The file is retained as the measured evidence behind
[`docs/findings/03-methodological-findings.md`](../../../../docs/findings/03-methodological-findings.md):
a cheap screen is only valid where cheapness does not change the quantity being screened — true
for Delta_Joint (frozen encoders), false for Delta_Awareness.

Its Delta_Joint figures (0.136-0.209) match the CPU grid's on four of five models, as they should:
the frozen states do not depend on the epoch budget. TabPFN-2.5 reads 0.138 here against 0.140 in
the CPU grid, and the cause is a data artifact rather than a real difference — this file has 101
rows because it carries an exact duplicate of the TabPFN-2.5 / `all` / fold-4 row, which
double-weights that fold in the mean. `deltas.normalize` does not de-duplicate (the Kaggle loader
does), so read this file with `.drop_duplicates(subset=['model','state','fold'])`.

## Reproduce

The pipeline runs on **Kaggle GPU**. Two pushes, split by model so that all four states for a given
learner share one session:

```bash
python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full --full-epochs 10 \
  --candidate "mariahalshiekh/udemy-course-academy-teaching=REG_TEXT_EDU_UDEMY_ACADEMY" \
  --folds 0,1,2,3,4 --models light,cat,tabm --states no_text,text_only,all,ft \
  --kernel-id talkraicer/multabench-udemy-lct

python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full --full-epochs 10 \
  --candidate "mariahalshiekh/udemy-course-academy-teaching=REG_TEXT_EDU_UDEMY_ACADEMY" \
  --folds 0,1,2,3,4 --models tabpfnv2,tabpfnv2p5 --states no_text,text_only,all,ft \
  --kernel-id talkraicer/multabench-udemy-pfn

python -m curation_lab.kaggle.verdict_from_runs \
  results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/grid_gpu_*.csv
```

Cost: **~0.65 GPU-hours** of measured session time for all 100 cells (~0.7-0.8 h billed including
container boot and output save), roughly 2-3% of the weekly Kaggle quota. `tar_cache` collapsed the
fine-tunings as designed — 15 `ft` cells needed only 5 fine-tunes, 10 needed 5 — because within a
fold the learners differ only in `USE_VAL_SPLIT`.

The superseded CPU grid is reproduced with `curation_lab.screen.verify` (see git history of this
file); it is not the recommended path for new work.

## Files in this folder

| file | schema | what it is |
|---|---|---|
| `grid_gpu_light_cat_tabm.csv` | kaggle | **Primary.** T4, `light,cat,tabm` x 4 states x folds 0-4 (60 cells) |
| `grid_gpu_tabpfn.csv` | kaggle | **Primary.** T4, `tabpfnv2,tabpfnv2p5` x 4 states x folds 0-4 (40 cells) |
| `grid.csv` | cpu | The earlier CPU grid at epochs=10; retained as the cross-environment comparison |
| `grid_epochs2_superseded.csv` | cpu | The epochs=2 sweep; superseded, retained as method evidence |
| `logs/kaggle_udemy_lct_gpu.log` | — | T4 kernel log, `light,cat,tabm` |
| `logs/kaggle_udemy_pfn_gpu.log` | — | T4 kernel log, both TabPFNs |
| `logs/kaggle_udemy_smoke_gpu.log` | — | 1-fold GPU smoke run that validated the path |
| `logs/verify_udemy_e10.log` | — | run log for the CPU grid |
| `logs/verify_udemy_frozen.log` | — | CPU frozen-states run log |
| `logs/verify_udemy_ft.log` | — | CPU `ft` run log |
| `logs/t3_udemy.log`, `logs/t3_udemy2.log`, `logs/t3_udemy_e20.log` | — | earlier T3 TAR probe logs |
