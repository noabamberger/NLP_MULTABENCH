# ACCEPTED — `REG_TEXT_EDU_UDEMY_ACADEMY`

**Verdict: ACCEPTED, 3 of 5 learners (quorum 3).**
Complete 5 models x 4 states x 5 folds grid, no missing cells. The verdict is computed by
`multabench.leaderboard.analysis.pass_matrix.passes()` — the repo's own implementation of the
criterion, never reimplemented here.

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

Domain: online course listings. No course/MOOC dataset exists in MulTaBench, so this is a
genuinely new domain rather than a re-slug of an existing one.

The curation spec was derived automatically by `curation_lab/screen/auto_spec.py`; there is no
hand-authored `annotated/` module. Leakage columns are dropped by a Spearman >= 0.95 rule against
the target.

## Result

Per-state means over folds 0-4, rounded to 3 decimals before differencing, delta = 0.001:

| model | Delta_Joint | Delta_Awareness | verdict |
|---|---|---|---|
| CatBoost | 0.194 | 0.010 | **PASS** |
| LightGBM | 0.209 | 0.006 | **PASS** |
| TabM | 0.208 | -0.007 | fail |
| TabPFN-2.5 | 0.140 | 0.016 | **PASS** |
| TabPFNv2 | 0.136 | -0.001 | fail |

3 of 5 pass. Quorum is 3. **ACCEPTED.**

Evidence: `grid.csv` (100 rows, cpu schema).

## Why this is not a marginal pass

**Delta_Joint is enormous on all five learners** — +0.136 to +0.209 against a delta of 0.001, i.e.
**136x to 209x the threshold**. It is a real complementarity result rather than an artifact of a
degenerate condition: both unimodal baselines are non-degenerate (`no_text` ~0.30, `text_only`
~0.20), so `all` genuinely beats *each* modality alone rather than winning because one of them
collapsed.

Delta_Awareness is the narrower margin (+0.006 to +0.016 on the three passing learners), which
matches the paper's own experience — TAR gain is the harder of the two criteria, and it is what
rejected most of the paper's 56-dataset pool.

The split is interpretable: the three passing learners are LightGBM, CatBoost and TabPFN-2.5;
TabM and TabPFNv2 fail on TAR while still showing large Delta_Joint.

## Deviation to disclose

E5 fine-tuning ran **10 epochs** rather than the `E5TrainArgs` default of 50 (patience 3), for CPU
feasibility.

This is conservative in the direction that matters. Delta_Awareness grew monotonically with the
epoch budget in our measurements — LightGBM fold 0 went **+0.0099 at 2 epochs to +0.0322 at 10** —
so the paper's full budget would be expected to *widen* the margin, not narrow it. The result
should be re-run at 50 epochs if GPU time becomes available.

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

Its Delta_Joint figures (0.136-0.209) match the accepted grid's on four of five models, as they
should: the frozen states do not depend on the epoch budget. TabPFN-2.5 reads 0.138 here against
0.140 in the accepted grid, and the cause is a data artifact rather than a real difference — this
file has 101 rows because it carries an exact duplicate of the TabPFN-2.5 / `all` / fold-4 row,
which double-weights that fold in the mean. `deltas.normalize` does not de-duplicate (the Kaggle
loader does), so read this file with `.drop_duplicates(subset=['model','state','fold'])`, as the
reproduce command below does.

## Reproduce

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m curation_lab.screen.verify \
  --ref mariahalshiekh/udemy-course-academy-teaching \
  --name REG_TEXT_EDU_UDEMY_ACADEMY \
  --out results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/grid.csv \
  --folds 0,1,2,3,4 --epochs 10
```

## Files in this folder

| file | schema | what it is |
|---|---|---|
| `grid.csv` | cpu | the accepted grid: 5 models x 4 states x 5 folds, epochs=10 |
| `grid_epochs2_superseded.csv` | cpu | the epochs=2 sweep; superseded, retained as method evidence |
| `logs/verify_udemy_e10.log` | — | run log for the accepted grid |
| `logs/verify_udemy_frozen.log` | — | frozen-states run log |
| `logs/verify_udemy_ft.log` | — | `ft` run log |
| `logs/t3_udemy.log`, `logs/t3_udemy2.log`, `logs/t3_udemy_e20.log` | — | earlier T3 TAR probe logs |
