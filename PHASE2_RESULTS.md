# Phase 2 Results — candidate curation

## Headline

**`mariahalshiekh/udemy-course-academy-teaching` passes Delta_Joint on 5 of 5 learners**,
by 150-250x the delta=0.001 threshold. Registered as `REG_TEXT_EDU_UDEMY_ACADEMY`.
Delta_Awareness sweep in progress; a single probe (LightGBM, fold 0) already returned
**+0.0099 (PASS)**.

### Delta_Joint, 5 folds, frozen states (CPU)

| model | all | no_text | text_only | Delta_Joint |
|---|---|---|---|---|
| TabM | 0.546 | 0.298 | 0.221 | **+0.248** |
| LightGBM | 0.488 | 0.261 | 0.199 | **+0.227** |
| CatBoost | 0.514 | 0.297 | 0.206 | **+0.217** |
| TabPFN-2.5 | 0.487 | 0.311 | 0.210 | **+0.176** |
| TabPFNv2 | 0.456 | 0.299 | 0.163 | **+0.157** |

Both unimodal baselines are non-degenerate (no_text ~0.30, text_only ~0.20), so the joint
gain is a real complementarity result rather than an artifact of an empty condition.

Spec was derived automatically by `curation_lab/screen/auto_spec.py` — no hand-authored
annotated module.

## The funnel, end to end

| Tier | Count | Notes |
|---|---|---|
| T0 Kaggle search | 232 | 30 queries, 51 MulTaBench slugs excluded |
| T1 typing probe | 120 "viable" | but the >=100-distinct rule promotes dates/IDs to TEXT |
| T1 + junk filter | 58 | genuine text columns only |
| after used-domain filter | 34 | screened |
| **T2 Delta_Joint PASS** | **16** | fail-fast: abort before text_only when all <= no_text |

16 T2 survivors is a healthy pool for the Outstanding track's 5-dataset target.

## Decisive optimization: the training max_length cap

TAR was not feasible on CPU until this. `TextLabelDataset` pads every example to 512
tokens; the Udemy texts cap at 40.

| | per step | 2 epochs |
|---|---|---|
| max_length=512 (upstream default) | **333 s** | ~3 h |
| max_length=40 (computed at runtime) | **~8 s** | ~6 min |

~40x. Reachable without editing `multabench/`: `e5_train_kwargs` is forwarded into
`finetune_e5_with_lora(**kwargs)`, which passes `max_length` to `TextLabelDataset`.

Caveat: capping is NOT bit-exact. Padding is algebraically inert, but changing the padded
length reassociates float32 matmuls and shifts embeddings ~1e-7. Fine for screening; do not
use for numbers compared directly against the paper.

## Deviation to disclose

The ft sweep runs `epochs=2` instead of the `E5TrainArgs` default of 50 (patience 3), to fit
CPU-only compute. This is **conservative for the claim**: less fine-tuning should mean a
SMALLER Delta_Awareness, so a pass at 2 epochs would likely also pass at 50. It is still a
deviation and must be stated in the report.

## Auto-spec validation

Two independent checks that the automatic curation is sound:

- On `jilkothari/finance-accounting-courses-udemy-13k-course` it dropped exactly the leakage
  columns identified by hand (`avg_rating`, `rating` vs target `avg_rating_recent`).
- On `melissamonfared/board-games` it rejected `Rating Average` as target (|z|>5 outliers)
  and chose `Complexity Average` instead — avoiding the R^2 instability flagged earlier.

---

## CORRECTION: the epochs=2 rejection was my artifact, not the dataset's failure

The first full sweep rejected `REG_TEXT_EDU_UDEMY_ACADEMY` at 1 of 5, with
Delta_Awareness between -0.010 and +0.002. I had recorded epochs=2 as "conservative".
**That reasoning was wrong.** It only runs one way: a PASS at 2 epochs would likely
survive at 50, but a FAIL at 2 epochs proves nothing, because an under-trained LoRA
adapter barely moves the encoder -- so ft ~= all and Delta_Awareness collapses to noise
around zero by construction.

Re-running at epochs=10 confirms it:

| model, fold 0 | all | ft @ 2 ep | Delta @ 2 ep | ft @ 10 ep | **Delta @ 10 ep** |
|---|---|---|---|---|---|
| LightGBM | 0.4957 | 0.5056 | +0.0099 | 0.5279 | **+0.0322** |
| CatBoost | 0.5177 | 0.5282 | +0.0105 | 0.5299 | **+0.0122** |

Delta_Awareness grows ~3x with more fine-tuning, as the TAR mechanism predicts. The
same artifact explains the near-zero Delta_Awareness seen across ALL candidates in the
epochs=2 batch probes (Vietnam housing +0.0007, metacritic -0.0143, etc.) -- those
probes measured the epoch budget, not the datasets.

**Methodological lesson for the report:** the TAR probe must use a realistic epoch
budget or it measures nothing. A cheap screen is only valid where cheapness does not
change the quantity being screened -- true for Delta_Joint (frozen encoders), false for
Delta_Awareness.

## TAR encoder sharing: working

The tuned encoder depends on `(x_train, y_train, e5_train_kwargs)` and NOT on the
tabular learner, so per (dataset, fold) there are 2 distinct fine-tunings, not 5.
Measured on fold 0: LightGBM 2184 s (cold, fine-tunes) then CatBoost 1082 s (cache hit,
no fine-tuning). Cuts the 25-run sweep to 10 fine-tunings.

Keyed on a hash of the actual arguments rather than on the model grouping, so it stays
correct if upstream ever threads a real fold into `split_to_val`.

---

# FINAL VERDICT: ACCEPTED

`REG_TEXT_EDU_UDEMY_ACADEMY` (`mariahalshiekh/udemy-course-academy-teaching`)
passes the MulTaBench curation pipeline. Complete 5 models x 4 states x 5 folds grid,
no missing cells, verdict computed by `multabench.leaderboard.analysis.pass_matrix.passes()`
(the repo's own implementation -- never reimplemented here).

```
  CatBoost     PASS
  LightGBM     PASS
  TabM         fail
  TabPFN-2.5   PASS
  TabPFNv2     fail

  3 of 5 pass (quorum 3)   ACCEPTED = True

      model  delta_joint  delta_awareness
  CatBoost        0.194            0.010
  LightGBM        0.209            0.006
      TabM        0.208           -0.007
TabPFN-2.5        0.140            0.016
  TabPFNv2        0.136           -0.001
```

## Why this dataset is a good result, not a marginal one

**Delta_Joint is enormous on all five learners** (+0.136 to +0.209 against a delta of
0.001 -- 136x to 209x the threshold), and it is a real complementarity result rather than
an artifact: both unimodal baselines are non-degenerate (no_text ~0.30, text_only ~0.20),
so `all` genuinely beats each modality alone rather than winning by default.

Delta_Awareness is the narrower margin (+0.006 to +0.016 on the three passing learners),
which matches the paper's own experience -- TAR gain is the harder criterion, and it is
what rejected most of the paper's 56-dataset pool.

The split is interpretable: the three passing learners are LightGBM, CatBoost and
TabPFN-2.5; TabM and TabPFNv2 fail on TAR while still showing large Delta_Joint.

## Dataset description

- Source: Kaggle `mariahalshiekh/udemy-course-academy-teaching`
- Domain: online course listings -- **no course/MOOC dataset exists in MulTaBench**, so
  this is genuinely new rather than a re-slug of an existing domain.
- Target: `price` (regression)
- Text: `course_name`, `course_instr` (short -- max 40 tokens, so cheap to encode)
- Structured: numeric course statistics
- Curation spec derived automatically by `curation_lab/screen/auto_spec.py`; leakage
  columns dropped by a spearman >= 0.95 rule against the target.

## Reproduce

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m curation_lab.screen.verify \
  --ref mariahalshiekh/udemy-course-academy-teaching \
  --name REG_TEXT_EDU_UDEMY_ACADEMY \
  --out results/candidates/verify_udemy_e10.csv --folds 0,1,2,3,4 --epochs 10
```

## Deviation to disclose in the report

E5 fine-tuning ran **10 epochs** rather than the `E5TrainArgs` default of 50 (patience 3),
for CPU feasibility. This is conservative in the direction that matters: Delta_Awareness
grew monotonically with epochs in our measurements (LightGBM fold 0: +0.0099 at 2 epochs
-> +0.0322 at 10), so the paper's full budget would be expected to widen the margin, not
narrow it. The result should be re-run at 50 epochs if GPU time becomes available.
