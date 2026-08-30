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
