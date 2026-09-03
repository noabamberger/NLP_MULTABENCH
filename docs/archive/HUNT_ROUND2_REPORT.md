> **Superseded 2026-09-02.** Replaced by `docs/findings/03-methodological-findings.md` and
> `results/curation/rejected/`.
> This file's four findings (fold-0 Delta_Awareness screens are invalid, joint signal is common but
> awareness is not, two junk-spec classes, and the local pool being thinner than it looks) are
> folded into the consolidated methodological findings; its per-candidate results now live under
> the per-dataset rejection folders.
> Kept verbatim below: the current document was written from it, and the
> judgment calls in that rewrite should stay checkable against this source.

# Second hunt round — screening the local candidate pool for a passing dataset

Goal: find more datasets that pass the full curation criterion, starting from the
candidates already explored locally. Everything below was measured on Kaggle T4 GPUs
with the corrected `auto_spec`, all four states on one machine, `ft` at 10 epochs.

**Outcome: no new dataset passes.** Eight candidates were screened, two were promoted
to a full grid, and both were rejected. The round's real value is three findings that
change how the next round should be run.

## Result table

Fold-0 screen (LightGBM + CatBoost, all four states):

| candidate | Delta_Joint | Delta_Awareness | outcome |
|---|---|---|---|
| board-games | +0.035 / +0.036 | +0.016 / +0.008 | promoted -> **REJECTED** (2/5) |
| anime popularity | +0.034 / +0.034 | +0.002 / +0.003 | promoted -> **REJECTED** (0/5) |
| horror-movie budget | +0.055 / +0.093 | -0.001 / -0.013 | fails Delta_Awareness |
| metacritic PC | +0.022 / +0.035 | -0.006 / -0.013 | fails Delta_Awareness |
| chess (women) | +0.003 / +0.015 | -0.004 / -0.008 | fails Delta_Awareness |
| vgsales user score | +0.001 / +0.004 | +0.001 / +0.002 | every delta at noise scale |
| kerala liquor | -0.001 / 0.000 | 0.000 | degenerate, `no_text` R^2 = 1.000 |
| global movies ROI | 0.000 | 0.000 | degenerate, `no_text` R^2 = 0.997 |

Full grids (4 models x 4 states x 5 folds, no gaps):

| board-games | no_text | text_only | all | ft | D_Joint | D_Awareness |
|---|---|---|---|---|---|---|
| LightGBM | 0.569 | 0.477 | 0.616 | 0.619 | +0.047 | +0.003 |
| TabM | 0.576 | 0.504 | 0.628 | 0.629 | +0.052 | +0.001 |
| CatBoost | 0.576 | 0.490 | 0.623 | 0.623 | +0.047 | 0.000 |
| TabPFNv2 | 0.585 | 0.504 | 0.640 | 0.639 | +0.055 | -0.001 |

2 of 5 pass, quorum is 3 -> REJECTED. And TabM's +0.001 is the same float knife-edge
documented for TabPFNv2 in `DJ_PROPERTY_TAR_REPORT.md`: the difference of the rounded
means is exactly delta and clears `>` only through float64 representation. Counting
honestly, board-games has **one** passing model.

Anime is cleaner cut: -0.001, 0.000, -0.002, -0.002 -> 0 of 5.

## Finding 1: a fold-0 Delta_Awareness screen is invalid

This is the same class of error as the `epochs=2` artifact already recorded in
`RESUME.md` — a cheap screen that changes the quantity being screened.

| board-games | cat | light | tabm | tabpfnv2 |
|---|---|---|---|---|
| fold 0 only | +0.0074 | +0.0163 | +0.0052 | +0.0010 |
| 5-fold mean | -0.0003 | +0.0021 | +0.0010 | -0.0015 |

Every model dropped, and the fold-0 reading pointed the opposite way for two of them.
The per-(model, fold) spread is **sigma = 0.0063**, over the range [-0.0124, +0.0163]
— roughly **6x the delta threshold of 0.001**. A single fold simply cannot resolve a
criterion whose threshold sits far inside its own noise band.

Delta_Joint does not have this problem: it is +0.03 to +0.09, an order of magnitude
above the same noise, which is why the cheap frozen screen remains sound for it.

**Consequence for the next round:** screen Delta_Awareness on all 5 folds, or at
minimum 3, and treat any |Delta_Awareness| under ~0.01 at fold 0 as unresolved rather
than as a pass. Screening one fold cost two full grids here.

## Finding 2: joint signal is common; target-awareness is not

Five of the eight candidates had a healthy Delta_Joint and then failed
Delta_Awareness. Combined with the accepted datasets, the pattern is consistent:
getting `all` above both unimodal baselines is comparatively easy, while making a
LoRA-tuned encoder beat a frozen one is the binding constraint.

That inverts the ordering assumption behind the current pipeline, which spends its
cheap frozen screen on Delta_Joint and reaches Delta_Awareness last. Since GPU makes
TAR affordable, the productive order is now the reverse: probe Delta_Awareness on a
few folds early, because it eliminates far more candidates.

## Finding 3: two junk-spec classes inflated the old Delta_Joint numbers

Both are fixed in `auto_spec.py`; see that commit for detail.

- **Identifier and serial targets.** `Sl No` (kerala) and `appid` (steam) were chosen
  as regression targets, which made their Delta_Joint meaningless. With the corrected
  target, kerala collapses entirely: `Special Fee` is a deterministic function of the
  structured columns, `no_text` scores R^2 = 1.000. A large delta measured against a
  junk target says nothing about the dataset.
- **Multi-column arithmetic leakage — still unfixed.** `roi_pct` in the movies dataset
  is revenue/budget, and both are structured columns, so `no_text` reaches 0.997.
  `find_leaks` only drops single columns that are near-copies of the target by rank
  correlation, and no individual column here is one. This class of leak is invisible
  to the current check.

A cheap guard for it, not yet implemented: flag any candidate whose `no_text` R^2
exceeds ~0.95 as saturated, since at that point no text signal can be demonstrated
regardless of the deltas.

## Finding 4: the local pool is thinner than it looks

`t1_batch.csv` reports 120 viable candidates, 86 unscreened. Re-scoring them with the
junk filter shows T1's `n_text` badly overcounts, because the profiler never applied
it: the top-ranked novel candidates have **no genuine text column at all** — their
only "text" is `Purchase Date`, `Record_ID`, `session_id`, `timestamp`. Two of the
better-looking ones (`global-restaurant-delivery-intelligence`, 62k rows;
`multicategory-electronics-product-new-egg`) produce no spec whatsoever.

After novelty and shape filtering, the local pool yields a handful of candidates, all
of which are now screened. **It is exhausted.**

## Recommended next step

A fresh Kaggle search (T0/T1) with the junk-aware profiler. The previous search
ranked candidates on a text-column count that counted dates and ids, so it was
scouting against a broken filter — the pool was never as rich as it appeared, and a
corrected pass would explore genuinely different ground.

Reproduce anything here with:

```bash
python -m curation_lab.screen.audit_specs --from-hunt results/candidates/hunt_full.csv
python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full \
  --candidate "<owner/slug>=REG_TEXT_<NAME>" --folds 0,1,2,3,4 \
  --models light,cat,tabm,tabpfnv2 --states no_text,text_only,all,ft
python -m curation_lab.kaggle.verdict_from_runs results/candidates/<file>.csv
```

Raw rows: `boardgames_full.csv`, `anime_full.csv`, `screen4_fold0.csv`,
`screen_wave2_fold0.csv`. Spec audits: `spec_audit2.csv`, `spec_audit_wave*.csv`.
