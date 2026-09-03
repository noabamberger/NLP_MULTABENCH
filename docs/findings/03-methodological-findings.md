# Methodological findings

> A cheap screen is only valid where cheapness does not change the quantity being screened. True
> for Delta_Joint, which uses frozen encoders and is exact. False for Delta_Awareness.

This principle was learned twice, at the cost of one wrong rejection and two full grids. Both
occurrences are recorded below because each is independent evidence for the same rule, on a
different axis of "cheap": fewer epochs, and fewer folds.

## Evidence 1 — the epochs budget

`epochs=2` leaves the LoRA adapter under-trained, so `ft ~= all` and Delta_Awareness collapses to
noise around zero *by construction*. The first sweep "rejected" `REG_TEXT_EDU_UDEMY_ACADEMY`
1-of-5 purely because of this.

| model, fold 0 | all | ft @ 2 ep | Delta @ 2 ep | ft @ 10 ep | Delta @ 10 ep |
|---|---|---|---|---|---|
| LightGBM | 0.4957 | 0.5056 | +0.0099 | 0.5279 | **+0.0322** |
| CatBoost | 0.5177 | 0.5282 | +0.0105 | 0.5299 | **+0.0122** |

Delta_Awareness grew roughly 3x with more fine-tuning, exactly as the TAR mechanism predicts. A
**fail at low epochs proves nothing; only a pass is informative** — a pass at 2 epochs would likely
survive at 50, but there is no direction in which an under-trained adapter can manufacture a false
pass.

This also invalidates every epochs=2 batch probe in `results/curation/screening/t3_tar/tar_probes.csv`
(Vietnam housing +0.0007, metacritic -0.0143, and others) — they measured the epoch budget, not the
datasets.

## Evidence 2 — the fold count

A fold-0 Delta_Awareness screen does not work either, and it is the same class of mistake — a
different axis of "cheap" collapsing the same way. Board games:

| board games | cat | light | tabm | tabpfnv2 |
|---|---|---|---|---|
| fold 0 only | +0.0074 | +0.0163 | +0.0052 | +0.0010 |
| 5-fold mean | -0.0003 | +0.0021 | +0.0010 | -0.0015 |

Every model dropped and two flipped sign. Anime showed the same reversal, more cleanly: a fold-0
screen of +0.002 / +0.003 (LightGBM / CatBoost) became 0 of 5 on the full grid, with three of the
four measured models landing negative. The per-(model, fold) spread is sigma = 0.0063 over
[-0.0124, +0.0163] — about 6x the delta threshold — so one fold cannot resolve a criterion whose
threshold sits deep inside its noise band. Screening one fold cost two full grids (board games,
then promoted anime). Delta_Joint is unaffected: at +0.03 to +0.09 it is an order of magnitude
above the same noise, which is why the cheap frozen screen stays sound for it.

## The remaining findings

**Over-deleting structured columns manufactures Delta_Joint.** The board-games table
(`results/curation/rejected/board_games/REJECTION.md`):

| spec | no_text | text_only | all | Delta_Joint |
|---|---|---|---|---|
| auto-spec (drops `Year Published`, `Play Time`) | 0.584 | 0.502 | 0.623 | **+0.0388** |
| `Year Published` restored | 0.613 | 0.502 | 0.636 | +0.0229 |
| full structured block restored | 0.684 | 0.502 | 0.684 | **-0.0005** |

`text_only` is unchanged throughout, so this is entirely a `no_text` effect: deleting a structured
feature does not merely weaken the unimodal baseline, it lets the text act as a proxy for the
deleted column, manufacturing a joint gain that vanishes the moment the columns come back.
Corollary: `hunt.py` is a triage net; its spec is not a curation decision, and any candidate whose
Delta_Joint came from it must be re-measured with the JUNK-deleted columns restored before it is
gridded.

**Identifier and date targets.** `Sl No`, `appid`, `globalReleaseDate`, `CustomerID` — the JUNK
regex uses word boundaries, so it misses camelCase and spaced abbreviations. Fixed in
`auto_spec.py` (`84a637e`).

**Multi-column arithmetic leakage, still unfixed.** `roi_pct` = revenue/budget; both are structured
columns, so no single column is a near-copy of the target and `find_leaks` cannot see it, while
`no_text` reaches R^2 0.997. Proposed guard, not yet implemented: flag any candidate whose
`no_text` R^2 exceeds ~0.95 as saturated.

**The float knife-edge.** Where the difference of rounded means equals delta exactly, a strict `>`
is decided by float64 representation (`0.647 - 0.646` -> `0.0010000000000000009`). Affects
Vietnam's TabPFNv2 (`mean(ft) - mean(all) = 0.647 - 0.646`, a pass only through float
representation — the verdict does not depend on it, since excluding that cell still leaves 4 of 5)
and board games' TabM (`0.629 - 0.628`, likewise not a real margin — counted honestly, board games
had only 1 of 5, not 2). Such cells must be reported, not counted.

**Delta_Awareness is the binding constraint.** Five of eight candidates in the second hunt round
cleared Delta_Joint and then failed TAR (horror-movie budget, metacritic PC, chess, vgsales,
alongside board games and anime once gridded). Now that GPU makes TAR affordable, the pipeline's
cheap-screen ordering should be reversed: probe Delta_Awareness on a few folds early, because it
eliminates far more candidates than Delta_Joint does.

**The typing rule fires in both directions.** The `>=100 distinct` arm promotes low-cardinality
columns into TEXT and blows the <=5 multimodal budget — Sephora's `brand` (324 distinct, 3.5%
unique) and `category` (143 distinct, 1.6% unique) both type as TEXT. This is the dominant real
failure, not the "short free text becomes categorical" case that was originally flagged as the
risk to watch for.

## Where these findings are evidenced per-dataset

- `results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/VERDICT.md` — the epochs=2/epochs=10
  comparison, and the `grid_epochs2_superseded.csv` file kept as method evidence.
- `results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md` — the TabPFNv2 float
  knife-edge, and the cross-machine (CPU/GPU) noise calibration used to decide the halves of a
  delta must share one environment.
- `results/curation/rejected/board_games/REJECTION.md`, `results/curation/rejected/anime/REJECTION.md`
  — both fold-0-screen reversals, and the manufactured-Delta_Joint artifact.
- `results/curation/rejected/metacritic/REJECTION.md` — the sentinel-target rejection.
- `results/curation/rejected/REJECTIONS.md` — the full screen-time rejection table, including the
  junk-target and arithmetic-leakage cases.
