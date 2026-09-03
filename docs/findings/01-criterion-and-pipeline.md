# The curation criterion, and why this runner's numbers can be trusted

## The criterion

Evaluate 5 learners — **TabM, CatBoost, LightGBM, TabPFN v2, TabPFN v2.5** — under 4 conditions.
A dataset **passes** if, for **at least 3 of the 5 learners** (`RHO = 3/5`):

```
Delta_Joint     = mean(all) - max(mean(no_text), mean(text_only))  > delta
Delta_Awareness = mean(ft)  - mean(all)                            > delta
```

with paper default `delta = 0.001`. Each state's mean is taken over 5 folds and **rounded to 3
decimals before differencing** — this is not a detail to drop when reproducing a number by hand;
the rounding is why some cells sit on a float knife-edge (see
[`03-methodological-findings.md`](03-methodological-findings.md)).

## Paper condition, CLI flag, CSV label

| Paper condition | CLI flag | CSV `multimodal_state` |
|---|---|---|
| Unimodal Structured (tabular only) | `no_text` | `no_text` |
| Unimodal Unstructured (text only) | `txt` or `text_only` | `text_only` |
| Joint Frozen (all feats, frozen E5) | `all` | `all` |
| Joint Target-Aware / TAR (LoRA-tuned E5) | `ft` or `ft-txt` | `ft` |

`all` and `ft` both map to `MultimodalState.ALL` internally; the conditions differ only in the
derived `tune_dino`/`tune_e5` flags. Never infer the condition from `MultimodalState` alone — read
the CSV's `multimodal_state` column.

## The rule that matters: reuse it, never reimplement it

`passes()` lives once, in `multabench/leaderboard/analysis/pass_matrix.py`. Every verdict in this
project — the two accepted datasets, the one in-progress dataset, and the three gridded rejections
— was computed by calling that function, not by reimplementing the arithmetic above. The
arithmetic looks trivial enough to copy by hand; the point of routing every verdict through one
function is that a rounding-order bug or a rho miscount would otherwise have to be caught
independently in every report that computes a verdict.

`passes()` also **asserts** row completeness (5 folds x 4 states per model) rather than averaging
over gaps. It carries exactly one hardcoded exception, `_KNOWN_MISSING_ROWS = {("TabPFNv2",
"REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO", "ft", 4)}` — a single dropped wandb run from the paper's
own 56-dataset pool, confirmed by exhaustively diffing the expected grid against the source data.
Any *other* assertion failure is a real data gap to investigate, not something to append to that
set. This project never needed to add a second entry: every grid gathered here is either complete
(the two accepted datasets, the three gridded rejections) or explicitly marked incomplete and
un-verdicted (the Metacritic partial grid, the MTG frozen-only grid).

## Validation: the harness reproduces the shipped ground truth

Before any candidate-dataset number was trusted, the harness was checked against artifacts the
paper itself ships:

- The shipped 56x10 `pass_matrix.csv` re-derives from `pool_scores_long.csv` with **0 mismatched
  cells**.
- `verdict()` returns `accepted=True`, 5 of 5, for the known-accept
  `MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT`.
- `verdict()` returns `accepted=False`, 0 of 5, for the known-reject
  `REG_TEXT_FOOD_RAMEN_RATINGS_2022`.

## Runner fidelity against the paper's own anchor

Full detail and the raw CSVs are in `results/curation/validation/`. Anchor dataset:
`MUL_TEXT_PRODUCT_SENTIMENT` (N=5091, 4 classes, 1 text + 1 structured column), one of the paper's
own accepted datasets — used here to prove the runner reproduces the paper before any number about
a *new* dataset is worth reading.

- **`no_text` reproduces the paper exactly for every model** — mean absolute difference 0.0000
  across a 36-run grid (LightGBM: `0.83454303717305`, identical to the paper's value). This is the
  encoder-free condition, so it isolates data loading, curation, the 90/10 split, the seeding and
  the learners themselves from anything encoder-related.
- `text_only` has mean signed difference **-0.0000** across the grid, range -0.046 to +0.038 —
  symmetric scatter, not bias. `all` carries a small genuine positive bias (mean +0.0058, max
  +0.0235) — real, but four times smaller than a single fold suggested.
- **Delta_Joint — the quantity the criterion actually consumes — reproduces within 0.012 across
  four models, every sign agreeing:**

  | model | Delta_Joint (ours) | Delta_Joint (paper) | sign agrees |
  |---|---|---|---|
  | CatBoost | 0.075 | 0.073 | yes |
  | LightGBM | 0.058 | 0.046 | yes |
  | TabM | 0.085 | 0.086 | yes |
  | TabPFNv2 | 0.076 | 0.079 | yes |

This 0.012 agreement, and the wider +/-0.015 fold-noise band it implies, is the calibration behind
every "is this delta real or noise" judgment made about a candidate dataset elsewhere in this
project (see [`03-methodological-findings.md`](03-methodological-findings.md)): a candidate whose
Delta_Joint sits below about 0.02 needs more folds, not a verdict on one reading.

The frozen embedding cache used throughout this project (~40x speedup) was proven bit-exact against
this same anchor before being trusted for any candidate — see `results/curation/validation/README.md`
for the cache-through-vs-uncached identity check.
