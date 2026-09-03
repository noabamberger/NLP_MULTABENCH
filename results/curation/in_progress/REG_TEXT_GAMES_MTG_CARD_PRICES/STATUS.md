# IN PROGRESS — `REG_TEXT_GAMES_MTG_CARD_PRICES`

**Status: NOT ACCEPTED — Delta_Awareness is unmeasured.**

This is not a rejection. Delta_Joint is measured, complete and positive on all five learners.
Delta_Awareness has never been run, and Delta_Joint alone does not accept a dataset.

## The dataset

| | |
|---|---|
| Source | Kaggle `douglascampospires/mtg-all-cards` |
| Registered as | `REG_TEXT_GAMES_MTG_CARD_PRICES` |
| Task | regression |
| Target | `price_usd_log10` (1,504 distinct, \|z\|max 5.36) |
| Text columns | `CARD_TEXT`, `TYPE` |
| Structured (numeric) | `CMC`, `NUMBER_OF_EDITIONS`, `power`, `toughness`, `first_edition_year` |
| Structured (categorical) | `RARITY`, `COLOR_PIE` |
| Rows | 28,507 |
| Preparation | `curation_lab/prep/mtg_cards.py` |

The raw file needs three repairs before the benchmark can consume it, all done in
`prep/mtg_cards.py` and none inside `multabench/`:

- `PRICES` is one packed string (`USD: $0.42 | USD_FOIL: $5.70 | ...`), so the target has to be
  extracted before anything else can happen.
- Card prices are log-normal (raw \|z\|max = 103). The target is regressed as **log10(USD)**.
- `POWER_TOUGHNESS` (`"5/5"`, `"*/*"`) and `FIRST_EDITION` (`"2004-11-19"`) pack two numerics and
  a date into strings, which the typing rule would promote to TEXT on cardinality alone.

### Target outliers

Even after the log transform, `price_usd_log10` carries **\|z\| up to 5.36** (top z-scores 5.358,
5.245, 4.700, ...). The repo only *warns* about \|z\| > 5 and **never clips**, so the warning
stands in the run logs and no clipping was applied. This is disclosed rather than corrected: the
log10 transform is the mitigation, and clipping would deviate from the paper protocol.

## What is measured: Delta_Joint, complete and positive on 5 of 5

`grid_frozen.csv` — 5 models x 3 frozen states x 5 folds = **75 of 75 cells, no gaps**:

| model | Delta_Joint |
|---|---|
| TabM | +0.075 |
| TabPFN-2.5 | +0.068 |
| CatBoost | +0.062 |
| LightGBM | +0.057 |
| TabPFNv2 | +0.050 |

All five are well clear of the +/-0.015 fold-noise band, though an order of magnitude below the
two accepted datasets.

## Why this is a good candidate

The text encodes **what the card does**; the structured block encodes **cost and scarcity**.
Those are independent price drivers — a cheap common with a powerful ability and an expensive rare
with a dull one are priced by different mechanisms, and neither channel can reconstruct the other.
That is the orthogonal-channel signature described in
[`docs/findings/02-mining-method-rules.md`](../../../../docs/findings/02-mining-method-rules.md),
and it is the structural reason to expect the joint condition to beat both unimodal ones — which
it does.

## What it needs

**A TAR grid at epochs >= 10 over >= 3 folds.**

A fold-0 screen will not resolve it. The per-(model, fold) spread of Delta_Awareness is
sigma = 0.0063 on board games, the widest measured (anime's own is 0.0026, all cells pooled
0.0047) — every one of them many times the delta threshold of 0.001, so a single fold cannot resolve a
criterion whose threshold sits far inside its own noise band. Screening one fold has already cost
two full grids elsewhere in this project; see
[`docs/findings/03-methodological-findings.md`](../../../../docs/findings/03-methodological-findings.md).

The same finding rules out a cheap epoch budget: at epochs=2 an under-trained LoRA adapter barely
moves the encoder, so `ft` ~= `all` and Delta_Awareness collapses to noise around zero by
construction. Any TAR run here must use at least 10 epochs.

## Files in this folder

| file | schema | rows | what it is |
|---|---|---|---|
| `grid_frozen.csv` | cpu | 75 | the complete frozen grid: 5 models x 3 states x 5 folds |
| `logs/dj_games.log` | — | — | frozen run log, including the target distribution and outlier warnings |
| `logs/dj_games_finish.log` | — | — | completion of the frozen sweep |
| `logs/probe_mtg.log` | — | — | initial spec/typing probe |
