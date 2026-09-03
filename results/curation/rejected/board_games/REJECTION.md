# REJECTED — board games (`melissamonfared/board-games`)

**Rejected twice over.** Two independent reasons, either of which is sufficient. Both are
recorded here because the first one is the more instructive: it is the reason the *screen* looked
promising at all.

The same source dataset was screened three ways under three registered names:

| registered name | evidence file | schema | what it was |
|---|---|---|---|
| `REG_TEXT_SOCIAL_BOARD_GAMES_BGG` | `grid_t2_screen.csv` | cpu | the aborted T2 screen (2 rows, `no_text` only) |
| `REG_TEXT_GAMES_BGG_DESCRIPTION` | `grid_bgg_description.csv` | cpu | LightGBM fold-0 probe on the `Description` text variant |
| `REG_TEXT_GAMES_BOARDGAMES_BGG` | `grid_gpu_full.csv` | kaggle | the T4 grid, 4 models x 4 states x 5 folds |
| `REG_TEXT_GAMES_BOARDGAMES_BGG` | `grid_pfn25.csv` | kaggle | TabPFN-2.5, 4 states x 5 folds, completing the committee |

## Reason 1: the screening Delta_Joint of +0.039 was an artifact

`auto_spec.JUNK` deletes any column whose name contains `year`, `time`, `rank`, `id`, ... On this
file that removes **`Year Published`** and **`Play Time`** — which are not identifiers. They are
two of the strongest structured predictors a board game has. (`BGG Rank` and `ID` are correctly
removed: rank is monotone in the ratings and 100% unique, so it would both leak and type as TEXT.)

LightGBM, fold 0, target `Complexity Average`, text = `Mechanics` + `Name`:

| spec | no_text | text_only | all | Delta_Joint |
|---|---|---|---|---|
| auto-spec (both deleted) | 0.584 | 0.502 | 0.623 | **+0.0388** |
| `Year Published` restored | 0.613 | 0.502 | 0.636 | +0.0229 |
| full structured block | 0.684 | 0.502 | 0.684 | **-0.0005** |

**`text_only` is unchanged throughout**, so this is entirely a `no_text` effect. Deleting a
structured feature does not merely weaken the unimodal baseline — it lets the text act as a
**proxy for the deleted column** (`Mechanics`/`Name` partly encode a game's era and length),
which manufactures a joint gain that vanishes the moment the columns come back.

The semantically natural target is worse still. With `Rating Average` and the full structured
block: `no_text` 0.567, `text_only` 0.335, `all` 0.567, Delta_Joint **+0.0006**.

Both honest specs land inside the +/-0.015 fold-noise band.

**Reusable lesson:** any candidate whose Delta_Joint came from `hunt.py` must be re-measured with
the JUNK-deleted structured columns restored before it is gridded. `hunt.py` is a triage net; its
spec is not a curation decision.

## Reason 2: the full grid fails Delta_Awareness

`grid_gpu_full.csv` + `grid_pfn25.csv` — 5 models x 4 states x 5 folds = **100 cells, no gaps**,
all on one T4, `ft` at 10 epochs. TabPFN-2.5 was added once the Prior Labs token unblocked it, so
no cell here is absent or counted as a non-pass by default:

| model | no_text | text_only | all | ft | Delta_Joint | Delta_Awareness |
|---|---|---|---|---|---|---|
| LightGBM | 0.569 | 0.477 | 0.616 | 0.619 | +0.047 | +0.003 |
| TabM | 0.576 | 0.504 | 0.628 | 0.629 | +0.052 | +0.001 |
| CatBoost | 0.576 | 0.490 | 0.623 | 0.623 | +0.047 | 0.000 |
| TabPFNv2 | 0.585 | 0.504 | 0.640 | 0.639 | +0.055 | -0.001 |
| TabPFN-2.5 | 0.586 | 0.514 | 0.645 | 0.645 | **+0.059** | 0.000 |

Delta_Joint is healthy (+0.047 to +0.059, TabPFN-2.5 strongest) and Delta_Awareness is not:
**2 of 5 against a quorum of 3 -> REJECTED.**

Completing the committee did not rescue it, and that is worth stating plainly: the earlier
four-model verdict counted TabPFN-2.5 as an absent non-pass, so one could have argued the
rejection was an artifact of a missing model. It was not. TabPFN-2.5 posts the dataset's
*largest* Delta_Joint and a Delta_Awareness of exactly 0.000.

And **counted honestly it is 1 of 5.** TabM's +0.001 is the same float knife-edge documented for
TabPFNv2 in `accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md`: the difference of the rounded
means is exactly delta and clears a strict `>` only through float64 representation. Only
LightGBM's +0.003 is a real margin.

## The fold-0 screen pointed the wrong way

This dataset was promoted to a full grid on a fold-0 Delta_Awareness screen of +0.016 / +0.008
(LightGBM / CatBoost). The five-fold means:

| board games | CatBoost | LightGBM | TabM | TabPFNv2 |
|---|---|---|---|---|
| fold 0 only | +0.0074 | +0.0163 | +0.0052 | +0.0010 |
| 5-fold mean | -0.0003 | +0.0021 | +0.0010 | -0.0015 |

Every model dropped, and the fold-0 reading pointed the *opposite* way for two of them. See
[`docs/findings/03-methodological-findings.md`](../../../docs/findings/03-methodological-findings.md).

## Files in this folder

| file | schema | rows | what it is |
|---|---|---|---|
| `grid_t2_screen.csv` | cpu | 2 | the aborted T2 screen, `REG_TEXT_SOCIAL_BOARD_GAMES_BGG` |
| `grid_bgg_description.csv` | cpu | 3 | LightGBM fold-0 `Description` variant, `REG_TEXT_GAMES_BGG_DESCRIPTION` |
| `grid_gpu_full.csv` | kaggle | 80 | the full T4 grid, `REG_TEXT_GAMES_BOARDGAMES_BGG` |
| `logs/t2_boardgames.log` | — | — | T2 screen log |
| `logs/dj_games_bgg.log` | — | — | `Description` variant log |
