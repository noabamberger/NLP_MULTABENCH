# Delta_Joint — games / leisure / hobbies lane

Frozen states only (`no_text`, `text_only`, `all`). **No fine-tuning was run**; every
number here is from a frozen E5 encoder, so Delta_Awareness is not measured.

*(work in progress — final verdict section is filled in at the end of the run)*

## Rejected: `melissamonfared/board-games` (the lane's PRIMARY)

**Rejected. Its screening `Delta_Joint` of +0.039 is an artifact of the auto-spec's
JUNK regex, not a property of the dataset.**

`auto_spec.JUNK` deletes any column whose name contains `year`, `time`, `rank`, `id`, ...
On this file that removes `Year Published` and `Play Time` — which are not identifiers,
they are two of the strongest structured predictors a board game has. `BGG Rank` and `ID`
are correctly removed (rank is monotone in the ratings and 100% unique, so it would both
leak and type as TEXT).

LightGBM, fold 0, target `Complexity Average`, text = `Mechanics` + `Name`:

| spec | no_text | text_only | all | Delta_Joint |
|---|---|---|---|---|
| auto-spec (drops `Year Published`, `Play Time`) | 0.584 | 0.502 | 0.623 | **+0.0388** |
| + `Year Published` restored | 0.613 | 0.502 | 0.636 | +0.0229 |
| + `Play Time` also restored (full structured block) | 0.684 | 0.502 | 0.684 | **-0.0005** |

`text_only` is unchanged across all three rows, so this is entirely a `no_text` effect:
deleting a structured feature does not merely weaken the unimodal baseline, it lets the
text act as a **proxy for the deleted column** (`Mechanics`/`Name` partly encode a game's
era and length), which manufactures a joint gain that vanishes as soon as the columns
come back.

The semantically natural target is worse still. With `Rating Average` as the target and
the full structured block:

| target | no_text | text_only | all | Delta_Joint |
|---|---|---|---|---|
| `Rating Average` | 0.567 | 0.335 | 0.567 | **+0.0006** |

Both honest specs land inside the ±0.015 fold-noise band, so this dataset carries no
joint signal worth a 75-cell grid. This also matches the earlier aborted T2 screen in
`results/candidates/t2_boardgames.csv`.

**Reusable lesson:** any candidate whose Delta_Joint came from `hunt.py` must be
re-measured with the JUNK-deleted structured columns restored before it is gridded.
`hunt.py` is a triage net, and its spec is not a curation decision.

## Rejected: `vikasojha98/top-women-chess-players` (FALLBACK 1)

Not gridded. Its screen (`hunt_full.csv`) reports `Delta_Joint = +0.024` with
`text_only` R² = 0.074 — the text (`Federation`, `Name`) is nearly inert. It is also
structurally the same trap as above: only two structured columns survive
(`Standard_Rating`, `Blitz_rating`), `Year_of_birth` is JUNK-deleted, and `Federation` is
a country code that types as TEXT purely on cardinality. A +0.024 mean against ±0.015
noise is weak evidence even before that.

## Rejected on inspection (no grid spent)

| dataset | reason |
|---|---|
| `muhammadaqeelkabir/steam-games-dataset-steamspy-api` | screen target was `appid` — a primary key. JUNK regex misses it (no word boundary in `appid`). With a real target there is still no free text: only `name`, `developer`, `publisher`, all proper nouns. |
| `lunthu/gog-com-video-games-dataset` | auto-target was `globalReleaseDate`, a date (camelCase defeats the JUNK regex). Re-screened `Delta_Joint = -0.0003`, and `text_only` R² 0.996 because the text columns were the release dates themselves. |
| `arnabchaki/popular-video-games-1980-2023` | 1,512 rows (too few for stable 5-fold), `Rating` has only 35 distinct values, and `Plays`/`Wishlist` are `"3.9K"` strings. |
| `thedevastator/get-your-game-on-metacritic-…` | `Metacritic` target is 82% sentinel zeros (found by a peer lane) — the "regression" is mostly a has-a-score indicator. |
| `mterzolo/lego-sets` | 12,261 rows are 744 products replicated across 21 countries; near-duplicate rows would straddle train and test in every fold. |
| `nikatomashvili/steam-games-dataset`, `rudrakumargupta/ultimate-games-…` | no numeric column qualifies as a target under the outlier/cardinality rule. |
