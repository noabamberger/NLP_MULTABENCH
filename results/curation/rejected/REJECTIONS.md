# Screen-time rejections

Candidates rejected during screening, before a full grid was spent on them. Each row names the
reason and the screening file that shows it. The three candidates that *were* gridded have their
own folders beside this file: [`board_games/`](board_games/REJECTION.md),
[`anime/`](anime/REJECTION.md), [`metacritic/`](metacritic/REJECTION.md).

Paths are relative to `results/curation/`.

| candidate | reason | evidence |
|---|---|---|
| `muhammadaqeelkabir/steam-games-dataset-steamspy-api` | screen target was `appid`, a primary key; the JUNK regex misses it (no word boundary in `appid`). No free text besides proper nouns — only `name`, `developer`, `publisher`. | `screening/t2_joint/hunt_full.csv` |
| `lunthu/gog-com-video-games-dataset` | auto-target was `globalReleaseDate`, a date — camelCase defeats the JUNK regex. Re-screened Delta_Joint **-0.0003**, with `text_only` R^2 **0.996** because the text *was* the release dates. | `screening/t2_joint/hunt_games2.csv` |
| `arnabchaki/popular-video-games-1980-2023` | 1,512 rows (too few for a stable 5-fold), `Rating` has only 35 distinct values, and `Plays`/`Wishlist` are `"3.9K"` strings. Screened as `skip` — no numeric column qualified as a target. | `screening/t2_joint/hunt_games2.csv` |
| `mterzolo/lego-sets` | 12,261 rows are 744 products replicated across 21 countries; near-duplicate rows would straddle train and test in every fold. **Rejected on inspection — no screen was run, so there is no evidence row.** | (none — see `docs/archive/DJ_GAMES_REPORT.md`) |
| `vikasojha98/top-women-chess-players` | Delta_Joint **+0.024** against a +/-0.015 noise band, with `text_only` R^2 **0.074** — the text (`Federation`, `Name`) is nearly inert. Only two structured columns survive, and `Federation` types as TEXT purely on cardinality. | `screening/t2_joint/hunt_full.csv` |
| `tolstoyjustin/kerala-bevco-liquor-price-list-2025-2026` | screen target was `Sl No`, a serial number. With the corrected target (`Special Fee`) the dataset is degenerate: `no_text` reaches R^2 **1.000**, because the fee is a deterministic function of the structured columns. | `screening/spec_audits/spec_audit2.csv` (corrected spec), `screening/t2_joint/screen4_fold0.csv` (the saturated R^2) |
| `neomatrix369/google-play-store-apps-extended` | target derived from its own text. | `screening/t2_joint/hunt_full.csv` |
| `nomanmunir/daraz-perfumes` | all-state R^2 negative (`no_text` -0.130, `all` -0.009, `text_only` -0.045) — no learner explains anything, so a delta between them is meaningless. | `screening/t2_joint/hunt_full.csv` |
| `rrokon/global-grocery-nutrition-dataset-2025` | Delta_Joint **+0.0005** with the baseline saturated at `no_text` R^2 **0.969** — no text signal can be demonstrated at that ceiling. | `screening/t2_joint/hunt_full.csv` |
| `suhanigupta04/global-movies-dataset-19502026` (`roi_pct`) | `roi_pct` is revenue/budget — an arithmetic identity over two structured columns, so `no_text` reaches R^2 **0.997**. Invisible to `find_leaks`, which only drops single columns that are near-copies of the target by rank correlation. | `screening/spec_audits/spec_audit_wave4.csv` (the target), `screening/t2_joint/screen_wave2_fold0.csv` (the saturated R^2) |
| `nikatomashvili/steam-games-dataset` | no numeric column qualifies as a target under the outlier/cardinality rule — screened as `skip`. | `screening/t2_joint/hunt_games2.csv` |
| `rudrakumargupta/ultimate-games-dataset-15k-games-43-features` | auto-target was `serial_no`, a serial number (15,000 distinct). Screened with it, Delta_Joint is **-0.0027** against a `no_text` R^2 of 0.983. | `screening/t2_joint/hunt_games2.csv` |
| `sujaykapadnis/horror-movies-profits-dataset` | Delta_Joint healthy (+0.055 / +0.093) but fold-0 Delta_Awareness **-0.001 / -0.013** — fails on target-awareness. | `screening/t2_joint/screen_wave2_fold0.csv` |
| `REG_TEXT_GAMES_VGSALES_USERSCORE` | every delta at noise scale: Delta_Joint +0.001 / +0.004, Delta_Awareness +0.001 / +0.002. | `screening/t2_joint/screen_wave2_fold0.csv` |

## Two recurring failure classes

**Junk targets.** `Sl No`, `appid`, `serial_no` and `globalReleaseDate` were all chosen as
regression targets by the automatic spec. A large delta measured against a junk target says
nothing about the dataset. Both the identifier and the date cases are now fixed in `auto_spec.py`.

**Multi-column arithmetic leakage — still unfixed.** `roi_pct` is revenue divided by budget, and
both are structured columns, so `no_text` reaches 0.997. `find_leaks` only drops single columns
that are near-copies of the target; no individual column here is one. A cheap guard exists on
paper and is not yet implemented: flag any candidate whose `no_text` R^2 exceeds ~0.95 as
saturated, since at that point no text signal can be demonstrated regardless of the deltas.

## Note on three rows above

Three entries differ from the summary tables in `docs/archive/`, and the evidence files were
checked directly rather than transcribed:

- `mterzolo/lego-sets` was rejected on inspection with **no grid and no screen spent**, so no
  results file records it. The archive's inspection table is the only source.
- The `roi_pct` target appears in `spec_audit_wave4.csv`, not `spec_audit_wave3.csv`; the
  saturated `no_text` R^2 that condemns it is in `screen_wave2_fold0.csv`.
- `rudrakumargupta/ultimate-games-*` was **not** skipped for want of a target: it was screened
  with `serial_no` and returned a negative Delta_Joint. Only `nikatomashvili/steam-games-dataset`
  (and `arnabchaki`, `douglascampospires`) were skipped outright.
