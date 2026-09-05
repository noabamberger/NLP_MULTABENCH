# Index of curation evidence

Every *evidence* file under `results/curation/` — grids, logs, screens and source tables — with
the name it had before the reorganization. The authored narrative documents (`VERDICT.md`,
`STATUS.md`, `REJECTION.md`, `REJECTIONS.md`, the `validation/README.md` and this index) are not
listed: they were written here rather than moved, so they have no prior name to record.

**Nothing here was reformatted.** The reorganization (commit `64bd9a5`) was renames only — no file
content changed, so no measured number moved.

Two separate claims are made about that move, each with its own evidence, and neither does the
other's job:

- **No content was destroyed** — proved by the content-addressed manifest
  (`docs/superpowers/plans/consolidation-manifest.json`; re-run it with
  `python -m curation_lab.tools.manifest check`). It asks whether each recorded file's content
  still exists *anywhere*, so it would not notice content moved somewhere nonsensical, and it says
  nothing about files it never recorded.
- **It went where this index says it went** — proved by the rename-only diff of `64bd9a5`
  (54 renames, 0 additions, 0 deletions) together with the original-filename column below. The *original filename* column is what keeps the
`--out` paths recorded in the archived reports and run logs traceable across the renames: a report
that says it wrote `results/candidates/dj_property_tar_all_ft.csv` is talking about
`accepted/REG_TEXT_HOUSES_VIETNAM_2024/grid_tar.csv`.

Because nothing was reformatted, two result schemas coexist:

- **`cpu`** — written by the local runner: `model, dataset, fold, multimodal_state, test_score,
  runtime, n_train, n_test, m_features, task_type, tune_e5`
- **`kaggle`** — written by the Kaggle GPU lane: `state, score, secs, epochs, dataset, model, fold`

**Use the right loader for each schema — neither one reads both.** Both canonicalize to
`[model, dataset, state, fold, test_score]`, so frames from the two can be merged afterwards, but
they are separate entry points:

- `cpu` files → `curation_lab.criterion.deltas.normalize`. It maps emoji `MODEL_NAME`s to short
  labels and renames `multimodal_state` to `state`. It does **not** rename `score`, so it raises
  `KeyError: 'test_score'` on a `kaggle` file.
- `kaggle` files → `curation_lab.kaggle.verdict_from_runs.load`. It renames `score` to
  `test_score`, maps short model keys to labels, and drops duplicate cells keeping the last.

Never hand-map the columns.

Some files under `results/curation/` are not benchmark results at all, and are labelled for what
they are rather than forced into one of the two result schemas: `screen` (one row per candidate,
the T2 Delta_Joint triage), `profile` (T0/T1 candidate profiling), `audit` (spec audits: chosen
target, typing, leak drops), `probe` (T3 TAR probes), `source` (a data table), `log` (plain text).

## accepted/

| new path | original filename | dataset | tier | schema | what it proves |
|---|---|---|---|---|---|
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/grid_gpu_light_cat_tabm.csv` | `tar_results.csv` (udemy-lct) | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | kaggle | **Primary evidence.** T4, `light,cat,tabm` x 4 states x folds 0-4 (60 cells), `ft` at 10 epochs. Split by model so every state for a learner is within-session. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/grid_gpu_tabpfn.csv` | `tar_results.csv` (udemy-pfn) | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | kaggle | **Primary evidence.** T4, `tabpfnv2,tabpfnv2p5` x 4 states x folds 0-4 (40 cells). With the row above: 100 cells, no gaps, ACCEPTED 3 of 5. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/grid.csv` | `verify_udemy_e10.csv` | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | cpu | The earlier CPU grid at epochs=10, also 3 of 5. Retained as the cross-environment comparison: Delta_Joint reproduces within 0.011 but two learners flip on Delta_Awareness. Not superseded — it is the evidence for that instability. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/grid_epochs2_superseded.csv` | `verify_udemy.csv` | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | cpu | The epochs=2 sweep that rejected the dataset 1 of 5. Superseded; retained as the evidence that a starved epoch budget measures the budget, not the dataset. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/logs/verify_udemy_e10.log` | `verify_udemy_e10.log` | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | log | Run log for the CPU grid: spec line, target distribution, per-cell scores. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/logs/verify_udemy_frozen.log` | `verify_udemy_frozen.log` | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | log | Frozen-state half of the sweep. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/logs/verify_udemy_ft.log` | `verify_udemy_ft.log` | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | log | The `ft` half of the sweep; shows the LoRA fine-tuning and the TAR encoder cache. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/logs/t3_udemy.log` | `t3_udemy.log` | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | log | Earliest T3 TAR probe. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/logs/t3_udemy2.log` | `t3_udemy2.log` | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | log | Second T3 TAR probe. |
| `accepted/REG_TEXT_EDU_UDEMY_ACADEMY/logs/t3_udemy_e20.log` | `t3_udemy_e20.log` | `REG_TEXT_EDU_UDEMY_ACADEMY` | accepted | log | Epoch-budget probe at 20 epochs. |
| `accepted/REG_TEXT_HOUSES_VIETNAM_2024/grid_frozen_cpu.csv` | `dj_property.csv` | `REG_TEXT_HOUSES_VIETNAM_2024` | accepted | cpu | The standalone Delta_Joint result: 75/75 CPU frozen cells, mean 0.2872, std 0.0247, 25/25 positive, t = 58.03. |
| `accepted/REG_TEXT_HOUSES_VIETNAM_2024/grid_frozen_gpu.csv` | `dj_property_tar_frozen.csv` | `REG_TEXT_HOUSES_VIETNAM_2024` | accepted | kaggle | The `no_text` + `text_only` half of the 100-cell same-machine grid (4 models x 5 folds). |
| `accepted/REG_TEXT_HOUSES_VIETNAM_2024/grid_tar.csv` | `dj_property_tar_all_ft.csv` | `REG_TEXT_HOUSES_VIETNAM_2024` | accepted | kaggle | The `all` + `ft` half at 10 epochs (4 models x 5 folds) — the Delta_Awareness evidence. |
| `accepted/REG_TEXT_HOUSES_VIETNAM_2024/grid_tar_pfn25.csv` | `dj_property_tar_pfn25.csv` | `REG_TEXT_HOUSES_VIETNAM_2024` | accepted | kaggle | TabPFN-2.5, all four states x 5 folds, 20/20 cells. Proves the fifth committee model is measured, not absent, and posts the largest Delta_Joint (+0.324). |
| `accepted/REG_TEXT_HOUSES_VIETNAM_2024/logs/dj_property.log` | `dj_property.log` | `REG_TEXT_HOUSES_VIETNAM_2024` | accepted | log | CPU frozen run log: spec line (target `Price`, 687 distinct, zmax 2.54) and per-cell scores. |

## in_progress/

| new path | original filename | dataset | tier | schema | what it proves |
|---|---|---|---|---|---|
| `in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/grid_frozen.csv` | `dj_games.csv` | `REG_TEXT_GAMES_MTG_CARD_PRICES` | in_progress | cpu | Delta_Joint complete and positive on all 5 models (75/75 cells, +0.050..+0.075). Delta_Awareness unmeasured, so no verdict. |
| `in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/logs/dj_games.log` | `dj_games.log` | `REG_TEXT_GAMES_MTG_CARD_PRICES` | in_progress | log | Frozen run log; records the target distribution and the zmax 5.36 outlier warning (never clipped). |
| `in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/logs/dj_games_finish.log` | `dj_games_finish.log` | `REG_TEXT_GAMES_MTG_CARD_PRICES` | in_progress | log | Completion of the frozen sweep. |
| `in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/logs/probe_mtg.log` | `probe_mtg.log` | `REG_TEXT_GAMES_MTG_CARD_PRICES` | in_progress | log | Initial spec/typing probe on the prepared CSV. |

## rejected/

| new path | original filename | dataset | tier | schema | what it proves |
|---|---|---|---|---|---|
| `rejected/board_games/grid_t2_screen.csv` | `t2_boardgames.csv` | `REG_TEXT_SOCIAL_BOARD_GAMES_BGG` | rejected | cpu | The aborted T2 screen (2 rows, `no_text` only) — the first pass at this source. |
| `rejected/board_games/grid_bgg_description.csv` | `dj_games_bgg.csv` | `REG_TEXT_GAMES_BGG_DESCRIPTION` | rejected | cpu | LightGBM fold-0 probe on the `Description` text variant, Delta_Joint +0.059 — a third registered name for the same source. |
| `rejected/board_games/grid_gpu_full.csv` | `boardgames_full.csv` | `REG_TEXT_GAMES_BOARDGAMES_BGG` | rejected | kaggle | The honest grid, 4 models x 4 states x 5 folds: Delta_Joint +0.047..+0.055, Delta_Awareness 2 of 5 (1 of 5 counted honestly) against a quorum of 3. |
| `rejected/board_games/grid_pfn25.csv` | `boardgames_pfn25.csv` | `REG_TEXT_GAMES_BOARDGAMES_BGG` | rejected | kaggle | TabPFN-2.5, 4 states x 5 folds, completing the committee to 100 cells. Delta_Joint +0.059 (largest of the five), Delta_Awareness 0.000 — the completed grid did not change the rejection. |
| `rejected/board_games/logs/t2_boardgames.log` | `t2_boardgames.log` | `REG_TEXT_SOCIAL_BOARD_GAMES_BGG` | rejected | log | T2 screen log. |
| `rejected/board_games/logs/dj_games_bgg.log` | `dj_games_bgg.log` | `REG_TEXT_GAMES_BGG_DESCRIPTION` | rejected | log | The `Description` variant log. |
| `rejected/anime/grid.csv` | `anime_full.csv` | `REG_TEXT_MEDIA_ANIME_POPULARITY` | rejected | kaggle | Full grid, 80/80 cells: Delta_Joint +0.031..+0.037 but Delta_Awareness 0 of 5, reversing a positive fold-0 screen. |
| `rejected/metacritic/derived_input.csv` | `derived/metacritic_scored.csv` | `REG_TEXT_MEDIA_METACRITIC_SCORED` | rejected | source | The repaired input the grid was built from: 2,267 scored rows after dropping 10,357 of 12,624 sentinel `Metacritic == 0` rows. |
| `rejected/metacritic/grid.csv` | `dj_media_metacritic.csv` | `REG_TEXT_MEDIA_METACRITIC_SCORED` | rejected | cpu | Partial frozen grid (52 rows, no `ft` state): LightGBM Delta_Joint +0.040, CatBoost -0.003. No verdict is claimed from it. |
| `rejected/metacritic/logs/dj_media_metacritic.log` | `dj_media_metacritic.log` | `REG_TEXT_MEDIA_METACRITIC_SCORED` | rejected | log | Run log; the `dropped 10357/12624 sentinel Metacritic==0 rows` line is the rejection reason in one place. |

## screening/

| new path | original filename | dataset | tier | schema | what it proves |
|---|---|---|---|---|---|
| `screening/t0_t1/t1_batch.csv` | `t1_batch.csv` | 232 candidates | screening | profile | The T0/T1 typing probe over the whole Kaggle search. Its `n_text` overcounts, because the profiler never applied the junk filter — dates and ids type as TEXT on cardinality alone. |
| `screening/t0_t1/t1_batch.log` | `t1_batch.log` | 232 candidates | screening | log | T0/T1 run log. |
| `screening/t0_t1/t1_shortlist.csv` | `t1_shortlist.csv` | 58 candidates | screening | profile | T1 survivors after the junk filter — genuine text columns only. |
| `screening/t0_t1/novelty_shortlist.csv` | `novelty_shortlist.csv` | 14 candidates | screening | profile | The shortlist after the used-domain (MulTaBench overlap) filter, with the ranking score. |
| `screening/t2_joint/hunt_full.csv` | `hunt_full.csv` | 34 candidates | screening | screen | The main T2 Delta_Joint triage. Holds the screen rows for steam/steamspy, chess, google-play, daraz and grocery cited in `rejected/REJECTIONS.md`. |
| `screening/t2_joint/hunt_full.log` | `hunt_full.log` | 34 candidates | screening | log | T2 run log for the above. |
| `screening/t2_joint/hunt_games2.csv` | `hunt_games2.csv` | 5 candidates | screening | screen | The games re-hunt: gog (Delta_Joint -0.0003, `text_only` R2 0.996), rudrakumargupta (`serial_no` target, -0.0027), and three `skip` rows including `douglascampospires/mtg-all-cards`. |
| `screening/t2_joint/hunt_games2.log` | `hunt_games2.log` | 5 candidates | screening | log | Games re-hunt log. |
| `screening/t2_joint/hunt_media.csv` | `hunt_media.csv` | 13 candidates | screening | screen | The media-lane T2 triage. |
| `screening/t2_joint/hunt_media.log` | `hunt_media.log` | 13 candidates | screening | log | Media-lane run log. |
| `screening/t2_joint/hunt_smoke.csv` | `hunt_smoke.csv` | 1 candidate | screening | screen | Single-candidate smoke test of the hunt harness. |
| `screening/t2_joint/screen4_fold0.csv` | `screen4_fold0.csv` | boardgames, metacritic PC, kerala liquor, chess women | screening | kaggle | Fold-0 four-state screen. Shows kerala saturated at `no_text` R2 1.000, and the board-games fold-0 Delta_Awareness that proved unrepresentative. |
| `screening/t2_joint/screen_wave2_fold0.csv` | `screen_wave2_fold0.csv` | vgsales, anime, horror budget, movies ROI | screening | kaggle | Fold-0 four-state screen. Shows movies-ROI saturated at `no_text` R2 0.997, and the positive anime fold-0 reading the full grid later reversed. |
| `screening/t3_tar/tar_probes.csv` | `tar_probes.csv` | vietnam, udemy, kerala, metacritic | screening | probe | The epochs=2 TAR batch probe. Every Delta_Awareness here (Vietnam +0.0007, metacritic -0.0143) measures the epoch budget, not the dataset. |
| `screening/t3_tar/batch_tar.log` | `batch_tar.log` | vietnam, udemy, kerala, metacritic | screening | log | Run log for the epochs=2 batch probe. |
| `screening/spec_audits/spec_audit.csv` | `spec_audit.csv` | 16 candidates | screening | audit | First spec audit: the chosen target, typing counts, and leak drops per candidate. |
| `screening/spec_audits/spec_audit2.csv` | `spec_audit2.csv` | 16 candidates | screening | audit | Re-audit after the junk-target fix. Holds the corrected kerala spec (`Special Fee` in place of `Sl No`). |
| `screening/spec_audits/spec_audit_wave2.csv` | `spec_audit_wave2.csv` | 10 candidates | screening | audit | Wave-2 spec audit. |
| `screening/spec_audits/spec_audit_wave3.csv` | `spec_audit_wave3.csv` | 3 candidates | screening | audit | Wave-3 spec audit, including `sujaykapadnis/horror-movies-profits-dataset`. |
| `screening/spec_audits/spec_audit_wave4.csv` | `spec_audit_wave4.csv` | 5 candidates | screening | audit | Wave-4 spec audit. Holds the `roi_pct` target row for `suhanigupta04/global-movies-dataset-19502026` — the multi-column arithmetic leak `find_leaks` cannot see. |

## validation/

Runs on the paper anchor `MUL_TEXT_PRODUCT_SENTIMENT`, not candidate screening. See
[`validation/README.md`](validation/README.md).

| new path | original filename | dataset | tier | schema | what it proves |
|---|---|---|---|---|---|
| `validation/phase1_lgbm.csv` | `phase1_lgbm.csv` | `MUL_TEXT_PRODUCT_SENTIMENT` | validation | cpu | The first fidelity check: LightGBM fold 0, `no_text` = 0.83454303717305, identical to the paper. |
| `validation/phase1_grid.csv` | `phase1_grid.csv` | `MUL_TEXT_PRODUCT_SENTIMENT` | validation | cpu | The 36-run frozen grid: `no_text` reproduces exactly for every model, and Delta_Joint agrees with the paper within 0.012 with all signs matching. |
| `validation/cache_check.csv` | `cache_check.csv` | `MUL_TEXT_PRODUCT_SENTIMENT` | validation | cpu | The frozen embedding cache is bit-exact: LightGBM `all` fold 0 returns 0.8618478948517495 through the cache, identical to the uncached run. |
| `validation/tabpfn25_retry.csv` | `tabpfn25_retry.csv` | `MUL_TEXT_PRODUCT_SENTIMENT` | validation | cpu | A historical record of the TabPFN-2.5 loading failure, from before a Prior Labs API key resolved it. Not a live limitation. |
