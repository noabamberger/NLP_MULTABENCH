# STATE — live handoff

Replaces `RESUME.md` (now in `docs/archive/`). Branch: **`curation-lab`**. Deadline
**2026-10-26** (Technion NLP course 097215, Spring 2026 final project, Track 2 — Benchmark
Track).

## Read these first, in order

1. `CLAUDE.md` — repo guide + this machine's environment constraints
2. `docs/findings/01-criterion-and-pipeline.md` — the criterion and why this runner's numbers
   can be trusted
3. `docs/findings/02-mining-method-rules.md` — the mining/screening rulebook
4. `docs/findings/03-methodological-findings.md` — the correction trail (epochs, folds, and
   what else went wrong)
5. `docs/findings/04-environment-and-performance.md` — environment constraints and performance
   economics, including the resolved TabPFN-2.5 blocker

## Verdicts

| dataset | status | quorum |
|---|---|---|
| `REG_TEXT_EDU_UDEMY_ACADEMY` | **ACCEPTED** | 3 of 5 |
| `REG_TEXT_HOUSES_VIETNAM_2024` | **ACCEPTED** | 5 of 5 |
| `REG_TEXT_GAMES_MTG_CARD_PRICES` | **IN PROGRESS** — Delta_Joint measured (75/75 cells, 5 of 5
  positive), Delta_Awareness never run | — |
| board games, anime, metacritic | **REJECTED**, full grids (board games, anime) or target-level
  (metacritic) | — |

Full detail: `results/curation/accepted/*/VERDICT.md`,
`results/curation/in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/STATUS.md`,
`results/curation/rejected/*/REJECTION.md`, `results/curation/rejected/REJECTIONS.md` (the
screen-time rejections that never reached a full grid).

**Standard scope (1 passing dataset) is met twice.** **Outstanding scope (>=5 passing datasets)
needs 3 more.**

## Deviation to carry into any writeup

E5 fine-tuning ran **10 epochs**, not the `E5TrainArgs` default of 50 (patience 3). It applies to
both acceptances, affects Delta_Awareness only, and is conservative — the measured trend is that
more epochs widen the delta. Re-run at 50 if GPU budget allows; nothing else is pending on it.

## Blockers

- **The local candidate pool is exhausted.** It needs a fresh T0/T1 Kaggle search with the
  junk-aware profiler, because the previous search ranked candidates on a text-column count that
  counted dates and ids as text — the pool was never as rich as it appeared.
- **The TabPFN-2.5 blocker is resolved.** It was a Prior Labs API key requirement, not an HF
  licence — see `docs/findings/04-environment-and-performance.md`. The full five-model committee
  is available for every future grid.

## Next steps, in order

1. **TAR grid on MTG at epochs >= 10 over >= 3 folds.** A fold-0 screen will not resolve it — see
   `docs/findings/03-methodological-findings.md` for why a single fold cannot resolve a criterion
   whose threshold sits inside the per-(model, fold) noise band.
2. **A fresh Kaggle search feeding the reversed pipeline** — probe Delta_Awareness on a few folds
   early instead of last, since GPU makes TAR affordable and it eliminates far more candidates
   than Delta_Joint does (see `docs/findings/03-methodological-findings.md`).

## Reproduce commands, against the new paths

```bash
python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full \
  --candidate "<owner/slug>=REG_TEXT_<NAME>" --folds 0,1,2,3,4 \
  --models light,cat,tabm,tabpfnv2 --states no_text,text_only,all,ft
python -m curation_lab.kaggle.verdict_from_runs results/curation/<path>.csv
```
