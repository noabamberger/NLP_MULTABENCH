# Final Project for the NLP Course (00970215) - Technion

A fork of the official **MulTaBench** benchmark
([arXiv 2605.10616](https://arxiv.org/abs/2605.10616)), used for the Technion NLP course
00970215 (Spring 2026) final project, **Track 2 — Benchmark Track**: curate new text-tabular
dataset(s) and prove they pass MulTaBench's own curation pipeline.

This repository contains two main components:

- **`multabench/` + `benchmark.py`** — the upstream benchmark. **Read-only** here; not
  modified for this project. Its own README is archived at
  [`docs/archive/README-upstream.md`](docs/archive/README-upstream.md).
- **`curation_lab/`** — this project's own code: dataset mining, screening, the Kaggle GPU
  harness, and the criterion/verdict logic built on top of `multabench`'s own
  `pass_matrix.passes()`. Its module map and funnel are documented in
  [`curation_lab/README.md`](curation_lab/README.md).

## Current state

Every grid below was measured on Kaggle Tesla T4 GPUs: 5 learners x 4 states x 5 folds, with all
four states for a learner run in one session and `ft` at 10 epochs.

| dataset | Delta_Joint | Delta_Awareness | verdict | evidence |
|---|---|---|---|---|
| `REG_TEXT_HOUSES_VIETNAM_2024` | +0.250..+0.324 (5/5) | +0.001..+0.015 (5/5) | **ACCEPTED** | [`results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/`](results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/) |
| `REG_TEXT_EDU_UDEMY_ACADEMY` | +0.136..+0.218 (5/5) | +0.002..+0.021 (3/5) | **ACCEPTED** | [`results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/`](results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/) |
| MTG card prices | +0.053..+0.074 (5/5) | -0.001..+0.006 (4/5 by `passes()`, 2 of them knife-edges) | **borderline** — not counted as accepted | [`results/curation/rejected/mtg_card_prices/`](results/curation/rejected/mtg_card_prices/) |
| board games | +0.047..+0.059 (5/5) | -0.001..+0.003 (2/5, one a knife-edge) | rejected | [`results/curation/rejected/board_games/`](results/curation/rejected/board_games/) |
| anime | +0.031..+0.037 (4/4 run) | -0.002..0.000 (0/5) | rejected | [`results/curation/rejected/anime/`](results/curation/rejected/anime/) |
| metacritic | — | — | rejected at the target (82% sentinel zeros) | [`results/curation/rejected/metacritic/`](results/curation/rejected/metacritic/) |

Every verdict is computed by `multabench.leaderboard.analysis.pass_matrix.passes()` — the
repo's own implementation of the criterion (>=3 of 5 learners, `delta = 0.001`, per-state
means over 5 folds rounded to 3 decimals before differencing) — never reimplemented here.

**Float knife-edges.** Some cells differ by exactly delta after rounding. They clear the strict
`>` only because float64 renders the difference as `0.0010000000000000009`. `passes()` counts
them as passes, and they are flagged wherever they appear:

- **Vietnam housing** (TabPFNv2) and **board games** (TabM) each have one. Neither verdict
  depends on it: without them Vietnam is 4 of 5 (still accepted) and board games 1 of 5 (still
  rejected).
- **MTG card prices** has two (CatBoost and LightGBM), and here the verdict does depend on them.
  `passes()` returns 4 of 5, but with both counted as fails the count is 2 of 5, below quorum. We
  report it as **borderline** and do not count it as an accepted dataset. Detail in
  [`results/curation/rejected/mtg_card_prices/VERDICT.md`](results/curation/rejected/mtg_card_prices/VERDICT.md).

**Udemy passes at exactly the quorum, and its Delta_Awareness is noisy.** Pooled over the 25
(learner, fold) cells it is +0.0036 +/- 0.0202 (t = 0.89, not significant), and it changes sign
within a single learner across folds. Cite the dataset-level verdict, not a per-learner one.
Detail in
[`results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/VERDICT.md`](results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/VERDICT.md).

**One deviation applies to every grid and must be disclosed in any writeup:** E5 fine-tuning ran
**10 epochs**, not the `E5TrainArgs` default of 50 (patience 3), for compute feasibility. This
affects only the `ft` state and therefore only Delta_Awareness — the narrower of the two
criteria, and the one Udemy passes at exactly quorum. Detail in
[`docs/findings/04-environment-and-performance.md`](docs/findings/04-environment-and-performance.md).

## Where the conclusions are

The canonical write-up of what was learned lives in four documents, in read order:

1. [`docs/findings/01-criterion-and-pipeline.md`](docs/findings/01-criterion-and-pipeline.md)
   — the criterion, and why this runner's numbers can be trusted.
2. [`docs/findings/02-mining-method-rules.md`](docs/findings/02-mining-method-rules.md) —
   the mining/screening rulebook, measured T0-T3 yields included.
3. [`docs/findings/03-methodological-findings.md`](docs/findings/03-methodological-findings.md)
   — the correction trail: what went wrong (epoch budget, fold count, manufactured deltas)
   and what it taught.
4. [`docs/findings/04-environment-and-performance.md`](docs/findings/04-environment-and-performance.md)
   — environment constraints and performance economics, including the resolved TabPFN-2.5
   blocker.

[`docs/archive/`](docs/archive/) holds every document these four superseded; each archived file
carries a header naming its replacement.

## Where the results are

Every measured grid, log and screen lives under `results/curation/`, organized into four
buckets — `accepted/`, `rejected/`, `screening/`, `validation/` — with
[`results/curation/INDEX.md`](results/curation/INDEX.md) as the file-level map (original
filenames, CSV schemas, and what each file proves). The GPU grids use the `kaggle` schema
(`state, score, secs, epochs, dataset, model, fold`).

These files were moved into that layout from a flat directory. That the move destroyed nothing
is a checkable claim, not an assurance — re-run it yourself:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m curation_lab.tools.manifest check   docs/superpowers/plans/consolidation-manifest.json   --except README.md RESUME.md PHASE2_RESULTS.md RESEARCH_NOTES.md            docs/AUTONOMOUS_MINER_RULES.md            results/candidates/DJ_PROPERTY_REPORT.md            results/candidates/DJ_PROPERTY_TAR_REPORT.md            results/candidates/DJ_GAMES_REPORT.md            results/candidates/HUNT_ROUND2_REPORT.md
# LOST: []
```

The manifest records the sha256 of every evidence file as it stood before the reorganization;
the check asks whether each recorded content still exists anywhere in the tree. The nine
exceptions are the documents that gained an archive header — the one intended content change.
Note what this proves and what it does not: it establishes **custody**, not correctness. A file
moved somewhere nonsensical still counts as present, and a file never recorded is invisible to
it. It answers "was anything destroyed", not "is everything where it belongs".

## How to run the harness

**The pipeline runs on Kaggle GPU.** A full 5 models x 4 states x 5 folds grid costs well under an
hour of T4 time (~0.65 GPU-h for Udemy's 100 cells, 2-3% of the ~30 h weekly quota).

```bash
python -m curation_lab.kaggle.push_code -m "why this push"   # re-version the code dataset FIRST

python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full --full-epochs 10 \
  --candidate "<owner/slug>=REG_TEXT_<NAME>" --folds 0,1,2,3,4 \
  --models light,cat,tabm --states no_text,text_only,all,ft \
  --kernel-id noabamberger/multabench-<name>-lct

python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full --full-epochs 10 \
  --candidate "<owner/slug>=REG_TEXT_<NAME>" --folds 0,1,2,3,4 \
  --models tabpfnv2,tabpfnv2p5 --states no_text,text_only,all,ft \
  --kernel-id noabamberger/multabench-<name>-pfn

python -m curation_lab.kaggle.verdict_from_runs results/curation/<path>/grid_gpu_*.csv
```

Four things about that shape are load-bearing:

- **`push_code` first.** The notebook imports `multabench`/`curation_lab` from the attached
  `noabamberger/multabench-code` dataset, not from a clone, so local changes are invisible to a run
  until the dataset is re-versioned — and the run silently uses the old code.
- **`machine-shape NvidiaTeslaT4`.** Kaggle's default P100 (sm_60) cannot launch kernels under
  this image's torch even though `torch.cuda.is_available()` returns `True`; the failure appears
  only once training starts.
- **Split by model, not by state.** All four states for a learner must share one session:
  Delta_Awareness is a difference of two means, and cross-machine drift lands directly in it.
  Splitting by model also keeps each kernel inside the session cap.
- **`--full-epochs 10` minimum.** A starved epoch budget measures the budget, not the dataset
  (`docs/findings/03-methodological-findings.md`).

The push and verdict tools run from the local checkout. Environment constraints for that (see
[`CLAUDE.md`](CLAUDE.md) for full detail):

- Use **`.venv/Scripts/python.exe`** — never the system Python (it hosts an unrelated
  project pinned to an incompatible pandas/numpy).
- **pandas must be 2.3.3** — pandas 3.x breaks feature-type detection.
- Always set **`PYTHONIOENCODING=utf-8`** — model names contain characters the console's
  cp1255 codepage can't print.

## The committee is complete

All five learners — TabM, CatBoost, LightGBM, TabPFN v2 and TabPFN v2.5 — are measured in every
accepted grid. TabPFN-2.5 was blocked for part of the project by a Prior Labs API key
requirement, misdiagnosed for weeks as Hugging Face gating; the key resolved it. No verdict here
rests on an absent model. The episode is written up as a methodological anecdote in
[`docs/findings/04-environment-and-performance.md`](docs/findings/04-environment-and-performance.md).
