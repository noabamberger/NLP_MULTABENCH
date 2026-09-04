# MulTaBench2 — Technion 097215 Track 2

A fork of the official **MulTaBench** benchmark
([arXiv 2605.10616](https://arxiv.org/abs/2605.10616)), used for the Technion NLP course
097215 (Spring 2026) final project, **Track 2 — Benchmark Track**: curate new text-tabular
dataset(s) and prove they pass MulTaBench's own curation pipeline. Deadline **2026-10-26**.

Two layers live in this repo:

- **`multabench/` + `benchmark.py`** — the upstream benchmark. **Read-only** here; not
  modified for this project. Its own README is archived at
  [`docs/archive/README-upstream.md`](docs/archive/README-upstream.md).
- **`curation_lab/`** — this project's own code: dataset mining, screening, the Kaggle GPU
  harness, and the criterion/verdict logic built on top of `multabench`'s own
  `pass_matrix.passes()`.

## Current state

| dataset | Delta_Joint | Delta_Awareness | verdict | evidence |
|---|---|---|---|---|
| `REG_TEXT_EDU_UDEMY_ACADEMY` | +0.136..+0.218 (5/5) | +0.002..+0.021 (3/5) | **ACCEPTED** | [`results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/`](results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/) |
| `REG_TEXT_HOUSES_VIETNAM_2024` | +0.250..+0.324 (5/5) | +0.001..+0.015 (5/5) | **ACCEPTED** | [`results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/`](results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/) |
| `REG_TEXT_GAMES_MTG_CARD_PRICES` | +0.050..+0.075 (5/5) | not measured | in progress | [`results/curation/in_progress/`](results/curation/in_progress/) |
| board games | +0.047..+0.059 | -0.001..+0.003 (2/5, one a knife-edge) | rejected | [`results/curation/rejected/board_games/`](results/curation/rejected/board_games/) |
| anime | +0.031..+0.037 | -0.002..0.000 (0/5) | rejected | [`results/curation/rejected/anime/`](results/curation/rejected/anime/) |
| metacritic | — | — | rejected (82% sentinel target) | [`results/curation/rejected/metacritic/`](results/curation/rejected/metacritic/) |

Every verdict is computed by `multabench.leaderboard.analysis.pass_matrix.passes()` — the
repo's own implementation of the criterion (>=3 of 5 learners, `delta = 0.001`, per-state
means over 5 folds rounded to 3 decimals before differencing) — never reimplemented here.
The counts above are what that function returns, including two cells it passes on a **float
knife-edge**: Vietnam housing's TabPFNv2 and board games' TabM each differ by exactly delta,
and clear a strict `>` only because float64 renders the difference as `0.0010000000000000009`.
Both are flagged where they appear, and neither verdict depends on its cell — dropping them
leaves Vietnam at 4 of 5 (still accepted) and board games at 1 of 5 (still rejected).

**Udemy was measured twice, on two machines, and accepted at 3 of 5 in both — but not by the same
three learners.** The Kaggle T4 grid is the primary evidence (100 cells, every state for a learner
within one session); the earlier CPU grid is retained as the cross-environment comparison.
Delta_Joint reproduced within 0.011 across the two lanes and `no_text` matched to three decimals
for all five models, but two learners flipped on Delta_Awareness in opposite directions (LightGBM
+0.006 -> -0.005, TabPFNv2 -0.001 -> +0.002). Both flips are sub-0.011 moves against a 0.001
threshold. Cite the dataset-level verdict; a per-learner Delta_Awareness on this dataset is not
reproducible across environments. Detail in
[`results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/VERDICT.md`](results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/VERDICT.md).

**One deviation applies to both acceptances and must be disclosed in any writeup:** E5
fine-tuning ran **10 epochs**, not the `E5TrainArgs` default of 50 (patience 3), for compute
feasibility. This affects the `ft` state and therefore Delta_Awareness — the narrower of the two
criteria, and the one Udemy passes at exactly quorum. It is conservative in the direction that
matters (Delta_Awareness grew with epochs in every measurement: LightGBM fold 0 went +0.0099 at 2
epochs to +0.0322 at 10), so the full budget would be expected to widen the margins rather than
narrow them. Detail in
[`docs/findings/04-environment-and-performance.md`](docs/findings/04-environment-and-performance.md).

**Standard scope (1 passing dataset) is met twice. Outstanding scope (>=5 passing datasets)
needs 3 more.**

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

[`docs/status/STATE.md`](docs/status/STATE.md) is the live handoff — read it for what's
actually next. [`docs/archive/`](docs/archive/) holds every document these four superseded;
each archived file carries a header naming its replacement.

## Where the results are

Every measured grid, log and screen lives under `results/curation/`, organized into five
buckets — `accepted/`, `in_progress/`, `rejected/`, `screening/`, `validation/` — with
[`results/curation/INDEX.md`](results/curation/INDEX.md) as the file-level map (original
filenames, both CSV schemas, and what each file proves).

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

## Where the paper is

[`paper/`](paper/) holds the Technion write-up: the assignment brief
(`paper/source/instructions.pdf`), a report **skeleton** (`paper/report.md` — evidence
pointers only, not a draft), and `paper/assets/` for generated tables/figures. See
[`paper/README.md`](paper/README.md).

## How to run the harness

**The pipeline runs on Kaggle GPU.** Every curation grid is measured there — a full 5 models x 4
states x 5 folds grid costs well under an hour of T4 time (~0.65 GPU-h for Udemy's 100 cells,
2-3% of the ~30 h weekly quota), so there is no longer a reason to run a grid locally.

```bash
python -m curation_lab.kaggle.push_code -m "why this push"   # re-version the code dataset FIRST

python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full --full-epochs 10 \
  --candidate "<owner/slug>=REG_TEXT_<NAME>" --folds 0,1,2,3,4 \
  --models light,cat,tabm --states no_text,text_only,all,ft \
  --kernel-id talkraicer/multabench-<name>-lct

python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full --full-epochs 10 \
  --candidate "<owner/slug>=REG_TEXT_<NAME>" --folds 0,1,2,3,4 \
  --models tabpfnv2,tabpfnv2p5 --states no_text,text_only,all,ft \
  --kernel-id talkraicer/multabench-<name>-pfn

python -m curation_lab.kaggle.verdict_from_runs results/curation/<path>/grid_gpu_*.csv
```

Four things about that shape are load-bearing:

- **`push_code` first.** The notebook imports `multabench`/`curation_lab` from the attached
  `talkraicer/multabench-code` dataset, not from a clone, so local changes are invisible to a run
  until the dataset is re-versioned — and the run silently uses the old code.
- **`machine-shape NvidiaTeslaT4`.** Kaggle's default P100 (sm_60) cannot launch kernels under
  this image's torch even though `torch.cuda.is_available()` returns `True`; the failure appears
  only once training starts.
- **Split by model, not by state.** All four states for a learner must share one session:
  Delta_Awareness is a difference of two means, and cross-machine drift lands directly in it.
  Splitting by model also keeps each kernel inside the session cap.
- **`--full-epochs 10` minimum.** A starved epoch budget measures the budget, not the dataset
  (`docs/findings/03-methodological-findings.md`).

**Local CPU is for frozen-only work and reproduction, not for grids.** There is no CUDA on the
local machine, frozen E5 embedding costs ~10 min per run, and TAR fine-tuning is far more
expensive — a full CPU grid takes days of serial wall-clock on the one machine that everything
else also needs. The CPU runner remains available for Delta_Joint-only measurements and for
reproducing the historical grids:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m curation_lab.screen.verify \
  --ref <owner/slug> --name REG_TEXT_<NAME> \
  --out results/curation/<path>.csv --folds 0,1,2,3,4 --epochs 10
```

Environment constraints (see [`CLAUDE.md`](CLAUDE.md) for full detail):

- Use **`.venv/Scripts/python.exe`** — never the system Python (it hosts an unrelated
  project pinned to an incompatible pandas/numpy).
- **pandas must be 2.3.3** — pandas 3.x breaks feature-type detection.
- Always set **`PYTHONIOENCODING=utf-8`** — model names contain characters the console's
  cp1255 codepage can't print.

## Open blockers

- **The local candidate pool is exhausted.** The next step is a fresh T0/T1 Kaggle search
  with the junk-aware profiler — the previous search ranked candidates on a text-column count
  that mistyped dates and ids as text, so the pool was never as rich as it appeared.
- **The TabPFN-2.5 blocker is resolved.** It was a Prior Labs API key requirement, not
  Hugging Face gating (see `docs/findings/04-environment-and-performance.md`). The full
  five-model committee is available for every future grid.
