# Repo consolidation — merge the lanes, organize the evidence, name the map

**Date:** 2026-09-02
**Status:** approved, pending implementation plan
**Scope:** repository structure only. No experiment is re-run, no measured number changes.

## Problem

The project's work is split across two diverged branches that do not know about each other's
results, and a third of the evidence is not in git at all.

- **CPU lane** — local branch `origin`, 5 commits unpushed. Holds the
  `REG_TEXT_EDU_UDEMY_ACADEMY` acceptance, the Vietnam housing Delta_Joint grid, the MTG
  card-price grid, and `AUTONOMOUS_MINER_RULES.md`.
- **GPU lane** — `origin/kaggle_work`, 13 commits. Holds the Kaggle notebook harness, the
  `REG_TEXT_HOUSES_VIETNAM_2024` acceptance, and the round-2 rejections.

Neither branch contains both accepted datasets. Additionally:

- `results/candidates/` is gitignored; files were force-added one at a time. 25 files —
  including `DJ_GAMES_REPORT.md`, every `.log`, `derived/metacritic_scored.csv`, and a
  `spec_audit.csv` that exists only inside the `tar-hunt` worktree — are untracked and one
  `git clean` from gone.
- The MTG grid completion (`dj_games.csv`, +12 rows, finishing the 75-cell grid) is
  uncommitted in the working tree.
- Eight conclusions documents partly supersede each other. `PHASE2_RESULTS.md` contains a
  rejected verdict, its correction, and then an acceptance. `DJ_PROPERTY_REPORT.md` states
  Delta_Awareness is unmeasured; `DJ_PROPERTY_TAR_REPORT.md` on the other branch measures it.
- The local branch is named `origin`, colliding with the remote of the same name. Every
  `git rev-list` against it prints `warning: refname 'origin' is ambiguous`.
- Five branches and four worktrees are live, one of them (`worktree-tar-gpu-remote`) sitting
  on plain `master` and never used.
- There is no home for the Track 2 write-up, and `instructions.pdf` sits untracked in the root.

## Goal

One branch containing all code and all evidence; results organized so that each claim sits
next to the file that proves it; conclusions reduced to a canonical, non-contradicting set
with the superseded versions preserved and labelled; a home for the paper; and a README that
tells a reader where everything is.

## Non-goals

- Writing the Track 2 report. `paper/report.md` is a **skeleton with evidence pointers**, not
  a draft. Drafting it is separate work with its own judgment calls.
- Re-running, re-measuring, or re-deriving any result.
- Refactoring `curation_lab/` beyond what the merge itself does.
- Touching `multabench/` — it stays read-only, per CLAUDE.md.
- Deleting remote branches. Local branches are retired; the remote is left alone.

## Design

### 1. Merge

The dry run (`git merge-tree --write-tree refs/heads/origin refs/remotes/origin/kaggle_work`)
returns a clean tree OID with no conflict report. **The merge is conflict-free.** The two
files that looked like clashes are one-sided additions the other lane never touched:
`PHASE2_RESULTS.md` (+68 lines on the CPU lane) and `verify_udemy_e10.csv` (+19 rows).

Of the three source files present on both lanes, each was modified by exactly one lane, so
git resolves all three without help:

| file | changed by | merge result |
|---|---|---|
| `curation_lab/screen/auto_spec.py` | GPU lane only (`84a637e`, `f8c38f6`) | Kaggle version |
| `curation_lab/ingest/candidate.py` | GPU lane only (`f8c38f6`) | Kaggle version |
| `curation_lab/screen/verify.py` | CPU lane only (`c52e10b`) | CPU version |

Order of operations. The sequence matters more than the merge does:

1. **Commit the MTG grid completion first.** `results/candidates/dj_games.csv` has 12
   uncommitted rows completing the 75-cell frozen grid. It is the newest result in the repo
   and exists only in the working tree. A dirty tree also blocks the merge.
2. **Tag safety nets.** `archive/cpu-lane` at the CPU tip and `archive/kaggle-lane` at
   `58a6b25`. Only two are needed: `kaggle_work_tar` (`b4b48cb`), `kaggle_work_frozenfix`
   (`d3fe4a1`) and `tar-hunt` (`58a6b25`) are all ancestors of the kaggle tip, and
   `worktree-tar-gpu-remote` is `master`. Nothing becomes unreachable.
3. **Create `curation-lab`** from the CPU lane tip *as of step 1* — that is, including the MTG
   commit — and merge `origin/kaggle_work` into it. `master` is left as a clean mirror of
   upstream so the official repo can still be pulled.
4. Capture, reorganize, rewrite, document — in that order (sections 2-6).

### 2. Capture before reorganize

`results/candidates/` is gitignored. Before any file moves, the ignore rule is narrowed and
everything is committed in a single **capture commit**:

- `.gitignore`: drop the `results/candidates/` line. Keep `data/candidates/`, `.emb_cache/`,
  `.venv-tools/`, `remote_login.env`, `remote_login.txt` ignored. Add `.tar_cache/`, which is
  a regenerable embedding cache currently untracked and unignored.
- Add all 25 untracked files under `results/candidates/`, plus `spec_audit.csv` copied out of
  the `tar-hunt` worktree, which is the only file unique to a worktree.
- Move `instructions.pdf` from the repo root into `paper/source/` and commit it.

Committing before moving means the reorganization shows up in git as renames rather than as
deletions of files git never knew about. It also means the safety tags cover the untracked
evidence, which they otherwise would not.

Untracked junk **not** captured, listed so the omission is deliberate: the shell-quoting
accidents in the working trees (`+0.0322`, `0.625`, and in the `tar-hunt` worktree `'`,
`100`, `FAILED'`, `0.001).sum()),-`, `browser_auth.ensure_license_accepted()`,
`list[tuple[str`). These are zero-information filenames produced by mis-quoted commands.

### 3. Target layout

```
README.md              the map: what is where, and what is currently true
CLAUDE.md              unchanged (agent instructions)
benchmark.py  multabench/  requirements*  init.sh   upstream, untouched

curation_lab/
  kaggle/              PRIMARY harness — push, build_notebook, verdict_from_runs
  discover/ ingest/ screen/ criterion/ runner/ prep/

paper/
  report.md            section skeleton, each section pointing at its evidence folder
  assets/              tables/figures generated from results/curation/
  source/              instructions.pdf (course brief)
  README.md            what to write, and where each number comes from

docs/
  findings/            canonical conclusions
    01-criterion-and-pipeline.md
    02-mining-method-rules.md
    03-methodological-findings.md
    04-environment-and-performance.md
  status/STATE.md      live handoff (replaces RESUME.md)
  archive/             superseded documents, each with a replacement header
  superpowers/         specs and plans, unchanged

results/curation/
  accepted/ in_progress/ rejected/ screening/ validation/
  INDEX.md
```

`results/` at the repo root contains only our work — the paper's own W&B exports live under
`multabench/leaderboard/results/` — so it can be restructured freely.

### 4. Results taxonomy

Five buckets. Four came from the approved layout; `validation/` was added during mapping,
because four CSVs are runs on the paper anchor `MUL_TEXT_PRODUCT_SENTIMENT` measuring
pipeline fidelity and cache bit-exactness. They are not candidate screening, and forcing them
into the funnel would misrepresent them.

**`accepted/REG_TEXT_EDU_UDEMY_ACADEMY/`**

| destination | source |
|---|---|
| `grid.csv` | `verify_udemy_e10.csv` — the accepted 5x4x5 grid, epochs=10 |
| `grid_epochs2_superseded.csv` | `verify_udemy.csv` — the epochs=2 sweep that produced the wrong 1-of-5 verdict; kept as the evidence for finding 03 |
| `logs/` | `t3_udemy.log`, `t3_udemy2.log`, `t3_udemy_e20.log`, `verify_udemy_e10.log`, `verify_udemy_frozen.log`, `verify_udemy_ft.log` |
| `VERDICT.md` | written from `PHASE2_RESULTS.md`'s final section |

**`accepted/REG_TEXT_HOUSES_VIETNAM_2024/`**

| destination | source |
|---|---|
| `grid_frozen_cpu.csv` | `dj_property.csv` — 75 frozen cells, CPU |
| `grid_frozen_gpu.csv` | `dj_property_tar_frozen.csv` (from the GPU lane) |
| `grid_tar.csv` | `dj_property_tar_all_ft.csv` (from the GPU lane) |
| `logs/` | `dj_property.log` |
| `VERDICT.md` | written from `DJ_PROPERTY_REPORT.md` + `DJ_PROPERTY_TAR_REPORT.md` |

`VERDICT.md` must carry two caveats the source reports establish: TabPFNv2's cell is decided
by float64 representation rather than by evidence and is **not** counted among the three
passes; TabPFN-2.5 failed all ten cells with `TabPFNLicenseError` and is counted as a
non-pass.

**`in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/`**

| destination | source |
|---|---|
| `grid_frozen.csv` | `dj_games.csv` — 75/75 cells, Delta_Joint +0.050..+0.075 on all 5 models |
| `logs/` | `dj_games.log`, `dj_games_finish.log`, `probe_mtg.log` |
| `STATUS.md` | states plainly that Delta_Awareness is unmeasured, so the dataset is not accepted |

**`rejected/`** — one folder each, with a `REJECTION.md` naming the reason:

| folder | grids | logs |
|---|---|---|
| `board_games/` | `boardgames_full.csv` (GPU, 4x4x5), `dj_games_bgg.csv`, `t2_boardgames.csv` | `dj_games_bgg.log`, `t2_boardgames.log` |
| `anime/` | `anime_full.csv` (`REG_TEXT_MEDIA_ANIME_POPULARITY`) | — |
| `metacritic/` | `dj_media_metacritic.csv`, `derived_input.csv` (was `derived/metacritic_scored.csv`) | `dj_media_metacritic.log` |

Board games accumulated three attempts under three different registered names —
`REG_TEXT_SOCIAL_BOARD_GAMES_BGG` (`t2_boardgames.csv`), `REG_TEXT_GAMES_BGG_DESCRIPTION`
(`dj_games_bgg.csv`) and `REG_TEXT_GAMES_BOARDGAMES_BGG` (`boardgames_full.csv`). They belong
in one folder because they are the same source dataset screened three ways, and the folder's
`REJECTION.md` must both record that fact and say that the +0.039 Delta_Joint was an artifact
of the JUNK regex deleting `Year Published` and `Play Time`, vanishing to -0.0005 once they
were restored.

Candidates rejected at screen time without a grid do not get folders. They are recorded in
`rejected/REJECTIONS.md` as a single table with the reason and the screening file that shows
it.

**`screening/`**

| subfolder | files |
|---|---|
| `t0_t1/` | `t1_batch.csv`, `t1_batch.log`, `t1_shortlist.csv`, `novelty_shortlist.csv` |
| `t2_joint/` | `hunt_full.csv/.log`, `hunt_games2.csv/.log`, `hunt_media.csv/.log`, `hunt_smoke.csv`, `screen4_fold0.csv`, `screen_wave2_fold0.csv` |
| `t3_tar/` | `tar_probes.csv`, `batch_tar.log` |
| `spec_audits/` | `spec_audit.csv`, `spec_audit2.csv`, `spec_audit_wave2.csv`, `spec_audit_wave3.csv`, `spec_audit_wave4.csv` |

**`validation/`** — `phase1_grid.csv`, `phase1_lgbm.csv`, `cache_check.csv`,
`tabpfn25_retry.csv`, all on `MUL_TEXT_PRODUCT_SENTIMENT`, plus a `README.md` stating what
they establish: the runner reproduces the paper's `no_text` score exactly
(`0.83454303717305`) and the frozen embedding cache is bit-exact.

**The two lanes wrote different CSV schemas**, and co-locating their files does not reconcile
them:

- CPU lane: `model, dataset, fold, multimodal_state, test_score, runtime, n_train, n_test,
  m_features, task_type, tune_e5`
- GPU lane: `state, score, secs, epochs, dataset, model, fold`

So `accepted/REG_TEXT_HOUSES_VIETNAM_2024/` will hold both shapes side by side. Nothing is
rewritten — converting measured files would violate the "no number changes" scope — but
`INDEX.md` records each file's schema, and any cross-lane read goes through the existing
`curation_lab/criterion/deltas.py::normalize`, which is what the verdict scripts already use.

**`INDEX.md`** — one row per file: new path, original filename, dataset, tier, schema, and
what it proves. The original-filename column is load-bearing: renaming `dj_property.csv` to
`grid_frozen_cpu.csv` breaks the `--out` paths recorded in the reports and logs, and the index
is what keeps those historical commands traceable. Each `VERDICT.md` additionally carries its
reproduce command rewritten against the new path.

### 5. Conclusions

Four canonical documents, each true as of 2026-09-02 with no internal contradictions:

- **`01-criterion-and-pipeline.md`** — the criterion, its validation against the shipped
  56x10 `pass_matrix.csv` with 0 mismatched cells, runner fidelity against the paper, and the
  rule that `passes()` is reused and never reimplemented.
- **`02-mining-method-rules.md`** — the funnel, measured yields, discovery and spec-derivation
  rules, and the positive signatures (baseline balance, lift, CV of delta, orthogonal
  channels). Sourced from `AUTONOMOUS_MINER_RULES.md`, updated where later results overtook
  it — notably its "MTG on track (grid completing)" line, now a finished 75-cell grid.
- **`03-methodological-findings.md`** — leads with the two corrections, because they are the
  most transferable output of the project and each cost real compute to learn:
  *a cheap screen is only valid where cheapness does not change the quantity being screened.*
  True for Delta_Joint under frozen encoders; false for Delta_Awareness, which the epochs=2
  budget and the fold-0 screen each broke in the same way. Also: the junk-regex artifact that
  manufactures Delta_Joint, the multi-column arithmetic leak `find_leaks` cannot see, and the
  float knife-edge where a rounded-mean difference equal to delta clears a strict `>`.
- **`04-environment-and-performance.md`** — the venv / pandas-2.3.3 / `PYTHONIOENCODING`
  constraints and the performance economics: frozen embedding cache ~40x (bit-exact),
  `max_length` cap ~7x (not bit-exact, screening only), TAR encoder sharing taking 25 runs down
  to 10 fine-tunings.

**`docs/status/STATE.md`** replaces `RESUME.md` as the live handoff: current verdicts, what is
running, open blockers (TabPFN-2.5 gated weights; exhausted local candidate pool), and next
steps.

Every superseded document moves to `docs/archive/` **unmodified except for a header** giving
the date, what replaced it, and why: `RESUME.md`, `PHASE2_RESULTS.md`, `RESEARCH_NOTES.md`,
`AUTONOMOUS_MINER_RULES.md`, `DJ_PROPERTY_REPORT.md`, `DJ_PROPERTY_TAR_REPORT.md`,
`DJ_GAMES_REPORT.md`, `HUNT_ROUND2_REPORT.md`.

`DJ_GAMES_REPORT.md` needs splitting rather than straight archiving: its MTG content goes to
the `in_progress` folder, its board-games and per-candidate rejections to the matching
`REJECTION.md` files, and its reusable lesson — that `hunt.py` is a triage net whose spec is
not a curation decision — to `03-methodological-findings.md`. The original still lands in the
archive intact.

### 6. README

Written as a map, not a tutorial:

1. What this repo is — the upstream benchmark plus our curation lab, and which directories
   belong to which.
2. Current state — a table of 2 accepted, 1 in progress, 3 gridded rejections, each linking
   into its evidence folder.
3. Where the conclusions are, and which four documents are canonical.
4. Where the paper lives.
5. How to run the harness — Kaggle GPU as the primary path with its `push` command, CPU as the
   legacy path that produced the Udemy and MTG grids and remains the way to reproduce them.
6. Open blockers.

The existing upstream `README.md` content is preserved in `docs/archive/README-upstream.md`,
since this is a fork and that text describes the benchmark rather than our work.

### 7. Verification

Before any file moves, write a manifest of every file under `results/` and every conclusions
document: path, size, SHA-256. After the reorganization, assert that every source SHA is still
present somewhere in the tree. "All results preserved" is then a checked claim rather than an
assertion.

Files whose content legitimately changes are enumerated as intended exceptions: the four
canonical `docs/findings/` documents, `docs/status/STATE.md`, the per-dataset `VERDICT.md` /
`STATUS.md` / `REJECTION.md` files, `README.md`, `INDEX.md`, `.gitignore`, and the archive
headers. Every other file must round-trip byte-identical.

A second check: `git status` on the new branch is clean, `git log --stat` shows the moves as
renames, and both accepted datasets' grids re-derive their verdicts through
`multabench.leaderboard.analysis.pass_matrix.passes()` — the same function, not a
reimplementation — confirming the reorganization did not disturb the numbers.

### 8. Cleanup

After `curation-lab` is pushed:

- Remove the four worktrees: `dj-frozen-fix`, `kaggle-tar`, `tar-gpu-remote`, `tar-hunt`.
- Delete the local branches `origin`, `kaggle_work_tar`, `kaggle_work_frozenfix`, `tar-hunt`,
  `worktree-tar-gpu-remote`. The ambiguous `origin` name goes away with them.
- `master` and `curation-lab` remain. Remote branches are left untouched.
- The safety tags stay until the user says otherwise.

`remote_login.env` still contains a plaintext password. It is gitignored and key auth works,
so it is deleted as part of cleanup.

## Risks

- **Renaming breaks recorded reproduce commands.** Mitigated by `INDEX.md`'s original-filename
  column and by rewriting each `VERDICT.md`'s command against the new path. The historical logs
  still name the old paths; that is why the index exists.
- **Rewriting conclusions can lose nuance.** Mitigated by archiving originals unmodified, so
  any judgment call in the rewrite stays checkable against its source.
- **The capture commit adds ~1.3 MB of logs to history.** Accepted deliberately: the logs are
  the only record of runtime, extreme-outlier warnings and cache-hit counts that the write-up
  cites, and the size is immaterial.
