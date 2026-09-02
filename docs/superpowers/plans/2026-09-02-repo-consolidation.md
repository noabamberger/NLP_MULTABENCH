# Repo Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the two diverged work lanes into one `curation-lab` branch, commit the evidence that is currently untracked, reorganize results so each claim sits beside the file that proves it, reduce eight overlapping conclusion documents to four canonical ones, create a home for the Track 2 write-up, and write a README that maps it all — proving by SHA manifest that no result was lost.

**Architecture:** Sequential git operations with a content-addressed safety net. A manifest of SHA-256 hashes is written before anything moves; because it is content-addressed, renames are invisible to it and only genuine loss shows up. Files are committed *before* they are moved so git records renames rather than deletions of files it never knew about. Conclusions are rewritten into canonical documents while the originals are preserved verbatim under a header naming their replacement.

**Tech Stack:** git (merge, mv, tags, worktrees), Python 3.11 in `.venv`, pytest, pandas 2.3.3.

**Spec:** `docs/superpowers/specs/2026-09-02-repo-consolidation-design.md`

## Global Constraints

- Use `.venv/Scripts/python.exe` for every Python invocation. Never the system Python — it holds pandas 3.0.3, which crashes this repo.
- Always set `PYTHONIOENCODING=utf-8`. The console codepage is cp1255 and emoji model names raise `UnicodeEncodeError`.
- Pass `encoding="utf-8"` to every `read_csv` / `to_csv`.
- `multabench/` is READ-ONLY. All new code lives in `curation_lab/`.
- **No measured number changes.** No experiment is re-run, no result CSV is rewritten or reformatted. Only file locations and prose change.
- Test suite baseline is **51 passed, 1 failed, 1 skipped**. The single failure is `tests/curation_lab/test_tar_cache.py::test_training_passage_matches_what_the_dataset_tokenizes`, pre-existing and identical on both lanes. It must stay the *only* failure; do not fix it (out of scope) and do not let the count grow.
- Run all git commands from the repo root `C:\Noa\nlp\project\MulTaBench2`. The Bash tool's working directory drifts between calls — prefer absolute paths or an explicit `cd` to the root at the start of each command.
- Commit messages end with the two attribution lines used throughout this repo:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
  ```

---

## File Structure

**New code (one file, one responsibility):**

- `curation_lab/tools/__init__.py` — package marker.
- `curation_lab/tools/manifest.py` — content-addressed manifest: build, write, check. This is the only new module. It exists because the claim "all results preserved" must be checked rather than asserted.
- `tests/curation_lab/test_manifest.py` — its tests.

**New documents:**

- `results/curation/INDEX.md` — every result file: new path, original filename, dataset, tier, schema, what it proves.
- `results/curation/{accepted,in_progress,rejected}/*/` — one `VERDICT.md` / `STATUS.md` / `REJECTION.md` per dataset folder.
- `results/curation/rejected/REJECTIONS.md` — screen-time rejections that never got a grid.
- `results/curation/validation/README.md` — what the anchor runs establish.
- `docs/findings/0{1,2,3,4}-*.md` — the four canonical conclusion documents.
- `docs/status/STATE.md` — live handoff, replaces `RESUME.md`.
- `docs/archive/*` — nine superseded documents, each with a replacement header.
- `paper/{report.md,README.md}`, `paper/source/instructions.pdf`, `paper/assets/.gitkeep`.
- `README.md` — rewritten as the map.

**Modified:** `.gitignore` (narrow the ignore rule).

---

## Task 1: Manifest tool

Builds the safety net before anything is touched. Content-addressed so that a rename is not a loss.

**Files:**
- Create: `curation_lab/tools/__init__.py`
- Create: `curation_lab/tools/manifest.py`
- Test: `tests/curation_lab/test_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `sha256_of(path: Path) -> str`
  - `build(roots: list[Path], repo_root: Path) -> dict[str, str]` — maps repo-relative POSIX path to SHA-256 hex.
  - `write_manifest(entries: dict[str, str], out: Path) -> None` — JSON, sorted keys.
  - `check(manifest_path: Path, repo_root: Path, exceptions: set[str]) -> list[str]` — returns repo-relative paths from the manifest whose content is no longer present anywhere under `repo_root`, excluding `exceptions`.

- [ ] **Step 1: Write the failing test**

Create `tests/curation_lab/test_manifest.py`:

```python
"""The manifest is what turns "nothing was lost" from a claim into a check."""
import json
from pathlib import Path

from curation_lab.tools.manifest import build, check, sha256_of, write_manifest


def _seed(root: Path) -> None:
    (root / "results").mkdir(parents=True)
    (root / "results" / "a.csv").write_text("alpha\n", encoding="utf-8")
    (root / "results" / "b.log").write_text("bravo\n", encoding="utf-8")


def test_sha256_is_stable_and_content_addressed(tmp_path: Path) -> None:
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    one.write_text("same\n", encoding="utf-8")
    two.write_text("same\n", encoding="utf-8")
    assert sha256_of(one) == sha256_of(two)


def test_build_uses_repo_relative_posix_paths(tmp_path: Path) -> None:
    _seed(tmp_path)
    entries = build([tmp_path / "results"], tmp_path)
    assert set(entries) == {"results/a.csv", "results/b.log"}


def test_a_renamed_file_is_not_reported_lost(tmp_path: Path) -> None:
    _seed(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(build([tmp_path / "results"], tmp_path), manifest)

    (tmp_path / "results" / "curation").mkdir()
    (tmp_path / "results" / "a.csv").rename(tmp_path / "results" / "curation" / "grid.csv")

    assert check(manifest, tmp_path, exceptions=set()) == []


def test_a_deleted_file_is_reported_lost(tmp_path: Path) -> None:
    _seed(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(build([tmp_path / "results"], tmp_path), manifest)

    (tmp_path / "results" / "b.log").unlink()

    assert check(manifest, tmp_path, exceptions=set()) == ["results/b.log"]


def test_an_intentionally_rewritten_file_is_excused_by_exceptions(tmp_path: Path) -> None:
    _seed(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(build([tmp_path / "results"], tmp_path), manifest)

    (tmp_path / "results" / "b.log").write_text("rewritten\n", encoding="utf-8")

    assert check(manifest, tmp_path, exceptions={"results/b.log"}) == []


def test_manifest_round_trips_as_sorted_json(tmp_path: Path) -> None:
    _seed(tmp_path)
    manifest = tmp_path / "manifest.json"
    entries = build([tmp_path / "results"], tmp_path)
    write_manifest(entries, manifest)

    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    assert loaded == entries
    assert list(loaded) == sorted(loaded)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/curation_lab/test_manifest.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'curation_lab.tools'`.

- [ ] **Step 3: Write the implementation**

Create `curation_lab/tools/__init__.py` (empty file).

Create `curation_lab/tools/manifest.py`:

```python
"""Content-addressed manifest of the repo's evidence files.

A reorganization moves and renames files. Recording paths alone would flag every
rename as a change; recording content hashes means only genuine loss shows up.
`check` therefore asks "does this content still exist anywhere?", not "is it still
at this path?".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def build(roots: list[Path], repo_root: Path) -> dict[str, str]:
    """Map repo-relative POSIX path -> sha256 for every file under `roots`."""
    entries: dict[str, str] = {}
    for root in roots:
        if root.is_file():
            entries[root.relative_to(repo_root).as_posix()] = sha256_of(root)
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                entries[path.relative_to(repo_root).as_posix()] = sha256_of(path)
    return dict(sorted(entries.items()))


def write_manifest(entries: dict[str, str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(dict(sorted(entries.items())), indent=2) + "\n",
        encoding="utf-8",
    )


def _hashes_present(repo_root: Path) -> set[str]:
    present: set[str] = set()
    skip = {".git", ".venv", "__pycache__", ".tar_cache", ".emb_cache"}
    for path in repo_root.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file():
            present.add(sha256_of(path))
    return present


def check(manifest_path: Path, repo_root: Path, exceptions: set[str]) -> list[str]:
    """Return manifest paths whose content is no longer anywhere in the tree."""
    recorded: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
    present = _hashes_present(repo_root)
    return [
        path
        for path, digest in sorted(recorded.items())
        if path not in exceptions and digest not in present
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/curation_lab/test_manifest.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git add curation_lab/tools tests/curation_lab/test_manifest.py && git commit -F - <<'MSG'
feat(tools): content-addressed manifest for the reorganization

The consolidation moves and renames nearly every evidence file. A path-based
record would flag each rename as a change and prove nothing; hashing content
means a rename is invisible and only genuine loss is reported.

`check` asks whether each recorded file's content still exists anywhere under
the repo, with an explicit exceptions set for the documents the plan
deliberately rewrites.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

---

## Task 2: Capture the working tree and set the safety net

Nothing may move until the MTG grid is committed, the untracked evidence is recorded, and the two lane tips are tagged.

**Files:**
- Modify: `results/candidates/dj_games.csv` (commit the 12 pending rows)
- Create: `docs/superpowers/plans/consolidation-manifest.json`

**Interfaces:**
- Consumes: `curation_lab.tools.manifest.build`, `write_manifest` from Task 1.
- Produces: `docs/superpowers/plans/consolidation-manifest.json`, tags `archive/cpu-lane` and `archive/kaggle-lane`.

- [ ] **Step 1: Verify the MTG grid is complete before committing it**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import pandas as pd
d = pd.read_csv('results/candidates/dj_games.csv', encoding='utf-8')
print('rows:', len(d))
print('cells:', d.groupby(['model','multimodal_state','fold']).ngroups)
print('states:', sorted(d.multimodal_state.unique()))
print('models:', sorted(d.model.unique()))
"
```

Expected: `rows: 75`, `cells: 75`, three frozen states (`all`, `no_text`, `text_only`), five models. If rows is not 75, stop — the grid is not what the spec assumed and the plan's `in_progress` classification needs revisiting.

- [ ] **Step 2: Commit the MTG grid completion**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git add results/candidates/dj_games.csv && git commit -F - <<'MSG'
data: MTG card prices frozen grid complete at 75/75 cells

Finishes the grid left at 63/75 in c52e10b. Delta_Joint is positive on all five
models, +0.050 to +0.075 against a threshold of 0.001:

  TabM +0.075  TabPFN-2.5 +0.068  CatBoost +0.062  LightGBM +0.057  TabPFNv2 +0.050

Frozen states only. Delta_Awareness is NOT measured, so this dataset is not
accepted -- it is one TAR grid away from a verdict.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

- [ ] **Step 3: Copy the worktree-only file into the main tree**

`spec_audit.csv` exists only inside the `tar-hunt` worktree and is in no branch.

```bash
cd /c/Noa/nlp/project/MulTaBench2 && cp .claude/worktrees/tar-hunt/results/candidates/spec_audit.csv results/candidates/spec_audit.csv && ls -la results/candidates/spec_audit.csv
```

Expected: the file exists in `results/candidates/`.

- [ ] **Step 4: Write the manifest**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
from pathlib import Path
from curation_lab.tools.manifest import build, write_manifest
root = Path('.').resolve()
roots = [root / 'results']
roots += [root / n for n in [
    'RESUME.md', 'PHASE2_RESULTS.md', 'RESEARCH_NOTES.md', 'README.md',
    'docs/AUTONOMOUS_MINER_RULES.md',
]]
entries = build(roots, root)
write_manifest(entries, root / 'docs/superpowers/plans/consolidation-manifest.json')
print('recorded', len(entries), 'files')
"
```

Expected: `recorded 45 files` (39 files + `derived/metacritic_scored.csv` + `spec_audit.csv` under `results/`, plus the 5 named documents). Any count is acceptable — record what it prints, it is the baseline for Task 9.

- [ ] **Step 5: Tag both lane tips**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git tag archive/cpu-lane HEAD && git tag archive/kaggle-lane origin/kaggle_work && git tag -l 'archive/*' --format='%(refname:short) %(objectname:short) %(contents:subject)'
```

Expected: two tags listed. `archive/cpu-lane` points at the MTG commit from Step 2; `archive/kaggle-lane` at `58a6b25`.

- [ ] **Step 6: Commit the manifest and the recovered file**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git add -f results/candidates/spec_audit.csv docs/superpowers/plans/consolidation-manifest.json && git commit -F - <<'MSG'
chore: manifest and safety tags before the consolidation

Records the sha256 of every evidence file so the reorganization can be checked
rather than trusted, and recovers spec_audit.csv, which existed only inside the
tar-hunt worktree and was in no branch at all.

Tags archive/cpu-lane and archive/kaggle-lane pin both lane tips so nothing
becomes unreachable once the branches are retired.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

---

## Task 3: Create `curation-lab` and merge the GPU lane

**Files:**
- No file edits. Branch creation and merge only.

**Interfaces:**
- Consumes: tags from Task 2.
- Produces: branch `curation-lab` containing both lanes.

- [ ] **Step 1: Confirm the merge is still clean**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git merge-tree --write-tree HEAD origin/kaggle_work
```

Expected: a single 40-character tree OID and nothing else. Any line containing `CONFLICT` means the assumption in the spec no longer holds — stop and re-verify before merging.

- [ ] **Step 2: Create the branch**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git switch -c curation-lab && git branch --show-current
```

Expected: `curation-lab`.

- [ ] **Step 3: Merge**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git merge --no-ff origin/kaggle_work -m "merge: fold the Kaggle GPU lane into curation-lab

Neither lane held both accepted datasets. The CPU lane carried the
REG_TEXT_EDU_UDEMY_ACADEMY acceptance, the Vietnam housing Delta_Joint grid and
the MTG grid; origin/kaggle_work carried the Kaggle notebook harness, the
REG_TEXT_HOUSES_VIETNAM_2024 acceptance and the round-2 rejections.

Conflict-free: the three shared source files were each modified by exactly one
lane, so auto_spec.py and ingest/candidate.py arrive from the Kaggle lane and
verify.py from the CPU lane.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo"
```

Expected: merge completes with no conflict prompt.

- [ ] **Step 4: Verify both lanes' code is present**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && ls curation_lab/kaggle/ curation_lab/prep/ && ls curation_lab/screen/ | sort
```

Expected: `curation_lab/kaggle/` holds `push.py`, `build_notebook.py`, `verdict_from_runs.py`, `push_code.py`, `compare_environments.py`; `curation_lab/prep/` holds `mtg_cards.py`; `curation_lab/screen/` holds both lanes' modules including `audit_specs.py`, `games_specs.py`, `media_specs.py`, `verify_games.py`, `verify_spec.py`.

- [ ] **Step 5: Verify the merge changed no measured numbers and broke no test**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/curation_lab/ -q 2>&1 | tail -3
```

Expected: `1 failed, 57 passed, 1 skipped` — the 51 baseline passes plus the 6 new manifest tests, with the same single pre-existing failure. If a *different* test fails, stop and investigate; the merge was supposed to be inert.

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import pandas as pd
d = pd.read_csv('results/candidates/verify_udemy_e10.csv', encoding='utf-8')
print('udemy rows:', len(d))
"
```

Expected: `udemy rows: 100` — the CPU lane's fuller version survived the merge, not the GPU lane's shorter one.

---

## Task 4: Capture the untracked evidence

**Files:**
- Modify: `.gitignore`
- Add: 25 untracked files under `results/candidates/`

**Interfaces:**
- Consumes: branch from Task 3.
- Produces: a fully tracked `results/` tree, ready to be moved as renames.

- [ ] **Step 1: Narrow the ignore rule**

In `.gitignore`, replace this block:

```
# curation_lab working data
data/candidates/
results/candidates/
.emb_cache/
```

with:

```
# curation_lab working data (results are tracked; caches and raw downloads are not)
data/candidates/
.emb_cache/
.tar_cache/
```

- [ ] **Step 2: Confirm what is about to be added**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git status --short results/ | sort
```

Expected: ~25 `??` lines including `DJ_GAMES_REPORT.md`, `derived/`, and every `.log`. There must be no `??` line for a filename that looks like a shell-quoting accident.

- [ ] **Step 3: Add the evidence, excluding junk**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git add .gitignore results/ && git status --short | grep -c '^A' && git status --short | grep -vE '^(A|M) ' | head
```

Expected: a count of roughly 25 added files, and the second command lists only the untracked junk in the repo root (`+0.0322`, `.tar_cache/`) which must **not** be staged. If `+0.0322` appears as staged, unstage it with `git restore --staged '+0.0322'`.

- [ ] **Step 4: Commit**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git commit -F - <<'MSG'
data: track the evidence that was only ever on disk

results/candidates/ was gitignored and files were force-added one at a time, so
25 files were untracked and one `git clean` from gone -- among them
DJ_GAMES_REPORT.md, every run log, and derived/metacritic_scored.csv.

The logs are not incidental: they are the only record of runtime, the
extreme-outlier warnings, and the cache-hit counts behind the claim that TAR
encoder sharing turns 25 runs into 10 fine-tunings. 1.3 MB total.

The ignore rule now covers only regenerable artifacts: raw downloads under
data/candidates/ and the two embedding caches.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

- [ ] **Step 5: Verify nothing is left behind**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git status --short results/ && echo "--- clean if empty above ---"
```

Expected: no output before the marker line.

- [ ] **Step 6: Regenerate the manifest over the complete tree**

The Task 2 manifest was taken *before* the merge, so it does not cover the 13 files that
arrived from the GPU lane (`anime_full.csv`, `boardgames_full.csv`, `dj_property_tar_*.csv`,
`novelty_shortlist.csv`, `screen*_fold0.csv`, `spec_audit_wave*.csv`, and the two reports).
Task 9 would pass while one of them was lost. Right now — merged, fully captured, nothing yet
moved — is the moment the tree holds everything at its original path.

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
from pathlib import Path
from curation_lab.tools.manifest import build, write_manifest
root = Path('.').resolve()
roots = [root / 'results']
roots += [root / n for n in [
    'RESUME.md', 'PHASE2_RESULTS.md', 'RESEARCH_NOTES.md', 'README.md',
    'docs/AUTONOMOUS_MINER_RULES.md',
]]
entries = build(roots, root)
write_manifest(entries, root / 'docs/superpowers/plans/consolidation-manifest.json')
print('recorded', len(entries), 'files')
"
```

Expected: a count larger than Task 2's, covering both lanes' files. Sanity-check that these
GPU-lane names now appear:

```bash
cd /c/Noa/nlp/project/MulTaBench2 && grep -c 'anime_full\|boardgames_full\|dj_property_tar\|spec_audit_wave\|HUNT_ROUND2' docs/superpowers/plans/consolidation-manifest.json
```

Expected: `8` or more.

- [ ] **Step 7: Commit the regenerated manifest**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git add docs/superpowers/plans/consolidation-manifest.json && git commit -F - <<'MSG'
chore: re-take the manifest now that both lanes are present

The first manifest predated the merge, so it covered neither the GPU lane's
grids nor its two reports -- the verification in Task 9 would have passed while
one of them was lost. This snapshot is taken at the one moment the tree holds
every file at its original path: merged, fully captured, nothing yet moved.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

---

## Task 5: Reorganize results into `results/curation/`

Pure `git mv`. No file content changes.

**Files:**
- Move: everything under `results/candidates/` to `results/curation/**`
- Move: four markdown reports out of `results/candidates/` to `docs/archive/`

**Interfaces:**
- Consumes: tracked tree from Task 4.
- Produces: the `results/curation/{accepted,in_progress,rejected,screening,validation}` tree consumed by Tasks 6 and 9.

- [ ] **Step 1: Create the directory skeleton**

```bash
cd /c/Noa/nlp/project/MulTaBench2/results && mkdir -p \
  curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/logs \
  curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/logs \
  curation/in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/logs \
  curation/rejected/board_games/logs \
  curation/rejected/anime \
  curation/rejected/metacritic/logs \
  curation/screening/t0_t1 curation/screening/t2_joint \
  curation/screening/t3_tar curation/screening/spec_audits \
  curation/validation && find curation -type d | sort
```

Expected: 15 directories listed.

- [ ] **Step 2: Move the accepted datasets**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && C=results/candidates && U=results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY && V=results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024 && \
git mv $C/verify_udemy_e10.csv $U/grid.csv && \
git mv $C/verify_udemy.csv $U/grid_epochs2_superseded.csv && \
git mv $C/t3_udemy.log $C/t3_udemy2.log $C/t3_udemy_e20.log $C/verify_udemy_e10.log $C/verify_udemy_frozen.log $C/verify_udemy_ft.log $U/logs/ && \
git mv $C/dj_property.csv $V/grid_frozen_cpu.csv && \
git mv $C/dj_property_tar_frozen.csv $V/grid_frozen_gpu.csv && \
git mv $C/dj_property_tar_all_ft.csv $V/grid_tar.csv && \
git mv $C/dj_property.log $V/logs/ && \
find results/curation/accepted -type f | sort
```

Expected: 12 files listed — 2 grids + 6 logs under Udemy, 3 grids + 1 log under Vietnam.

- [ ] **Step 3: Move the in-progress dataset**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && C=results/candidates && M=results/curation/in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES && \
git mv $C/dj_games.csv $M/grid_frozen.csv && \
git mv $C/dj_games.log $C/dj_games_finish.log $C/probe_mtg.log $M/logs/ && \
find $M -type f | sort
```

Expected: `grid_frozen.csv` plus three logs.

- [ ] **Step 4: Move the rejections**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && C=results/candidates && R=results/curation/rejected && \
git mv $C/boardgames_full.csv $R/board_games/grid_gpu_full.csv && \
git mv $C/dj_games_bgg.csv $R/board_games/grid_bgg_description.csv && \
git mv $C/t2_boardgames.csv $R/board_games/grid_t2_screen.csv && \
git mv $C/dj_games_bgg.log $C/t2_boardgames.log $R/board_games/logs/ && \
git mv $C/anime_full.csv $R/anime/grid.csv && \
git mv $C/dj_media_metacritic.csv $R/metacritic/grid.csv && \
git mv $C/derived/metacritic_scored.csv $R/metacritic/derived_input.csv && \
git mv $C/dj_media_metacritic.log $R/metacritic/logs/ && \
find $R -type f | sort
```

Expected: 9 files across three folders.

- [ ] **Step 5: Move the screening and validation files**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && C=results/candidates && S=results/curation/screening && \
git mv $C/t1_batch.csv $C/t1_batch.log $C/t1_shortlist.csv $C/novelty_shortlist.csv $S/t0_t1/ && \
git mv $C/hunt_full.csv $C/hunt_full.log $C/hunt_games2.csv $C/hunt_games2.log $C/hunt_media.csv $C/hunt_media.log $C/hunt_smoke.csv $C/screen4_fold0.csv $C/screen_wave2_fold0.csv $S/t2_joint/ && \
git mv $C/tar_probes.csv $C/batch_tar.log $S/t3_tar/ && \
git mv $C/spec_audit.csv $C/spec_audit2.csv $C/spec_audit_wave2.csv $C/spec_audit_wave3.csv $C/spec_audit_wave4.csv $S/spec_audits/ && \
git mv $C/phase1_grid.csv $C/phase1_lgbm.csv $C/cache_check.csv $C/tabpfn25_retry.csv results/curation/validation/ && \
find $S results/curation/validation -type f | sort
```

Expected: 20 screening files + 4 validation files.

- [ ] **Step 6: Move the four reports to the archive and confirm the old tree is empty**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && mkdir -p docs/archive && C=results/candidates && \
git mv $C/DJ_PROPERTY_REPORT.md $C/DJ_PROPERTY_TAR_REPORT.md $C/DJ_GAMES_REPORT.md $C/HUNT_ROUND2_REPORT.md docs/archive/ && \
find $C -type f | sort && echo "--- results/candidates is empty if nothing above ---"
```

Expected: no files listed before the marker. If anything remains, it was missed by the mapping — file it before continuing rather than leaving it behind.

- [ ] **Step 7: Verify git recorded moves as renames, then commit**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git status --short | grep -c '^R' && git commit -F - <<'MSG'
refactor(results): one folder per claim

results/candidates/ was a flat pile of 45 files with names like dj_games_bgg.csv
that said nothing about what they proved. Each dataset now owns a folder holding
its grids, its logs and (next commit) its verdict, so a reader opening
accepted/REG_TEXT_HOUSES_VIETNAM_2024/ finds the claim beside the file that
supports it.

Five buckets: accepted, in_progress, rejected, screening, validation. The last
was not in the original layout -- phase1_grid, phase1_lgbm, cache_check and
tabpfn25_retry are all runs on the paper anchor MUL_TEXT_PRODUCT_SENTIMENT
measuring pipeline fidelity, not candidate screening, and filing them under the
funnel would have misrepresented them.

Renames only. No file content changed, so no measured number moved.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

Expected: the count printed before the commit is ~45, and `git log --stat -1` shows renames rather than add/delete pairs.

---

## Task 6: Per-dataset verdict documents and the index

Each folder gets the document that says what it proves. All numbers below are transcribed from the archived reports — do not recompute them.

**Files:**
- Create: `results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/VERDICT.md`
- Create: `results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md`
- Create: `results/curation/in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/STATUS.md`
- Create: `results/curation/rejected/{board_games,anime,metacritic}/REJECTION.md`
- Create: `results/curation/rejected/REJECTIONS.md`
- Create: `results/curation/validation/README.md`
- Create: `results/curation/INDEX.md`

**Interfaces:**
- Consumes: the tree from Task 5; source facts from `docs/archive/*.md`.
- Produces: documents linked from `README.md` in Task 8.

- [ ] **Step 1: Write the Udemy verdict**

`results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/VERDICT.md` must contain, verbatim from `docs/archive/PHASE2_RESULTS.md`:

- Source `mariahalshiekh/udemy-course-academy-teaching`, target `price` (regression), text `course_name` + `course_instr`, structured course statistics.
- **ACCEPTED**, 3 of 5 (quorum 3), complete 5×4×5 grid with no missing cells, verdict computed by `multabench.leaderboard.analysis.pass_matrix.passes()`.
- The per-model table:

  | model | Delta_Joint | Delta_Awareness | verdict |
  |---|---|---|---|
  | CatBoost | 0.194 | 0.010 | PASS |
  | LightGBM | 0.209 | 0.006 | PASS |
  | TabM | 0.208 | -0.007 | fail |
  | TabPFN-2.5 | 0.140 | 0.016 | PASS |
  | TabPFNv2 | 0.136 | -0.001 | fail |

- Why it is not marginal: Delta_Joint is 136x-209x the delta of 0.001, and both unimodal baselines are non-degenerate (`no_text` ~0.30, `text_only` ~0.20), so `all` beats each modality rather than winning by default.
- The deviation to disclose: E5 fine-tuning ran **10 epochs** rather than the `E5TrainArgs` default of 50 (patience 3), for CPU feasibility. Conservative in the direction that matters — LightGBM fold 0 went +0.0099 at 2 epochs to +0.0322 at 10 — so the full budget would be expected to widen the margin.
- A pointer to `grid_epochs2_superseded.csv` explaining that it is the epochs=2 sweep whose 1-of-5 rejection was an artifact, retained as the evidence for `docs/findings/03-methodological-findings.md`.
- Reproduce command, rewritten against the new path:

  ```bash
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m curation_lab.screen.verify \
    --ref mariahalshiekh/udemy-course-academy-teaching \
    --name REG_TEXT_EDU_UDEMY_ACADEMY \
    --out results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/grid.csv \
    --folds 0,1,2,3,4 --epochs 10
  ```

- [ ] **Step 2: Write the Vietnam verdict**

`results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md`, merging `docs/archive/DJ_PROPERTY_REPORT.md` and `docs/archive/DJ_PROPERTY_TAR_REPORT.md`:

- Source `nguyentiennhan/vietnam-housing-dataset-2024`, target `Price` (687 distinct, |z|max 2.54), text `Address` — genuine free text that no structured column duplicates. 6 numeric + 4 categorical survive; no leakage columns detected.
- **ACCEPTED**, 3 of 5. All four states were measured on one machine (Kaggle T4), 4 models × 4 states × 5 folds, no gaps — the GPU `ft` half was deliberately *not* differenced against the CPU `all` half.
- The per-model table:

  | model | no_text | text_only | all | ft | Delta_Joint | Delta_Awareness |
  |---|---|---|---|---|---|---|
  | LightGBM | 0.342 | 0.310 | 0.592 | 0.607 | +0.250 | +0.015 |
  | TabM | 0.312 | 0.336 | 0.633 | 0.638 | +0.297 | +0.005 |
  | CatBoost | 0.341 | 0.322 | 0.632 | 0.636 | +0.291 | +0.004 |
  | TabPFNv2 | 0.342 | 0.340 | 0.646 | 0.647 | +0.304 | +0.001 |
  | TabPFN-2.5 | — | — | — | — | — | — |

- **Two caveats that must appear.** TabPFNv2 is *not* counted among the three passes: the difference of its rounded means is exactly delta and clears a strict `>` only because float64 renders `0.647 - 0.646` as `0.0010000000000000009` — decided by representation, not evidence. TabPFN-2.5 failed all ten cells with `TabPFNLicenseError` (gated weights) and is counted as a non-pass, the paper's treatment of an empty cell and the conservative direction.
- The CPU frozen grid (`grid_frozen_cpu.csv`) stands on its own: mean Delta_Joint 0.2872, std 0.0247, 25/25 cells positive, t = 58.03, against a fold-noise band of about ±0.015.
- The domain-novelty caveat: MulTaBench already contains four housing datasets; this is a different market and a different task, but the domain overlaps. Novelty matters for the writeup, not for whether the criterion is met.
- Note the two CSV schemas: `grid_frozen_cpu.csv` uses the CPU columns, `grid_frozen_gpu.csv` and `grid_tar.csv` the Kaggle columns. Read both through `curation_lab.criterion.deltas.normalize`.

- [ ] **Step 3: Write the MTG status**

`results/curation/in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/STATUS.md`:

- Source `douglascampospires/mtg-all-cards`, target `price_usd_log10`, text `CARD_TEXT` + `TYPE`, structured cost and scarcity (CMC, editions, power, toughness, first-edition year, rarity, colour).
- **NOT ACCEPTED — Delta_Awareness is unmeasured.** Frozen grid complete at 75/75 cells; Delta_Joint positive on all five models: TabM +0.075, TabPFN-2.5 +0.068, CatBoost +0.062, LightGBM +0.057, TabPFNv2 +0.050.
- Why it is a good candidate: text encodes *what the card does* while the structured block encodes *cost and scarcity* — independent price drivers, the orthogonal-channel signature in `docs/findings/02-mining-method-rules.md`.
- What it needs: a TAR grid at epochs ≥ 10 over ≥ 3 folds. A fold-0 screen will not resolve it — see finding 03.
- The `price_usd_log10` target carries |z| up to 5.36; the target was log10-transformed and outliers were warned about, never clipped.

- [ ] **Step 4: Write the three rejection notes and the screen-time rejections table**

`rejected/board_games/REJECTION.md` — the same source dataset screened three ways under three registered names: `REG_TEXT_SOCIAL_BOARD_GAMES_BGG` (`grid_t2_screen.csv`), `REG_TEXT_GAMES_BGG_DESCRIPTION` (`grid_bgg_description.csv`), `REG_TEXT_GAMES_BOARDGAMES_BGG` (`grid_gpu_full.csv`). Two independent reasons for rejection, both of which must be stated:

1. Its screening Delta_Joint of +0.039 was an artifact. `auto_spec.JUNK` deleted `Year Published` and `Play Time` — not identifiers but two of the strongest structured predictors a board game has. Restoring them:

   | spec | no_text | text_only | all | Delta_Joint |
   |---|---|---|---|---|
   | auto-spec (both deleted) | 0.584 | 0.502 | 0.623 | +0.0388 |
   | `Year Published` restored | 0.613 | 0.502 | 0.636 | +0.0229 |
   | full structured block | 0.684 | 0.502 | 0.684 | -0.0005 |

   `text_only` is unchanged throughout, so this is entirely a `no_text` effect: the text acted as a proxy for the deleted columns.

2. The full GPU grid gives Delta_Joint +0.047..+0.055 but Delta_Awareness +0.003 / +0.001 / 0.000 / -0.001 — **2 of 5** against a quorum of 3, and counted honestly 1 of 5, because the TabM cell is the same float knife-edge documented for TabPFNv2.

`rejected/anime/REJECTION.md` — `REG_TEXT_MEDIA_ANIME_POPULARITY`. Delta_Joint +0.031..+0.037 but Delta_Awareness -0.001 / 0.000 / -0.002 / -0.002, **0 of 5**. Its fold-0 screen had looked positive (+0.002 / +0.003); the full grid reversed it.

`rejected/metacritic/REJECTION.md` — `REG_TEXT_MEDIA_METACRITIC_SCORED`. The `Metacritic` target is 82% sentinel zeros, so the "regression" is mostly a has-a-score indicator. `derived_input.csv` is the scored file the grid was built from.

`rejected/REJECTIONS.md` — a table of screen-time rejections with reason and the screening file that shows it, transcribed from `docs/archive/DJ_GAMES_REPORT.md` and `docs/archive/HUNT_ROUND2_REPORT.md`:

| candidate | reason | evidence |
|---|---|---|
| `muhammadaqeelkabir/steam-games-dataset-steamspy-api` | screen target was `appid`, a primary key; JUNK regex misses it (no word boundary). No free text besides proper nouns. | `screening/t2_joint/hunt_full.csv` |
| `lunthu/gog-com-video-games-dataset` | auto-target was `globalReleaseDate`, a date (camelCase defeats the JUNK regex). Re-screened Delta_Joint -0.0003; `text_only` R² 0.996 because the text *was* the release dates. | `screening/t2_joint/hunt_games2.csv` |
| `arnabchaki/popular-video-games-1980-2023` | 1,512 rows, `Rating` has 35 distinct values, `Plays`/`Wishlist` are `"3.9K"` strings. | `screening/t2_joint/hunt_games2.csv` |
| `mterzolo/lego-sets` | 12,261 rows are 744 products replicated across 21 countries; near-duplicates would straddle train and test in every fold. | `screening/t0_t1/t1_batch.csv` |
| `vikasojha98/top-women-chess-players` | Delta_Joint +0.024 against a ±0.015 noise band, with `text_only` R² 0.074 — the text is nearly inert. | `screening/t2_joint/hunt_full.csv` |
| `tolstoyjustin/kerala-bevco-liquor-price-list` | target was `Sl No`, a serial number. With the corrected target `no_text` reaches R² 1.000. | `screening/spec_audits/spec_audit2.csv` |
| `neomatrix369/google-play-store-apps-extended` | target derived from its own text. | `screening/t2_joint/hunt_full.csv` |
| `nomanmunir/daraz-perfumes` | all-state R² negative. | `screening/t2_joint/hunt_full.csv` |
| `rrokon/global-grocery-nutrition-2025` | Delta_Joint +0.0005 with the baseline saturated at 0.969. | `screening/t2_joint/hunt_full.csv` |
| movies (`roi_pct`) | `roi_pct` is revenue/budget, an arithmetic identity over two structured columns; `no_text` reaches R² 0.997. Invisible to `find_leaks`. | `screening/spec_audits/spec_audit_wave3.csv` |
| `nikatomashvili/steam-games-dataset`, `rudrakumargupta/ultimate-games-*` | no numeric column qualifies as a target under the outlier/cardinality rule. | `screening/t2_joint/hunt_games2.csv` |

- [ ] **Step 5: Write the validation README**

`results/curation/validation/README.md` — these four CSVs are runs on the paper anchor `MUL_TEXT_PRODUCT_SENTIMENT`, not candidate screening. They establish:

- The runner reproduces the paper exactly for `no_text`: `0.83454303717305` for every model. Anchor values, LightGBM fold 0: `no_text=0.83454303717305`, `text_only=0.7658517388790819`, `all=0.8618478948517495`.
- Delta_Joint agrees with the paper within 0.012 across four models, all signs matching.
- `cache_check.csv` proves the frozen embedding cache is bit-exact (~40x speedup, 600s cold to 11s warm).
- `tabpfn25_retry.csv` records a TabPFN-2.5 retry against the gated-weights blocker.

- [ ] **Step 6: Write INDEX.md**

`results/curation/INDEX.md` — one row per file under `results/curation/`, with columns: **new path | original filename | dataset | tier | schema | what it proves**. The original-filename column is what keeps the `--out` paths recorded in the archived reports and logs traceable after the renames. The schema column takes one of two values:

- `cpu` — `model, dataset, fold, multimodal_state, test_score, runtime, n_train, n_test, m_features, task_type, tune_e5`
- `kaggle` — `state, score, secs, epochs, dataset, model, fold`

State at the top of the file that nothing was reformatted, and that any cross-schema read goes through `curation_lab.criterion.deltas.normalize`.

- [ ] **Step 7: Verify every result file is indexed, then commit**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
from pathlib import Path
index = Path('results/curation/INDEX.md').read_text(encoding='utf-8')
files = [p for p in Path('results/curation').rglob('*') if p.is_file()
         and p.name not in {'INDEX.md','VERDICT.md','STATUS.md','REJECTION.md','REJECTIONS.md','README.md'}]
missing = [p.as_posix() for p in files if p.name not in index]
print('files:', len(files), '| unindexed:', missing)
"
```

Expected: `unindexed: []`.

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git add results/curation && git commit -F - <<'MSG'
docs(results): a verdict beside every grid

Each dataset folder now states what it proves, with the numbers transcribed from
the archived reports rather than recomputed.

Two caveats are recorded rather than smoothed over, because both change a
verdict. Vietnam housing's TabPFNv2 cell clears the criterion only through
float64 representation -- the difference of its rounded means is exactly delta --
so it is not counted among the three passes. Board games is rejected twice over:
its screening Delta_Joint of +0.039 was an artifact of the JUNK regex deleting
Year Published and Play Time, and its honest full grid passes Delta_Awareness on
one model, not three.

INDEX.md keeps the original filename of every file, so the --out paths recorded
in the archived reports and logs stay traceable through the renames, and records
which of the two CSV schemas each file uses.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

---

## Task 7: Canonical findings and the marked archive

**Files:**
- Create: `docs/findings/01-criterion-and-pipeline.md`, `02-mining-method-rules.md`, `03-methodological-findings.md`, `04-environment-and-performance.md`
- Create: `docs/status/STATE.md`
- Move: `RESUME.md`, `PHASE2_RESULTS.md`, `RESEARCH_NOTES.md`, `docs/AUTONOMOUS_MINER_RULES.md` into `docs/archive/`
- Modify: the nine files in `docs/archive/` (header only)

**Interfaces:**
- Consumes: source documents listed per file below.
- Produces: documents linked from `README.md` in Task 8.

- [ ] **Step 1: Move the remaining superseded documents into the archive**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && mkdir -p docs/findings docs/status && \
git mv RESUME.md PHASE2_RESULTS.md RESEARCH_NOTES.md docs/archive/ && \
git mv docs/AUTONOMOUS_MINER_RULES.md docs/archive/ && \
ls docs/archive/
```

Expected: eight `.md` files — the four moved here plus the four moved in Task 5 Step 6.

- [ ] **Step 2: Write `docs/findings/01-criterion-and-pipeline.md`**

Sources: `docs/superpowers/plans/phase1-findings.md`, `CLAUDE.md`. Contents:

- The criterion: 5 learners (TabM, CatBoost, LightGBM, TabPFN v2, TabPFN v2.5) × 4 conditions; a dataset passes if for ≥3 of 5 (`RHO = 3/5`), `Delta_Joint = mean(all) - max(mean(no_text), mean(text_only)) > δ` and `Delta_Awareness = mean(ft) - mean(all) > δ`, with `δ = 0.001` and per-state means over 5 folds rounded to 3 decimals before differencing.
- The condition-to-flag mapping table from `CLAUDE.md` (paper condition | CLI flag | CSV label).
- **The rule that matters:** `passes()` lives once, in `multabench/leaderboard/analysis/pass_matrix.py`. Reuse it; never reimplement it.
- Validation: the harness re-derives the shipped 56×10 `pass_matrix.csv` with **0 mismatched cells**, and `verdict()` accepts a known 5-of-5 and rejects a known 0-of-5.
- Runner fidelity, with the anchor numbers, pointing at `results/curation/validation/`.
- `passes()` asserts row completeness (5 folds × 4 states) rather than averaging over gaps; a new assertion failure is a real data gap, not something to add to `_KNOWN_MISSING_ROWS`.

- [ ] **Step 3: Write `docs/findings/02-mining-method-rules.md`**

Source: `docs/archive/AUTONOMOUS_MINER_RULES.md`, updated where later results overtook it. Carry over sections 0-10 substantially intact — the funnel yields, discovery rules R1.1-R1.6, ingestion, spec derivation, screening, TAR, performance, environment traps, the case file, and the positive signatures R10.1-R10.5.

Two updates are required:

1. The case file's MTG row reads "on track (grid completing)". The grid is now complete at 75/75; update it to the finished figures (+0.050..+0.075 on all five models, Delta_Awareness still unmeasured).
2. Add board games' and anime's full-grid outcomes to the case file, which predate their rejection.

Everything asserted in the positive-signatures section (baseline balance as the strongest single signal, lift over raw delta, low CV, orthogonal channels, one rich text column over several thin ones) carries over unchanged — it is measured, not inferred.

- [ ] **Step 4: Write `docs/findings/03-methodological-findings.md`**

Sources: `docs/archive/PHASE2_RESULTS.md`, `docs/archive/HUNT_ROUND2_REPORT.md`, `docs/archive/DJ_GAMES_REPORT.md`, `docs/archive/RESUME.md`.

Lead with the governing principle, stated once and then evidenced twice:

> A cheap screen is only valid where cheapness does not change the quantity being screened. True for Delta_Joint, which uses frozen encoders and is exact. False for Delta_Awareness.

**Evidence 1 — the epochs budget.** `epochs=2` leaves the LoRA adapter under-trained, so `ft ≈ all` and Delta_Awareness collapses to noise around zero *by construction*. The first sweep "rejected" Udemy 1-of-5 purely because of this.

| model, fold 0 | all | ft @ 2 ep | Delta @ 2 ep | ft @ 10 ep | Delta @ 10 ep |
|---|---|---|---|---|---|
| LightGBM | 0.4957 | 0.5056 | +0.0099 | 0.5279 | **+0.0322** |
| CatBoost | 0.5177 | 0.5282 | +0.0105 | 0.5299 | **+0.0122** |

This also invalidates every epochs=2 batch probe in `results/curation/screening/t3_tar/tar_probes.csv` — they measured the epoch budget, not the datasets.

**Evidence 2 — the fold count.** A fold-0 Delta_Awareness screen does not work either, and it is the same class of mistake. Board games:

| board games | cat | light | tabm | tabpfnv2 |
|---|---|---|---|---|
| fold 0 only | +0.0074 | +0.0163 | +0.0052 | +0.0010 |
| 5-fold mean | -0.0003 | +0.0021 | +0.0010 | -0.0015 |

Every model dropped and two flipped sign. The per-(model, fold) spread is σ = 0.0063 over [-0.0124, +0.0163] — about 6× the delta threshold — so one fold cannot resolve a criterion whose threshold sits deep inside its noise band. Screening one fold cost two full grids. Delta_Joint is unaffected: at +0.03 to +0.09 it is an order of magnitude above the same noise, which is why the cheap frozen screen stays sound for it.

Then the remaining findings, each with its evidence:

- **Over-deleting structured columns manufactures Delta_Joint.** The board-games table from Task 6. Corollary: `hunt.py` is a triage net; its spec is not a curation decision, and any candidate whose Delta_Joint came from it must be re-measured with the JUNK-deleted columns restored before it is gridded.
- **Identifier and date targets.** `Sl No`, `appid`, `globalReleaseDate`, `CustomerID` — the JUNK regex uses word boundaries, so it misses camelCase and spaced abbreviations. Fixed in `auto_spec.py` (`84a637e`).
- **Multi-column arithmetic leakage, still unfixed.** `roi_pct` = revenue/budget; both are structured columns, so no single column is a near-copy of the target and `find_leaks` cannot see it, while `no_text` reaches R² 0.997. Proposed guard, not yet implemented: flag any candidate whose `no_text` R² exceeds ~0.95 as saturated.
- **The float knife-edge.** Where the difference of rounded means equals delta exactly, a strict `>` is decided by float64 representation (`0.647 - 0.646` → `0.0010000000000000009`). Affects Vietnam's TabPFNv2 and board games' TabM. Such cells must be reported, not counted.
- **Delta_Awareness is the binding constraint.** Five of eight candidates cleared Delta_Joint and then failed TAR. Now that GPU makes TAR affordable, the pipeline's cheap-screen ordering should be reversed.
- **The typing rule fires in both directions.** The `>=100 distinct` arm promotes low-cardinality columns into TEXT and blows the ≤5 multimodal budget — Sephora's `brand` (324 distinct, 3.5% unique) and `category` (143 distinct, 1.6% unique) both type as TEXT. This is the dominant real failure, not the "short free text becomes categorical" case.

- [ ] **Step 5: Write `docs/findings/04-environment-and-performance.md`**

Sources: `CLAUDE.md`, `docs/superpowers/plans/phase1-findings.md`, `docs/archive/RESUME.md`.

- Environment: `.venv/Scripts/python.exe` only; pandas pinned at 2.3.3 (under pandas 3.x string columns get the new `str` dtype and `tabstar.preprocessing.feat_types.is_numerical_feature` raises, taking down all feature detection and therefore the `no_text`/`text_only` states); `PYTHONIOENCODING=utf-8` because the console codepage is cp1255; no CUDA locally, so `DEVICE` is `None`.
- Performance economics:

  | optimization | speedup | status |
  |---|---|---|
  | frozen embedding cache | ~40x | done, bit-exact (4 tests) |
  | `max_length` cap, frozen encode | ~7x | off by default |
  | `max_length` cap, TAR training loop | 333 s → ~8 s per step | done |
  | TAR encoder sharing (25 runs → 10 fine-tunings) | 2.5x | done; measured 2184 s cold then 1082 s on a cache hit |

- **The `max_length` cap is not bit-exact.** Masked padding is algebraically inert, but changing the padded length reassociates float32 matmuls and moves embeddings ~1e-7. Compare with `atol=1e-5`, never `array_equal`. Fine for screening; never for numbers compared against the paper.
- Why encoder sharing is correct: fine-tuning happens only in the embedding step, so the tuned encoder is a function of `(x_train, y_train, e5_train_kwargs)` alone. `USE_VAL_SPLIT` is True for LightGBM/CatBoost/TabM and False for both TabPFNs, giving 2 distinct fine-tunings per fold rather than 5. The cache is keyed on argument content, not on model grouping, so it stays correct if upstream ever threads a real fold into `split_to_val`.
- Open blockers: TabPFN-2.5's gated HuggingFace weights (`browser_auth.py::_poll_for_token` calls `select.select` on stdin, which fails on Windows with `OSError: WinError 10038` in any non-interactive context) — needs a one-time licence acceptance; and `multabench/e5/e5_finetune.py:245` asserting `CUDA_VISIBLE_DEVICES`, bypassed by `run_one(..., cpu_ft=True)`.
- The known test failure: `test_training_passage_matches_what_the_dataset_tokenizes` compares detokenized text (which reinserts spaces around punctuation) against the raw string. A flaw in the test, not the pipeline. Baseline is 1 failed.

- [ ] **Step 6: Write `docs/status/STATE.md`**

The live handoff replacing `RESUME.md`:

- Verdicts: 2 accepted (Udemy, Vietnam), 1 in progress (MTG, needs a TAR grid), 3 gridded rejections (board games, anime, metacritic), plus the screen-time rejections table.
- Standard scope (1 passing dataset) is met twice; outstanding scope (≥5) needs 3 more.
- Blockers: TabPFN-2.5 licence; the local candidate pool is exhausted and needs a fresh T0/T1 Kaggle search with the junk-aware profiler, because the previous search ranked on a text-column count that counted dates and ids.
- Next steps, in order: TAR grid on MTG at epochs ≥ 10 over ≥ 3 folds; then a fresh search feeding the reversed pipeline (TAR first).
- Deadline 2026-10-26.
- The reproduce commands, against the new paths:

  ```bash
  python -m curation_lab.kaggle.push --machine-shape NvidiaTeslaT4 --full \
    --candidate "<owner/slug>=REG_TEXT_<NAME>" --folds 0,1,2,3,4 \
    --models light,cat,tabm,tabpfnv2 --states no_text,text_only,all,ft
  python -m curation_lab.kaggle.verdict_from_runs results/curation/<path>.csv
  ```

- [ ] **Step 7: Add the replacement header to each archived document**

Prepend to each of the nine files in `docs/archive/` a header of exactly this shape, filled in per file:

```markdown
> **Superseded 2026-09-02.** Replaced by `<path>`.
> <one sentence on what changed>.
> Kept verbatim below: the current document was written from it, and the
> judgment calls in that rewrite should stay checkable against this source.
```

Replacements:

| archived file | replaced by |
|---|---|
| `RESUME.md` | `docs/status/STATE.md` |
| `PHASE2_RESULTS.md` | `results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/VERDICT.md` and `docs/findings/03-methodological-findings.md` |
| `RESEARCH_NOTES.md` | `docs/findings/02-mining-method-rules.md` |
| `AUTONOMOUS_MINER_RULES.md` | `docs/findings/02-mining-method-rules.md` |
| `DJ_PROPERTY_REPORT.md` | `results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md` |
| `DJ_PROPERTY_TAR_REPORT.md` | `results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/VERDICT.md` |
| `DJ_GAMES_REPORT.md` | `results/curation/in_progress/REG_TEXT_GAMES_MTG_CARD_PRICES/STATUS.md`, `results/curation/rejected/board_games/REJECTION.md`, `results/curation/rejected/REJECTIONS.md` |
| `HUNT_ROUND2_REPORT.md` | `docs/findings/03-methodological-findings.md` and `results/curation/rejected/` |
| `README-upstream.md` | created in Task 8 |

`README-upstream.md` does not exist yet; add its header in Task 8.

- [ ] **Step 8: Verify no canonical document contradicts a verdict, then commit**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && grep -rn "Delta_Awareness has NOT been measured\|TAR) has NOT been measured" docs/findings/ results/curation/accepted/ && echo "FOUND -- investigate" || echo "OK: no accepted dataset claims TAR is unmeasured"
```

Expected: `OK: ...`. The phrase legitimately survives in `docs/archive/DJ_PROPERTY_REPORT.md` and in the MTG `STATUS.md`; it must not appear in `docs/findings/` or under `accepted/`.

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git add docs/ && git commit -F - <<'MSG'
docs: four canonical findings, and an archive that says what replaced it

Eight documents partly superseded each other. PHASE2_RESULTS.md held a rejected
verdict, its correction, and then an acceptance; DJ_PROPERTY_REPORT.md said
Delta_Awareness was unmeasured while a report on the other branch measured it. A
reader had no way to tell which half of any file was still true.

Now: 01 criterion and pipeline, 02 mining method rules, 03 methodological
findings, 04 environment and performance -- each true as of today, none
contradicting another. docs/status/STATE.md replaces RESUME.md as the handoff.

03 leads with the correction trail rather than burying it. That a cheap screen
is only valid where cheapness does not change the quantity being screened is the
most transferable thing this project produced, and it was learned twice at the
cost of a wrong rejection and two full grids.

Originals are kept verbatim under a header naming their replacement, so every
judgment call in the rewrite stays checkable against its source.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

---

## Task 8: The paper directory and the README

**Files:**
- Create: `paper/report.md`, `paper/README.md`, `paper/assets/.gitkeep`
- Move: `instructions.pdf` → `paper/source/instructions.pdf`
- Move: `README.md` → `docs/archive/README-upstream.md`
- Create: `README.md`

**Interfaces:**
- Consumes: documents from Tasks 6 and 7.
- Produces: the repo's entry point.

- [ ] **Step 1: Create the paper directory**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && mkdir -p paper/source paper/assets && \
git mv instructions.pdf paper/source/instructions.pdf 2>/dev/null || mv instructions.pdf paper/source/instructions.pdf && \
touch paper/assets/.gitkeep && ls -R paper
```

Expected: `paper/source/instructions.pdf` exists. (`instructions.pdf` is untracked, so `git mv` fails and the plain `mv` runs — both outcomes are fine.)

- [ ] **Step 2: Write `paper/README.md`**

States: this directory holds the Technion 097215 Track 2 write-up. `source/instructions.pdf` is the assignment brief. `report.md` is a **skeleton, not a draft** — drafting it is separate work. `assets/` holds tables and figures generated from `results/curation/`; nothing there is hand-typed, so every number in the report traces to a committed CSV. Note the two deviations that must be disclosed in any submission: E5 fine-tuning ran 10 epochs rather than the `E5TrainArgs` default of 50, and TabPFN-2.5 could not run locally because of gated weights.

- [ ] **Step 3: Write `paper/report.md` as a skeleton with evidence pointers**

Each section is a heading plus a note naming the evidence folder it draws from — no prose. Sections:

1. **Introduction and track** — Track 2, Benchmark Track. Evidence: `paper/source/instructions.pdf`.
2. **The curation criterion** — Evidence: `docs/findings/01-criterion-and-pipeline.md`.
3. **Method: an automated mining pipeline** — the T0→T3 funnel and its measured yields. Evidence: `docs/findings/02-mining-method-rules.md`, `results/curation/screening/`.
4. **Result 1: `REG_TEXT_EDU_UDEMY_ACADEMY`** — Evidence: `results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/`.
5. **Result 2: `REG_TEXT_HOUSES_VIETNAM_2024`** — Evidence: `results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/`.
6. **In progress: `REG_TEXT_GAMES_MTG_CARD_PRICES`** — Evidence: `results/curation/in_progress/`.
7. **Negative results and what they taught** — Evidence: `results/curation/rejected/`, `docs/findings/03-methodological-findings.md`.
8. **Methodological contribution** — the cheap-screen principle and its two instances. Evidence: `docs/findings/03-methodological-findings.md`.
9. **Reproducibility and deviations** — Evidence: `docs/findings/04-environment-and-performance.md`, `results/curation/validation/`.
10. **Limitations** — TabPFN-2.5 unavailable; 10 epochs not 50; the float knife-edge cells.

- [ ] **Step 4: Archive the upstream README and add its header**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git mv README.md docs/archive/README-upstream.md && ls docs/archive/ | wc -l
```

Expected: 9 files. Then prepend the Task 7 Step 7 header to `docs/archive/README-upstream.md`, replaced by `README.md`, noting that it describes the upstream benchmark rather than this fork's work.

- [ ] **Step 5: Write the new `README.md`**

A map, in this order:

1. **What this is** — a fork of the official MulTaBench repo (arXiv 2605.10616) used for the Technion 097215 Track 2 project. Two layers: `multabench/` + `benchmark.py` are upstream and read-only; `curation_lab/` is ours.
2. **Current state** — a table linking into evidence:

   | dataset | Delta_Joint | Delta_Awareness | verdict | evidence |
   |---|---|---|---|---|
   | `REG_TEXT_EDU_UDEMY_ACADEMY` | +0.136..+0.209 (5/5) | +0.006..+0.016 (3/5) | **ACCEPTED** | `results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/` |
   | `REG_TEXT_HOUSES_VIETNAM_2024` | +0.249..+0.324 (5/5) | +0.004..+0.015 (3/5) | **ACCEPTED** | `results/curation/accepted/REG_TEXT_HOUSES_VIETNAM_2024/` |
   | `REG_TEXT_GAMES_MTG_CARD_PRICES` | +0.050..+0.075 (5/5) | not measured | in progress | `results/curation/in_progress/` |
   | board games | +0.047..+0.055 | +0.003..-0.001 (1/5) | rejected | `results/curation/rejected/board_games/` |
   | anime | +0.031..+0.037 | -0.002..0.000 (0/5) | rejected | `results/curation/rejected/anime/` |
   | metacritic | — | — | rejected (82% sentinel target) | `results/curation/rejected/metacritic/` |

3. **Where the conclusions are** — the four `docs/findings/` documents are canonical; `docs/status/STATE.md` is the live handoff; `docs/archive/` is superseded and every file there says what replaced it.
4. **Where the results are** — the five buckets, and `results/curation/INDEX.md` as the file-level map.
5. **Where the paper is** — `paper/`, with the note that `report.md` is a skeleton.
6. **How to run the harness** — Kaggle GPU as the primary path (`curation_lab.kaggle.push`, then `verdict_from_runs`), CPU as the legacy path that produced the Udemy and MTG grids and remains the way to reproduce them (`curation_lab.screen.verify`). Include the environment constraints inline: `.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, pandas 2.3.3.
7. **Open blockers** — TabPFN-2.5 gated weights; exhausted local candidate pool.

- [ ] **Step 6: Verify every README link resolves, then commit**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import re
from pathlib import Path
bad = []
for md in [Path('README.md'), Path('paper/report.md'), Path('paper/README.md')]:
    for target in re.findall(r'\]\(([^)#][^)]*)\)', md.read_text(encoding='utf-8')):
        if target.startswith(('http', 'mailto')):
            continue
        if not (md.parent / target).exists():
            bad.append(f'{md}: {target}')
print('broken links:', bad)
"
```

Expected: `broken links: []`.

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git add -A README.md paper docs/archive && git commit -F - <<'MSG'
docs: a README that says where everything is, and a home for the paper

The repo root previously offered the upstream benchmark's README, which
describes MulTaBench rather than this fork's work, and gave no way to find the
two accepted datasets. The new README is a map: current verdicts linking into
their evidence folders, which four documents are canonical, where the results
live, and which harness path to use.

paper/ is new. report.md is a skeleton with evidence pointers, deliberately not
a draft -- every section names the folder its numbers must come from, so nothing
in the writeup is hand-typed. The course brief moves to paper/source/ and is
committed rather than sitting untracked in the root.

The upstream README is preserved at docs/archive/README-upstream.md.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

---

## Task 9: Verify nothing was lost

**Files:**
- No edits. Verification only.

**Interfaces:**
- Consumes: `docs/superpowers/plans/consolidation-manifest.json` from Task 2; `curation_lab.tools.manifest.check` from Task 1.

- [ ] **Step 1: Run the manifest check**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
from pathlib import Path
from curation_lab.tools.manifest import check
root = Path('.').resolve()
exceptions = {
    'README.md',
    'RESUME.md', 'PHASE2_RESULTS.md', 'RESEARCH_NOTES.md',
    'docs/AUTONOMOUS_MINER_RULES.md',
    'results/candidates/DJ_PROPERTY_REPORT.md',
    'results/candidates/DJ_PROPERTY_TAR_REPORT.md',
    'results/candidates/DJ_GAMES_REPORT.md',
    'results/candidates/HUNT_ROUND2_REPORT.md',
}
lost = check(root / 'docs/superpowers/plans/consolidation-manifest.json', root, exceptions)
print('LOST:', lost)
"
```

Expected: `LOST: []`.

The exceptions are exactly the nine documents that gained an archive header, so their content legitimately changed. **Every other file must round-trip byte-identical** — a non-empty list means a result was lost and the task fails. Do not add entries to the exception set to make it pass.

- [ ] **Step 2: Confirm the moves were recorded as renames**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git log --diff-filter=R --name-status --oneline -1 $(git log --format=%H --grep='one folder per claim' -1) | head -20
```

Expected: `R` status lines, not `A`/`D` pairs.

- [ ] **Step 3: Re-derive both acceptances through the repo's own `passes()`**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import pandas as pd
from curation_lab.criterion.deltas import normalize, verdict
d = normalize(pd.read_csv('results/curation/accepted/REG_TEXT_EDU_UDEMY_ACADEMY/grid.csv', encoding='utf-8'))
d = d.drop_duplicates(subset=['model','state','fold'])
print('UDEMY:', verdict(d))
"
```

Expected: 3 of 5 passing, `accepted` True — identical to the verdict recorded in `VERDICT.md`. If the numbers moved, the reorganization touched data it should not have.

- [ ] **Step 4: Run the full test suite**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/curation_lab/ -q 2>&1 | tail -3
```

Expected: `1 failed, 57 passed, 1 skipped` — the same single pre-existing failure, no new ones.

- [ ] **Step 5: Confirm the tree is clean**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git status --short && echo "--- clean if only untracked junk above ---"
```

Expected: no tracked modifications. The shell-quoting junk (`+0.0322`, `.tar_cache/`) may still appear as untracked; `.tar_cache/` should now be ignored.

- [ ] **Step 6: Record the verification result**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git commit --allow-empty -F - <<'MSG'
chore: verify the consolidation lost nothing

Content-addressed check against the manifest taken before anything moved: every
recorded file's content is still present in the tree. The only exceptions are
the nine documents that gained an archive header, which is the one intended
content change.

Both acceptances re-derive to the same verdicts through the repo's own
pass_matrix.passes(), the moves are recorded as renames rather than add/delete
pairs, and the test suite is unchanged at 1 pre-existing failure.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BCGmNdi44q9z3kU46BM9bo
MSG
```

---

## Task 10: Retire the old branches and worktrees

Destructive. Run only after Task 9 passes.

**Files:**
- Delete: `remote_login.env`
- Remove: four worktrees, five local branches

**Interfaces:**
- Consumes: verified branch from Task 9; tags from Task 2.

- [ ] **Step 1: Push the new branch**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git push -u origin curation-lab && git push origin archive/cpu-lane archive/kaggle-lane
```

Expected: branch and both tags created on the remote.

- [ ] **Step 2: Confirm both lane tips are reachable before deleting anything**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git merge-base --is-ancestor archive/cpu-lane curation-lab && echo "cpu-lane contained" && git merge-base --is-ancestor archive/kaggle-lane curation-lab && echo "kaggle-lane contained"
```

Expected: both lines print. If either does not, **stop** — work would be lost by the deletions below.

- [ ] **Step 3: Remove the worktrees**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && for w in dj-frozen-fix kaggle-tar tar-gpu-remote tar-hunt; do git worktree remove --force .claude/worktrees/$w; done && git worktree list
```

Expected: only the main worktree remains.

- [ ] **Step 4: Delete the retired local branches**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git branch -D origin kaggle_work_tar kaggle_work_frozenfix tar-hunt worktree-tar-gpu-remote && git branch -vv
```

Expected: only `master` and `curation-lab` remain, and `git rev-list` no longer warns that `origin` is ambiguous.

- [ ] **Step 5: Delete the plaintext credential file**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && rm -f remote_login.env && ls remote_login.env 2>&1 | head -1
```

Expected: `No such file or directory`. It is gitignored and never entered history; key auth at `~/.ssh/multabench_remote` continues to work.

- [ ] **Step 6: Final state check**

```bash
cd /c/Noa/nlp/project/MulTaBench2 && git branch -a && git tag -l 'archive/*' && git status --short
```

Expected: `master` + `curation-lab` locally, the remote branches still present (deleting them was explicitly out of scope), both archive tags, and a clean tree.

---

## Self-Review

**Spec coverage.** Section 1 (merge) → Tasks 2-3. Section 2 (capture) → Task 4, with the worktree-only `spec_audit.csv` handled in Task 2 Step 3. Section 3 (layout) → Tasks 5, 7, 8. Section 4 (taxonomy, all five buckets, INDEX.md, both schemas) → Tasks 5-6. Section 5 (conclusions, four canonical docs, archive headers) → Task 7. Section 6 (README) → Task 8. Section 7 (verification) → Tasks 1 and 9. Section 8 (cleanup) → Task 10. No gaps.

**Interface consistency.** `build`, `write_manifest`, `check`, `sha256_of` are defined in Task 1 and used with matching signatures in Tasks 2 and 9. `check`'s `exceptions` parameter is a `set[str]` of repo-relative paths in both. The nine exception paths in Task 9 Step 1 are exactly the nine files given archive headers in Task 7 Step 7 and Task 8 Step 4, recorded at their *pre-move* paths, which is what the manifest holds.

**Ordering.** The manifest must exist before any move (Task 2 before Task 5), and files must be tracked before they are moved (Task 4 before Task 5) so git records renames. Task 10 is gated on Task 9.
