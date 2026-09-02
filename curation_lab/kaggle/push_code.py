"""Publish the repo's Python tree as a new version of the `multabench-code` dataset.

The Kaggle notebook does not clone this repo -- it imports `multabench` and
`curation_lab` from an attached dataset. So any local change to curation_lab is
INVISIBLE to a run until that dataset is re-versioned, and the run silently uses
the old code. That is a nasty failure mode: a fixed spec rule appears not to work,
and nothing in the log says why. Re-push after touching anything the notebook
imports.

Only source is uploaded: no .venv, no caches, no results, no data.

Usage:
    python -m curation_lab.kaggle.push_code -m "auto_spec: reject identifier targets"
    python -m curation_lab.kaggle.push_code --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from curation_lab.kaggle.push import KAGGLE_EXE, REPO_ROOT, _env, load_token

DATASET_ID = "talkraicer/multabench-code"
INCLUDE = ("multabench", "curation_lab", "benchmark.py")
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".pytest_cache",
             ".emb_cache", ".tar_cache", "catboost_info"}
# The notebook only ever imports Python; shipping data or notebooks would bloat the
# dataset and, worse, put stale result CSVs where the spec cell globs for candidate
# CSVs under /kaggle/input.
KEEP_SUFFIXES = (".py", ".txt")


def stage(dest: str) -> int:
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    n = 0
    for item in INCLUDE:
        src = os.path.join(REPO_ROOT, item)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, item))
            n += 1
            continue
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if not f.endswith(KEEP_SUFFIXES):
                    continue
                rel = os.path.relpath(os.path.join(root, f), REPO_ROOT)
                out = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(os.path.join(root, f), out)
                n += 1
    with open(os.path.join(dest, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump({"title": "multabench-code", "id": DATASET_ID,
                   "licenses": [{"name": "CC0-1.0"}]}, fh, indent=2)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-m", "--message", default="update code")
    p.add_argument("--dir", default=os.path.join(REPO_ROOT, "kaggle_uploads", "code"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    n = stage(args.dir)
    print(f"staged {n} files -> {args.dir}")
    if args.dry_run:
        return
    env = _env(KAGGLE_API_TOKEN=load_token())
    proc = subprocess.run(
        [KAGGLE_EXE, "datasets", "version", "-p", args.dir, "-m", args.message,
         "--dir-mode", "zip"],
        capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out)
    if proc.returncode != 0:
        raise SystemExit(f"upload failed with exit {proc.returncode}")
    print("NOTE: Kaggle needs a minute to finish processing before a kernel run "
          "will see the new version.")


if __name__ == "__main__":
    main()
