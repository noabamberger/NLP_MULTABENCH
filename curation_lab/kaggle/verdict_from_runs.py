"""Merge the Kaggle TAR run CSVs into one grid and apply the curation criterion.

The notebook emits one CSV per push (a run measures a subset of the four states),
so the full 5x4x5 grid arrives in pieces. This stitches them together and hands
the result to the criterion -- it does NOT reimplement the rule:
curation_lab.criterion.deltas delegates to
multabench.leaderboard.analysis.pass_matrix.passes().

deltas.verdict() insists on the complete five-model committee. TabPFN-2.5 cannot
be measured here (its weights are behind a one-time licence acceptance), so we
apply passes() per model over the models that DID run and report the quorum
against the full committee of five, counting the unmeasured model as a non-pass --
which is how the paper treats a (model, dataset) cell with no data. That is the
conservative direction: it can only make acceptance harder.

Usage:
    python -m curation_lab.kaggle.verdict_from_runs results/candidates/dj_property_tar_*.csv
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd

from curation_lab.criterion.deltas import (CURATION_MODELS, DELTA, FOLDS, QUORUM, STATES,
                                           screen_deltas)
from multabench.leaderboard.analysis.pass_matrix import passes

# The notebook writes SHORT_NAMEs; the criterion speaks the canonical labels that
# committee_pool._MODEL_LABELS produces from the emoji MODEL_NAMEs.
SHORT_TO_LABEL = {"light": "LightGBM", "cat": "CatBoost", "tabm": "TabM",
                  "tabpfnv2": "TabPFNv2", "tabpfnv2p5": "TabPFN-2.5"}


def load(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8")
        frames.append(df)
    if not frames:
        raise SystemExit("no input CSVs matched")
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].map(lambda m: SHORT_TO_LABEL.get(m, m))
    df = df.rename(columns={"score": "test_score"})
    # A re-run of the same cell supersedes the earlier one.
    df = df.drop_duplicates(subset=["model", "state", "fold"], keep="last")
    return df[["model", "dataset", "state", "fold", "test_score"]]


def report(df: pd.DataFrame, delta: float = DELTA) -> int:
    dataset = df["dataset"].unique()
    if len(dataset) != 1:
        raise SystemExit(f"expected one dataset, got {list(dataset)}")
    print(f"dataset: {dataset[0]}\n")

    print(screen_deltas(df).to_string(index=False))
    print()

    per_model: dict[str, bool] = {}
    for model in CURATION_MODELS:
        sub = df[df["model"] == model]
        have = set(zip(sub["state"], sub["fold"]))
        want = {(s, f) for s in STATES for f in FOLDS}
        if have != want:
            print(f"  {model:12s} NOT MEASURED ({len(want - have)} of {len(want)} cells absent)")
            per_model[model] = False
            continue
        per_model[model] = bool(passes(sub, delta=delta))
        print(f"  {model:12s} {'PASS' if per_model[model] else 'fail'}")

    n_pass = sum(per_model.values())
    accepted = n_pass >= QUORUM
    print(f"\n{n_pass} of {len(CURATION_MODELS)} models pass both criteria "
          f"(quorum {QUORUM}, delta {delta})")
    print(f"VERDICT: {'ACCEPTED' if accepted else 'REJECTED'}")
    return 0 if accepted else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="Result CSVs (globs allowed).")
    p.add_argument("--delta", type=float, default=DELTA)
    args = p.parse_args()
    paths = [p_ for pat in args.paths for p_ in glob.glob(pat)] or args.paths
    raise SystemExit(report(load(paths), delta=args.delta))


if __name__ == "__main__":
    main()
