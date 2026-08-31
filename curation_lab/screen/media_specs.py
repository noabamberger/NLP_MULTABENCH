"""Hand-built CandidateSpecs for the digital-media / software / apps lane.

`auto_spec.build_spec` is the default path, but it has two blind spots that matter
here and cannot be fixed by tweaking a regex:

1. **Sentinel targets.** `thedevastator/get-your-game-on-metacritic-...` looks like a
   clean regression target (`Metacritic`, 72 distinct, |z|max 2.96) but 82% of its rows
   are literal `0` standing in for "no score". The auto-picker sees spread, not meaning,
   so it happily selects a task that is 82% has-a-score indicator. Repairing it needs a
   ROW filter, and `CandidateSpec` only knows about columns.
2. **camelCase junk.** `auto_spec.JUNK` uses word boundaries, so `ReleaseDate` is not
   recognised as a date and gets promoted to TEXT by multabench's `>=100 distinct` rule.

Both repairs are recorded here as code, not as a one-off notebook cell, so the exact
frame that was benchmarked can be regenerated from the raw Kaggle file at any time.
Derived CSVs are written under `results/candidates/derived/` and are pure functions of
the raw download.
"""
from __future__ import annotations

import glob
import os
from typing import Callable

import pandas as pd

from curation_lab.ingest.candidate import CandidateSpec

DERIVED_DIR = os.path.join("results", "candidates", "derived")


def _raw_csv(ref: str) -> str:
    """Largest CSV of a kagglehub download."""
    from curation_lab.discover.kaggle_search import _load_env
    _load_env()
    import kagglehub
    d = kagglehub.dataset_download(ref)
    return max(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True), key=os.path.getsize)


def _materialize(df: pd.DataFrame, stem: str) -> str:
    os.makedirs(DERIVED_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(DERIVED_DIR, f"{stem}.csv"))
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def metacritic_repaired() -> tuple[CandidateSpec, str]:
    """Metacritic game scores with the missing-score sentinel rows removed.

    `Metacritic == 0` is not a score of zero, it is "Steam reported no Metacritic
    score" -- 10,357 of 12,624 rows. Keeping them turns the regression into a
    has-a-score classifier wearing a regression's clothes, so every condition would be
    measuring the wrong thing. Dropping them leaves 2,267 rows spanning 20..96.

    `ReleaseDate` is dropped: it is a date string that multabench types as TEXT.
    `GenreIsNonGame` is constant after the filter and carries no information.
    """
    raw = _raw_csv("thedevastator/get-your-game-on-metacritic-recommendations-and")
    df = pd.read_csv(raw, low_memory=False)
    n0 = len(df)
    df = df[df["Metacritic"] > 0].reset_index(drop=True)
    path = _materialize(df, "metacritic_scored")
    numeric = [c for c in df.columns
               if c.startswith("GenreIs") and df[c].nunique() > 1] + \
              ["IsFree", "PriceInitial", "RecommendationCount"]
    spec = CandidateSpec(
        name="REG_TEXT_MEDIA_METACRITIC_SCORED",
        csv_path=path, target="Metacritic", task="REG",
        cols_to_drop=["ReleaseDate", "GenreIsNonGame"],
        text_cols=["ResponseName"], numeric_cols=numeric, categorical_cols=[],
    )
    return spec, (f"dropped {n0 - len(df)}/{n0} sentinel Metacritic==0 rows -> {len(df)} rows; "
                  f"text=['ResponseName'] n_num={len(numeric)}")


CANDIDATES: dict[str, Callable[[], tuple[CandidateSpec, str]]] = {
    "metacritic": metacritic_repaired,
}
