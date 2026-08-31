"""Build a curation-ready CSV from `douglascampospires/mtg-all-cards`.

The raw file stores three things this benchmark cannot consume directly:

* `PRICES` is one packed string (``USD: $0.42 | USD_FOIL: $5.70 | ...``), so the
  regression target has to be extracted before anything else can happen.
* Card prices are log-normal (raw |z|max = 103), and this repo only *warns* about
  |z|>5 and never clips -- an unlogged target would let a handful of $1000 cards
  dominate R^2 in every condition. We regress log10(USD).
* `POWER_TOUGHNESS` ("5/5", "*/*") and `FIRST_EDITION` ("2004-11-19") pack two
  numerics and a date into strings; multabench's typing rule would promote both
  to TEXT on cardinality alone.

Everything else is left exactly as the raw file has it. This is the equivalent of
an annotated module's LOADING_FUNC/PROCESSING_FUNC, kept out of `multabench/`.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

REF = "douglascampospires/mtg-all-cards"

TARGET = "price_usd_log10"
TEXT_COLS = ["CARD_TEXT", "TYPE"]
NUMERIC_COLS = ["CMC", "NUMBER_OF_EDITIONS", "power", "toughness", "first_edition_year"]
CATEGORICAL_COLS = ["RARITY", "COLOR_PIE"]


def raw_csv() -> str:
    import kagglehub

    from curation_lab.discover.kaggle_search import _load_env

    _load_env()
    d = kagglehub.dataset_download(REF)
    return max(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True), key=os.path.getsize)


def build(out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv(), low_memory=False)

    usd = df["PRICES"].astype(str).str.extract(r"USD:\s*\$([0-9.]+)")[0].astype(float)
    df[TARGET] = np.log10(usd.where(usd > 0))

    # One joke card (Gleemax) has CMC 1,000,000; the real ceiling is 16. Left in, it
    # is a 169-sigma outlier that dominates every standardising learner's scaling.
    df["CMC"] = pd.to_numeric(df["CMC"], errors="coerce").where(lambda s: s <= 16)

    pt = df["POWER_TOUGHNESS"].astype(str).str.split("/", n=1, expand=True)
    df["power"] = pd.to_numeric(pt[0], errors="coerce")        # "*" -> NaN, as intended
    df["toughness"] = pd.to_numeric(pt[1], errors="coerce")
    df["first_edition_year"] = pd.to_datetime(
        df["FIRST_EDITION"], errors="coerce").dt.year.astype("Float64").astype(float)

    keep = [TARGET] + TEXT_COLS + NUMERIC_COLS + CATEGORICAL_COLS
    out = df.loc[df[TARGET].notna(), keep].reset_index(drop=True)
    # A card with no rules text is a vanilla creature; "" is the honest encoding,
    # and leaving NaN would make the text column look half-missing to E5.
    for c in TEXT_COLS:
        out[c] = out[c].fillna("").astype(str)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    return out


def spec(csv_path: str, name: str = "REG_TEXT_GAMES_MTG_CARD_PRICES"):
    from curation_lab.ingest.candidate import CandidateSpec

    return CandidateSpec(
        name=name, csv_path=csv_path, target=TARGET, task="REG",
        cols_to_drop=[], text_cols=list(TEXT_COLS), numeric_cols=list(NUMERIC_COLS),
        categorical_cols=list(CATEGORICAL_COLS), read_kwargs={},
        context="Magic: The Gathering trading cards; predict market price from rules text and card stats.",
    )


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "curation_lab/prep/data/mtg_cards.csv"
    d = build(out)
    print(f"wrote {out} rows={len(d)}")
    print(d[[TARGET] + NUMERIC_COLS].describe().T.to_string())
    for c in CATEGORICAL_COLS + TEXT_COLS:
        print(f"{c:22} nuniq={d[c].nunique():>6} nulls={d[c].isna().sum():>5}")
