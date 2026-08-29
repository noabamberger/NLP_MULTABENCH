"""Derive a CandidateSpec from a raw CSV automatically.

This is the "smart mechanism" the Outstanding track needs: no hand-authored
annotated module per dataset. Everything here is a deterministic rule, so the
same code screens one candidate or fifty.

Rules encode what Phase 1/2 actually taught us:

- Junk text: multabench's `>=100 distinct` rule PROMOTES dates and IDs into TEXT,
  which is the dominant false positive (not the documented "short text becomes
  categorical" case). Drop them by name pattern.
- Leakage: any numeric column with |corr| >= LEAK_CORR against the target is a
  near-copy and must go, or the task is trivial and Delta_Joint collapses. This
  is what would otherwise let `BGG Rank` or `avg_rating_recent` through.
- Target choice: prefer a numeric column with many distinct values and NO extreme
  outliers, because the repo only warns about |z|>5 and never clips -- outlier-heavy
  targets dominate R^2 and destabilise every condition.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from curation_lab.ingest.candidate import CandidateSpec

JUNK = re.compile(
    r"(?:^|_|\b)(date|time|timestamp|year|id|ids|sku|url|link|code|identifier|isbn|asin|"
    r"uuid|key|ref|rank|index|unnamed)(?:$|_|\b)", re.I)
LEAK_CORR = 0.95
MAX_ABS_Z = 5.0
MIN_TARGET_UNIQUE = 20


def _is_junk(col: str) -> bool:
    return bool(JUNK.search(str(col)))


def pick_target(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[str | None, str]:
    """Choose the numeric column that makes the best regression target.

    Returns (column, reason). Rejects outlier-heavy columns outright: the repo only
    warns about |z|>5 and never clips, so those wreck R^2.
    """
    best, best_score, why = None, -1.0, "no numeric column qualified"
    for c in numeric_cols:
        if _is_junk(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 100 or s.nunique() < MIN_TARGET_UNIQUE or s.std(ddof=0) == 0:
            continue
        zmax = float(((s - s.mean()).abs() / s.std(ddof=0)).max())
        if zmax > MAX_ABS_Z:
            continue
        # Prefer well-spread targets; nunique ratio is a decent proxy.
        score = min(s.nunique() / len(s), 0.5) * (1.0 / (1.0 + zmax / MAX_ABS_Z))
        if score > best_score:
            best, best_score, why = c, score, f"nuniq={s.nunique()} zmax={zmax:.2f}"
    return best, why


def find_leaks(df: pd.DataFrame, target: str, numeric_cols: list[str]) -> list[str]:
    """Numeric columns that are near-copies (or monotone functions) of the target."""
    y = pd.to_numeric(df[target], errors="coerce")
    leaks = []
    for c in numeric_cols:
        if c == target:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        pair = pd.concat([y, s], axis=1).dropna()
        if len(pair) < 50:
            continue
        # Spearman catches monotone-but-nonlinear relations (e.g. a rank column).
        try:
            r = abs(float(pair.corr(method="spearman").iloc[0, 1]))
        except Exception:
            continue
        if np.isfinite(r) and r >= LEAK_CORR:
            leaks.append(c)
    return leaks


def build_spec(df: pd.DataFrame, name: str, csv_path: str,
               read_kwargs: dict | None = None, max_text_cols: int = 3) -> tuple[CandidateSpec | None, str]:
    """Return (spec, reason). spec is None when the dataset cannot be used."""
    from tabstar.preprocessing.feat_types import detect_numerical_features

    from multabench.preprocessing.feat_types import classify_semantic_features

    numeric = sorted(detect_numerical_features(df.select_dtypes(exclude=["datetime", "datetimetz"])))
    sem = classify_semantic_features(x=df, numerical_features=set(numeric))
    text_all = sorted(sem.text_features)
    text = [c for c in text_all if not _is_junk(c)]
    cats = [c for c in sorted(sem.categorical_features) if not _is_junk(c)]

    if not text:
        return None, f"no genuine text column (raw text cols: {text_all[:4]})"
    target, why = pick_target(df, numeric)
    if target is None:
        return None, f"no usable target ({why})"

    leaks = find_leaks(df, target, numeric)
    nums = [c for c in numeric if c != target and c not in leaks and not _is_junk(c)]
    if not nums and not cats:
        return None, "no structured column survives -> no_text would be empty"

    # Keep the cheapest text columns: fewer and shorter is better.
    if len(text) > max_text_cols:
        text = sorted(text, key=lambda c: df[c].astype(str).str.len().mean())[:max_text_cols]

    drop = [c for c in df.columns if c not in set(text) | set(nums) | set(cats) | {target}]
    spec = CandidateSpec(
        name=name, csv_path=csv_path, target=target, task="REG",
        cols_to_drop=drop, text_cols=text, numeric_cols=nums,
        categorical_cols=cats, read_kwargs=read_kwargs or {},
    )
    return spec, f"target={target} ({why}) leaks_dropped={leaks} text={text} n_num={len(nums)}"
