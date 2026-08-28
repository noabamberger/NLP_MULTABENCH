"""Delta computation and pass/fail verdicts.

The curation rule itself is NEVER reimplemented here -- verdict() delegates to
multabench.leaderboard.analysis.pass_matrix.passes(). Two entry points are kept
deliberately separate so a cheap partial screen can never emit something that
looks like a verdict:

    screen_deltas()  any fold count, raw deltas, NO verdict     (T2 screen)
    verdict()        full 5-fold x 4-state grid, real pass/fail (T3)
"""
from __future__ import annotations

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import _MODEL_LABELS
from multabench.leaderboard.analysis.pass_matrix import passes

STATES: tuple[str, ...] = ("no_text", "text_only", "all", "ft")
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
CURATION_MODELS: tuple[str, ...] = ("LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5")
QUORUM: int = 3
DELTA: float = 0.001

_CANONICAL = ["model", "dataset", "state", "fold", "test_score"]


def normalize(df: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame:
    """Canonicalize any results frame to [model, dataset, state, fold, test_score].

    Accepts either emoji MODEL_NAMEs (as our runner and the shipped W&B exports
    write) or the short labels used in pool_scores_long.csv, and either a
    `multimodal_state` or `state` column.
    """
    out = df.copy()
    out["model"] = out["model"].astype(str).str.strip().map(lambda m: _MODEL_LABELS.get(m, m))
    if "multimodal_state" in out.columns and "state" not in out.columns:
        out = out.rename(columns={"multimodal_state": "state"})
    if dataset is not None:
        out["dataset"] = dataset
    if "dataset" not in out.columns:
        raise ValueError("No `dataset` column and no `dataset=` argument given.")
    out["fold"] = out["fold"].astype(int)
    out["test_score"] = out["test_score"].astype(float)
    return out[_CANONICAL].reset_index(drop=True)


def screen_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    """Raw Delta_Joint / Delta_Awareness per model. No completeness check, NO verdict.

    Means are rounded to 3 decimals before differencing, matching passes().
    Delta_Awareness is NaN when the `ft` state is absent (the frozen-only screen).
    """
    rows = []
    for model, sub in scores.groupby("model"):
        means = sub.groupby("state")["test_score"].mean().round(3)
        unimodal = [means[s] for s in ("no_text", "text_only") if s in means.index]
        if "all" in means.index and unimodal:
            delta_joint = float(means["all"] - max(unimodal))
        else:
            delta_joint = float("nan")
        if "ft" in means.index and "all" in means.index:
            delta_awareness = float(means["ft"] - means["all"])
        else:
            delta_awareness = float("nan")
        rows.append({
            "model": model,
            "n_rows": len(sub),
            "delta_joint": delta_joint,
            "delta_awareness": delta_awareness,
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def missing_cells(scores: pd.DataFrame, models: tuple[str, ...] = CURATION_MODELS) -> list[tuple[str, str, int]]:
    """(model, state, fold) triples absent from the full grid, sorted."""
    have = set(zip(scores["model"], scores["state"], scores["fold"]))
    want = {(m, s, f) for m in models for s in STATES for f in FOLDS}
    return sorted(want - have)


def verdict(scores: pd.DataFrame, delta: float = DELTA) -> dict:
    """Real pass/fail. Requires the complete 5-model x 4-state x 5-fold grid.

    Raises ValueError naming the absent cells rather than letting passes() fail
    with an opaque AssertionError -- the message is a re-queue list.
    """
    datasets = scores["dataset"].unique()
    if len(datasets) != 1:
        raise ValueError(f"verdict() takes exactly one dataset, got {list(datasets)}")
    absent = missing_cells(scores)
    if absent:
        raise ValueError(
            f"Incomplete grid for {datasets[0]}: {len(absent)} missing (model, state, fold) "
            f"cells -- re-queue these runs: {absent}"
        )
    per_model = {
        model: bool(passes(sub, delta=delta))
        for model, sub in scores[scores["model"].isin(CURATION_MODELS)].groupby("model")
    }
    n_pass = sum(per_model.values())
    return {
        "dataset": str(datasets[0]),
        "per_model": per_model,
        "n_pass": n_pass,
        "quorum": QUORUM,
        "accepted": n_pass >= QUORUM,
        "deltas": screen_deltas(scores),
    }
