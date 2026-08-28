"""Compare our runner's scores against the shipped paper results.

Tolerance is asymmetric by design. The frozen states are seeded and must agree
closely. `ft` is stochastic LoRA fine-tuning and the shipped numbers came from
different hardware and library versions, so it is reported but never asserted.

Note on the frozen tolerance: states that touch no encoder (`no_text`) reproduce
essentially exactly, while embedding-dependent states (`text_only`, `all`) carry
CPU-vs-GPU floating point drift through E5 -> PCA -> the downstream learner. The
default tolerance accommodates the latter.
"""
from __future__ import annotations

import os

import pandas as pd

from curation_lab.criterion.deltas import normalize

SHIPPED_DIR = "multabench/leaderboard/results/text"
FROZEN_STATES = ("no_text", "text_only", "all")
FROZEN_TOL = 0.03


def compare_to_shipped(ours_csv: str, dataset: str, frozen_tol: float = FROZEN_TOL) -> pd.DataFrame:
    """One row per (model, state, fold) we ran, with the shipped score alongside."""
    shipped_path = os.path.join(SHIPPED_DIR, f"{dataset}.csv")
    shipped = normalize(pd.read_csv(shipped_path, encoding="utf-8"), dataset=dataset)
    ours = normalize(pd.read_csv(ours_csv, encoding="utf-8"))
    ours = ours[ours["dataset"] == dataset]

    merged = ours.merge(
        shipped, on=["model", "dataset", "state", "fold"],
        how="left", suffixes=("_ours", "_paper"),
    )
    merged["abs_diff"] = (merged["test_score_ours"] - merged["test_score_paper"]).abs()
    merged["frozen"] = merged["state"].isin(FROZEN_STATES)
    merged["within_tol"] = merged["frozen"] & (merged["abs_diff"] <= frozen_tol)
    return merged.sort_values(["model", "state", "fold"]).reset_index(drop=True)


def delta_joint_agreement(ours_csv: str, dataset: str) -> pd.DataFrame:
    """Per-model Delta_Joint computed from our runs and from the paper's, side by side.

    This is the check that actually matters for curation: the criterion consumes
    Delta_Joint, not individual state scores, so agreement in sign and rough
    magnitude here is worth more than 4-decimal equality on any single state.
    """
    from curation_lab.criterion.deltas import screen_deltas

    shipped = normalize(
        pd.read_csv(os.path.join(SHIPPED_DIR, f"{dataset}.csv"), encoding="utf-8"),
        dataset=dataset,
    )
    ours = normalize(pd.read_csv(ours_csv, encoding="utf-8"))
    ours = ours[ours["dataset"] == dataset]

    # Restrict the paper side to the exact (model, state, fold) cells we ran, so
    # the comparison is like-for-like rather than 5 folds against 1.
    ran = set(zip(ours["model"], ours["state"], ours["fold"]))
    shipped = shipped[[k in ran for k in zip(shipped["model"], shipped["state"], shipped["fold"])]]

    a = screen_deltas(ours).rename(columns={"delta_joint": "delta_joint_ours"})
    b = screen_deltas(shipped).rename(columns={"delta_joint": "delta_joint_paper"})
    out = a[["model", "n_rows", "delta_joint_ours"]].merge(
        b[["model", "delta_joint_paper"]], on="model", how="outer"
    )
    out["sign_agrees"] = (out["delta_joint_ours"] > 0) == (out["delta_joint_paper"] > 0)
    return out
