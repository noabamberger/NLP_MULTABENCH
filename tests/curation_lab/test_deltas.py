"""Criterion tests. Offline: reads only shipped CSVs, fits no models."""
import pandas as pd
import pytest

from curation_lab.criterion.deltas import (
    CURATION_MODELS,
    missing_cells,
    normalize,
    screen_deltas,
    verdict,
)

POOL_CSV = "multabench/leaderboard/results/analysis_curation_sensitivity/pool_scores_long.csv"
MATRIX_CSV = "multabench/leaderboard/results/analysis_curation_sensitivity/pass_matrix.csv"

KNOWN_ACCEPT = "MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT"   # 5 of 5 in shipped matrix
KNOWN_REJECT = "REG_TEXT_FOOD_RAMEN_RATINGS_2022"      # 0 of 5 in shipped matrix


@pytest.fixture(scope="module")
def pool():
    return pd.read_csv(POOL_CSV, encoding="utf-8")


def _one(pool, dataset):
    sub = pool[(pool["dataset"] == dataset) & (pool["model"].isin(CURATION_MODELS))]
    return normalize(sub)


def test_normalize_maps_emoji_model_names_and_state_column():
    raw = pd.DataFrame({
        "model": ["LightGBM 💡", "CatBoost 😸"],
        "multimodal_state": ["all", "ft"],
        "fold": [0, 0],
        "test_score": [0.5, 0.6],
    })
    out = normalize(raw, dataset="D")
    assert list(out.columns) == ["model", "dataset", "state", "fold", "test_score"]
    assert set(out["model"]) == {"LightGBM", "CatBoost"}
    assert set(out["state"]) == {"all", "ft"}
    assert set(out["dataset"]) == {"D"}


def test_verdict_accepts_known_positive(pool):
    got = verdict(_one(pool, KNOWN_ACCEPT))
    assert got["accepted"] is True
    assert sum(got["per_model"].values()) == 5


def test_verdict_rejects_known_negative(pool):
    got = verdict(_one(pool, KNOWN_REJECT))
    assert got["accepted"] is False
    assert sum(got["per_model"].values()) == 0


def test_verdict_reports_missing_cells_instead_of_asserting(pool):
    scores = _one(pool, KNOWN_ACCEPT)
    trimmed = scores[~((scores["model"] == "TabM") & (scores["state"] == "ft") & (scores["fold"] == 3))]
    with pytest.raises(ValueError) as err:
        verdict(trimmed)
    assert "TabM" in str(err.value) and "ft" in str(err.value)


def test_missing_cells_empty_on_complete_grid(pool):
    assert missing_cells(_one(pool, KNOWN_ACCEPT)) == []


def test_screen_deltas_gives_no_verdict_without_ft(pool):
    scores = _one(pool, KNOWN_ACCEPT)
    frozen = scores[scores["state"] != "ft"]
    out = screen_deltas(frozen)
    assert "accepted" not in out.columns
    assert out["delta_joint"].notna().all()
    assert out["delta_awareness"].isna().all()


def test_reproduces_shipped_pass_matrix_exactly():
    """The full 10-model matrix must re-derive with zero mismatched cells."""
    from multabench.leaderboard.analysis.pass_matrix import build_pass_matrix

    df = pd.read_csv(POOL_CSV, encoding="utf-8")
    expected = pd.read_csv(MATRIX_CSV, index_col="dataset", encoding="utf-8")
    got = build_pass_matrix(df).reindex(index=expected.index, columns=expected.columns)
    mismatches = (got.fillna("NA").astype(str) != expected.fillna("NA").astype(str)).values.sum()
    assert int(mismatches) == 0
