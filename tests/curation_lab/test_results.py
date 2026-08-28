import pandas as pd

from curation_lab.runner.results import COLUMNS, append_row, row_from_summary

SUMMARY = {
    "model": "LightGBM 💡",
    "dataset": "MUL_TEXT_PRODUCT_SENTIMENT",
    "fold": 0,
    "test_score": 0.9123,
    "multimodal_state": "all 🔥",     # the enum value -- must be overridden
    "runtime": 12.5,
    "n_train": 4582,
    "n_test": 509,
    "m_features": 31,
    "task_type": "MUL",
    "tune_e5": False,
}


def test_state_flag_overrides_the_enum_value():
    row = row_from_summary(SUMMARY, state_flag="all")
    assert row["multimodal_state"] == "all"      # NOT "all 🔥"


def test_model_keeps_the_emoji_name():
    row = row_from_summary(SUMMARY, state_flag="all")
    assert row["model"] == "LightGBM 💡"


def test_row_has_exactly_the_declared_columns():
    assert list(row_from_summary(SUMMARY, state_flag="ft").keys()) == COLUMNS


def test_append_writes_utf8_and_roundtrips(tmp_path):
    path = tmp_path / "out.csv"
    append_row(str(path), row_from_summary(SUMMARY, state_flag="no_text"))
    append_row(str(path), row_from_summary(SUMMARY, state_flag="text_only"))
    back = pd.read_csv(path, encoding="utf-8")
    assert len(back) == 2
    assert list(back.columns) == COLUMNS
    assert back["model"].iloc[0] == "LightGBM 💡"
    assert set(back["multimodal_state"]) == {"no_text", "text_only"}
