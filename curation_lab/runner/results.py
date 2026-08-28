"""The run-row CSV schema.

Two fields are load-bearing for the existing analysis layer to consume our output
unmodified:

  multimodal_state  the CLI-style flag (no_text/text_only/all/ft), NOT the
                    MultimodalState enum value ("all 🔥"), which
                    pass_matrix._STATES will not match.
  model             the emoji MODEL_NAME, so committee_pool._MODEL_LABELS maps it.

Everything is written UTF-8: the model names contain emoji and the Windows
console/default codepage here is cp1255.
"""
from __future__ import annotations

import os

import pandas as pd

COLUMNS: list[str] = [
    "model",
    "dataset",
    "fold",
    "multimodal_state",
    "test_score",
    "runtime",
    "n_train",
    "n_test",
    "m_features",
    "task_type",
    "tune_e5",
]

VALID_STATES = {"no_text", "text_only", "all", "ft"}


def row_from_summary(summary: dict, state_flag: str) -> dict:
    """Project an evaluate_on_loaded_dataset() summary onto the schema.

    `state_flag` overrides whatever `multimodal_state` the summary carries,
    because evaluate_on_loaded_dataset() logs the MultimodalState enum and both
    `all` and `ft` map to MultimodalState.ALL -- the enum alone cannot tell them
    apart.
    """
    if state_flag not in VALID_STATES:
        raise ValueError(f"state_flag must be one of {sorted(VALID_STATES)}, got {state_flag!r}")
    row = {c: summary.get(c) for c in COLUMNS}
    row["multimodal_state"] = state_flag
    return row


def append_row(csv_path: str, row: dict) -> None:
    """Append one row, writing the header only when creating the file."""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        csv_path, mode="a", header=write_header, index=False, encoding="utf-8"
    )
