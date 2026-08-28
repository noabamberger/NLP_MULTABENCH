"""Load a MulTaBench paper dataset at one of the four curation conditions.

Phase 1 only needs the already-curated Kaggle-hosted datasets; spec-driven
ingestion of new candidates is Phase 2.
"""
from __future__ import annotations

from multabench.benchmark.load import load_multabench_dataset
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.curation import MultimodalDataset
from multabench.datasets.multimodal import MultimodalState

# `all` and `ft` deliberately share MultimodalState.ALL: they use identical
# features and differ only in whether the text encoder is fine-tuned (tune_e5).
STATE_BY_FLAG: dict[str, MultimodalState] = {
    "no_text": MultimodalState.NO_TEXT,
    "text_only": MultimodalState.TEXT_ONLY,
    "all": MultimodalState.ALL,
    "ft": MultimodalState.ALL,
}


def load_paper_dataset(name: str, state_flag: str) -> MultimodalDataset:
    if state_flag not in STATE_BY_FLAG:
        raise ValueError(f"Unknown state flag {state_flag!r}; expected one of {sorted(STATE_BY_FLAG)}")
    try:
        dataset_id = MulTaBenchDatasetID[name]
    except KeyError:
        raise ValueError(f"{name!r} is not a MulTaBenchDatasetID member") from None
    return load_multabench_dataset(dataset_id, multimodal_state=STATE_BY_FLAG[state_flag])
