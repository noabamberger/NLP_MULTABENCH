"""Register a candidate CSV as a MulTaBench dataset, without touching multabench/.

The repo's normal path needs an enum member PLUS a module in
`multabench/datasets/annotated/`, and `curation_mapping.py` auto-imports every module
in that package and re-raises on failure -- so one malformed generated file would break
dataset loading for everything. We therefore build a `CuratedDataset` in memory and
inject it into the `CURATIONS` dict at runtime instead.

This still reuses the repo's ENTIRE curation path (`curate_dataset`): column drops,
target extraction, value/type curation, null-target row removal, and
`filter_by_multimodality` for the requested state.

Dataset names must be `{BIN|MUL|REG}_TEXT_{REST}`: the prefix is load-bearing, since
`evaluate.py:63` derives task_type from `name[:3]`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from multabench.datasets.curation import MultimodalDataset, curate_dataset
from multabench.datasets.curation_mapping import CURATIONS
from multabench.datasets.curation_objects import CuratedDataset, CuratedFeature, CuratedTarget
from multabench.datasets.multimodal import MultimodalState
from multabench.datasets.objects import FeatureType, SupervisedTask

STATE_BY_FLAG: dict[str, MultimodalState] = {
    "no_text": MultimodalState.NO_TEXT,
    "text_only": MultimodalState.TEXT_ONLY,
    "all": MultimodalState.ALL,
    "ft": MultimodalState.ALL,  # same features; differs only by tune_e5
}

_TASKS = {
    "REG": SupervisedTask.REGRESSION,
    "BIN": SupervisedTask.BINARY,
    "MUL": SupervisedTask.MULTICLASS,
}


class CandidateID(str, Enum):
    """Minimal stand-in for a MultimodalDatasetID.

    Only `.name` is consumed downstream (logging, and the task_type prefix slice),
    so a str-Enum built per candidate is sufficient and keeps us out of
    multabench/datasets/all_datasets.py entirely.
    """


def _make_id(name: str) -> Enum:
    if name[:3] not in _TASKS:
        raise ValueError(f"Candidate name must start with BIN_/MUL_/REG_, got {name!r}")
    return Enum(name, {name: name}, type=str)[name]


@dataclass
class CandidateSpec:
    """Everything a hand-written annotated/ module would declare."""

    name: str
    csv_path: str
    target: str
    task: str                                     # "REG" | "BIN" | "MUL"
    cols_to_drop: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)      # force FeatureType.TEXT
    numeric_cols: list[str] = field(default_factory=list)   # force FeatureType.NUMERIC
    categorical_cols: list[str] = field(default_factory=list)
    read_kwargs: dict = field(default_factory=dict)         # sep/encoding/decimal/...
    context: str = ""
    # "" or "log1p". The repo only WARNS about |z|>5 targets and never clips
    # (check_extreme_outliers), so a heavy right tail -- every price dataset -- lets a
    # handful of rows dominate R^2 and swamp both deltas. This is the CandidateSpec
    # equivalent of an annotated module's PROCESSING_FUNC.
    #
    # It changes the task: scores become R^2 on log price, not on price. That is a
    # curation choice and has to be stated wherever the numbers are reported.
    target_transform: str = ""

    def features(self) -> list[CuratedFeature]:
        feats = []
        for c in self.text_cols:
            feats.append(CuratedFeature(raw_name=c, feat_type=FeatureType.TEXT))
        for c in self.numeric_cols:
            feats.append(CuratedFeature(raw_name=c, feat_type=FeatureType.NUMERIC))
        for c in self.categorical_cols:
            feats.append(CuratedFeature(raw_name=c, feat_type=FeatureType.CATEGORICAL))
        return feats


def register(spec: CandidateSpec) -> Enum:
    """Inject the candidate into CURATIONS and return its dataset id."""
    dataset_id = _make_id(spec.name)
    CURATIONS[spec.name] = CuratedDataset(
        name=spec.name,
        target=CuratedTarget(raw_name=spec.target, task_type=_TASKS[spec.task]),
        features=spec.features(),
        cols_to_drop=list(spec.cols_to_drop),
        context=spec.context,
    )
    return dataset_id


def apply_target_transform(df: pd.DataFrame, spec: CandidateSpec) -> pd.DataFrame:
    """Apply spec.target_transform to the target column, in place on a copy.

    Applied to the raw frame BEFORE curate_dataset, so every downstream consumer --
    the split, the models, the metric, and the TAR discretisation into 20 bins --
    sees the same transformed target. Transforming later would silently give the
    fine-tuning bins a different target from the one being scored.
    """
    if not spec.target_transform:
        return df
    if spec.target_transform != "log1p":
        raise ValueError(f"Unknown target_transform {spec.target_transform!r}")
    out = df.copy()
    y = pd.to_numeric(out[spec.target], errors="coerce")
    if (y.dropna() < 0).any():
        raise ValueError(
            f"log1p target_transform needs a non-negative target; {spec.target!r} "
            f"has negative values (min {float(y.min())})")
    out[spec.target] = np.log1p(y)
    return out


def load_candidate(spec: CandidateSpec, state_flag: str) -> MultimodalDataset:
    """Read the CSV and run it through the repo's full curation path."""
    if state_flag not in STATE_BY_FLAG:
        raise ValueError(f"Unknown state {state_flag!r}; expected one of {sorted(STATE_BY_FLAG)}")
    dataset_id = register(spec)
    df = pd.read_csv(spec.csv_path, low_memory=False, **spec.read_kwargs)
    df = apply_target_transform(df, spec)
    return curate_dataset(
        x=df, y=None, dataset_id=dataset_id,
        multimodal_state=STATE_BY_FLAG[state_flag],
    )
