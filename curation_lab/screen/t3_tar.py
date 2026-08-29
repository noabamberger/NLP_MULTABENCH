"""T3: measure Delta_Awareness (TAR) for a candidate that already passed T2.

Delta_Awareness = mean(ft) - mean(all)

This is the expensive half of the criterion and the one that kills most datasets
that clear Delta_Joint, so it is worth testing on ONE model/fold before committing
to a full 5x4x5 sweep.

CPU-only by design: `cpu_ft=True` bypasses the upstream GPU-only assertion in
multabench/e5/e5_finetune.py:245. The embedding cache is NOT used for ft (a tuned
encoder yields different vectors for the same text), but the max_length cap IS,
since it is independent of tuning and is where the ~7x saving comes from.
"""
from __future__ import annotations

import os
import time

import torch
from tabstar.training.devices import get_device

from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
from multabench.baselines.lgbm import LightGBM
from multabench.constants import DEVICE
from multabench.finetune.train_args import E5TrainArgs

from curation_lab.ingest.candidate import STATE_BY_FLAG, CandidateSpec, load_candidate
from curation_lab.runner.cache import disable_cache, enable_dynamic_max_length


def _cpu_device() -> torch.device:
    """Satisfy the upstream 'Single GPU only' assertion without a GPU present."""
    if torch.cuda.is_available():
        return get_device(device=DEVICE)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return torch.device("cpu")


def tar_probe(spec: CandidateSpec, all_score: float, fold: int = 0, epochs: int = 2,
              model_cls=LightGBM) -> dict:
    """One ft run; returns the score and Delta_Awareness against a known `all` score."""
    kwargs = E5TrainArgs().to_dict()
    kwargs["epochs"] = epochs
    t0 = time.time()
    enable_dynamic_max_length()
    try:
        loaded = load_candidate(spec, "ft")
        summary = evaluate_on_loaded_dataset(
            model_cls=model_cls, dataset=loaded, fold=fold, device=_cpu_device(),
            train_examples=DOWNSTREAM_EXAMPLES,
            multimodal_state=STATE_BY_FLAG["ft"], tune_e5=True, e5_train_kwargs=kwargs,
        )
    finally:
        disable_cache()
    ft = float(summary["test_score"])
    return {"ft": ft, "all": all_score, "delta_awareness": round(ft - all_score, 4),
            "secs": round(time.time() - t0, 1), "epochs": epochs}
