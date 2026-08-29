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


def _training_max_length(loaded, text_cols: list[str]) -> int:
    """Cap for the TRAINING tokenizer (TextLabelDataset), not just the encode pass.

    Without this the fine-tuning loop pads every example to 512 regardless of true
    length -- measured at 333 s/step on an 8-core CPU, i.e. ~3 h for 2 epochs. The
    cap is reachable because fit_text_encoders_tuned forwards e5_train_kwargs into
    finetune_e5_with_lora(**kwargs), which passes max_length down to TextLabelDataset.

    Training strings are built as f"{col}: {val}" and then prefixed with
    "passage: " by TextLabelDataset -- exactly format_e5_passage(col, val).
    """
    from transformers import AutoTokenizer

    from multabench.e5.constants import E5_SMALL_V2

    from curation_lab.runner.cache import compute_max_length_cap

    tok = AutoTokenizer.from_pretrained(E5_SMALL_V2)
    caps = [compute_max_length_cap(loaded.x[c].astype(str).tolist(), c, tok)
            for c in text_cols if c in loaded.x.columns]
    return max(caps) if caps else 512


def tar_probe(spec: CandidateSpec, all_score: float, fold: int = 0, epochs: int = 2,
              model_cls=LightGBM) -> dict:
    """One ft run; returns the score and Delta_Awareness against a known `all` score."""
    kwargs = E5TrainArgs().to_dict()
    kwargs["epochs"] = epochs
    t0 = time.time()
    enable_dynamic_max_length()
    try:
        loaded = load_candidate(spec, "ft")
        cap = _training_max_length(loaded, spec.text_cols)
        kwargs["max_length"] = cap
        print(f"[tar] training max_length cap = {cap} (was 512) -> ~{512/cap:.1f}x", flush=True)
        summary = evaluate_on_loaded_dataset(
            model_cls=model_cls, dataset=loaded, fold=fold, device=_cpu_device(),
            train_examples=DOWNSTREAM_EXAMPLES,
            multimodal_state=STATE_BY_FLAG["ft"], tune_e5=True, e5_train_kwargs=kwargs,
        )
    finally:
        disable_cache()
    ft = float(summary["test_score"])
    return {"ft": ft, "all": all_score, "cap": cap, "delta_awareness": round(ft - all_score, 4),
            "secs": round(time.time() - t0, 1), "epochs": epochs}
