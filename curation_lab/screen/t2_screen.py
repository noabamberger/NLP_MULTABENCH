"""T2: the cheap frozen screen that produces a real Delta_Joint for a candidate.

Only the three FROZEN states are run (no_text / text_only / all). Delta_Joint needs
nothing else, and none of them require a GPU -- which is the whole point of the
funnel: a candidate must prove joint signal on CPU before any TAR/GPU time is spent
on it.

The gate is deliberately looser than the real criterion (>0 rather than >0.001):
a 1-2 fold estimate is noisy, and a discarded good dataset is unrecoverable while a
passed-through bad one only costs cluster time.
"""
from __future__ import annotations

import time

import pandas as pd
from tabstar.training.devices import get_device

from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
from multabench.baselines.catboost import CatBoost
from multabench.baselines.lgbm import LightGBM
from multabench.constants import DEVICE

from curation_lab.criterion.deltas import normalize, screen_deltas
from curation_lab.ingest.candidate import STATE_BY_FLAG, CandidateSpec, load_candidate
from curation_lab.runner.cache import disable_cache, enable_cache
from curation_lab.runner.results import append_row, row_from_summary

FROZEN_STATES = ("no_text", "text_only", "all")
SCREEN_MODELS = {"light": LightGBM, "cat": CatBoost}


def screen(spec: CandidateSpec, out_csv: str, folds=(0,), models=SCREEN_MODELS,
           cache_dir: str = ".emb_cache") -> pd.DataFrame:
    """Run the frozen grid and return per-model Delta_Joint."""
    device = get_device(device=DEVICE)
    for state in FROZEN_STATES:
        for fold in folds:
            for key, cls in models.items():
                t0 = time.time()
                enable_cache(cache_dir, frozen_only=True)
                try:
                    loaded = load_candidate(spec, state)
                    summary = evaluate_on_loaded_dataset(
                        model_cls=cls, dataset=loaded, fold=fold, device=device,
                        train_examples=DOWNSTREAM_EXAMPLES,
                        multimodal_state=STATE_BY_FLAG[state],
                        tune_e5=False, e5_train_kwargs=None,
                    )
                finally:
                    disable_cache()
                append_row(out_csv, row_from_summary(summary, state_flag=state))
                print(f"  {key:<6} {state:<10} fold={fold} score={summary['test_score']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    df = normalize(pd.read_csv(out_csv, encoding="utf-8"))
    return screen_deltas(df)
