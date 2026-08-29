"""Full curation verdict for one candidate: 5 models x 4 states x 5 folds.

Frozen states are cheap (embedding cache) and run first, so a candidate that loses
Delta_Joint on the full grid is rejected before any ft time is spent. Only then does
the expensive TAR half run.

CPU-only. Uses the training max_length cap, which took a step from 333s to ~8s on
this machine -- without it the ft half is not feasible on CPU at all.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
import warnings

import pandas as pd


def _run_all(spec, out_csv: str, folds, do_ft: bool, epochs: int | None) -> None:
    import torch
    from tabstar.training.devices import get_device

    from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
    from multabench.baselines.catboost import CatBoost
    from multabench.baselines.lgbm import LightGBM
    from multabench.baselines.tabm import TabM
    from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
    from multabench.constants import DEVICE
    from multabench.finetune.train_args import E5TrainArgs

    from curation_lab.ingest.candidate import STATE_BY_FLAG, load_candidate
    from curation_lab.runner.cache import disable_cache, enable_cache, enable_dynamic_max_length
    from curation_lab.runner.results import append_row, row_from_summary
    from curation_lab.screen.t3_tar import _cpu_device, _training_max_length

    models = {"light": LightGBM, "cat": CatBoost, "tabm": TabM,
              "tabpfnv2": TabPFNv2, "tabpfnv2p5": TabPFNv2p5}
    done = set()
    if os.path.exists(out_csv):  # resumable: skip cells already recorded
        prev = pd.read_csv(out_csv, encoding="utf-8")
        done = set(zip(prev["model"], prev["multimodal_state"], prev["fold"]))

    states = ["no_text", "text_only", "all"] + (["ft"] if do_ft else [])
    for state in states:
        tune = state == "ft"
        for fold in folds:
            for key, cls in models.items():
                if (cls.MODEL_NAME, state, fold) in done:
                    continue
                t0 = time.time()
                kwargs = None
                if tune:
                    enable_dynamic_max_length()
                else:
                    enable_cache(".emb_cache", frozen_only=True, dynamic_max_length=True)
                try:
                    loaded = load_candidate(spec, state)
                    if tune:
                        kwargs = E5TrainArgs().to_dict()
                        if epochs:
                            kwargs["epochs"] = epochs
                        kwargs["max_length"] = _training_max_length(loaded, spec.text_cols)
                    dev = _cpu_device() if tune else get_device(device=DEVICE)
                    s = evaluate_on_loaded_dataset(
                        model_cls=cls, dataset=loaded, fold=fold, device=dev,
                        train_examples=DOWNSTREAM_EXAMPLES,
                        multimodal_state=STATE_BY_FLAG[state],
                        tune_e5=tune, e5_train_kwargs=kwargs,
                    )
                    append_row(out_csv, row_from_summary(s, state_flag=state))
                    print(f"  {key:<10} {state:<10} f{fold} = {s['test_score']:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  {key:<10} {state:<10} f{fold} SKIP {type(e).__name__}: "
                          f"{str(e)[:70]}", flush=True)
                finally:
                    disable_cache()


def main() -> None:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=None, help="override E5 epochs")
    p.add_argument("--frozen-only", action="store_true")
    args = p.parse_args()

    from curation_lab.discover.kaggle_search import _load_env
    _load_env()
    import kagglehub

    from curation_lab.criterion.deltas import normalize, screen_deltas
    from curation_lab.screen.auto_spec import build_spec
    from curation_lab.screen.batch_profile import _read_any_csv

    d = kagglehub.dataset_download(args.ref)
    csv = max(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True), key=os.path.getsize)
    df, enc, _ = _read_any_csv(csv)
    rk = {} if enc == "utf-8" else {"encoding": enc.split("/")[0]}
    spec, why = build_spec(df, name=args.name, csv_path=csv, read_kwargs=rk)
    print(f"{args.name}: rows={len(df)} | {why}", flush=True)

    folds = tuple(int(f) for f in args.folds.split(","))
    _run_all(spec, args.out, folds, do_ft=not args.frozen_only, epochs=args.epochs)

    df_out = normalize(pd.read_csv(args.out, encoding="utf-8"))
    print("\n=== deltas ===")
    print(screen_deltas(df_out).to_string(index=False))


if __name__ == "__main__":
    main()
