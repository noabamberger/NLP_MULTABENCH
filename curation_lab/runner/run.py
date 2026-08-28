"""Execute one (dataset, model, state, fold) run and append a result row.

Enters the existing pipeline at evaluate_on_loaded_dataset() so the paper's
protocol -- seeds, 90/10 split, 2000-row test cap, metric selection -- is
inherited unchanged.

On W&B: this module never calls wandb_run(), which is what would demand
credentials. The `wandb` package itself is still imported transitively, because
multabench/baselines/benchmarks/evaluate.py:17 imports get_current_commit_hash
from multabench.utils.logging, which imports wandb at module level. Installing
the package is enough; no credentials are needed.
"""
from __future__ import annotations

import argparse

from tabstar.training.devices import get_device

from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
from multabench.baselines.catboost import CatBoost
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.constants import DEVICE
from multabench.finetune.train_args import E5TrainArgs

from curation_lab.runner.cache import disable_cache, enable_cache
from curation_lab.runner.paper import STATE_BY_FLAG, load_paper_dataset
from curation_lab.runner.results import append_row, row_from_summary

MODELS: dict[str, type] = {
    LightGBM.SHORT_NAME: LightGBM,        # "light"
    CatBoost.SHORT_NAME: CatBoost,        # "cat"
    TabM.SHORT_NAME: TabM,                # "tabm"
    TabPFNv2.SHORT_NAME: TabPFNv2,        # "tabpfnv2"
    TabPFNv2p5.SHORT_NAME: TabPFNv2p5,    # "tabpfnv2p5"
}


def run_one(dataset: str, model_key: str, state_flag: str, fold: int, out_csv: str,
            e5_overrides: dict | None = None, use_cache: bool = True,
            cache_dir: str = ".emb_cache") -> dict:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model {model_key!r}; expected one of {sorted(MODELS)}")
    tune_e5 = state_flag == "ft"
    e5_train_kwargs = None
    if tune_e5:
        e5_train_kwargs = E5TrainArgs().to_dict()
        e5_train_kwargs.update(e5_overrides or {})

    # The cache is only sound for frozen encoders; a LoRA-tuned E5 returns
    # different vectors for the same text under the same base model name.
    cache_on = use_cache and not tune_e5
    if cache_on:
        enable_cache(cache_dir, frozen_only=True)
    try:
        loaded = load_paper_dataset(dataset, state_flag)
        summary = evaluate_on_loaded_dataset(
            model_cls=MODELS[model_key],
            dataset=loaded,
            fold=fold,
            device=get_device(device=DEVICE),
            train_examples=DOWNSTREAM_EXAMPLES,
            multimodal_state=STATE_BY_FLAG[state_flag],
            tune_e5=tune_e5,
            e5_train_kwargs=e5_train_kwargs,
        )
    finally:
        if cache_on:
            disable_cache()
    append_row(out_csv, row_from_summary(summary, state_flag=state_flag))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run one MulTaBench evaluation without W&B.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--state", required=True, choices=sorted(STATE_BY_FLAG))
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--out", default="results/candidates/phase1.csv")
    p.add_argument("--e5-epochs", type=int, default=None,
                   help="Override E5 fine-tuning epochs (ft only); use to timebox CPU runs.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the frozen-embedding cache (it is off for ft regardless).")
    args = p.parse_args()
    overrides = {"epochs": args.e5_epochs} if args.e5_epochs is not None else None
    summary = run_one(args.dataset, args.model, args.state, args.fold, args.out, overrides,
                      use_cache=not args.no_cache)
    print(f"{args.model} {args.state} fold={args.fold} score={summary['test_score']:.4f} "
          f"runtime={summary['runtime']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
