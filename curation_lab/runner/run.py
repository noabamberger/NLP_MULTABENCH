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
import os

import torch
from tabstar.training.devices import get_device

from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
from multabench.baselines.catboost import CatBoost
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.constants import DEVICE
from multabench.finetune.train_args import E5TrainArgs

from curation_lab.runner.cache import disable_cache, enable_cache, enable_dynamic_max_length
from curation_lab.runner.paper import STATE_BY_FLAG, load_paper_dataset
from curation_lab.runner.results import append_row, row_from_summary

MODELS: dict[str, type] = {
    LightGBM.SHORT_NAME: LightGBM,        # "light"
    CatBoost.SHORT_NAME: CatBoost,        # "cat"
    TabM.SHORT_NAME: TabM,                # "tabm"
    TabPFNv2.SHORT_NAME: TabPFNv2,        # "tabpfnv2"
    TabPFNv2p5.SHORT_NAME: TabPFNv2p5,    # "tabpfnv2p5"
}


def _force_cpu_for_finetuning() -> torch.device:
    """Opt-in bypass of the upstream GPU-only guard, for CPU validation runs.

    multabench/e5/e5_finetune.py:245 asserts `"CUDA_VISIBLE_DEVICES" in os.environ`
    ("Single GPU only" per its docstring). The assertion tests only that the name is
    present, not that a GPU exists, so setting it satisfies the guard; and because
    torch.cuda.is_available() is False on this machine, nothing can silently land on
    a GPU. We also return an explicit torch.device so tabstar's get_device()
    short-circuits instead of inferring "cuda" from the variable we just set.

    This exists to answer "does the TAR code path run at all on CPU?". Numbers it
    produces are for pipeline validation only -- real Delta_Awareness measurements
    belong on the GPU cluster.
    """
    if torch.cuda.is_available():
        raise RuntimeError("--cpu-ft is for CPU-only machines; CUDA is available here.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return torch.device("cpu")


def run_one(dataset: str, model_key: str, state_flag: str, fold: int, out_csv: str,
            e5_overrides: dict | None = None, use_cache: bool = True,
            cache_dir: str = ".emb_cache", cpu_ft: bool = False,
            max_length_cap: bool = False) -> dict:
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
    # The max_length cap is OFF by default: it is ~7x faster but NOT bit-exact
    # (see test_max_length.py -- shrinking the padded sequence reassociates the
    # float32 matmuls and moves embeddings by ~1e-7). Opt in only where a
    # last-bit change is acceptable; never for numbers compared to the paper.
    patched = cache_on or max_length_cap
    if cache_on:
        enable_cache(cache_dir, frozen_only=True, dynamic_max_length=max_length_cap)
    elif max_length_cap:
        enable_dynamic_max_length()
    device = _force_cpu_for_finetuning() if (tune_e5 and cpu_ft) else get_device(device=DEVICE)
    try:
        loaded = load_paper_dataset(dataset, state_flag)
        summary = evaluate_on_loaded_dataset(
            model_cls=MODELS[model_key],
            dataset=loaded,
            fold=fold,
            device=device,
            train_examples=DOWNSTREAM_EXAMPLES,
            multimodal_state=STATE_BY_FLAG[state_flag],
            tune_e5=tune_e5,
            e5_train_kwargs=e5_train_kwargs,
        )
    finally:
        if patched:
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
    p.add_argument("--cache-dir", default=".emb_cache",
                   help="Directory for the frozen-embedding cache.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the frozen-embedding cache (it is off for ft regardless).")
    p.add_argument("--max-length-cap", action="store_true",
                   help="Pad to the longest text present instead of 512 (~7x faster encode). "
                        "NOT bit-exact: embeddings move by ~1e-7. Off by default.")
    p.add_argument("--cpu-ft", action="store_true",
                   help="Bypass the upstream GPU-only guard to run ft on CPU. Validation only; "
                        "real Delta_Awareness numbers belong on the GPU cluster.")
    args = p.parse_args()
    overrides = {"epochs": args.e5_epochs} if args.e5_epochs is not None else None
    summary = run_one(args.dataset, args.model, args.state, args.fold, args.out, overrides,
                      use_cache=not args.no_cache, cache_dir=args.cache_dir,
                      cpu_ft=args.cpu_ft, max_length_cap=args.max_length_cap)
    print(f"{args.model} {args.state} fold={args.fold} score={summary['test_score']:.4f} "
          f"runtime={summary['runtime']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
