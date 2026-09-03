"""GPU smoke test for the TAR (`ft`) code path only.

Runs `multabench.e5.e5_finetune.finetune_e5_with_lora` -- the *only* part of the
pipeline that a GPU changes -- for a small number of epochs, then proves the
returned backbone still encodes. Nothing downstream (LightGBM/CatBoost/TabM/PCA)
is involved, so a failure here is unambiguously a TAR failure.

Two data sources:

  --source synthetic   two lexically-separable classes, generated in-process.
                       No dataset download; isolates "does LoRA fine-tuning run
                       on this GPU at all". This is the smoke test.
  --source anchor      real text columns from a MulTaBench dataset, built
                       through the same `passage: {col}: {val}` formatting and
                       the same 20-bin discretisation the real `ft` run uses.
                       Confirms the real data path, still without downstream
                       models.

`CUDA_VISIBLE_DEVICES` is set before torch is imported: e5_finetune.py:245
asserts the name is present, and torch reads it at first CUDA init.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "anchor"], default="synthetic")
    p.add_argument("--dataset", default="MUL_TEXT_PRODUCT_SENTIMENT",
                   help="MulTaBenchDatasetID name, used only with --source anchor.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--text-layers", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--gpu", default="0", help="Value for CUDA_VISIBLE_DEVICES.")
    return p.parse_args()


ARGS = _parse_args()
# Must precede `import torch` (and thus any transitive multabench import).
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu
# multabench.constants reads GPU to build DEVICE="cuda:{GPU}".
os.environ.setdefault("GPU", ARGS.gpu)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _fail(msg: str) -> "None":
    print(f"\nSMOKE FAIL: {msg}")
    sys.exit(1)


def _synthetic(n_train: int, n_val: int) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Two lexically-separable classes, so 1 epoch is enough to move the loss."""
    rng = np.random.default_rng(0)
    pos = ["excellent battery life", "works flawlessly, very happy", "great value, fast delivery",
           "sturdy build and crisp screen", "exceeded expectations completely"]
    neg = ["stopped working after a week", "terrible quality, returned it",
           "screen cracked immediately", "waste of money, very slow", "arrived broken and dirty"]

    def build(n: int) -> tuple[list[str], np.ndarray]:
        labels = np.array([i % 2 for i in range(n)])
        texts = [
            f"review: {(pos if lab else neg)[rng.integers(0, 5)]} (sample {i})"
            for i, lab in enumerate(labels)
        ]
        return texts, labels

    tr_x, tr_y = build(n_train)
    va_x, va_y = build(n_val)
    return tr_x, tr_y, va_x, va_y


def _anchor(dataset: str, n_train: int, n_val: int):
    """Real text columns, formatted exactly as fit_text_encoders_tuned does."""
    import pandas as pd
    from multabench.preprocessing.discretize import discretize_numerical
    from multabench.preprocessing.feat_types import detect_text_features
    from multabench.preprocessing.splits import split_to_val

    from curation_lab.runner.paper import load_paper_dataset

    loaded = load_paper_dataset(dataset, "ft")
    x, y = loaded.x, loaded.y
    text_cols = sorted(detect_text_features(x))
    if not text_cols:
        _fail(f"{dataset} has no detected text columns; nothing for TAR to tune.")
    print(f"[anchor] {dataset}: {len(x)} rows, text columns = {text_cols}")

    is_cls = loaded.task_type.is_cls if hasattr(loaded.task_type, "is_cls") else True
    if not is_cls:
        y = discretize_numerical(y, n_bins=20)
    x_tr, x_val, y_tr, y_val = split_to_val(x=x, y=y, is_cls=True)

    def flatten(xs, ys, cap):
        texts, labels = [], []
        for idx in xs.index:
            for col in text_cols:
                val = str(xs.loc[idx, col]).strip() if pd.notna(xs.loc[idx, col]) else ""
                texts.append(f"{col}: {val}")
                labels.append(ys.loc[idx])
        return texts[:cap], np.array(labels[:cap])

    tr_x, tr_y = flatten(x_tr, y_tr, n_train)
    va_x, va_y = flatten(x_val, y_val, n_val)
    return tr_x, tr_y, va_x, va_y


def main() -> None:
    a = ARGS
    print("=" * 72)
    print("TAR (ft) GPU smoke test -- LoRA fine-tuning of E5 only")
    print("=" * 72)

    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() is False. This test is GPU-only; "
              "for a CPU sanity run use `run.py --state ft --cpu-ft` instead.")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[gpu] {name}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
    if cap[0] < 8:
        print("[gpu] note: pre-Ampere card -- no bf16 and no tensor cores; fp32 only.")

    from transformers import AutoTokenizer

    from multabench.e5.constants import E5_SMALL_V2
    from multabench.e5.e5_finetune import encode_texts_with_e5, finetune_e5_with_lora

    if a.source == "synthetic":
        tr_x, tr_y, va_x, va_y = _synthetic(a.n_train, a.n_val)
    else:
        tr_x, tr_y, va_x, va_y = _anchor(a.dataset, a.n_train, a.n_val)
    print(f"[data] source={a.source}  train={len(tr_x)}  val={len(va_x)}  "
          f"classes={sorted(set(map(str, tr_y)))[:6]}")

    tokenizer = AutoTokenizer.from_pretrained(E5_SMALL_V2)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    backbone, tuned_tok = finetune_e5_with_lora(
        train_texts=tr_x,
        train_y=tr_y,
        val_texts=va_x,
        val_y=va_y,
        device=device,
        tokenizer=tokenizer,
        model_name=E5_SMALL_V2,
        lora_rank=a.lora_rank,
        text_layers=a.text_layers,
        epochs=a.epochs,
        batch_size=a.batch_size,
        max_length=a.max_length,
        dataloader_num_workers=a.num_workers,
    )
    train_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    # The point of TAR is a *usable tuned encoder*, so encoding must work too.
    t1 = time.time()
    emb = encode_texts_with_e5(texts=va_x[:32], col_name="smoke", model=backbone,
                               tokenizer=tuned_tok, device=device, batch_size=16,
                               max_length=a.max_length)
    encode_s = time.time() - t1

    on_cuda = next(backbone.parameters()).device.type
    print("\n" + "-" * 72)
    print(f"[result] train {a.epochs} epoch(s) : {train_s:7.1f}s")
    print(f"[result] encode 32 texts      : {encode_s:7.1f}s")
    print(f"[result] peak GPU memory      : {peak_gb:7.2f} GB")
    print(f"[result] backbone device      : {on_cuda}")
    print(f"[result] embedding shape      : {emb.shape}  dtype={emb.dtype}")

    if on_cuda != "cuda":
        _fail(f"tuned backbone ended up on {on_cuda!r}, not cuda -- it never used the GPU.")
    if emb.shape[0] != len(va_x[:32]):
        _fail(f"embedding row count {emb.shape[0]} != {len(va_x[:32])}")
    if not np.isfinite(emb).all():
        _fail("tuned encoder produced non-finite embeddings (NaN/Inf).")
    if float(np.abs(emb).sum()) == 0.0:
        _fail("tuned encoder produced all-zero embeddings.")

    print("\nSMOKE PASS: one TAR training epoch ran on the GPU and the tuned "
          "encoder produces finite embeddings.")


if __name__ == "__main__":
    main()
