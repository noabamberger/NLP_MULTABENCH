"""Cross-process cache for the TAR fine-tuned E5 encoder, plus a max_length cap
for the fine-tuning loop.

Both are installed by monkeypatching a single function,
`multabench.e5.e5_finetune.finetune_e5_with_lora`. That is the seam because it is
the *only* producer of a tuned encoder, and because its own arguments are the
complete set of determinants of what it returns:

    (train_texts, train_y, val_texts, val_y, device, tokenizer, **hyperparams)

`fit_text_encoders_tuned` imports it inside the function body, so rebinding the
module attribute is picked up at call time.

1. TUNED-ENCODER CACHE. TAR fine-tuning happens only in the embedding step, never
   end-to-end: `TabularModel.fit` runs `fit_preprocessor` (which fine-tunes E5
   against an auxiliary head on the target) strictly before `fit_model`, and
   LightGBM/CatBoost are not differentiable, so nothing can flow back. The tuned
   encoder is therefore a pure function of the arguments above -- the tabular
   learner is not one of them.

   In a curation sweep that means the 25 `ft` runs (5 models x 5 folds) contain
   only 10 distinct fine-tunings, because per (dataset, fold) the models differ
   only in `USE_VAL_SPLIT` (True for LightGBM/CatBoost/TabM, False for the two
   TabPFNs) and hence in `x_train`.

   We do NOT key on that grouping. We key on a hash of the actual arguments, so
   the cache stays correct even if upstream later threads a real `fold` into
   `split_to_val` and the groups change or vanish. `tests/curation_lab/
   test_tar_cache.py` verifies the grouping empirically *with the cache off*,
   because a wrong grouping would corrupt every Delta_Awareness number silently.

   WHAT IS PERSISTED: the LoRA adapter only (~1.3 MB), via PEFT's own
   `save_pretrained`, plus a sha256 of the full backbone `state_dict()`. A hit
   rebuilds the model with the same `_load_fresh_e5` upstream uses and reapplies
   the adapter; the sha256 is then recomputed and must match, otherwise the entry
   is treated as a miss. The base weights are frozen during LoRA training
   (`bias="none"`, only lora_A/lora_B and the discarded auxiliary head require
   grad), which is what makes adapter-only persistence lossless -- and the
   checksum is what proves it rather than assuming it.

   PCA is NOT cached. `fit_text_encoders_tuned` refits it per call from the tuned
   embeddings, exactly as before; only the fine-tuning is skipped.

2. TRAINING max_length CAP. `TextLabelDataset` tokenizes with
   padding="max_length", max_length=512 while real passages are far shorter
   (anchor dataset: 71 tokens), so ~7x of the most expensive compute in the
   pipeline is spent on masked padding. We compute the cap from the actual
   training texts and assert zero truncation.

   The training string format is NOT the frozen-encode format. Frozen encoding
   uses `format_e5_passage(col, val)`; training passes pre-joined `f"{col}: {val}"`
   strings through `TextLabelDataset`, which does `prefix + (str(t).strip() or " ")`
   -- an extra strip that removes the trailing space of an empty value. Hence
   `training_passages()` here rather than `cache.longest_passage_tokens()`; the
   tokenizing and truncation-checking helpers are reused from `cache.py`.

   Like the frozen cap, this is NOT bit-exact: changing the padded length
   reassociates the float32 matmuls, which moves activations by ~1e-7 and hence
   the gradients too. It is therefore opt-in, and it participates in the cache key
   so weights trained under different caps can never be served for one another.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import warnings
from typing import Any, Iterable

import numpy as np
import torch

from multabench.e5 import e5_finetune as _e5f
from multabench.e5.constants import E5_PASSAGE_PREFIX

from curation_lab.runner.cache import (
    CAP_MULTIPLE,
    MODEL_MAX_LENGTH,
    _token_lengths,
    check_no_truncation,
)

# Bump when the stored artefact's meaning changes, so old entries can never be
# served under a new interpretation.
CACHE_VERSION = "tar-1"

_original_finetune = _e5f.finetune_e5_with_lora

_cache_dir: str | None = None
_dynamic_max_length: bool = False
_verify_on_write: bool = True

STATS: dict[str, int] = {"hits": 0, "misses": 0, "corrupt": 0}

# The fine-tuning corpus is one flat list of passages, not per-column, so there is
# no column name to name in a truncation message.
_CORPUS_LABEL = "<e5 fine-tuning corpus>"

# Arguments that carry the training data itself; hashed separately from the
# scalar hyperparameters.
_DATA_ARGS = ("train_texts", "train_y", "val_texts", "val_y", "device", "tokenizer")


# ---------------------------------------------------------------------------
# max_length cap for the training loop
# ---------------------------------------------------------------------------

def training_passages(texts: Iterable[str], prefix: str = E5_PASSAGE_PREFIX) -> list[str]:
    """The exact strings `TextLabelDataset` tokenizes, for the given raw texts.

    Mirrors `TextLabelDataset.__init__` (`[str(t).strip() or " " for t in texts]`)
    and `__getitem__` (`self.prefix + self.texts[idx]`). Kept in one place so the
    cap is measured on the same string the trainer sees, prefix included.
    """
    return [prefix + (str(t).strip() or " ") for t in texts]


def longest_training_tokens(texts: Iterable[str], tokenizer) -> int:
    """Tokenized length of the longest training passage, no truncation."""
    passages = training_passages(texts)
    if not passages:
        return 0
    return max(_token_lengths(tokenizer, passages))


def round_to_cap(longest: int, multiple: int = CAP_MULTIPLE,
                 ceiling: int = MODEL_MAX_LENGTH) -> int:
    """Smallest multiple-of-`multiple` cap holding `longest`, never past `ceiling`.

    Same arithmetic as `cache.compute_max_length_cap`; that one is bound to the
    `format_e5_passage` string format, which is not the training format.
    `test_tar_cache.py::test_cap_rounding_matches_the_frozen_path` pins the two
    together so they cannot drift.
    """
    return min(ceiling, max(multiple, -(-longest // multiple) * multiple))


def compute_training_max_length_cap(texts: Iterable[str], tokenizer) -> int:
    """Cap for the fine-tuning loop, asserted not to truncate anything."""
    longest = longest_training_tokens(texts, tokenizer)
    cap = round_to_cap(longest)
    check_no_truncation(longest, cap, _CORPUS_LABEL)
    return cap


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def effective_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve every hyperparameter of `finetune_e5_with_lora`, defaults included.

    Read off the upstream signature rather than restated here, so a changed
    upstream default invalidates the cache instead of being silently ignored.
    """
    resolved: dict[str, Any] = {}
    for name, param in inspect.signature(_original_finetune).parameters.items():
        if name in _DATA_ARGS or param.kind is param.VAR_KEYWORD:
            continue
        resolved[name] = kwargs[name] if name in kwargs else param.default
    for name, value in kwargs.items():
        if name not in resolved and name not in _DATA_ARGS:
            resolved[name] = value
    return resolved


def _update_texts(h: "hashlib._Hash", texts: Iterable[str]) -> None:
    items = [str(t) for t in texts]
    h.update(str(len(items)).encode("utf-8"))
    for t in items:
        b = t.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)


def _update_labels(h: "hashlib._Hash", y: Any) -> None:
    arr = np.asarray(y).ravel()
    h.update(f"{arr.dtype}|{arr.shape}".encode("utf-8"))
    if arr.dtype.kind in "OUSMm":
        _update_texts(h, [str(v) for v in arr])
    else:
        h.update(np.ascontiguousarray(arr).tobytes())


def _library_tag() -> str:
    """Versions that can change trained weights for identical inputs.

    Stored weights are far more version-sensitive than a frozen forward pass, and
    a library bump is rare, so paying a full recompute on one is the right trade.
    """
    try:
        import peft
        peft_version = peft.__version__
    except Exception:                                   # pragma: no cover
        peft_version = "unknown"
    return f"torch={torch.__version__}|peft={peft_version}"


def compute_key(train_texts, train_y, val_texts, val_y, device,
                eff_kwargs: dict[str, Any]) -> str:
    """Content hash of everything `finetune_e5_with_lora`'s output depends on."""
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(torch.device(device)).encode("utf-8"))
    h.update(b"\x00")
    h.update(_library_tag().encode("utf-8"))
    h.update(b"\x00")
    for name in sorted(eff_kwargs):
        h.update(f"{name}={eff_kwargs[name]!r}\x00".encode("utf-8"))
    h.update(b"\x01")
    _update_texts(h, train_texts)
    h.update(b"\x02")
    _update_labels(h, train_y)
    h.update(b"\x03")
    _update_texts(h, val_texts)
    h.update(b"\x04")
    _update_labels(h, val_y)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def state_dict_sha256(module: torch.nn.Module) -> str:
    """Checksum of every parameter and buffer, keys included.

    This is the guarantee that adapter-only persistence is lossless: it is
    recomputed after a cache hit is rebuilt and compared against the value
    recorded when the model was trained.
    """
    h = hashlib.sha256()
    sd = module.state_dict()
    for key in sorted(sd):
        tensor = sd[key].detach().to("cpu")
        h.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}\x00".encode("utf-8"))
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def _adapter_dir(entry_dir: str) -> str:
    return os.path.join(entry_dir, "adapter")


def _meta_path(entry_dir: str) -> str:
    return os.path.join(entry_dir, "meta.json")


def _rebuild(entry_dir: str, device, model_name: str) -> torch.nn.Module | None:
    """Reload a cached tuned backbone, or None if it cannot be trusted."""
    try:
        from peft import PeftModel
        with open(_meta_path(entry_dir), "r", encoding="utf-8") as f:
            meta = json.load(f)
        base, _ = _e5f._load_fresh_e5(device, model_name=model_name)
        model = PeftModel.from_pretrained(base, _adapter_dir(entry_dir))
        model.to(device)
        model.eval()
    except Exception as exc:                            # pragma: no cover
        warnings.warn(f"TAR cache entry {entry_dir} could not be loaded ({exc}); "
                      "re-running the fine-tuning.", RuntimeWarning, stacklevel=2)
        return None
    got = state_dict_sha256(model)
    if got != meta.get("state_dict_sha256"):
        warnings.warn(
            f"TAR cache entry {entry_dir} failed its checksum "
            f"(stored {meta.get('state_dict_sha256')}, rebuilt {got}). Ignoring it "
            "and re-running the fine-tuning; the entry is not trustworthy.",
            RuntimeWarning, stacklevel=2,
        )
        return None
    return model


def _store(entry_dir: str, backbone: torch.nn.Module, device, meta: dict[str, Any]) -> None:
    """Persist the adapter + checksum atomically; never let a failure break a run."""
    if not hasattr(backbone, "save_pretrained"):        # pragma: no cover
        warnings.warn("Tuned backbone is not a PEFT model; nothing cached.",
                      RuntimeWarning, stacklevel=2)
        return
    tmp_dir = entry_dir + ".tmp"
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        backbone.save_pretrained(_adapter_dir(tmp_dir))
        meta = dict(meta, state_dict_sha256=state_dict_sha256(backbone))
        with open(_meta_path(tmp_dir), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True, default=str)
        if _verify_on_write and _rebuild(tmp_dir, device, meta["model_name"]) is None:
            raise RuntimeError("write-back verification failed")
        if os.path.isdir(entry_dir):
            # Another process won the race; its entry is keyed identically, so
            # either is correct. Keep the existing one.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return
        os.replace(tmp_dir, entry_dir)
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        warnings.warn(f"Could not write TAR cache entry {entry_dir}: {exc}. The run "
                      "is unaffected; the next one will re-fine-tune.",
                      RuntimeWarning, stacklevel=2)


# ---------------------------------------------------------------------------
# The patch
# ---------------------------------------------------------------------------

def _wrapped_finetune(train_texts, train_y, val_texts, val_y, device, tokenizer,
                      **kwargs):
    kwargs = dict(kwargs)
    train_texts, val_texts = list(train_texts), list(val_texts)

    if _dynamic_max_length and "max_length" not in kwargs:
        kwargs["max_length"] = compute_training_max_length_cap(
            train_texts + val_texts, tokenizer)
    elif "max_length" in kwargs:
        # Caller was explicit; still refuse to truncate silently.
        check_no_truncation(longest_training_tokens(train_texts + val_texts, tokenizer),
                            int(kwargs["max_length"]), _CORPUS_LABEL)

    eff = effective_kwargs(kwargs)

    def run_original():
        return _original_finetune(train_texts=train_texts, train_y=train_y,
                                  val_texts=val_texts, val_y=val_y, device=device,
                                  tokenizer=tokenizer, **kwargs)

    if _cache_dir is None:
        return run_original()

    key = compute_key(train_texts, train_y, val_texts, val_y, device, eff)
    entry_dir = os.path.join(_cache_dir, key)
    model_name = eff["model_name"]

    if os.path.isdir(entry_dir):
        model = _rebuild(entry_dir, device, model_name)
        if model is not None:
            STATS["hits"] += 1
            print(f"[tar_cache] hit {key[:12]} -- skipping E5 fine-tuning "
                  f"({len(train_texts)} train examples)")
            return model, tokenizer
        STATS["corrupt"] += 1
        shutil.rmtree(entry_dir, ignore_errors=True)

    STATS["misses"] += 1
    backbone, out_tokenizer = run_original()
    _store(entry_dir, backbone, device, {
        "key": key,
        "version": CACHE_VERSION,
        "model_name": model_name,
        "device": str(torch.device(device)),
        "libraries": _library_tag(),
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "kwargs": eff,
    })
    return backbone, out_tokenizer


def enable_tar_cache(cache_dir: str = ".tar_cache", dynamic_max_length: bool = False) -> None:
    """Reuse a previously fine-tuned E5 whose training inputs hash identically.

    `dynamic_max_length` additionally caps the training-loop padding to the
    longest passage present (~7x faster, NOT bit-exact -- see module docstring).
    It is part of the cache key, so capped and uncapped weights never mix.
    """
    global _cache_dir, _dynamic_max_length
    os.makedirs(cache_dir, exist_ok=True)
    _cache_dir = cache_dir
    _dynamic_max_length = dynamic_max_length
    _e5f.finetune_e5_with_lora = _wrapped_finetune


def enable_training_max_length_cap() -> None:
    """Cap the fine-tuning padding without caching anything."""
    global _cache_dir, _dynamic_max_length
    _cache_dir = None
    _dynamic_max_length = True
    _e5f.finetune_e5_with_lora = _wrapped_finetune


def disable_tar_cache() -> None:
    """Restore the unpatched fine-tuning function."""
    global _cache_dir, _dynamic_max_length
    _cache_dir = None
    _dynamic_max_length = False
    _e5f.finetune_e5_with_lora = _original_finetune


def is_enabled() -> bool:
    return _e5f.finetune_e5_with_lora is not _original_finetune


def cache_stats() -> dict[str, int]:
    return dict(STATS)


def reset_stats() -> None:
    for k in STATS:
        STATS[k] = 0
