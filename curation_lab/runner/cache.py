"""Optional frozen-embedding cache.

Frozen E5 output for a given (model, column, text) is deterministic and identical
across every learner and every fold, so caching it is result-preserving: PCA is
still refitted per fold from the cached vectors.

VALID ONLY WHEN tune_e5 IS False. A LoRA fine-tuned encoder produces different
vectors for the same text while reporting the same base model name, so enabling
this during a TAR run would silently serve wrong embeddings. run.py enables it
only for frozen states, and enable_cache() refuses without an explicit
frozen_only acknowledgement.

Keying is PER STRING, not per call. The train split changes with the fold, so a
whole-list key would miss on every fold; a per-string key lets all 5 folds and
all 5 learners share one encode of each unique text.

Ships DISABLED. Enable only after test_cache_is_bit_exact passes.
"""
from __future__ import annotations

import hashlib
import os
import threading

import numpy as np

from multabench.baselines.preprocessing import text_embeddings as _te

_original_encode = _te.encode_texts_with_e5
_lock = threading.Lock()


def _text_key(model_name: str, col_name: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(col_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _store_path(cache_dir: str, model_name: str, col_name: str) -> str:
    tag = hashlib.sha256(f"{model_name}\x00{col_name}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{tag}.npz")


def _load_store(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as z:
        keys, vecs = z["keys"], z["vecs"]
    return {str(k): vecs[i] for i, k in enumerate(keys)}


def _save_store(path: str, store: dict[str, np.ndarray]) -> None:
    if not store:
        return
    keys = np.array(list(store.keys()))
    vecs = np.stack([store[k] for k in store])
    tmp = path + ".tmp.npz"
    np.savez(tmp, keys=keys, vecs=vecs)
    os.replace(tmp, path)


def enable_cache(cache_dir: str = ".emb_cache", frozen_only: bool = False) -> None:
    """Patch encode_texts_with_e5 to serve per-string cached vectors.

    `frozen_only` must be True: it is a caller acknowledgement that no fine-tuned
    encoder will be used while the cache is active.
    """
    if not frozen_only:
        raise ValueError(
            "enable_cache(frozen_only=True) is required. The cache is only valid for "
            "frozen encoders; a LoRA-tuned E5 yields different vectors for the same text."
        )
    os.makedirs(cache_dir, exist_ok=True)

    def cached_encode_texts(texts, model, tokenizer, device, col_name):
        texts = list(texts)
        model_name = str(getattr(getattr(model, "config", None), "_name_or_path", "unknown"))
        path = _store_path(cache_dir, model_name, str(col_name))
        with _lock:
            store = _load_store(path)
            keys = [_text_key(model_name, str(col_name), t) for t in texts]
            missing_idx = [i for i, k in enumerate(keys) if k not in store]
            if missing_idx:
                fresh = _original_encode(
                    texts=[texts[i] for i in missing_idx], model=model,
                    tokenizer=tokenizer, device=device, col_name=col_name,
                )
                fresh = np.asarray(fresh)
                for slot, i in enumerate(missing_idx):
                    store[keys[i]] = fresh[slot]
                _save_store(path, store)
            return np.stack([store[k] for k in keys])

    _te.encode_texts_with_e5 = cached_encode_texts


def disable_cache() -> None:
    _te.encode_texts_with_e5 = _original_encode


def is_enabled() -> bool:
    return _te.encode_texts_with_e5 is not _original_encode
