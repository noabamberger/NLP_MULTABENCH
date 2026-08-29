"""Optional frozen-embedding cache and dynamic max_length capping.

Two independent, result-preserving speedups for the E5 encode path, both
installed by monkeypatching `text_embeddings.encode_texts_with_e5` (which is
where every call site resolves the name, including `E5ColumnEncoder.encode_texts`).

1. PER-STRING CACHE. Frozen E5 output for a given (model, column, text) is
   deterministic and identical across every learner and every fold, so caching it
   is result-preserving: PCA is still refitted per fold from the cached vectors.

   VALID ONLY WHEN tune_e5 IS False. A LoRA fine-tuned encoder produces different
   vectors for the same text while reporting the same base model name, so enabling
   this during a TAR run would silently serve wrong embeddings. run.py enables it
   only for frozen states, and enable_cache() refuses without an explicit
   frozen_only acknowledgement.

   Keying is PER STRING, not per call. The train split changes with the fold, so a
   whole-list key would miss on every fold; a per-string key lets all 5 folds and
   all 5 learners share one encode of each unique text.

2. DYNAMIC max_length (OFF BY DEFAULT -- see below). Upstream tokenizes with
   padding="max_length", max_length=512, so every text is padded to 512 tokens
   regardless of its true length. We compute the cap at runtime from the actual
   texts -- ceil(longest/8)*8 -- and assert zero truncation, rather than
   hardcoding a number that would silently truncate a longer-texted dataset.
   Unlike the cache, this is independent of whether the encoder was tuned: it
   depends only on the tokenizer and the strings.

   NOT BIT-EXACT. Padding is masked out of the attention softmax exactly, so the
   mathematics is unchanged -- but shrinking the padded sequence changes the K
   dimension of every matmul, and the BLAS kernel reassociates its float32 sums
   differently at a different shape. Embeddings move by ~1e-7 (float32 epsilon is
   1.19e-7), i.e. last-bit rounding. That is not thread non-determinism: encoding
   is exactly reproducible run-to-run at any fixed max_length, and single-threading
   does not make two caps agree. Because this repo's whole purpose is reproducing
   the paper's numbers, the cap therefore defaults to OFF and must be opted into
   (`run.py --max-length-cap`). See tests/curation_lab/test_max_length.py.

The cap participates in the cache key, so vectors computed under different caps
can never be served for one another.
"""
from __future__ import annotations

import hashlib
import os
import threading
import warnings

import numpy as np

from multabench.baselines.preprocessing import text_embeddings as _te
from multabench.e5.constants import format_e5_passage

_original_encode = _te.encode_texts_with_e5
_lock = threading.Lock()

# E5-small-v2 is a BERT encoder with 512 learned position embeddings, so 512 is a
# hard ceiling, not a tunable. It is also the upstream default, which makes
# cap == MODEL_MAX_LENGTH exactly the unoptimized behaviour.
MODEL_MAX_LENGTH = 512
# Round caps up to a multiple of 8 for AVX/MKL (and tensor-core) alignment.
CAP_MULTIPLE = 8

# Tokenize in chunks so a large column does not build one huge python list of ids.
_LEN_CHUNK = 2048


def _token_lengths(tokenizer, passages: list[str]) -> list[int]:
    """True tokenized length (with special tokens) of each passage, no truncation."""
    lengths: list[int] = []
    for i in range(0, len(passages), _LEN_CHUNK):
        chunk = passages[i : i + _LEN_CHUNK]
        with warnings.catch_warnings():
            # HF warns when a sequence exceeds the model max; we handle that
            # explicitly below and do not want it to look like a failure here.
            warnings.simplefilter("ignore")
            out = tokenizer(chunk, padding=False, truncation=False,
                            add_special_tokens=True)
        lengths.extend(len(ids) for ids in out["input_ids"])
    return lengths


def longest_passage_tokens(texts, col_name: str, tokenizer) -> int:
    """Length in tokens of the longest `passage: {col}: {val}` string in `texts`.

    Built with format_e5_passage so it measures the exact string
    encode_texts_with_e5 will tokenize, prefix included.
    """
    texts = list(texts)
    if not texts:
        return 0
    passages = [format_e5_passage(str(col_name), col_val=t) for t in texts]
    return max(_token_lengths(tokenizer, passages))


def _cap_for_length(longest: int, multiple: int = CAP_MULTIPLE,
                    ceiling: int = MODEL_MAX_LENGTH) -> int:
    """Round a measured token length up to the cap actually used."""
    cap = max(multiple, -(-longest // multiple) * multiple)
    return min(cap, ceiling)


def compute_max_length_cap(texts, col_name: str, tokenizer,
                           multiple: int = CAP_MULTIPLE,
                           ceiling: int = MODEL_MAX_LENGTH) -> int:
    """Smallest multiple-of-`multiple` cap that holds every passage untruncated.

    Never exceeds `ceiling` (the model's position-embedding limit). If the longest
    text does not fit under the ceiling, the returned cap is the ceiling -- which
    is exactly what upstream would have used -- and check_no_truncation() will
    warn rather than raise, because that truncation is upstream behaviour and not
    something this optimization introduced.
    """
    longest = longest_passage_tokens(texts, col_name, tokenizer)
    return _cap_for_length(longest, multiple=multiple, ceiling=ceiling)


def check_no_truncation(longest: int, cap: int, col_name: str) -> None:
    """Fail loudly if the chosen cap would truncate text that 512 would not have.

    A silent truncation corrupts every downstream score invisibly, so a cap below
    the model ceiling that does not fit the data is a hard error. Text longer than
    the model ceiling is a different case: upstream truncates it too, so we warn
    and match the unoptimized behaviour instead of breaking a dataset that
    otherwise runs.
    """
    if longest <= cap:
        return
    if cap >= MODEL_MAX_LENGTH:
        warnings.warn(
            f"Column {col_name!r}: longest passage is {longest} tokens, above the "
            f"E5 limit of {MODEL_MAX_LENGTH}. Truncating at {MODEL_MAX_LENGTH}, "
            "which is what the unoptimized pipeline does as well.",
            RuntimeWarning, stacklevel=2,
        )
        return
    raise ValueError(
        f"max_length cap {cap} would truncate column {col_name!r}: longest passage "
        f"is {longest} tokens. The cap must never truncate -- that would change "
        "embeddings, and therefore scores, invisibly."
    )


def _text_key(model_name: str, col_name: str, text: str, cap: int) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(col_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(cap).encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _store_path(cache_dir: str, model_name: str, col_name: str) -> str:
    tag = hashlib.sha256(f"{model_name}\x00{col_name}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{tag}.npz")


def _load_store(path: str) -> tuple[dict[str, np.ndarray], int | None]:
    """Return (entries, cap). `cap` is None for a store written before capping."""
    if not os.path.exists(path):
        return {}, None
    with np.load(path, allow_pickle=False) as z:
        keys, vecs = z["keys"], z["vecs"]
        cap = int(z["cap"]) if "cap" in z.files else None
    return {str(k): vecs[i] for i, k in enumerate(keys)}, cap


def _save_store(path: str, store: dict[str, np.ndarray], cap: int) -> None:
    if not store:
        return
    keys = np.array(list(store.keys()))
    vecs = np.stack([store[k] for k in store])
    tmp = path + ".tmp.npz"
    np.savez(tmp, keys=keys, vecs=vecs, cap=np.array(cap))
    os.replace(tmp, path)


def _install(cache_dir: str | None, dynamic_max_length: bool) -> None:
    """Install the encode wrapper for the requested combination of features."""

    def wrapped_encode(texts, model, tokenizer, device, col_name, max_length=None,
                       **kwargs):
        texts = list(texts)
        col = str(col_name)

        # Resolve the cap first: it feeds both the encode call and the cache key.
        # `longest` stays None when nothing needs checking -- an uncapped call is
        # byte-for-byte upstream behaviour, which does not truncation-check either.
        longest = None
        if max_length is not None:
            cap = int(max_length)                      # caller was explicit; obey it
            longest = longest_passage_tokens(texts, col, tokenizer)
        elif dynamic_max_length:
            longest = longest_passage_tokens(texts, col, tokenizer)
            cap = _cap_for_length(longest)
        else:
            cap = MODEL_MAX_LENGTH

        if cache_dir is None:
            if longest is not None:
                check_no_truncation(longest, cap, col)
            return _original_encode(texts=texts, model=model, tokenizer=tokenizer,
                                    device=device, col_name=col_name,
                                    max_length=cap, **kwargs)

        model_name = str(getattr(getattr(model, "config", None), "_name_or_path", "unknown"))
        path = _store_path(cache_dir, model_name, col)
        with _lock:
            store, stored_cap = _load_store(path)
            # The cap only ever grows. Otherwise a fold whose subset happens to
            # exclude the longest text would ask for a smaller cap, miss every
            # entry and re-encode the whole column.
            if stored_cap is not None and stored_cap > cap:
                cap = stored_cap
            if stored_cap != cap:
                # Entries were keyed at another cap (or by a pre-cap version of
                # this module) and can never be hit again. Drop them.
                store = {}
            if longest is not None:
                check_no_truncation(longest, cap, col)

            keys = [_text_key(model_name, col, t, cap) for t in texts]
            missing_idx = [i for i, k in enumerate(keys) if k not in store]
            if missing_idx:
                fresh = _original_encode(
                    texts=[texts[i] for i in missing_idx], model=model,
                    tokenizer=tokenizer, device=device, col_name=col_name,
                    max_length=cap, **kwargs,
                )
                fresh = np.asarray(fresh)
                for slot, i in enumerate(missing_idx):
                    store[keys[i]] = fresh[slot]
                _save_store(path, store, cap)
            return np.stack([store[k] for k in keys])

    _te.encode_texts_with_e5 = wrapped_encode


def enable_cache(cache_dir: str = ".emb_cache", frozen_only: bool = False,
                 dynamic_max_length: bool = False) -> None:
    """Patch encode_texts_with_e5 to serve per-string cached vectors.

    `frozen_only` must be True: it is a caller acknowledgement that no fine-tuned
    encoder will be used while the cache is active.

    `dynamic_max_length` also caps padding to the longest text actually present.
    It defaults to False because capping is NOT bit-exact against the upstream 512
    (see tests/curation_lab/test_max_length.py); the cache alone is. The cap is part
    of the cache key, so capped and uncapped vectors can never be served for one
    another.
    """
    if not frozen_only:
        raise ValueError(
            "enable_cache(frozen_only=True) is required. The cache is only valid for "
            "frozen encoders; a LoRA-tuned E5 yields different vectors for the same text."
        )
    os.makedirs(cache_dir, exist_ok=True)
    _install(cache_dir=cache_dir, dynamic_max_length=dynamic_max_length)


def enable_dynamic_max_length() -> None:
    """Patch encode_texts_with_e5 to cap padding, without caching anything.

    Safe for tuned encoders too: the cap is derived from the tokenizer and the
    strings, never from model weights.
    """
    _install(cache_dir=None, dynamic_max_length=True)


def disable_cache() -> None:
    """Restore the unpatched encode function (removes cache and cap alike)."""
    _te.encode_texts_with_e5 = _original_encode


def is_enabled() -> bool:
    return _te.encode_texts_with_e5 is not _original_encode
