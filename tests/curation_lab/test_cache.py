"""Bit-exactness tests for the frozen embedding cache.

If any of these fail, the cache must be deleted rather than fixed-and-trusted:
a cache that changes results is worse than no cache.
"""
import numpy as np
import pytest
import torch

from multabench.baselines.preprocessing.text_embeddings import encode_texts_with_e5
from multabench.e5.constants import E5_SMALL_V2
from multabench.e5.e5_finetune import get_vanilla_e5

from curation_lab.runner import cache as cache_mod

TEXTS = [
    "great product, arrived fast",
    "terrible, broke in a day",
    "ok for the price",
    "great product, arrived fast",   # deliberate duplicate
]


@pytest.fixture(scope="module")
def e5():
    device = torch.device("cpu")
    model, tokenizer = get_vanilla_e5(device, model_name=E5_SMALL_V2)
    return model, tokenizer, device


def test_enable_requires_frozen_only_acknowledgement(tmp_path):
    with pytest.raises(ValueError):
        cache_mod.enable_cache(str(tmp_path / "c"), frozen_only=False)
    assert not cache_mod.is_enabled()


def test_cache_is_bit_exact(tmp_path, e5):
    model, tokenizer, device = e5
    uncached = encode_texts_with_e5(texts=TEXTS, model=model, tokenizer=tokenizer,
                                    device=device, col_name="review")
    cache_mod.enable_cache(str(tmp_path / "emb"), frozen_only=True)
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        miss = te.encode_texts_with_e5(texts=TEXTS, model=model, tokenizer=tokenizer,
                                       device=device, col_name="review")
        hit = te.encode_texts_with_e5(texts=TEXTS, model=model, tokenizer=tokenizer,
                                      device=device, col_name="review")
    finally:
        cache_mod.disable_cache()
    assert np.array_equal(np.asarray(uncached), np.asarray(miss))
    assert np.array_equal(np.asarray(uncached), np.asarray(hit))


def test_cache_serves_subsets_in_correct_order(tmp_path, e5):
    """The real workload: a later fold requests a re-ordered subset of cached texts."""
    model, tokenizer, device = e5
    cache_mod.enable_cache(str(tmp_path / "emb2"), frozen_only=True)
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        te.encode_texts_with_e5(texts=TEXTS, model=model, tokenizer=tokenizer,
                                device=device, col_name="review")
        subset = [TEXTS[2], TEXTS[0]]
        got = te.encode_texts_with_e5(texts=subset, model=model, tokenizer=tokenizer,
                                      device=device, col_name="review")
    finally:
        cache_mod.disable_cache()
    expected = encode_texts_with_e5(texts=subset, model=model, tokenizer=tokenizer,
                                    device=device, col_name="review")
    assert np.array_equal(np.asarray(expected), np.asarray(got))


def test_cache_separates_columns(tmp_path, e5):
    """Same text under a different column name must not collide."""
    model, tokenizer, device = e5
    cache_mod.enable_cache(str(tmp_path / "emb3"), frozen_only=True)
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        a = te.encode_texts_with_e5(texts=["hello"], model=model, tokenizer=tokenizer,
                                    device=device, col_name="col_a")
        b = te.encode_texts_with_e5(texts=["hello"], model=model, tokenizer=tokenizer,
                                    device=device, col_name="col_b")
    finally:
        cache_mod.disable_cache()
    expected_b = encode_texts_with_e5(texts=["hello"], model=model, tokenizer=tokenizer,
                                      device=device, col_name="col_b")
    assert np.array_equal(np.asarray(expected_b), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(b))
