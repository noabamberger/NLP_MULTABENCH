"""Tests for the dynamic max_length cap on the frozen E5 encode path.

HEADLINE RESULT: capping is NOT bit-exact. The cap is therefore OFF by default
(`run.py --max-length-cap` opts in). See test_capped_encoding_is_not_bit_exact,
which pins the observed behaviour rather than asserting the property we wanted.

Why it is not exact: padding is masked out of the attention softmax exactly (the
mask adds finfo.min, whose softmax weight underflows to 0.0), so the *mathematics*
is unchanged. But shrinking the padded sequence changes the K dimension of every
matmul, and the BLAS kernel reassociates its float32 sums differently at a
different shape. The result moves by ~1e-7 -- around float32 epsilon (1.19e-7),
i.e. last-bit rounding, not a modelling change. It is not thread non-determinism:
encoding is exactly reproducible run-to-run at any fixed max_length, and forcing
torch to a single thread does not make the two caps agree.
"""
import numpy as np
import pytest
import torch

from multabench.e5.constants import E5_SMALL_V2, format_e5_passage
from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5

from curation_lab.runner import cache as cache_mod

SHORT_TEXTS = [
    "great product, arrived fast",
    "terrible, broke in a day",
    "ok for the price",
    "great product, arrived fast",   # deliberate duplicate
]

LONG_TEXT = (
    "I have been using this laptop for about three weeks now and honestly the battery "
    "life is the single best thing about it; it lasts a full working day with plenty "
    "of browser tabs open, which is more than I can say for the previous model."
)

COL = "review"


@pytest.fixture(scope="module")
def e5():
    device = torch.device("cpu")
    model, tokenizer = get_vanilla_e5(device, model_name=E5_SMALL_V2)
    return model, tokenizer, device


# --------------------------------------------------------------------------
# Cap computation
# --------------------------------------------------------------------------

def test_cap_is_measured_on_the_formatted_passage(e5):
    """The cap must measure `passage: {col}: {val}`, not the bare value."""
    _, tokenizer, _ = e5
    bare = len(tokenizer(["x"], padding=False, truncation=False)["input_ids"][0])
    measured = cache_mod.longest_passage_tokens(["x"], COL, tokenizer)
    expected = len(tokenizer([format_e5_passage(COL, col_val="x")],
                             padding=False, truncation=False)["input_ids"][0])
    assert measured == expected
    assert measured > bare, "prefix and column name must be counted"


def test_cap_rounds_up_to_multiple_of_eight(e5):
    _, tokenizer, _ = e5
    longest = cache_mod.longest_passage_tokens(SHORT_TEXTS, COL, tokenizer)
    cap = cache_mod.compute_max_length_cap(SHORT_TEXTS, COL, tokenizer)
    assert cap % cache_mod.CAP_MULTIPLE == 0
    assert cap >= longest
    assert cap - longest < cache_mod.CAP_MULTIPLE
    assert cap < cache_mod.MODEL_MAX_LENGTH, "short texts should cap well under 512"


def test_cap_never_exceeds_the_model_ceiling(e5):
    """E5-small-v2 has 512 position embeddings; the cap may not go past them."""
    _, tokenizer, _ = e5
    huge = " ".join(["word"] * 4000)
    cap = cache_mod.compute_max_length_cap([huge], COL, tokenizer)
    assert cap == cache_mod.MODEL_MAX_LENGTH


def test_cap_handles_empty_input(e5):
    _, tokenizer, _ = e5
    assert cache_mod.compute_max_length_cap([], COL, tokenizer) == cache_mod.CAP_MULTIPLE


# --------------------------------------------------------------------------
# Zero-truncation guard
# --------------------------------------------------------------------------

def test_truncation_check_raises_when_a_text_exceeds_the_cap():
    """A cap below the ceiling that does not fit the data is a hard error."""
    with pytest.raises(ValueError, match="would truncate"):
        cache_mod.check_no_truncation(longest=73, cap=72, col_name=COL)


def test_truncation_check_passes_when_everything_fits():
    cache_mod.check_no_truncation(longest=71, cap=72, col_name=COL)
    cache_mod.check_no_truncation(longest=72, cap=72, col_name=COL)


def test_truncation_beyond_the_model_ceiling_only_warns():
    """Upstream truncates at 512 too, so matching it is not a regression."""
    with pytest.warns(RuntimeWarning, match="above the E5 limit"):
        cache_mod.check_no_truncation(longest=900, cap=cache_mod.MODEL_MAX_LENGTH,
                                      col_name=COL)


def test_encode_path_raises_rather_than_truncating(tmp_path, e5, monkeypatch):
    """End-to-end: a cap too small for the data must fail loudly, not truncate.

    Stubbing the cap to a value the data cannot fit simulates the bug this guard
    exists to catch -- a cap computed from one set of texts applied to a longer one.
    """
    model, tokenizer, device = e5
    assert cache_mod.longest_passage_tokens(SHORT_TEXTS, COL, tokenizer) > 8

    cache_mod.enable_cache(str(tmp_path / "emb"), frozen_only=True,
                           dynamic_max_length=True)
    monkeypatch.setattr(cache_mod, "_cap_for_length", lambda longest: 8)
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        with pytest.raises(ValueError, match="would truncate"):
            te.encode_texts_with_e5(texts=SHORT_TEXTS, model=model, tokenizer=tokenizer,
                                    device=device, col_name=COL)
    finally:
        cache_mod.disable_cache()


# --------------------------------------------------------------------------
# The correctness claim the whole optimization rests on
# --------------------------------------------------------------------------

def test_encoding_is_reproducible_at_a_fixed_max_length(e5):
    """Baseline: at a fixed cap, encoding is exactly reproducible.

    This is what makes the per-string cache sound, and it isolates the cause of
    the inexactness below to the max_length change itself.
    """
    model, tokenizer, device = e5
    kw = dict(texts=SHORT_TEXTS, model=model, tokenizer=tokenizer, device=device,
              col_name=COL)
    assert np.array_equal(encode_texts_with_e5(max_length=512, **kw),
                          encode_texts_with_e5(max_length=512, **kw))
    assert np.array_equal(encode_texts_with_e5(max_length=72, **kw),
                          encode_texts_with_e5(max_length=72, **kw))


def test_capped_encoding_is_not_bit_exact(e5):
    """THE reason the cap ships disabled.

    We wanted np.array_equal against the uncapped 512 encode. It does not hold.
    This test pins what actually happens so the finding cannot be quietly lost:
    the vectors agree to ~1e-7 (float32 epsilon) but differ in their last bits.
    """
    model, tokenizer, device = e5
    texts = SHORT_TEXTS + [LONG_TEXT]
    cap = cache_mod.compute_max_length_cap(texts, COL, tokenizer)
    assert cap < 512

    kw = dict(texts=texts, model=model, tokenizer=tokenizer, device=device, col_name=COL)
    uncapped = encode_texts_with_e5(max_length=512, **kw)
    capped = encode_texts_with_e5(max_length=cap, **kw)

    assert not np.array_equal(uncapped, capped), (
        "Capped encoding is now bit-exact. That is good news, but it invalidates "
        "this test and the reasoning behind the cap defaulting to OFF -- re-run "
        "the anchor fidelity check and reconsider enabling it by default."
    )
    diff = np.abs(uncapped - capped).max()
    assert diff < 1e-6, f"drift {diff:.3e} is far above last-bit rounding"
    assert np.allclose(uncapped, capped, rtol=0, atol=1e-6)


def test_single_thread_does_not_restore_exactness(e5):
    """Rules out thread-count reduction order as the cause of the drift."""
    model, tokenizer, device = e5
    texts = SHORT_TEXTS + [LONG_TEXT]
    cap = cache_mod.compute_max_length_cap(texts, COL, tokenizer)
    kw = dict(texts=texts, model=model, tokenizer=tokenizer, device=device, col_name=COL)

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        uncapped = encode_texts_with_e5(max_length=512, **kw)
        capped = encode_texts_with_e5(max_length=cap, **kw)
    finally:
        torch.set_num_threads(original_threads)
    assert not np.array_equal(uncapped, capped)


# --------------------------------------------------------------------------
# Cap x cache interaction
# --------------------------------------------------------------------------

def test_cache_is_still_bit_exact_under_a_fixed_cap(tmp_path, e5):
    """With the cap on, cached vectors must still match a direct capped encode.

    The cap costs last-bit fidelity against 512, but it must not add any error of
    its own on top: a cache hit has to reproduce the capped encode exactly.
    """
    model, tokenizer, device = e5
    texts = SHORT_TEXTS + [LONG_TEXT]
    cap = cache_mod.compute_max_length_cap(texts, COL, tokenizer)
    direct = encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                  device=device, col_name=COL, max_length=cap)

    cache_mod.enable_cache(str(tmp_path / "emb"), frozen_only=True,
                           dynamic_max_length=True)
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        miss = te.encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                       device=device, col_name=COL)
        hit = te.encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                      device=device, col_name=COL)
    finally:
        cache_mod.disable_cache()
    assert np.array_equal(direct, miss)
    assert np.array_equal(direct, hit)


def test_cap_does_not_shrink_for_a_subset(tmp_path, e5):
    """A fold whose subset excludes the longest text must reuse the stored cap.

    Otherwise every fold would compute a smaller cap, miss the whole cache and
    re-encode the column -- and would do so at a cap that disagrees in the last
    bits with the vectors the other folds used.
    """
    model, tokenizer, device = e5
    texts = SHORT_TEXTS + [LONG_TEXT]
    cap = cache_mod.compute_max_length_cap(texts, COL, tokenizer)
    subset = [SHORT_TEXTS[2], SHORT_TEXTS[0]]
    assert cache_mod.compute_max_length_cap(subset, COL, tokenizer) < cap

    cache_dir = tmp_path / "emb"
    cache_mod.enable_cache(str(cache_dir), frozen_only=True, dynamic_max_length=True)
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        full = te.encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                       device=device, col_name=COL)
        got = te.encode_texts_with_e5(texts=subset, model=model, tokenizer=tokenizer,
                                      device=device, col_name=COL)
    finally:
        cache_mod.disable_cache()

    # Served from the store at the wider cap, i.e. the same rows as the full call.
    assert np.array_equal(full[2], got[0])
    assert np.array_equal(full[0], got[1])

    stores = list(cache_dir.glob("*.npz"))
    assert len(stores) == 1, "the subset must not have opened a second store"
    with np.load(stores[0], allow_pickle=False) as z:
        assert int(z["cap"]) == cap


def test_cap_change_invalidates_the_store(tmp_path, e5):
    """Vectors computed at one cap must never be served at another."""
    model, tokenizer, device = e5
    cache_dir = tmp_path / "emb"

    cache_mod.enable_cache(str(cache_dir), frozen_only=True, dynamic_max_length=True)
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        te.encode_texts_with_e5(texts=SHORT_TEXTS, model=model, tokenizer=tokenizer,
                                device=device, col_name=COL)
        small_cap = cache_mod.compute_max_length_cap(SHORT_TEXTS, COL, tokenizer)
        # A longer text forces a wider cap; the old entries must be dropped, not reused.
        wider = SHORT_TEXTS + [LONG_TEXT]
        got = te.encode_texts_with_e5(texts=wider, model=model, tokenizer=tokenizer,
                                      device=device, col_name=COL)
    finally:
        cache_mod.disable_cache()

    big_cap = cache_mod.compute_max_length_cap(wider, COL, tokenizer)
    assert big_cap > small_cap
    expected = encode_texts_with_e5(texts=wider, model=model, tokenizer=tokenizer,
                                    device=device, col_name=COL, max_length=big_cap)
    assert np.array_equal(expected, got)

    stores = list(cache_dir.glob("*.npz"))
    with np.load(stores[0], allow_pickle=False) as z:
        assert int(z["cap"]) == big_cap
        assert len(z["keys"]) == len(wider) - 1   # one duplicate text in SHORT_TEXTS


def test_dynamic_max_length_without_cache(e5):
    """The cap can be installed on its own, with no caching."""
    model, tokenizer, device = e5
    texts = SHORT_TEXTS + [LONG_TEXT]
    cap = cache_mod.compute_max_length_cap(texts, COL, tokenizer)
    expected = encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                    device=device, col_name=COL, max_length=cap)
    cache_mod.enable_dynamic_max_length()
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        assert cache_mod.is_enabled()
        got = te.encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                      device=device, col_name=COL)
    finally:
        cache_mod.disable_cache()
    assert not cache_mod.is_enabled()
    assert np.array_equal(expected, got)


def test_cache_without_cap_is_bit_exact_against_512(tmp_path, e5):
    """dynamic_max_length=False must reproduce the unoptimized encode exactly.

    This is the configuration run.py uses by default, and it is what keeps the
    already-validated anchor scores reproducible.
    """
    model, tokenizer, device = e5
    texts = SHORT_TEXTS + [LONG_TEXT]
    uncapped = encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                    device=device, col_name=COL)
    cache_mod.enable_cache(str(tmp_path / "emb"), frozen_only=True,
                           dynamic_max_length=False)
    try:
        from multabench.baselines.preprocessing import text_embeddings as te
        miss = te.encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                       device=device, col_name=COL)
        hit = te.encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer,
                                      device=device, col_name=COL)
    finally:
        cache_mod.disable_cache()
    assert np.array_equal(uncapped, miss)
    assert np.array_equal(uncapped, hit)
