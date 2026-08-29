"""Tests for the TAR tuned-encoder cache and the training-loop max_length cap.

The cache replaces a fine-tuning with a stored one, so its failure mode is
silent: a wrong entry would change every Delta_Awareness number without any
error. Three properties therefore have to hold, and are asserted here rather
than assumed:

  1. fine-tuning is DETERMINISTIC in its inputs (otherwise "reuse" is not the
     same thing as "recompute") -- test_finetuning_is_deterministic;
  2. adapter-only persistence is LOSSLESS -- test_cache_hit_is_bit_exact
     compares the sha256 of the whole backbone state_dict, not just outputs;
  3. the key is CONTENT-addressed, so any change to the training inputs or
     hyperparameters misses -- test_key_* .

The grouping claim that motivates the cache (LightGBM/CatBoost/TabM share one
fine-tuning, the two TabPFNs share another) is checked in
test_encoder_sharing_groups_are_real, with the cache OFF -- it must be observed,
not arranged. That test fits three real learners and takes ~10 minutes on CPU, so
it runs only under MULTABENCH_TAR_SLOW=1.
"""
import contextlib
import io
import json
import os

import numpy as np
import pytest
import torch

# The upstream fine-tuner asserts CUDA_VISIBLE_DEVICES is present ("single GPU
# only"); it checks only for the name, and torch.cuda.is_available() is False
# here, so nothing can land on a GPU. Same bypass as run.py --cpu-ft.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from transformers import AutoTokenizer

from multabench.e5 import e5_finetune as e5f
from multabench.e5.constants import E5_PASSAGE_PREFIX, E5_SMALL_V2, format_e5_passage
from multabench.e5.e5_finetune import TextLabelDataset

from curation_lab.runner import cache as cache_mod
from curation_lab.runner import tar_cache as tc

RUN_SLOW = os.environ.get("MULTABENCH_TAR_SLOW") == "1"
slow = pytest.mark.skipif(not RUN_SLOW, reason="set MULTABENCH_TAR_SLOW=1 (minutes of CPU)")

DEVICE = torch.device("cpu")

# Tiny, fast fine-tuning settings. The point of these tests is equality between
# runs, never score quality.
TRAIN_KWARGS = dict(epochs=1, batch_size=16, dataloader_num_workers=0,
                    text_layers=1, lora_rank=8, max_length=32)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(E5_SMALL_V2)


def _make_corpus(n: int, seed: int):
    """(texts, labels) in the shape fit_text_encoders_tuned builds: "col: value"."""
    rng = np.random.default_rng(seed)
    words = ["fast", "broke", "great", "cheap", "awful", "fine", "sturdy", "loud"]
    texts = [
        "review: " + " ".join(rng.choice(words, size=int(rng.integers(3, 12))))
        for _ in range(n)
    ]
    labels = np.array([t.count("a") % 3 for t in texts])
    return texts, labels


@pytest.fixture(scope="module")
def corpus():
    train_texts, train_y = _make_corpus(48, 1)
    val_texts, val_y = _make_corpus(16, 2)
    return train_texts, train_y, val_texts, val_y


def _finetune(corpus, tokenizer, **overrides):
    train_texts, train_y, val_texts, val_y = corpus
    kwargs = dict(TRAIN_KWARGS, **overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return e5f.finetune_e5_with_lora(
            train_texts=train_texts, train_y=train_y, val_texts=val_texts,
            val_y=val_y, device=DEVICE, tokenizer=tokenizer, **kwargs)


@pytest.fixture(autouse=True)
def _clean_patch():
    """No test may leak the patch into another."""
    tc.disable_tar_cache()
    tc.reset_stats()
    yield
    tc.disable_tar_cache()
    tc.reset_stats()


# ---------------------------------------------------------------------------
# The training-loop max_length cap (Task 1)
# ---------------------------------------------------------------------------

def test_training_passage_matches_what_the_dataset_tokenizes(tokenizer):
    """The cap must be measured on TextLabelDataset's string, not on ours.

    Training strings are pre-joined "col: value" and are stripped a second time
    inside the dataset, so they are NOT format_e5_passage(col, value) -- an empty
    value loses its trailing space. Decode what the dataset actually produced and
    require it to equal training_passages().
    """
    raw = ["Product_Description: great sound", "Product_Description: ", "  ", ""]
    ds = TextLabelDataset(raw, np.zeros(len(raw), dtype=int), tokenizer, max_length=64)
    expected = tc.training_passages(raw)
    for i, want in enumerate(expected):
        ids = ds[i]["input_ids"]
        mask = ds[i]["attention_mask"]
        got = tokenizer.decode(ids[mask.bool()], skip_special_tokens=True)
        assert got == want.lower().strip(), f"row {i}: {got!r} != {want!r}"


def test_cap_holds_every_training_passage_with_no_truncation(tokenizer):
    raw = ["review: " + "word " * n for n in (1, 5, 40)]
    longest = tc.longest_training_tokens(raw, tokenizer)
    cap = tc.compute_training_max_length_cap(raw, tokenizer)
    assert cap >= longest
    assert cap % cache_mod.CAP_MULTIPLE == 0
    assert cap - longest < cache_mod.CAP_MULTIPLE
    # No attention mask may run to the edge of the cap: that would mean truncation.
    ds = TextLabelDataset(raw, np.zeros(len(raw), dtype=int), tokenizer, max_length=cap)
    lengths = [int(ds[i]["attention_mask"].sum()) for i in range(len(raw))]
    assert max(lengths) == longest
    assert max(lengths) <= cap


def test_cap_rounding_matches_the_frozen_encode_path(tokenizer):
    """round_to_cap must not drift from cache.compute_max_length_cap.

    Fed a text whose training passage and format_e5_passage form coincide, the
    two must agree exactly -- the formats differ, the arithmetic must not.
    """
    col, val = "review", "great sound quality"
    assert tc.training_passages([f"{col}: {val}"]) == [format_e5_passage(col, val)]
    frozen = cache_mod.compute_max_length_cap([val], col, tokenizer)
    mine = tc.compute_training_max_length_cap([f"{col}: {val}"], tokenizer)
    assert mine == frozen


def test_cap_never_exceeds_the_model_ceiling(tokenizer):
    huge = "review: " + " ".join(["word"] * 4000)
    with pytest.warns(RuntimeWarning, match="above the"):
        cap = tc.compute_training_max_length_cap([huge], tokenizer)
    assert cap == cache_mod.MODEL_MAX_LENGTH


def test_an_explicit_cap_that_would_truncate_is_refused(corpus, tokenizer):
    """A caller-supplied max_length is still checked; silent truncation is the
    one failure this whole module exists to prevent."""
    tc.enable_tar_cache(cache_dir=os.devnull + "_unused", dynamic_max_length=False)
    tc.disable_tar_cache()
    tc.enable_training_max_length_cap()
    with pytest.raises(ValueError, match="would truncate"):
        _finetune(corpus, tokenizer, max_length=8)


def test_dynamic_cap_is_injected_and_recorded(tmp_path, corpus, tokenizer, monkeypatch):
    """With dynamic_max_length on, the cap reaches finetune_e5_with_lora."""
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capturing")

    monkeypatch.setattr(tc, "_original_finetune", spy)
    tc.enable_training_max_length_cap()
    with pytest.raises(RuntimeError, match="stop after capturing"):
        train_texts, train_y, val_texts, val_y = corpus
        kwargs = {k: v for k, v in TRAIN_KWARGS.items() if k != "max_length"}
        e5f.finetune_e5_with_lora(train_texts=train_texts, train_y=train_y,
                                  val_texts=val_texts, val_y=val_y, device=DEVICE,
                                  tokenizer=tokenizer, **kwargs)
    expected = tc.compute_training_max_length_cap(
        list(corpus[0]) + list(corpus[2]), tokenizer)
    assert seen["max_length"] == expected
    assert expected < cache_mod.MODEL_MAX_LENGTH


# ---------------------------------------------------------------------------
# The cache key (Task 2, invalidation)
# ---------------------------------------------------------------------------

def _key(corpus, **overrides):
    train_texts, train_y, val_texts, val_y = corpus
    eff = tc.effective_kwargs(dict(TRAIN_KWARGS, **overrides))
    return tc.compute_key(train_texts, train_y, val_texts, val_y, DEVICE, eff)


def test_effective_kwargs_resolves_upstream_defaults(tokenizer):
    eff = tc.effective_kwargs({"epochs": 1})
    assert eff["epochs"] == 1
    # Read off the upstream signature, so an upstream default change invalidates.
    assert eff["lora_rank"] == 16
    assert eff["seed"] == 42
    assert eff["model_name"] == E5_SMALL_V2
    assert "train_texts" not in eff and "device" not in eff


def test_key_is_content_addressed_not_identity(corpus):
    train_texts, train_y, val_texts, val_y = corpus
    eff = tc.effective_kwargs(dict(TRAIN_KWARGS))
    a = tc.compute_key(train_texts, train_y, val_texts, val_y, DEVICE, eff)
    b = tc.compute_key(list(train_texts), np.array(train_y), list(val_texts),
                       np.array(val_y), torch.device("cpu"), dict(eff))
    assert a == b


@pytest.mark.parametrize("override", [
    {"epochs": 2},
    {"max_length": 64},
    {"lora_rank": 16},
    {"learning_rate": 2e-4},
    {"seed": 43},
])
def test_key_changes_with_every_hyperparameter(corpus, override):
    assert _key(corpus) != _key(corpus, **override)


def test_key_changes_with_training_data(corpus):
    train_texts, train_y, val_texts, val_y = corpus
    eff = tc.effective_kwargs(dict(TRAIN_KWARGS))
    base = tc.compute_key(train_texts, train_y, val_texts, val_y, DEVICE, eff)

    other_texts = list(train_texts)
    other_texts[0] = other_texts[0] + "!"
    assert tc.compute_key(other_texts, train_y, val_texts, val_y, DEVICE, eff) != base

    other_y = np.array(train_y)
    other_y[0] = (other_y[0] + 1) % 3
    assert tc.compute_key(train_texts, other_y, val_texts, val_y, DEVICE, eff) != base

    # A dropped row must not collide with the full set (length is hashed).
    assert tc.compute_key(train_texts[:-1], train_y[:-1], val_texts, val_y,
                          DEVICE, eff) != base

    # The val split is a determinant too: it drives early stopping.
    assert tc.compute_key(train_texts, train_y, val_texts[:-1], val_y[:-1],
                          DEVICE, eff) != base


def test_key_ignores_ordering_of_kwargs_only(corpus):
    eff1 = tc.effective_kwargs({"epochs": 1, "lora_rank": 8})
    eff2 = tc.effective_kwargs({"lora_rank": 8, "epochs": 1})
    train_texts, train_y, val_texts, val_y = corpus
    assert (tc.compute_key(train_texts, train_y, val_texts, val_y, DEVICE, eff1)
            == tc.compute_key(train_texts, train_y, val_texts, val_y, DEVICE, eff2))


def test_disable_restores_the_original_function(tmp_path):
    assert not tc.is_enabled()
    tc.enable_tar_cache(str(tmp_path / "tar"))
    assert tc.is_enabled()
    tc.disable_tar_cache()
    assert not tc.is_enabled()
    assert e5f.finetune_e5_with_lora is tc._original_finetune


# ---------------------------------------------------------------------------
# The cache itself (Task 2, correctness). Each fine-tune is ~25 s on CPU.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_twice(corpus, tokenizer):
    """Two independent fine-tunings of identical inputs, cache OFF."""
    tc.disable_tar_cache()
    a = _finetune(corpus, tokenizer)[0]
    b = _finetune(corpus, tokenizer)[0]
    return tc.state_dict_sha256(a), tc.state_dict_sha256(b)


def test_finetuning_is_deterministic(trained_twice):
    """The premise of the whole cache: reuse == recompute.

    If this ever fails, the cache is not merely an optimization -- it changes
    which encoder a run gets -- and the sharing test below becomes meaningless.
    """
    a, b = trained_twice
    assert a == b


def test_cache_hit_is_bit_exact(tmp_path, corpus, tokenizer):
    cache_dir = str(tmp_path / "tar")
    tc.enable_tar_cache(cache_dir)
    miss_model, _ = _finetune(corpus, tokenizer)
    assert tc.cache_stats() == {"hits": 0, "misses": 1, "corrupt": 0}
    hit_model, hit_tok = _finetune(corpus, tokenizer)
    assert tc.cache_stats() == {"hits": 1, "misses": 1, "corrupt": 0}

    # Whole-model equality, not just equality of the outputs we happened to probe.
    assert tc.state_dict_sha256(miss_model) == tc.state_dict_sha256(hit_model)
    assert hit_tok is tokenizer

    probe = ["great sound quality", "battery died fast"]
    kw = dict(col_name="review", tokenizer=tokenizer, device=DEVICE)
    a = e5f.encode_texts_with_e5(texts=probe, model=miss_model, **kw)
    b = e5f.encode_texts_with_e5(texts=probe, model=hit_model, **kw)
    assert np.array_equal(a, b)


def test_changed_hyperparameters_miss_the_cache(tmp_path, corpus, tokenizer):
    """Invalidation: a different epoch count must retrain, not serve the old run."""
    tc.enable_tar_cache(str(tmp_path / "tar"))
    _finetune(corpus, tokenizer, epochs=1)
    assert tc.cache_stats()["misses"] == 1
    _finetune(corpus, tokenizer, epochs=2)
    assert tc.cache_stats() == {"hits": 0, "misses": 2, "corrupt": 0}
    # ... and the 1-epoch entry is still there and still served.
    _finetune(corpus, tokenizer, epochs=1)
    assert tc.cache_stats() == {"hits": 1, "misses": 2, "corrupt": 0}


def test_a_tampered_entry_is_rejected_rather_than_served(tmp_path, corpus, tokenizer):
    """The checksum is the guarantee; prove it actually rejects."""
    cache_dir = tmp_path / "tar"
    tc.enable_tar_cache(str(cache_dir))
    _finetune(corpus, tokenizer)
    entries = [p for p in cache_dir.iterdir() if p.is_dir()]
    assert len(entries) == 1
    meta_path = entries[0] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["state_dict_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="failed its checksum"):
        assert tc._rebuild(str(entries[0]), DEVICE, E5_SMALL_V2) is None


def test_cached_adapter_is_small(tmp_path, corpus, tokenizer):
    """Persisting the adapter, not the backbone: ~1 MB, not ~130 MB per entry."""
    cache_dir = tmp_path / "tar"
    tc.enable_tar_cache(str(cache_dir))
    _finetune(corpus, tokenizer)
    total = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
    assert total < 20e6, f"{total/1e6:.1f} MB per entry is too large to keep 10 of"


# ---------------------------------------------------------------------------
# The grouping claim, observed rather than arranged (slow: fits real learners)
# ---------------------------------------------------------------------------

@slow
def test_encoder_sharing_groups_are_real():
    """With the cache OFF, do same-group learners really tune the same encoder?

    LightGBM and CatBoost both set USE_VAL_SPLIT=True, so fit() hands
    fit_preprocessor the same 90% slice; TabPFNv2 sets it False and passes the
    full train set. If the first pair did NOT agree, the cache would be handing
    models an encoder they would never have trained, and every Delta_Awareness
    number would be quietly wrong.
    """
    from multabench.baselines.catboost import CatBoost
    from multabench.baselines.lgbm import LightGBM
    from multabench.baselines.tabpfnv2 import TabPFNv2
    from curation_lab.runner.paper import load_paper_dataset

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ds = load_paper_dataset("MUL_TEXT_PRODUCT_SENTIMENT", "all")
    x, y = ds.x.iloc[:300].copy(), ds.y.iloc[:300].copy()
    probe = ["great sound quality", "battery died fast", "average tablet"]

    def tuned(cls):
        model = cls(problem_type=ds.task_type, device=DEVICE, dataset=ds.dataset_id,
                    tune_e5=True, e5_train_kwargs=dict(TRAIN_KWARGS, batch_size=64))
        with contextlib.redirect_stdout(buf):
            model.fit(x.copy(), y.copy())
        col = sorted(model.text_transformers)[0]
        enc = model.text_transformers[col]
        return e5f.encode_texts_with_e5(texts=probe, col_name=col, model=enc.model,
                                        tokenizer=enc.tokenizer, device=DEVICE)

    light, cat, pfn = tuned(LightGBM), tuned(CatBoost), tuned(TabPFNv2)
    assert np.allclose(light, cat, atol=1e-5), (
        "LightGBM and CatBoost tuned DIFFERENT encoders -- the sharing assumption "
        "is false and the cache must not be used.")
    assert not np.allclose(light, pfn, atol=1e-5), (
        "TabPFNv2 (no val split) matched the val-split group; the two groups are "
        "not distinct, so the key is not capturing x_train.")
