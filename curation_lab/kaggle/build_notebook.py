"""Generate the Kaggle GPU notebook that runs the TAR (`ft`) condition.

Why a generator and not a checked-in .ipynb: the notebook is iterated on many
times against real Kaggle runs, and hand-editing JSON with embedded source
strings is where mistakes hide. Cells are plain Python here; the .ipynb is a
build artifact.

The notebook does only the TAR half. Everything else in the curation criterion
(no_text / text_only / all) is frozen-encoder work that already runs on CPU --
GPU time is scarce and is spent on the one condition that needs it.

Usage:
    python -m curation_lab.kaggle.build_notebook
    python -m curation_lab.kaggle.build_notebook --out kaggle_uploads/tar-gpu
"""
from __future__ import annotations

import argparse
import json
import os

KERNEL_ID = "noabamberger/multabench-tar-gpu"
CODE_DATASET = "noabamberger/multabench-code"
# The candidate must be pre-attached: an API-launched kernel is non-interactive
# and Kaggle rejects kagglehub.dataset_download() for a not-yet-attached dataset.
# Which candidate is a build-time choice (--dataset-ref / --dataset-name); it has
# to appear in BOTH the notebook parameters and kernel-metadata dataset_sources,
# which is exactly the pair that is easy to desynchronise by hand.
CANDIDATE_DATASET = "mariahalshiekh/udemy-course-academy-teaching"
CANDIDATE_NAME = "REG_TEXT_EDU_UDEMY_ACADEMY"
NOTEBOOK_FILE = "tar_gpu.ipynb"

# --------------------------------------------------------------------------
# Cells. Each entry is ("markdown"|"code", source).
# --------------------------------------------------------------------------

MD_HEADER = """\
# MulTaBench — TAR (`ft`) on GPU

Runs **only** the Target-Aware condition: LoRA-fine-tune `intfloat/e5-small-v2`
on the target, re-embed the text columns, then fit the downstream learner.

The other three conditions (`no_text`, `text_only`, `all`) use a frozen encoder
and are cheap on CPU, so they are deliberately not run here.

**Requires:** GPU accelerator ON, Internet ON, and the `multabench-code`
dataset attached.
"""

PARAMS = '''\
# ---------------------------------------------------------------- parameters
SMOKE          = __SMOKE__    # True: one training epoch, just prove the path runs
REQUIRE_GPU    = __REQUIRE_GPU__    # False = CPU validation run (no GPU quota spent)
CANDIDATES     = __CANDIDATES__   # (kaggle ref, MulTaBench name); the {BIN|MUL|REG}_TEXT_
                                  # prefix is load-bearing -- evaluate.py slices name[:3]
FOLDS          = __FOLDS__
SMOKE_EPOCHS   = 1
FULL_EPOCHS    = __FULL_EPOCHS__   # PHASE2_RESULTS.md: epochs=2 under-trains the adapter
                         # and collapses Delta_Awareness to noise; 10 is the working value.
MODELS         = __MODELS__   # SHORT_NAMEs of the curation committee members to run
STATES         = __STATES__   # curation conditions to measure this run
CODE_DIR       = "/kaggle/input/multabench-code"
OUT_CSV        = "/kaggle/working/tar_results.csv"
'''

ENV = '''\
# ------------------------------------------------- environment (before torch)
# multabench/e5/e5_finetune.py asserts CUDA_VISIBLE_DEVICES is *present* and the
# finetune path is documented "single GPU only" -- Kaggle hands out 2x T4, so pin
# to one. This must happen before torch initialises CUDA.
import os, subprocess, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TabPFN 2.5/2.6 weights are gated behind a Prior Labs licence, checked by
# tabpfn/model_loading.py -> browser_auth.ensure_license_accepted(). Without a token
# that raises TabPFNLicenseError, which is why TabPFN-2.5 had no cells in the earlier
# grids. browser_auth.get_cached_token() reads TABPFN_TOKEN first. TabPFN v2 is NOT
# gated -- it is absent from _HF_REPOS -- so it never needed this.
# Injected at build time from .env; the generator in git carries only the placeholder.
if "__TABPFN_TOKEN__":
    os.environ["TABPFN_TOKEN"] = "__TABPFN_TOKEN__"

try:
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        capture_output=True, text=True).stdout or "!! nvidia-smi produced no output")
except FileNotFoundError:
    print("no nvidia-smi -- this session has no accelerator attached")
print("python:", sys.version.split()[0])
'''

DEPS = '''\
# ------------------------------------------------------------------ packages
import importlib, subprocess, sys

def version_of(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "(no __version__)")
    except Exception as e:
        return f"MISSING ({type(e).__name__})"

NEEDED = ["torch", "transformers", "peft", "pandas", "numpy", "sklearn",
          "lightgbm", "skrub", "openml", "kagglehub", "wandb", "dotenv", "tabstar"]
print("--- before install ---")
for m in NEEDED:
    print(f"  {m:14s} {version_of(m)}")

# Pin torch so pip cannot swap the CUDA build out from under us while resolving
# tabstar's dependency tree. transformers must be new enough for DINOv3 symbols,
# which multabench/baselines/preprocessing/image_embeddings.py imports eagerly.
import torch
with open("/kaggle/working/constraints.txt", "w") as fh:
    fh.write(f"torch=={torch.__version__.split('+')[0]}\\n")

EXTRAS = {"tabm": ["pytabkit"], "tabpfnv2": ["tabpfn"], "tabpfnv2p5": ["tabpfn"]}
wanted = sorted({pkg for m in MODELS for pkg in EXTRAS.get(m, [])})
if wanted:
    print("extra learner packages for", MODELS, "->", wanted)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts",
    "-c", "/kaggle/working/constraints.txt",
    "tabstar", "skrub", "openml", "peft", "python-dotenv", "transformers>=4.56",
    *wanted,
])

# peft's LoRA dispatch calls is_torchao_available(), which *raises* rather than
# returning False when torchao is present but older than its minimum. The Kaggle
# image ships torchao 0.10.0 against peft 0.19.1 (minimum 0.16.0), so LoRA
# injection dies before training starts. Nothing in the TAR path uses torchao
# quantisation, and the same helper returns False cleanly when the module is
# absent -- so removing it is the least invasive fix.
try:
    from importlib.metadata import version as _pkg_version
    _tao = _pkg_version("torchao")
    if tuple(int(p) for p in _tao.split(".")[:2]) < (0, 16):
        print(f"removing torchao {_tao} (incompatible with peft {version_of('peft')})")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
except Exception as e:
    print("torchao check skipped:", e)

print("--- after install ---")
for m in NEEDED:
    print(f"  {m:14s} {version_of(m)}")
'''

LOCATE = '''\
# ------------------------------------------------------- locate the code tree
# Kaggle's mount layout is not stable across API versions: it has used both
# /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug>, and it decides
# for itself whether to extract an uploaded .zip. So walk and look for the tree.
import glob, os, zipfile

def _looks_like_repo(d):
    return os.path.isdir(os.path.join(d, "multabench")) and os.path.isdir(os.path.join(d, "curation_lab"))

print("--- /kaggle/input ---")
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.rstrip("/").count("/") - 2
    if depth > 3:
        dirs[:] = []
        continue
    print("  " * depth, root, "|", sorted(dirs)[:6], sorted(files)[:4])

resolved = None
for root, dirs, _ in os.walk("/kaggle/input"):
    if _looks_like_repo(root):
        resolved = root
        break

if resolved is None:   # dataset arrived as an un-extracted archive
    for zp in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        dest = "/kaggle/working/code"
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
        if _looks_like_repo(dest):
            resolved = dest
            print("extracted", zp, "->", dest)
            break

assert resolved, ("Could not find the code tree. Is the `multabench-code` dataset "
                  "attached to this notebook?")
CODE_DIR = resolved
print("CODE_DIR =", CODE_DIR)
'''

IMPORTS = '''\
# -------------------------------------------------------------- import chain
import sys
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import multabench.constants as _c   # calls load_dotenv(); sets HF_TOKEN="" when absent
if not os.environ.get("HF_TOKEN"):
    os.environ.pop("HF_TOKEN", None)   # empty string confuses huggingface_hub auth

import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
if not REQUIRE_GPU:
    # CPU validation mode: proves the TAR code path end to end without spending
    # GPU quota. t3_tar._cpu_device() supplies the CUDA_VISIBLE_DEVICES name that
    # multabench/e5/e5_finetune.py asserts on, and returns torch.device("cpu").
    print("REQUIRE_GPU=False -- running on CPU, numbers are for plumbing only.")
else:
    assert torch.cuda.is_available(), "No GPU visible -- turn the accelerator on."
    print("device:", torch.cuda.get_device_name(0))

# is_available() is True even when the card's compute capability is not in the
# installed torch build (Kaggle's P100 is sm_60; torch 2.10+cu128 ships sm_70+).
# In that state every kernel launch fails, but only once training starts -- so
# force the failure here, cheaply, with a real launch.
if REQUIRE_GPU:
  print("compute capability:", torch.cuda.get_device_capability(0))
  print("torch arch list   :", torch.cuda.get_arch_list())
  try:
    _probe = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    print(f"CUDA kernel launch OK (probe={_probe:.3f})")
  except Exception as e:
    raise SystemExit(
        f"GPU present but UNUSABLE by this torch build: {type(e).__name__}: {e} "
        f"-- Kaggle's P100 is sm_60 and torch 2.10+cu128 ships sm_70+. "
        f"Set the accelerator to 'GPU T4 x2' (sm_75) in the notebook settings."
    )

# Importing curation_mapping executes every module in multabench/datasets/annotated/,
# so an ImportError anywhere in that package surfaces here rather than mid-run.
from curation_lab.screen.auto_spec import build_spec
from curation_lab.screen.batch_profile import _read_any_csv
from curation_lab.screen.t3_tar import tar_probe
print("import chain OK")
'''

SPEC = '''\
# ------------------------------------------------- candidate CSV -> spec
# NOT kagglehub.dataset_download(): a kernel started through the API runs
# non-interactively, and Kaggle refuses to attach a new datasource from there
# ("New Datasets cannot be attached in non-interactive sessions"). The candidate
# is declared in kernel-metadata.json dataset_sources instead and is already
# mounted, so we just locate its CSV.
import glob

all_csvs = [p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
            if not p.startswith(CODE_DIR)]

def spec_for(ref, name):
    """Locate one candidate's CSV among the mounted datasources and auto-spec it.

    Matching is by slug rather than by taking the largest CSV overall, because a
    multi-candidate run mounts several datasets at once and "largest" would hand
    every candidate the same file.
    """
    slug = ref.split("/")[-1]
    matching = [p for p in all_csvs if slug in p]
    if not matching:
        return None, None, f"no CSV mounted for {ref}; is it in dataset_sources?"
    csv = max(matching, key=os.path.getsize)
    df, enc, err = _read_any_csv(csv)
    if df is None:
        return None, None, f"could not read {csv}: {err}"
    read_kwargs = {} if enc == "utf-8" else {"encoding": enc.split("/")[0]}
    spec, why = build_spec(df, name=name, csv_path=csv, read_kwargs=read_kwargs)
    if spec is None:
        return None, df, f"spec rejected: {why}"
    return spec, df, why

# Build every spec up front, so a bad candidate fails here rather than after the
# first one has already spent an hour of GPU.
SPECS = {}
for _ref, _name in CANDIDATES:
    _spec, _df, _why = spec_for(_ref, _name)
    print(f"\\n=== {_name}  ({_ref}) ===")
    if _spec is None:
        print("  SKIPPED:", _why)
        continue
    SPECS[_name] = _spec
    print(f"  rows={len(_df)} cols={_df.shape[1]}  csv={os.path.basename(_spec.csv_path)}")
    print(f"  {_why}")
    print(f"  target={_spec.target!r}  text={_spec.text_cols}")
    print(f"  numeric={_spec.numeric_cols}")
    print(f"  categorical={_spec.categorical_cols}")

assert SPECS, "no candidate produced a usable spec"
print(f"\\n{len(SPECS)} of {len(CANDIDATES)} candidates specced")
'''

SMOKE = '''\
# ------------------------------------------------------------- TAR smoke test
# One epoch. all_score=0.0 is a placeholder: this cell proves the fine-tune path
# executes on GPU, it does NOT produce a usable Delta_Awareness.
import time

if SMOKE:
    t0 = time.time()
    result = tar_probe(next(iter(SPECS.values())), all_score=0.0,
                       fold=FOLDS[0], epochs=SMOKE_EPOCHS)
    print("\\n=== SMOKE OK ===")
    print(f"  ft score      : {result['ft']:.4f}")
    print(f"  max_length cap: {result['cap']} (upstream default 512)")
    print(f"  wall clock    : {result['secs']}s for {result['epochs']} epoch(s)")
    print(f"  total elapsed : {time.time() - t0:.0f}s")
else:
    print("SMOKE=False -- skipping.")
'''

FULL = '''\
# ------------------------------------------------------- full TAR measurement
# Runs `all` (frozen E5) and `ft` (LoRA-tuned E5) under identical splits, so the
# difference is Delta_Awareness for each (model, fold).
#
# Both states are measured HERE rather than reusing the CPU `all` numbers from the
# frozen Delta_Joint grid: Delta_Awareness is a difference of two means, so the two
# halves must come from the same machine, the same torch build and the same library
# versions, or float drift between them lands directly in the delta.
import time
import traceback

import importlib

import pandas as pd
from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
from multabench.finetune.train_args import E5TrainArgs
from curation_lab.ingest.candidate import STATE_BY_FLAG, load_candidate
from curation_lab.runner.cache import disable_cache, enable_dynamic_max_length
from curation_lab.runner.tar_cache import (cache_stats, disable_tar_cache,
                                           enable_tar_cache, reset_stats)

# Fine-tuning happens only in the embedding step, never end to end, so the tuned
# encoder is a function of (train texts, train labels, device, hyperparameters) and
# NOT of the tabular learner. Per fold the five committee models differ only in
# USE_VAL_SPLIT -- True for LightGBM/CatBoost/TabM, False for both TabPFNs -- so a
# fold contains 2 distinct fine-tunings, not 5, and a 5x5 sweep contains 10, not 25.
#
# The cache still keys on a hash of the actual arguments rather than on that
# grouping, so if the grouping is ever wrong (or upstream threads a real fold into
# split_to_val) the effect is a wasted miss, never a model served an encoder it
# would not have trained. cache_stats() is printed at the end so the hits are
# evidence rather than an assumption.
TAR_CACHE_DIR = "/kaggle/working/.tar_cache"

# Imported lazily and one at a time. Each baseline module pulls its own learner at
# import (tabm -> pytabkit, tabpfnv2 -> tabpfn), and those are NOT all in the Kaggle
# image -- so an eager import of the whole committee makes a LightGBM-only run fail
# on a dependency it never uses.
_MODEL_PATHS = {"light": ("multabench.baselines.lgbm", "LightGBM"),
                "cat": ("multabench.baselines.catboost", "CatBoost"),
                "tabm": ("multabench.baselines.tabm", "TabM"),
                "tabpfnv2": ("multabench.baselines.tabpfnv2", "TabPFNv2"),
                "tabpfnv2p5": ("multabench.baselines.tabpfnv2", "TabPFNv2p5")}

def model_cls_for(name):
    module, attr = _MODEL_PATHS[name]
    cls = getattr(importlib.import_module(module), attr)
    if name.startswith("tabpfn"):
        cls = _without_hf_token_assignment(cls)
    return cls

def _without_hf_token_assignment(cls):
    """multabench/baselines/tabpfnv2.py:28 does `os.environ["HF_TOKEN"] = HF_TOKEN`.

    HF_TOKEN comes from multabench.constants, which reads it out of a .env that
    does not exist on Kaggle -- so it is None and the assignment raises
    `TypeError: str expected, not NoneType` in initialize_model(), before a single
    weight is fetched. Skip just that assignment when there is no token to set;
    TabPFN v2's weights are public, so the download proceeds anonymously. (TabPFN
    2.5's are gated and will still fail at fetch time -- that is the known licence
    blocker in RESUME.md, not anything about this dataset.)
    """
    class _Patched(cls):
        def initialize_model(self):
            if os.environ.get("HF_TOKEN"):
                return super().initialize_model()
            return None
    _Patched.__name__ = cls.__name__
    _Patched.MODEL_NAME = cls.MODEL_NAME
    _Patched.SHORT_NAME = cls.SHORT_NAME
    return _Patched

# e5_finetune asserts CUDA_VISIBLE_DEVICES exists regardless of the device it is
# handed; the environment cell sets it, so CPU validation satisfies the guard too.
DEVICE = torch.device("cuda:0") if REQUIRE_GPU else torch.device("cpu")

def run_state(spec, model_cls, state_flag, fold, epochs=None):
    enable_dynamic_max_length()
    if state_flag == "ft":
        # dynamic_max_length here caps the TRAINING loop's padding (512 -> the longest
        # passage actually present). It participates in the cache key, so weights
        # trained under different caps can never be served for one another.
        enable_tar_cache(TAR_CACHE_DIR, dynamic_max_length=True)
    try:
        loaded = load_candidate(spec, state_flag)
        kwargs = None
        if state_flag == "ft":
            kwargs = E5TrainArgs().to_dict()
            kwargs["epochs"] = epochs
            print(f"[{state_flag}] epochs={epochs}, max_length capped by tar_cache", flush=True)
        t0 = time.time()
        summary = evaluate_on_loaded_dataset(
            model_cls=model_cls, dataset=loaded, fold=fold,
            device=DEVICE, train_examples=DOWNSTREAM_EXAMPLES,
            multimodal_state=STATE_BY_FLAG[state_flag],
            tune_e5=(state_flag == "ft"), e5_train_kwargs=kwargs,
        )
    finally:
        disable_cache()
        disable_tar_cache()
    return {"state": state_flag, "score": float(summary["test_score"]),
            "secs": round(time.time() - t0, 1), "epochs": epochs}

def flush(rows):
    """Write after every cell. A Kaggle session that hits its wall clock still
    leaves every completed (model, fold) behind in the output."""
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

def summarise(df_r):
    """Per-model deltas for one dataset. Means are rounded to 3 decimals BEFORE
    differencing -- the paper's rule, per pass_matrix.passes()."""
    piv = df_r.pivot_table(index="model", columns="state", values="score",
                           aggfunc="mean").round(3)
    unimodal = [s for s in ("no_text", "text_only") if s in piv.columns]
    if "all" in piv.columns and unimodal:
        piv["D_Joint"] = (piv["all"] - piv[unimodal].max(axis=1)).round(4)
    if {"all", "ft"}.issubset(piv.columns):
        piv["D_Awareness"] = (piv["ft"] - piv["all"]).round(4)
    return piv

if not SMOKE:
    rows = []
    reset_stats()
    for dname, spec in SPECS.items():
        print(f"\\n########## {dname} ##########", flush=True)
        for fold in FOLDS:
            for name in MODELS:
                try:
                    cls = model_cls_for(name)
                except Exception as e:
                    print(f"!! {name}: cannot import ({type(e).__name__}: {e})", flush=True)
                    continue
                for state in STATES:
                    ep = FULL_EPOCHS if state == "ft" else None
                    tag = f"{dname[:22]}/{name}/{state}/f{fold}"
                    try:
                        r = run_state(spec, cls, state, fold, epochs=ep)
                    except SystemExit:
                        # is_invalid_model_dataset_pair() calls exit() for a legitimately
                        # skipped (model, dataset) cell; that is data, not a failure.
                        print(f"!! {tag}: model skipped this dataset", flush=True)
                        continue
                    except Exception as e:
                        # MultimodalError is a soft skip meaning "this state does not
                        # apply here", not a crash -- it is reported the same way and
                        # simply leaves the cell absent.
                        print(f"!! {tag} FAILED: {type(e).__name__}: {e}", flush=True)
                        traceback.print_exc()
                        continue
                    r.update(dataset=dname, model=name, fold=fold)
                    rows.append(r)
                    flush(rows)
                    print(f"  {tag:52s} = {r['score']:.4f}  ({r['secs']}s)", flush=True)

    st = cache_stats()
    print(f"\\n[tar_cache] {st} -- {st['hits']} of {st['hits'] + st['misses']} ft runs "
          f"reused an encoder instead of fine-tuning it again")

    if rows:
        df_all = pd.DataFrame(rows)
        for dname, sub in df_all.groupby("dataset"):
            print(f"\\n================ {dname} ================")
            print(summarise(sub).to_string())
            print(f"folds: {sorted(sub.fold.unique())}")
    print("\\nwrote", OUT_CSV)
else:
    print("SMOKE=True -- skipping the full run.")
'''

CELLS: list[tuple[str, str]] = [
    ("markdown", MD_HEADER),
    ("code", PARAMS),
    ("code", ENV),
    ("code", DEPS),
    ("code", LOCATE),
    ("code", IMPORTS),
    ("code", SPEC),
    ("code", SMOKE),
    ("code", FULL),
]


def _tabpfn_token() -> str:
    """Prior Labs API key from the environment or .env, or "" if absent.

    Read at BUILD time and baked into the generated notebook, which lives under the
    gitignored kaggle_uploads/. The key therefore never enters git -- but it does
    travel to Kaggle inside the notebook source, so the kernel must stay private.
    """
    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if token:
        return token
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.isfile(env_path):
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("TABPFN_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def _cells(require_gpu: bool, smoke: bool = True, full_epochs: int = 10,
           candidates: tuple[tuple[str, str], ...] = ((CANDIDATE_DATASET, CANDIDATE_NAME),),
           folds: tuple[int, ...] = (0,),
           models: tuple[str, ...] = ("light",),
           states: tuple[str, ...] = ("all", "ft")) -> list[tuple[str, str]]:
    subs = {"__REQUIRE_GPU__": str(bool(require_gpu)),
            "__SMOKE__": str(bool(smoke)),
            "__FULL_EPOCHS__": str(int(full_epochs)),
            "__CANDIDATES__": repr([list(c) for c in candidates]),
            "__FOLDS__": repr(list(folds)),
            "__MODELS__": repr(list(models)),
            "__STATES__": repr(list(states)),
            "__TABPFN_TOKEN__": _tabpfn_token()}
    out = []
    for kind, src in CELLS:
        for k, v in subs.items():
            src = src.replace(k, v)
        out.append((kind, src))
    return out


def _check_cells_compile() -> None:
    """Compile every code cell locally before it can cost a Kaggle run.

    The cell bodies are ordinary (non-raw) triple-quoted strings, so a `\\n`
    meant for the *generated* code has to be written `\\\\n` -- getting that
    wrong yields an unterminated f-string that only surfaces on the server,
    several minutes and one GPU-quota slot later.
    """
    for require_gpu in (True, False):
        for smoke in (True, False):
            for idx, (kind, src) in enumerate(_cells(require_gpu, smoke)):
                if kind != "code":
                    continue
                try:
                    compile(src, f"<cell {idx}>", "exec")
                except SyntaxError as e:
                    raise SystemExit(f"cell {idx} (REQUIRE_GPU={require_gpu}, SMOKE={smoke}) "
                                     f"does not compile: {e}\n---\n{src}\n---")


def build_notebook(require_gpu: bool = True, smoke: bool = True, full_epochs: int = 10,
                   **candidate) -> dict:
    _check_cells_compile()
    return {
        "cells": [_cell(kind, src)
                  for kind, src in _cells(require_gpu, smoke, full_epochs, **candidate)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata(enable_gpu: bool = True, machine_shape: str | None = None,
                   dataset_refs: tuple[str, ...] = (CANDIDATE_DATASET,),
                   kernel_id: str | None = None) -> dict:
    # The CPU validation variant lives in its own kernel so that pushing it does
    # not overwrite the GPU kernel's accelerator setting. A caller can also name a
    # separate kernel to run a second sweep concurrently: one kernel only ever runs
    # one version at a time, so pushing a screen to the kernel already running a
    # full grid would displace it.
    kid = kernel_id or (KERNEL_ID if enable_gpu else KERNEL_ID.replace("-tar-gpu", "-tar-cpu"))
    meta = {
        "id": kid,
        "title": kid.split("/")[-1],
        "code_file": NOTEBOOK_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [CODE_DATASET, *dataset_refs],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    # Kaggle hands out a P100 by default, whose sm_60 the image's torch no longer
    # supports. The accepted names are an undocumented server-side enum (the CLI
    # forwards the string unvalidated), so this stays a plain override.
    if machine_shape:
        meta["machine_shape"] = machine_shape
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="kaggle_uploads/tar-gpu")
    p.add_argument("--machine-shape", default=None,
                   help="Accelerator override written into kernel-metadata.json.")
    p.add_argument("--cpu", action="store_true",
                   help="Build the CPU validation variant (no accelerator, no GPU quota).")
    p.add_argument("--full", action="store_true",
                   help="SMOKE=False: run the all vs ft measurement instead of the smoke test.")
    p.add_argument("--full-epochs", type=int, default=10,
                   help="E5 fine-tuning epochs for the full run.")
    p.add_argument("--candidate", action="append", default=[], metavar="REF=NAME",
                   help="Repeatable. Kaggle owner/slug=MulTaBench name. Every ref is also "
                        "written into dataset_sources, so one kernel run can screen several "
                        "candidates and pay the container/pip setup cost once.")
    p.add_argument("--folds", default="0", help="Comma-separated folds, e.g. 0,1,2,3,4.")
    p.add_argument("--models", default="light",
                   help="Comma-separated SHORT_NAMEs from the curation committee: "
                        "light,cat,tabm,tabpfnv2,tabpfnv2p5.")
    p.add_argument("--kernel-id", default=None,
                   help="owner/slug of the kernel to write. Use a second kernel to run "
                        "a sweep concurrently with one already running.")
    p.add_argument("--states", default="all,ft",
                   help="Curation conditions to measure: no_text,text_only,all,ft. "
                        "Run all four on one machine when the output feeds passes().")
    args = p.parse_args()

    pairs = args.candidate or [f"{CANDIDATE_DATASET}={CANDIDATE_NAME}"]
    candidates = []
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--candidate must be REF=NAME, got {item!r}")
        ref, name = item.split("=", 1)
        ref, name = ref.strip(), name.strip()
        if name[:3] not in ("BIN", "MUL", "REG"):
            raise SystemExit(f"candidate name must start with BIN_/MUL_/REG_, got {name!r}")
        if ref.count("/") != 1:
            raise SystemExit(f"candidate ref must be owner/slug, got {ref!r}")
        candidates.append((ref, name))
    dup = {n for _, n in candidates if [x for _, x in candidates].count(n) > 1}
    if dup:
        raise SystemExit(f"duplicate candidate name(s) {sorted(dup)}; results key on name")
    folds = tuple(int(f) for f in args.folds.split(",") if f.strip() != "")
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    known = {"light", "cat", "tabm", "tabpfnv2", "tabpfnv2p5"}
    unknown = sorted(set(models) - known)
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; expected from {sorted(known)}")
    states = tuple(s.strip() for s in args.states.split(",") if s.strip())
    known_states = {"no_text", "text_only", "all", "ft"}
    bad_states = sorted(set(states) - known_states)
    if bad_states:
        raise SystemExit(f"unknown state(s) {bad_states}; expected from {sorted(known_states)}")

    require_gpu = not args.cpu
    os.makedirs(args.out, exist_ok=True)
    nb_path = os.path.join(args.out, NOTEBOOK_FILE)
    meta_path = os.path.join(args.out, "kernel-metadata.json")
    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(build_notebook(require_gpu=require_gpu, smoke=not args.full,
                                 full_epochs=args.full_epochs,
                                 candidates=tuple(candidates),
                                 folds=folds, models=models, states=states), fh, indent=1)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(build_metadata(enable_gpu=require_gpu,
                                 machine_shape=args.machine_shape,
                                 dataset_refs=tuple(r for r, _ in candidates),
                                 kernel_id=args.kernel_id), fh, indent=2)
    print(f"wrote {nb_path}\nwrote {meta_path}")


if __name__ == "__main__":
    main()
