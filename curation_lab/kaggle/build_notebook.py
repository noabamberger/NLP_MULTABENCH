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

KERNEL_ID = "talkraicer/multabench-tar-gpu"
CODE_DATASET = "talkraicer/multabench-code"
# The candidate must be pre-attached: an API-launched kernel is non-interactive
# and Kaggle rejects kagglehub.dataset_download() for a not-yet-attached dataset.
CANDIDATE_DATASET = "mariahalshiekh/udemy-course-academy-teaching"
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
DATASET_REF    = "mariahalshiekh/udemy-course-academy-teaching"
DATASET_NAME   = "REG_TEXT_EDU_UDEMY_ACADEMY"   # {BIN|MUL|REG}_TEXT_* prefix is load-bearing
FOLD           = 0
SMOKE_EPOCHS   = 1
FULL_EPOCHS    = __FULL_EPOCHS__   # PHASE2_RESULTS.md: epochs=2 under-trains the adapter
                         # and collapses Delta_Awareness to noise; 10 is the working value.
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

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts",
    "-c", "/kaggle/working/constraints.txt",
    "tabstar", "skrub", "openml", "peft", "python-dotenv", "transformers>=4.56",
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

slug = DATASET_REF.split("/")[-1]
all_csvs = [p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
            if not p.startswith(CODE_DIR)]
matching = [p for p in all_csvs if slug in p] or all_csvs
assert matching, f"no candidate CSV mounted; is {DATASET_REF} in dataset_sources?"
csv = max(matching, key=os.path.getsize)
print("csv:", csv)

df, enc, err = _read_any_csv(csv)
assert df is not None, f"could not read {csv}: {err}"

read_kwargs = {} if enc == "utf-8" else {"encoding": enc.split("/")[0]}
spec, why = build_spec(df, name=DATASET_NAME, csv_path=csv, read_kwargs=read_kwargs)
assert spec is not None, f"spec rejected: {why}"

print(f"rows={len(df)}  cols={df.shape[1]}")
print(f"reason: {why}")
print(f"target={spec.target!r}  task={spec.task}")
print(f"text={spec.text_cols}")
print(f"numeric={spec.numeric_cols}")
print(f"categorical={spec.categorical_cols}")
'''

SMOKE = '''\
# ------------------------------------------------------------- TAR smoke test
# One epoch. all_score=0.0 is a placeholder: this cell proves the fine-tune path
# executes on GPU, it does NOT produce a usable Delta_Awareness.
import time

if SMOKE:
    t0 = time.time()
    result = tar_probe(spec, all_score=0.0, fold=FOLD, epochs=SMOKE_EPOCHS)
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
# difference is Delta_Awareness for this (dataset, model, fold).
import time

import pandas as pd
from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
from multabench.baselines.lgbm import LightGBM
from multabench.finetune.train_args import E5TrainArgs
from curation_lab.ingest.candidate import STATE_BY_FLAG, load_candidate
from curation_lab.runner.cache import disable_cache, enable_dynamic_max_length
from curation_lab.screen.t3_tar import _training_max_length

# e5_finetune asserts CUDA_VISIBLE_DEVICES exists regardless of the device it is
# handed; the environment cell sets it, so CPU validation satisfies the guard too.
DEVICE = torch.device("cuda:0") if REQUIRE_GPU else torch.device("cpu")

def run_state(state_flag, epochs=None):
    enable_dynamic_max_length()
    try:
        loaded = load_candidate(spec, state_flag)
        kwargs = None
        if state_flag == "ft":
            kwargs = E5TrainArgs().to_dict()
            kwargs["epochs"] = epochs
            kwargs["max_length"] = _training_max_length(loaded, spec.text_cols)
            print(f"[{state_flag}] max_length={kwargs['max_length']}, epochs={epochs}", flush=True)
        t0 = time.time()
        summary = evaluate_on_loaded_dataset(
            model_cls=LightGBM, dataset=loaded, fold=FOLD,
            device=DEVICE, train_examples=DOWNSTREAM_EXAMPLES,
            multimodal_state=STATE_BY_FLAG[state_flag],
            tune_e5=(state_flag == "ft"), e5_train_kwargs=kwargs,
        )
    finally:
        disable_cache()
    return {"state": state_flag, "score": float(summary["test_score"]),
            "secs": round(time.time() - t0, 1), "epochs": epochs}

if not SMOKE:
    rows = [run_state("all"), run_state("ft", epochs=FULL_EPOCHS)]
    scores = {r["state"]: r["score"] for r in rows}
    delta = round(scores["ft"] - scores["all"], 4)
    for r in rows:
        print(f"  {r['state']:>4s}: {r['score']:.4f}  ({r['secs']}s)")
    print(f"\\nDelta_Awareness = {delta}  -> {'PASS' if delta > 0.001 else 'FAIL'} (delta=0.001)")

    out = pd.DataFrame(rows)
    out["dataset"] = DATASET_NAME
    out["fold"] = FOLD
    out["delta_awareness"] = delta
    out.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_CSV)
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


def _cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def _cells(require_gpu: bool, smoke: bool = True, full_epochs: int = 10) -> list[tuple[str, str]]:
    subs = {"__REQUIRE_GPU__": str(bool(require_gpu)),
            "__SMOKE__": str(bool(smoke)),
            "__FULL_EPOCHS__": str(int(full_epochs))}
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


def build_notebook(require_gpu: bool = True, smoke: bool = True, full_epochs: int = 10) -> dict:
    _check_cells_compile()
    return {
        "cells": [_cell(kind, src)
                  for kind, src in _cells(require_gpu, smoke, full_epochs)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata(enable_gpu: bool = True, machine_shape: str | None = None) -> dict:
    # The CPU validation variant lives in its own kernel so that pushing it does
    # not overwrite the GPU kernel's accelerator setting.
    kid = KERNEL_ID if enable_gpu else KERNEL_ID.replace("-tar-gpu", "-tar-cpu")
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
        "dataset_sources": [CODE_DATASET, CANDIDATE_DATASET],
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
    args = p.parse_args()

    require_gpu = not args.cpu
    os.makedirs(args.out, exist_ok=True)
    nb_path = os.path.join(args.out, NOTEBOOK_FILE)
    meta_path = os.path.join(args.out, "kernel-metadata.json")
    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(build_notebook(require_gpu=require_gpu, smoke=not args.full,
                                 full_epochs=args.full_epochs), fh, indent=1)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(build_metadata(enable_gpu=require_gpu,
                                 machine_shape=args.machine_shape), fh, indent=2)
    print(f"wrote {nb_path}\nwrote {meta_path}")


if __name__ == "__main__":
    main()
