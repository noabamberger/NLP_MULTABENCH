"""Fail-fast candidate hunter: screen many datasets for Delta_Joint, cheaply.

Ordering exploits the criterion's own shape:

    Delta_Joint = mean(all) - max(mean(no_text), mean(text_only))

so if `all` <= `no_text` the dataset FAILS no matter what `text_only` does. We run
`no_text` first (no encoder, seconds), then `all` (the one expensive encode), and
abort before `text_only` when the comparison is already lost. Survivors get
`text_only` almost free, because the embedding cache is already warm from `all`.

Screening uses the max_length cap (~7x faster, ~1e-7 embedding drift). That drift is
irrelevant for a triage gate but must NOT be used for numbers reported against the
paper.
"""
from __future__ import annotations

import glob
import os
import time
import traceback

import pandas as pd
from tabstar.training.devices import get_device

from multabench.baselines.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, evaluate_on_loaded_dataset
from multabench.baselines.lgbm import LightGBM
from multabench.constants import DEVICE

from curation_lab.ingest.candidate import STATE_BY_FLAG, load_candidate
from curation_lab.runner.cache import disable_cache, enable_cache
from curation_lab.screen.auto_spec import build_spec
from curation_lab.screen.batch_profile import _read_any_csv

# Abort if `all` fails to beat `no_text` by at least this much. Slightly negative:
# a 1-fold estimate is noisy and a discarded good dataset is unrecoverable, while a
# passed-through bad one only costs one more cheap run.
ABORT_MARGIN = -0.005


def _one_run(spec, state: str, fold: int, device, cache_dir: str) -> float:
    enable_cache(cache_dir, frozen_only=True, dynamic_max_length=True)
    try:
        loaded = load_candidate(spec, state)
        s = evaluate_on_loaded_dataset(
            model_cls=LightGBM, dataset=loaded, fold=fold, device=device,
            train_examples=DOWNSTREAM_EXAMPLES,
            multimodal_state=STATE_BY_FLAG[state], tune_e5=False, e5_train_kwargs=None,
        )
    finally:
        disable_cache()
    return float(s["test_score"])


def hunt_one(ref: str, out_rows: list[dict], cache_dir: str = ".emb_cache") -> dict:
    """Profile, auto-spec, and frozen-screen one Kaggle dataset with early abort."""
    import kagglehub

    row = {"ref": ref, "verdict": "", "detail": "", "no_text": None, "all": None,
           "text_only": None, "delta_joint": None, "secs": 0.0}
    t0 = time.time()
    try:
        d = kagglehub.dataset_download(ref)
        csvs = [f for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
                if os.path.getsize(f) / 1e6 <= 200]
        if not csvs:
            row.update(verdict="skip", detail="no csv")
            return row
        path = max(csvs, key=os.path.getsize)
        df, enc, err = _read_any_csv(path)
        if df is None:
            row.update(verdict="skip", detail=f"unreadable: {err[:60]}")
            return row
        rk = {}
        if enc and enc not in ("utf-8",):
            rk["encoding"] = enc.split("/")[0]
            if "sep=" in enc:
                rk["sep"] = enc.split("sep=")[1].strip("'\"")
        name = "REG_TEXT_AUTO_" + "".join(ch if ch.isalnum() else "_" for ch in ref.upper())[:44]
        spec, why = build_spec(df, name=name, csv_path=path, read_kwargs=rk)
        if spec is None:
            row.update(verdict="skip", detail=why)
            return row
        row["detail"] = why

        device = get_device(device=DEVICE)
        row["no_text"] = _one_run(spec, "no_text", 0, device, cache_dir)
        row["all"] = _one_run(spec, "all", 0, device, cache_dir)
        if row["all"] - row["no_text"] < ABORT_MARGIN:
            row.update(verdict="FAIL", detail=f"{why} | all<=no_text, aborted before text_only")
            return row
        row["text_only"] = _one_run(spec, "text_only", 0, device, cache_dir)
        row["delta_joint"] = round(row["all"] - max(row["no_text"], row["text_only"]), 4)
        row["verdict"] = "PASS" if row["delta_joint"] > 0 else "FAIL"
    except Exception as e:
        row.update(verdict="error", detail=f"{type(e).__name__}: {str(e)[:80]}")
        traceback.print_exc(limit=1)
    finally:
        row["secs"] = round(time.time() - t0, 1)
        out_rows.append(row)
    return row


def hunt(refs: list[str], out_csv: str) -> pd.DataFrame:
    rows: list[dict] = []
    for i, ref in enumerate(refs, 1):
        r = hunt_one(ref, rows)
        print(f"[{i}/{len(refs)}] {r['verdict']:<6} dj={r['delta_joint']} "
              f"nt={r['no_text']} all={r['all']} to={r['text_only']} "
              f"{r['secs']}s {ref[:44]} :: {r['detail'][:90]}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    return pd.DataFrame(rows)
