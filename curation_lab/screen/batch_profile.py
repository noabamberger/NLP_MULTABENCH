"""Batch T1: download and typing-profile every T0 candidate.

T1 costs seconds per dataset, so there is no reason to hand-pick a handful from
the T0 list -- run the whole set and let the gates decide. Results are written
incrementally so a long run can be inspected while still in progress.

Failures are recorded, never silently dropped: an unreadable file is a finding
(it feeds the parser-fallback work), not a non-result.
"""
from __future__ import annotations

import glob
import os
import traceback

import pandas as pd

from curation_lab.discover.kaggle_search import _load_env
from curation_lab.screen.profile import profile_frame

# Encodings tried in order; the first that parses wins. Recorded per dataset so
# the Phase 3 scraper can learn which sources need which.
_ENCODINGS = ("utf-8", "latin-1", "cp1252")
_MAX_CSV_MB = 200.0


def _read_any_csv(path: str) -> tuple[pd.DataFrame | None, str, str]:
    """Return (df, encoding_used, error). Tries several encodings and separators."""
    last = ""
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if df.shape[1] == 1:  # probably the wrong separator
                for sep in (";", "\t", "|"):
                    try:
                        alt = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                        if alt.shape[1] > 1:
                            return alt, f"{enc}/sep={sep!r}", ""
                    except Exception:
                        pass
            return df, enc, ""
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:110]}"
    return None, "", last


def profile_ref(ref: str) -> dict:
    """Download one Kaggle dataset and profile its largest CSV."""
    import kagglehub

    row: dict = {"ref": ref, "status": "", "encoding": "", "csv": "", "n_rows": 0,
                 "n_text": 0, "n_num": 0, "n_cat": 0, "text_cols": "", "problems": "",
                 "viable": False}
    try:
        d = kagglehub.dataset_download(ref)
    except Exception as e:
        row["status"] = f"download_failed: {type(e).__name__}: {str(e)[:80]}"
        return row

    csvs = [f for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
            if os.path.getsize(f) / 1e6 <= _MAX_CSV_MB]
    if not csvs:
        row["status"] = "no_usable_csv"
        return row
    path = max(csvs, key=os.path.getsize)
    row["csv"] = os.path.basename(path)

    df, enc, err = _read_any_csv(path)
    if df is None:
        row["status"] = f"read_failed: {err}"
        return row
    row["encoding"] = enc

    try:
        p = profile_frame(df, name=ref)
    except Exception as e:
        row["status"] = f"profile_failed: {type(e).__name__}: {str(e)[:80]}"
        return row

    row.update({
        "status": "ok",
        "n_rows": p.n_rows,
        "n_text": len(p.text_cols),
        "n_num": len(p.numeric_cols),
        "n_cat": len(p.categorical_cols),
        "text_cols": "|".join(p.text_cols[:6]),
        "problems": "; ".join(p.problems),
        "viable": p.viable,
    })
    return row


def run(refs: list[str], out_csv: str) -> pd.DataFrame:
    _load_env()
    rows: list[dict] = []
    for i, ref in enumerate(refs, 1):
        try:
            row = profile_ref(ref)
        except Exception:
            row = {"ref": ref, "status": "crashed: " + traceback.format_exc(limit=1)[:100],
                   "viable": False}
        rows.append(row)
        print(f"[{i}/{len(refs)}] {ref:<55} {row.get('status','')[:40]:<40} "
              f"rows={row.get('n_rows',0):<7} text={row.get('n_text',0)} "
              f"viable={row.get('viable')}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    return pd.DataFrame(rows)
