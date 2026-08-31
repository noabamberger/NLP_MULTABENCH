"""Frozen-only grid driver for `games_specs.CANDIDATES`.

Same contract as `verify_spec.py` (which reads the media lane's registry): it
delegates to `verify._run_all`, so there is one implementation of the grid loop,
the cache wiring and the resumable skip. Frozen states only -- no `--epochs`, no
path to `tune_e5=True`. TAR is measured elsewhere, on a GPU.
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd


def main() -> None:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--models", default="", help="comma-separated subset, for triage")
    args = p.parse_args()

    from curation_lab.criterion.deltas import normalize, screen_deltas
    from curation_lab.screen.games_specs import CANDIDATES
    from curation_lab.screen.verify import _run_all

    if args.candidate not in CANDIDATES:
        raise SystemExit(f"unknown candidate {args.candidate!r}; have {sorted(CANDIDATES)}")
    spec, why = CANDIDATES[args.candidate]()
    print(f"{spec.name}: {why}", flush=True)
    print(f"  target={spec.target} text={spec.text_cols} num={spec.numeric_cols} "
          f"cat={spec.categorical_cols} drop={spec.cols_to_drop}", flush=True)

    df = pd.read_csv(spec.csv_path, **spec.read_kwargs)
    y = pd.to_numeric(df[spec.target], errors="coerce").dropna()
    z = float(((y - y.mean()).abs() / y.std(ddof=0)).max())
    top = y.value_counts(normalize=True).head(1)
    print(f"  target dist: n={len(y)} nuniq={y.nunique()} min={y.min()} max={y.max()} "
          f"mean={y.mean():.3f} std={y.std():.3f} |z|max={z:.2f} "
          f"most_common={top.index[0]}@{top.iloc[0]*100:.1f}%", flush=True)

    folds = tuple(int(f) for f in args.folds.split(","))
    only = [m.strip() for m in args.models.split(",") if m.strip()] or None
    _run_all(spec, args.out, folds, do_ft=False, epochs=None, only_models=only)

    out = normalize(pd.read_csv(args.out, encoding="utf-8"))
    print("\n=== deltas ===")
    print(screen_deltas(out).to_string(index=False))


if __name__ == "__main__":
    main()
