"""Is a positive Delta_Joint distinguishable from fold noise?

`screen_deltas` reports ONE number per model: the difference of 5-fold means,
rounded to 3 decimals first (the official rule). That headline number carries no
uncertainty, so a marginal +0.002 and a decisive +0.2 look alike in the table.

This module keeps the official statistic untouched and adds a paired,
fold-level view alongside it:

    delta_f(model, fold) = all_f - max(no_text_f, text_only_f)

Pairing within a fold is what makes it a fair test -- all three states share the
same seed (`SEED + fold`), the same subsample and the same split, so fold-to-fold
variance cancels. The one-sided t-test asks whether mean(delta_f) > 0; the
bootstrap CI is reported too because 25 cells is small and the per-model n is 5.

Calibration note from Phase 1: fold-level noise on Delta_Joint is about +/-0.015,
so a mean below ~0.02 is weak evidence regardless of what the t-statistic says.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEAK_EVIDENCE_BAND = 0.02  # measured fold-level noise on Delta_Joint (~+/-0.015)


def fold_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    """Per (model, fold) paired Delta_Joint. Rows with a missing state are dropped."""
    wide = scores.pivot_table(index=["model", "fold"], columns="state",
                              values="test_score", aggfunc="mean")
    need = ["all", "no_text", "text_only"]
    missing = [c for c in need if c not in wide.columns]
    if missing:
        raise ValueError(f"states absent from results: {missing}")
    wide = wide.dropna(subset=need)
    out = wide[need].copy()
    out["delta_f"] = out["all"] - out[["no_text", "text_only"]].max(axis=1)
    return out.reset_index()


def _one_sided_t(x: np.ndarray) -> tuple[float, float]:
    """(t, p) for H0: mean <= 0 vs H1: mean > 0. Falls back to NaN p without scipy."""
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return float("inf") if x.mean() > 0 else float("nan"), float("nan")
    t = float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))
    try:
        from scipy import stats
        p = float(stats.t.sf(t, df=n - 1))
    except Exception:
        p = float("nan")
    return t, p


def _boot_ci(x: np.ndarray, iters: int = 20000, alpha: float = 0.05,
             seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(iters, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def significance(scores: pd.DataFrame) -> dict:
    """Pooled and per-model evidence that Delta_Joint > 0.

    Returns {"pooled": {...}, "per_model": DataFrame}. The pooled t-test treats the
    25 (model, fold) cells as exchangeable, which they are not -- models on the same
    fold are correlated -- so read it as a descriptive summary and rely on the
    per-model column (all 5 positive?) for the criterion itself.
    """
    fd = fold_deltas(scores)
    x = fd["delta_f"].to_numpy(dtype=float)
    t, p = _one_sided_t(x)
    lo, hi = _boot_ci(x)
    pooled = {
        "n_cells": int(len(x)),
        "n_positive": int((x > 0).sum()),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
        "t": t,
        "p_one_sided": p,
        "boot_ci95_lo": lo,
        "boot_ci95_hi": hi,
        "weak_evidence": bool(x.mean() < WEAK_EVIDENCE_BAND),
    }
    rows = []
    for model, sub in fd.groupby("model"):
        v = sub["delta_f"].to_numpy(dtype=float)
        tm, pm = _one_sided_t(v)
        rows.append({
            "model": model, "n_folds": len(v),
            "mean_delta_f": float(v.mean()),
            "std_delta_f": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "min_delta_f": float(v.min()), "n_positive": int((v > 0).sum()),
            "t": tm, "p_one_sided": pm,
        })
    return {"pooled": pooled,
            "per_model": pd.DataFrame(rows).sort_values("model").reset_index(drop=True),
            "fold_deltas": fd}
