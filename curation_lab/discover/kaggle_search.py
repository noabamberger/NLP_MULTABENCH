"""T0 metadata filter: search Kaggle for text-tabular candidates.

Selection is driven by the encoding-cost model discovered in Phase 1:

    cost  ~  n_rows x n_text_columns x mean_tokens

so we want SHORT, few text columns, and a moderate row count -- but not so short
that multabench's _is_text_feature (>=100 distinct values OR >=80% unique ratio)
demotes the column to categorical, which would collapse the text_only condition.

This module only ranks *candidates* from metadata. Nothing here decides anything:
the real gates are the T1 typing probe and the T2 frozen screen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_USED_SLUGS_PATH = Path(__file__).with_name("_used_slugs.txt")


def _load_env(path: str = ".env") -> None:
    """Populate KAGGLE_* env vars before importing kaggle (it authenticates on import)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k.startswith("KAGGLE_") and v:
            os.environ.setdefault(k, v)


def used_slugs() -> set[str]:
    """owner/slug pairs already used by MulTaBench — these cannot be the deliverable."""
    if not _USED_SLUGS_PATH.exists():
        return set()
    return {s.strip().lower() for s in _USED_SLUGS_PATH.read_text(encoding="utf-8").splitlines() if s.strip()}


@dataclass
class Candidate:
    ref: str
    title: str
    subtitle: str
    size_mb: float
    downloads: int
    votes: int
    usability: float
    license_name: str

    def __str__(self) -> str:
        return (f"{self.ref:<52} {self.size_mb:>7.1f}MB  votes={self.votes:<5} "
                f"dl={self.downloads:<7} {self.title[:42]}")


def search(queries: list[str], max_size_mb: float = 120.0, min_size_mb: float = 0.05,
           per_query: int = 20) -> list[Candidate]:
    """Search Kaggle for each query, drop already-used and out-of-size-range datasets.

    The size window is a proxy for row count: under ~50KB is too small to give stable
    5-fold estimates; over ~120MB usually means either huge row counts (wasted, since
    training subsamples to 10k) or long documents (expensive to encode).
    """
    _load_env()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    used = used_slugs()
    seen: set[str] = set()
    out: list[Candidate] = []
    for q in queries:
        try:
            results = api.dataset_list(search=q, file_type="csv")
        except Exception as e:
            print(f"  [warn] query {q!r} failed: {type(e).__name__}: {e}")
            continue
        for d in results[:per_query]:
            ref = str(d.ref)
            key = ref.lower()
            if key in used or key in seen:
                continue
            seen.add(key)
            # kaggle 2.x uses snake_case, and its server-side max_size filter is
            # not applied, so size is filtered here.
            size_mb = float(getattr(d, "total_bytes", 0) or 0) / 1e6
            if not (min_size_mb <= size_mb <= max_size_mb):
                continue
            out.append(Candidate(
                ref=ref,
                title=str(getattr(d, "title", "") or ""),
                subtitle=str(getattr(d, "subtitle", "") or ""),
                size_mb=size_mb,
                downloads=int(getattr(d, "download_count", 0) or 0),
                votes=int(getattr(d, "vote_count", 0) or 0),
                usability=float(getattr(d, "usability_rating", 0) or 0),
                license_name=str(getattr(d, "license_name", "") or ""),
            ))
    return out


def rank(cands: list[Candidate]) -> list[Candidate]:
    """Popularity is a weak proxy for 'clean and well documented', which matters for curation."""
    return sorted(cands, key=lambda c: (c.votes, c.downloads), reverse=True)
