"""Rank T1-viable candidates by domain novelty and by shape.

Two filters, both learned the hard way and recorded in RESUME.md:

- Slug exclusion is NOT enough. MulTaBench already contains wine, restaurants, used
  cars, video-game sales, job postings, anime, airbnb, spotify, laptop prices and
  more under *other* slugs, so a candidate has to be matched on domain words, not on
  whether its ref was seen before.
- "reviews" as a search term is an anti-pattern: it returns single-column text
  corpora with no structured block, which fail the joint criterion by construction.

Shape rules come from what actually passed: 1-3 text columns (the multimodal budget
is 5, and every extra column is another encode), a non-empty structured block so
`no_text` is a real baseline rather than an empty condition, and enough rows that a
5-fold split is meaningful.

Usage:
    python -m curation_lab.discover.novelty --t1 results/candidates/t1_batch.csv \
        --exclude results/candidates/hunt_full.csv --top 20
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

# Domains already in MulTaBench under some slug, plus the shapes that reliably fail.
USED_DOMAIN = re.compile(
    r"wine|zomato|restaurant|food-delivery|used-car|car-price|auto-scout|vehicle|"
    r"video-game-sales|game-sales|job|salary|salaries|hiring|linkedin|resume|"
    r"book|goodreads|rotten|tomato|imdb|movie|anime|manga|airbnb|mercari|"
    r"clothing|product-sentiment|accident|spotify|music|song|track|toxic|jigsaw|"
    r"kickstarter|chocolate|coffee|beer|ramen|laptop|amazon|university|ranking|"
    r"play-store|google-play|app-store|udemy|coursera|course", re.I)

# Terms that describe a text corpus rather than a table with a text column.
CORPUS_SHAPED = re.compile(r"review|tweet|sentiment|news|headline|article|comment|"
                           r"chat|dialog|corpus|nlp|text-classification", re.I)


def rank(t1: pd.DataFrame, exclude_refs: set[str], min_rows: int = 2000,
         max_rows: int = 300_000) -> pd.DataFrame:
    d = t1[t1["viable"] == True].copy()  # noqa: E712 - pandas mask, not identity
    d = d[~d["ref"].isin(exclude_refs)]

    d["n_struct"] = d["n_num"].fillna(0) + d["n_cat"].fillna(0)
    d["used_domain"] = d["ref"].str.contains(USED_DOMAIN)
    d["corpus_shaped"] = d["ref"].str.contains(CORPUS_SHAPED)

    reasons = []
    for _, r in d.iterrows():
        why = []
        if r["used_domain"]:
            why.append("used-domain")
        if r["corpus_shaped"]:
            why.append("corpus-shaped")
        if not (min_rows <= r["n_rows"] <= max_rows):
            why.append(f"rows={int(r['n_rows'])}")
        if not (1 <= r["n_text"] <= 3):
            why.append(f"n_text={int(r['n_text'])}")
        if r["n_struct"] < 3:
            why.append(f"n_struct={int(r['n_struct'])}")
        reasons.append(",".join(why))
    d["rejected_for"] = reasons
    d["keep"] = d["rejected_for"] == ""

    # Among survivors, prefer a small text budget and a substantial structured block:
    # that is the shape where a positive Delta_Joint means complementarity rather
    # than an artifact of one side being empty.
    d["score"] = (d["n_struct"].clip(upper=15) / 15.0) - 0.15 * d["n_text"]
    return d.sort_values(["keep", "score"], ascending=[False, False])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t1", required=True)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    excl: set[str] = set()
    for path in args.exclude:
        excl |= set(pd.read_csv(path, encoding="utf-8")["ref"])

    d = rank(pd.read_csv(args.t1, encoding="utf-8"), excl)
    keep = d[d["keep"]]
    print(f"{len(keep)} kept of {len(d)} viable-unscreened\n")
    cols = ["ref", "n_rows", "n_text", "n_num", "n_cat", "score"]
    print(keep[cols].head(args.top).to_string(index=False))
    if args.out:
        keep.to_csv(args.out, index=False, encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
