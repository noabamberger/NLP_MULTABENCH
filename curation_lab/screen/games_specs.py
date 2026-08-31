"""Hand-built specs for the games / leisure / hobbies lane.

`auto_spec.build_spec` cannot express these three things, and every candidate in
this lane needed at least one of them:

1. **A parsed target.** MTG's price lives inside a packed string and is log-normal;
   BGG's raw file needs nothing parsed but does need columns chosen by meaning.
2. **Keeping structured columns the JUNK regex would delete.** That regex drops any
   column whose name contains `year`, `time`, `rank`, `id`, ... For a board game,
   `Year Published` and `Play Time` are not identifiers, they are the two strongest
   structured predictors. Deleting them does not just weaken `no_text` -- it lets
   the text act as a proxy for the deleted feature, which manufactures a
   Delta_Joint that disappears the moment the columns come back (measured:
   +0.0388 -> +0.0229 -> -0.0005 as `Year Published` and `Play Time` are restored).
3. **Choosing the semantically meaningful target.** The auto picker maximises a
   spread heuristic and picked `Complexity Average` over `Rating Average`, and a
   release *date* for the GOG catalogue.

Each entry returns (spec, why) so the driver can print the rationale next to the
numbers it produces.
"""
from __future__ import annotations

import glob
import os

from curation_lab.ingest.candidate import CandidateSpec


def _kaggle_csv(ref: str, filename: str | None = None) -> str:
    import kagglehub

    from curation_lab.discover.kaggle_search import _load_env

    _load_env()
    d = kagglehub.dataset_download(ref)
    csvs = glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
    if filename:
        csvs = [c for c in csvs if os.path.basename(c) == filename]
    return max(csvs, key=os.path.getsize)


def mtg_cards() -> tuple[CandidateSpec, str]:
    """Magic: The Gathering cards -> log10(USD market price)."""
    from curation_lab.prep.mtg_cards import build, spec

    out = "curation_lab/prep/data/mtg_cards.csv"
    if not os.path.exists(out):
        build(out)
    return spec(out), ("douglascampospires/mtg-all-cards; target log10(USD) parsed out of the "
                       "packed PRICES string, CMC clipped at the real ceiling of 16")


# The BGG dump that actually carries prose. `melissamonfared/board-games` has only
# a name and a comma-joined mechanics list; this one has the full rulebook blurb.
_BGG_TEXT = ["description", "mechanic", "category"]
_BGG_NUM = ["year_published", "min_players", "max_players", "min_playtime",
            "max_playtime", "playing_time", "min_age", "users_rated"]


def bgg_description() -> tuple[CandidateSpec, str]:
    """BoardGameGeek game description -> average user rating."""
    csv = _kaggle_csv("sujaykapadnis/board-games", "board_games.csv")
    keep = set(_BGG_TEXT) | set(_BGG_NUM) | {"average_rating"}
    import pandas as pd
    cols = list(pd.read_csv(csv, nrows=0).columns)
    return CandidateSpec(
        name="REG_TEXT_GAMES_BGG_DESCRIPTION", csv_path=csv, target="average_rating",
        task="REG", cols_to_drop=[c for c in cols if c not in keep],
        text_cols=list(_BGG_TEXT), numeric_cols=list(_BGG_NUM), categorical_cols=[],
        context="BoardGameGeek entries; predict the average user rating from the game's "
                "description, mechanics and categories plus its published attributes.",
    ), ("sujaykapadnis/board-games (10,532 rows); every structured column is kept, "
        "including year_published and the playtime columns the JUNK regex would drop")


def bgg_ratings_full() -> tuple[CandidateSpec, str]:
    """The lane's PRIMARY, with the JUNK regex's deletions restored. Kept for the record."""
    csv = _kaggle_csv("melissamonfared/board-games")
    num = ["Year Published", "Min Players", "Max Players", "Play Time", "Min Age",
           "Users Rated", "Owned Users", "Complexity Average"]
    return CandidateSpec(
        name="REG_TEXT_GAMES_BGG_RATING", csv_path=csv, target="Rating Average", task="REG",
        cols_to_drop=["ID", "BGG Rank"], text_cols=["Mechanics", "Name"],
        numeric_cols=num, categorical_cols=["Domains"], read_kwargs={"encoding": "latin-1"},
    ), "melissamonfared/board-games with the full structured block restored"


CANDIDATES = {
    "mtg_cards": mtg_cards,
    "bgg_description": bgg_description,
    "bgg_ratings_full": bgg_ratings_full,
}
