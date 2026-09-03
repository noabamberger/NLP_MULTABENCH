> **Superseded 2026-09-02.** Replaced by `docs/findings/02-mining-method-rules.md`.
> This file's per-candidate scouting notes and its "corrections to my stated assumptions" section
> are folded into the consolidated rulebook, generalized as numbered rules with their evidence.
> Kept verbatim below: the current document was written from it, and the
> judgment calls in that rewrite should stay checkable against this source.

# RESEARCH_NOTES.md — Phase 2 dataset scouting

Log of T0 (metadata) and T1 (typing probe) work for the Curation Lab
(`docs/superpowers/specs/2026-08-28-curation-lab-design.md`, §4).

All profiling used `.venv/Scripts/python.exe` with `PYTHONIOENCODING=utf-8`, via
`curation_lab/screen/profile.py::profile_frame` / `token_profile` — i.e. multabench's own
detectors (`detect_numerical_features`, `classify_semantic_features`, `detect_image_features`),
never a reimplementation of the heuristics.

"TEXT" below always means *detected as text by `_is_text_feature`* (>=100 distinct values OR
>=80% unique ratio), not "looks like prose to a human". Nothing else matters for the
`no_text` / `text_only` split.

---

## 1. Summary — candidates ranked by viability

| # | Candidate | Kaggle ref | Rows (post-curation) | TEXT | NUM | CAT | Target | z-max | tok/pass | T1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Board games (BGG)** | `melissamonfared/board-games` | 20,343 | 2 | 8 | 1 | `Rating Average` (REG) | 5.72 (7 rows) | 525k | **PASS** |
| 2 | **Sephora products** | `raghadalharbi/all-products-available-on-sephora-website` | 9,168 | 4 | 8 | 1 | `rating` (REG) | 3.96 (0 rows) | 386k | **PASS** |
| 3 | **Fragrantica perfumes** | `olgagmiufana1/fragrantica-com-fragrance-dataset` (`fra_cleaned.csv`) | 24,063 | 5 | 2 | 7 | `Rating Value` (REG) | 9.16 (11 rows) | 1.64M | **PASS** (target needs trim) |
| 4 | Fragrantica (`fra_perfumes.csv`) | same ref, other file | 63,922 | 4 | **0** | 1 | `Rating Value` | 5.69 (296) | 5.15M | weak — see §2.4 |
| 5 | Luxury watches | `philmorekoung11/luxury-watch-listings` | 284,491 | ~4 | 0 raw | — | `price` (REG) | **136.9 (906)** | — | **FAIL** (outliers + all-string cols) |
| 6 | Google Play Store apps | `bhavikjikadara/google-play-store-applications` | 9,367 | ~2 | 1 raw | — | `Rating` (REG) | **27.6 (24)** | — | **FAIL** (dirty; needs parsers) |
| 7 | LEGO sets | `mterzolo/lego-sets` | 12,261 | ~3 | 8 | 2 | `star_rating` | — | — | **FAIL** (744 real products x 21 countries) |
| 8 | Drugs.com side effects | `jithinanievarghese/drugs-side-effects-and-medical-condition` | **1,583** | ~5 | 1 | 5 | `rating` | — | — | **FAIL** (<2000 rows) |

`tok/pass` = total E5 tokens for one full encoding pass over the whole frame, using the exact
string the pipeline builds (`"passage: {col}: {val}"`). Training subsamples to 10k rows
(`DOWNSTREAM_EXAMPLES`), so the effective per-run cost is roughly `tok/pass * 10000/n_rows`.

**Recommended backups: #1 Board games, then #2 Sephora, then #3 Fragrantica.**

---

## 2. Per-candidate detail

### 2.1 Board games (BGG) — `melissamonfared/board-games`  [TOP PICK]

Untouched domain. One file, `BGG_Data_Set.csv`, 20,343 x 14.

**Ingestion:** UTF-8 fails (`invalid continuation byte` at byte 3 — the header itself).
**`encoding="cp1252"` works**; `latin-1` also decodes but mangles the accented designer names.

**Spec:** `cols_to_drop: [ID, "BGG Rank"]`, `target: "Rating Average"`, task `REG`.

```
TEXT (2): Mechanics, Name
NUM  (8): Complexity Average, Max Players, Min Age, Min Players,
          Owned Users, Play Time, Users Rated, Year Published
CAT  (1): Domains
```

- **`BGG Rank` MUST be dropped — it is the target.** BGG's site rank is a monotone function of
  the Bayesian-adjusted average rating. It is also perfectly unique (20,343/20,343), so if left
  in it would additionally be typed as TEXT and leak into *every* condition. This is the single
  most dangerous column in the file.
- `Complexity Average` is the strongest legitimate structured predictor (|corr| = 0.481). It is a
  separate user-voted quantity, not a rating restatement — keep it. Its presence is a feature, not
  a bug: it gives `no_text` a real, non-degenerate baseline, which makes a positive Δ_Joint
  meaningful rather than an artifact of an empty structured condition.
- Target `Rating Average`: min 1.05, max 9.58, mean 6.40, std 0.936. `|z|max = 5.72`, **7 rows**
  over the |z|>5 line. Trim those 7 (or accept the warning) and the target is clean. Do **not**
  log-transform: `log1p` makes it far worse (|z|max 9.59, 24 rows) because the tail is on the low
  side.
- `Domains` is 50% null and only 39 distinct → lands as CATEGORICAL, which is what we want (it
  keeps the multimodal budget at 2).
- Token cost: `Name` mean 10.6 / max 33 / cap 40; `Mechanics` mean 15.2 / max 84 / cap 88.
  525,349 tokens/pass over 20k rows → ~258k per 10k-row run. 5.8x cheaper than a 512 cap.

**Why it should produce deltas:** `Mechanics` ("Action Queue, Deck Building, Worker Placement…")
is semantically dense and genuinely predictive of how well a game is rated, and it is *not*
recoverable from the eight numeric columns. `Name` adds franchise/expansion signal. Both are the
kind of short, target-relevant text that TAR fine-tuning can sharpen — which is the Δ_Awareness
story.

### 2.2 Sephora products — `raghadalharbi/all-products-available-on-sephora-website`  [BEST COST]

9,168 x 21, plain UTF-8, no parse issues at all.

Raw file has **9 columns that type as TEXT**, which blows the `MAX_MULTIMODAL = 5` budget.
Dropping the four long ones plus the identifier fixes it:

**Spec:** `cols_to_drop: [id, URL, details, how_to_use, ingredients, options, value_price]`,
`target: rating`, task `REG`.

```
TEXT (4): brand, category, name, size
NUM  (8): MarketingFlags, exclusive, limited_edition, limited_time_offer,
          love, number_of_reviews, online_only, price
CAT  (1): MarketingFlags_content
```

Drop rationale, column by column:

| Column | Why dropped |
|---|---|
| `id` | identifier, 99.4% unique → would type TEXT |
| `URL` | 100% unique → types TEXT, pure noise, and encodes the product name verbatim |
| `details`, `how_to_use`, `ingredients` | long marketing/INCI prose; ~96%/71%/76% distinct. Each alone would dominate the encoding budget |
| `options` | 564 distinct so it *does* type TEXT, but **91.4% of rows are the literal string `"no options"`**. Near-zero information, and it was the entire reason the cap was 128 instead of 40 |
| `value_price` | **leakage risk against a `price` target** — `corr(price, value_price) = 0.983`, and the two are literally equal in 94% of rows. Harmless if the target is `rating`, but dropped so the target shortlist can include `price` without re-curating |

Surprises worth recording:

- **`brand` (324 distinct) and `category` (143 distinct) type as TEXT, not categorical.** The
  `>=100 distinct` arm of `_is_text_feature` fires even though the unique ratio is only 3.5% /
  1.6%. This is the mirror image of the usual failure — columns a human would call categorical
  get *promoted* into the multimodal budget. It is also why the raw file has 9 text columns.
  They are cheap (cap 16) and semantically real, so keeping them is fine, but the budget
  accounting has to expect it.
- Structured features are almost uninformative about `rating`: strongest |corr| is 0.130
  (`limited_edition`). A weak `no_text` baseline is good for Δ_Joint headroom but also means the
  overall R² will be low, so the absolute deltas may be small.
- **Target caveat — the `rating == 0` sentinel.** 398 rows have `rating = 0`, and 397 of them
  have `number_of_reviews = 0`. These are unrated products, not badly-rated ones. Two options,
  and they trade off against each other:
  - *Keep them* (recommended for a first T2): target is clean (`|z|max = 3.96`, zero rows over
    the line), but `number_of_reviews == 0` is a near-perfect structured shortcut for
    `rating == 0`, which lifts the `no_text` baseline and eats Δ_Joint.
  - *Filter `number_of_reviews > 0`*: n = 8,771, target becomes unimodal (mean 4.17, std 0.555)
    but picks up 24 rows with |z| > 5. Removes the shortcut.
- Token cost is the best of any candidate: every column caps at <=40, 385,943 tokens/pass,
  **12.8x** cheaper than a 512 cap.
- Alternative target worth a shortlist slot: **`rating >= 4.5` as BIN** (48.0% positive, well
  balanced) — ROC-AUC is bounded and often gives steadier fold means than a low-R² regression.

### 2.3 Fragrantica perfumes — `olgagmiufana1/fragrantica-com-fragrance-dataset` (`fra_cleaned.csv`)

Untouched domain, and the most *structurally* interesting candidate. See §3 for the three-way
ingestion trap that made this file initially unreadable.

**Spec:** `loader: {file: fra_cleaned.csv, sep: ";", encoding: latin-1, decimal: ","}`,
`cols_to_drop: [url, Perfumer1, Perfumer2]`, `target: "Rating Value"`, task `REG`.

```
TEXT (5): Base, Brand, Middle, Perfume, Top
NUM  (2): Rating Count, Year
CAT  (7): Country, Gender, mainaccord1..mainaccord5
```

- `Top` / `Middle` / `Base` are the fragrance pyramid — comma-separated note lists
  ("bulgarian rose, egyptian jasmine, lily-of-the-valley"). Dense, domain-specific, and exactly
  the kind of text a frozen general-purpose E5 embeds only approximately, which is where TAR
  fine-tuning should earn Δ_Awareness.
- `mainaccord1..5` are single accord words (66-77 distinct each) → they land as CATEGORICAL and
  give the `no_text` state five real features. This is a genuinely nice structure: the categorical
  accords are a coarse summary of the same information the note text carries in full.
- Exactly at the multimodal budget: **5 TEXT columns, no headroom.** Dropping `Brand` (a
  1,060-value brand-prestige signal) is the obvious lever if a 6th text column is ever needed.
- `Perfumer1` (869 distinct, but the modal value is the literal `"unknown"`) and `Perfumer2`
  (94% null) are dropped — both would otherwise type TEXT and blow the budget to 7.
- Target `Rating Value`: mean 3.960, **std only 0.277**. `|z|max = 9.16` but just **11 rows**
  over the line; trim them, or filter on `Rating Count`. The narrow spread is the real concern —
  structured |corr| maxes at 0.109 (`Year`), so every condition may sit near R² = 0 and the deltas
  could be small in absolute terms.
- Token cost: 1,644,661/pass over 24k rows → ~683k per 10k-row run. Cap 96, driven by `Top` and
  `Middle` (max 89). 5.3x cheaper than 512.

### 2.4 Fragrantica `fra_perfumes.csv` — the same ref's *other* file, and why it loses

70,103 x 8, ordinary UTF-8 comma CSV — so it reads without complaint, which makes it the
tempting choice. It is the worse file:

```
TEXT (4): Main Accords, Name, Perfumers, Rating Count   <-- Rating Count!
NUM  (0)
CAT  (1): Gender
```

**`Rating Count` arrives as `object` dtype**, so `detect_numerical_features` rejects it, and its
2,913 distinct values then trip the `>=100 distinct` text rule. The result is a dataset with
**zero numeric features** — `no_text` would be reduced to the 3-value `Gender` column alone.
That is not a Δ_Joint measurement, it is a degenerate baseline. Recorded as a clean example of
gate 2 (§4 of the design doc) firing for a non-obvious reason: it wasn't that no structured column
existed, it was that a numeric one got *promoted* to TEXT by a dtype accident.

`fra_cleaned.csv` covers the same domain with a third of the rows, a real structured block, and a
correctly-typed `Rating Count`. Use it instead.

### 2.5 Rejected, with reasons

**Luxury watches — `philmorekoung11/luxury-watch-listings`** (284,491 x 14). Attractive on
metadata; fails on two counts. (a) `price` arrives as `"$43,500"` strings; parsed, it runs to
$9,000,000 against a $6,899 median — `|z|max = 136.9` with **906 rows** over |z|>5. `log(price)`
would fix it (|z|max 5.27, 5 rows) but the repo never transforms targets, so that needs a
`PROCESSING_FUNC`, i.e. real curation code rather than a spec. (b) `size` ("42 mm") and `yop`
(year) are also strings, and `ref` (33,223 distinct serial numbers) would type TEXT while carrying
no semantics. `name` is 25% null. Salvageable with effort; not a backup.

**Google Play Store apps — `bhavikjikadara/google-play-store-applications`** (10,841 x 14).
`Reviews`, `Size` ("19M"), `Installs` ("10,000+"), `Price` ("$4.99") are all strings needing four
separate parsers, `Installs` is only 22 distinct so it degrades to categorical, 1,181 duplicate
app names, and `Rating` contains the well-known corrupted row (`|z|max = 27.6`, 24 rows over the
line). Too much cleaning for a backup.

**LEGO sets — `mterzolo/lego-sets`** (12,261 x 14). Looks like 12k rows; is actually
**744 products x 21 countries**. The same product appears ~16 times, so any fold split puts
near-identical rows in both train and test — the scores would be meaningless. Separately,
`val_star_rating` correlates 0.728 with `star_rating` and `play_star_rating` 0.608, so any one of
the three as target leaks through the other two. Reject.

**Drugs.com side effects — `jithinanievarghese/drugs-side-effects-and-medical-condition`**
(2,931 x 17). Genuinely nice schema — `drug_name`, `generic_name`, `drug_classes`, `side_effects`
text alongside `rx_otc` / `pregnancy_category` / `csa` categoricals. But `rating` is only 54%
non-null → **1,583 rows** after the null-target drop, under the 2,000 floor. Recorded in case the
floor is ever relaxed, or a larger drugs.com scrape turns up.

---

## 3. Ingestion pitfalls observed

| # | Symptom | Dataset | Cause | Fix |
|---|---|---|---|---|
| 1 | `UnicodeDecodeError: invalid continuation byte 0xe9` at **byte 3** | `melissamonfared/board-games` | Windows-1252 file (accented designer/publisher names); the bad byte is in the header | `encoding="cp1252"`. `latin-1` also decodes but corrupts the accents |
| 2 | `ParserError: Expected 7 fields in line 5, saw 8` | `fra_cleaned.csv` | **Not ragged rows.** The file is `;`-delimited, so pandas' `,` sniffing split the embedded commas in the note lists | `sep=";"` |
| 3 | Ratings all NaN after fixing #2 | `fra_cleaned.csv` | European decimal comma — `Rating Value` is `1,42` | `decimal=","` |
| 4 | `UnicodeDecodeError` at byte 7752 after fixing #2/#3 | `fra_cleaned.csv` | latin-1 file as well | `encoding="latin-1"` |
| 5 | `ValueError: The 'low_memory' option is not supported with the 'python' engine` | (scouting script, not committed code) | the `ParserError` retry path passed `low_memory=False` alongside `engine="python"` | drop `low_memory` whenever `engine="python"` is used. This bug **masked** pitfalls #2-#4 and made the file look unrecoverable — worth guarding against in `ingest/parsers.py` |
| 6 | Numeric column typed as TEXT | `fra_perfumes.csv`, `googleplaystore.csv` | `object` dtype from thousands separators / unit suffixes → `detect_numerical_features` rejects → >=100 distinct → TEXT | coerce to numeric during curation *before* typing, or drop |
| 7 | `DtypeWarning: Columns (13) have mixed types` | `Watches.csv` | mixed `condition` column | `low_memory=False` |

**Lesson for `ingest/parsers.py`:** the three fragrance failures were a *compound* — separator,
decimal mark, and encoding all wrong at once — and each fix only exposed the next error. A
fallback reader that tries one axis at a time will conclude the file is unreadable. Try the
cross-product `{utf-8, cp1252, latin-1} x {",", ";", "\t"} x {".", ","}` and keep the read whose
result has the most numeric columns, not merely the first that does not raise.

---

## 4. Queries that worked vs. were noisy

Runs used `curation_lab/discover/kaggle_search.search(..., max_size_mb=120, min_size_mb=0.15,
per_query=20)`.

### Worked

| Query | Yield |
|---|---|
| `board game` | the top pick, plus 8 further BGG variants — an unusually dense cluster |
| `perfume fragrance` | Fragrantica (#3) and a Russian-language sibling (`aromo-ru-fragrance-dataset`) |
| `board game geek ratings` | more BGG variants; confirms the domain is deep enough for the Outstanding scope |
| `lego sets` | 8 hits — right *shape*, wrong *content* (see the LEGO rejection); still a good shape-query |
| `pokemon cards`, `magic the gathering cards` | trading-card catalogues: card name + rules text + numeric stats. Not yet profiled; the most promising unexplored lead |
| `watches price` | luxury watches + Rolex/Chrono24 — right shape, price-target outliers kill them |

### Noisy

| Query | Failure mode |
|---|---|
| `drug reviews` | returns *review corpora* — one long free-text column and a sentiment label. No structured block at all, so `no_text` is empty. **"reviews" in a query is an anti-pattern**: it selects for the unstructured-only shape |
| `podcast` | transcripts (Lex Fridman, Skeptoid, Office Ladies) — document dumps, not tables |
| `google play store apps` | 12+ near-identical forks of the same dirty 2018 scrape |
| `recipes nutrition ratings`, `hiking trails`, `cocktail recipes` | returned **nothing** on-topic; the round's result set was entirely `drug reviews` + `podcast` hits. Multi-word natural-language queries get swamped by the co-submitted queries' stronger terms |
| `art auction paintings`, `shark tank pitches`, `houseplants gardening`, `craft cocktails` | same — zero on-topic hits; the round was 100% olympics/university |
| `olympic athletes` | 10+ hits, all useless: the only text is athlete and country **names**, which are proper nouns with no semantic content for E5. High row counts, zero multimodality |
| `university programs tuition` | institution names — same proper-noun problem |
| `steam store games` | adjacent to MulTaBench's existing video-game-sales domain; excluded on novelty |

### Rules this suggests for `discover/rules.py` (Phase 3)

1. **Query for catalogues, not corpora.** `{item-type} {attribute}` ("board game", "lego sets",
   "watches price") returns item-name + attributes + numeric-target tables. Queries containing
   `review`, `sentiment`, `transcript`, `tweets`, `comments` return single-column text corpora
   that fail gate 2 (no structured column survives). Make those terms a hard negative filter.
2. **Proper nouns are not text.** Athlete, university, city and person names pass
   `_is_text_feature` on uniqueness while carrying almost no signal E5 can exploit. A domain whose
   only string column is a proper-noun identifier should be scored down at T0, before download.
3. **Rank by within-query cluster density.** `board game` returned 9 distinct datasets of the same
   shape; `craft cocktails` returned 0. A dense cluster means an active community, cleaner files,
   and — for the Outstanding scope — several near-siblings that can be curated with one spec.
4. **One query per call.** Batching 5-6 queries into one `search()` let the strongest terms
   monopolise the merged, deduplicated result list; the weak queries returned nothing at all
   rather than a few weak hits. Search per query and merge afterwards.
5. **Size is a weak proxy for row count and a bad one for cost.** LEGO looked like 12k rows at
   4.5MB and was 744 products; Sephora is 9k genuine rows in a comparable file. The row count and
   the duplicate-key check both have to happen after download.
6. **Add a duplicate-key gate to T1.** `n_rows` vs. `nunique` of the most identifier-like column
   caught the LEGO 744x21 replication, which nothing in the current `profile_frame` would flag.
   This is a cheap, high-value addition.
7. **Check the target's `object`-dtype siblings before rejecting a dataset.** Two candidates lost
   their entire numeric block to string formatting (`"1,234"`, `"19M"`, `"$4.99"`), not to any
   real property of the data.

---

## 5. Suggested next actions

1. **T2 on Board games** — `cols_to_drop: [ID, "BGG Rank"]`, target `Rating Average`, trim the
   7 rows with |z| > 5. Highest expected Δ_Joint: `Mechanics` carries information the eight
   numeric columns provably do not.
2. **T2 on Sephora** in parallel — it is the cheapest to encode by a wide margin (12.8x under a
   512 cap, every column <=40 tokens), so it is nearly free to screen. Run the `rating == 0`
   keep/filter variants as two separate target-shortlist entries.
3. **Fragrantica third.** Structurally the most interesting (categorical accords vs. full note
   text is a clean "does the text add anything" contrast) but the 0.277 target std is a real
   risk to delta magnitude.
4. **Unexplored lead:** trading-card catalogues (`adampq/pokemon-tcg-all-cards-1999-2023`,
   `patrickgendotti/mtg-all-cards`, `earlarredondo/magic-the-gathering-standrd-cardsjul-2012present`).
   Card rules text is short, dense, and mechanically tied to numeric power/toughness/cost stats —
   the best remaining shape match, and untouched by MulTaBench.
