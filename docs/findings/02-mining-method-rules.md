# Rules for an autonomous MulTaBench dataset miner

Everything here was **measured**, not assumed. Each rule states its evidence, because several
contradict the intuition the project started with — including two cases where its own tooling
manufactured false positives. Carried over from `docs/archive/AUTONOMOUS_MINER_RULES.md`
(sections 0-10 substantially intact) with two updates: the case file's MTG row now reads the
finished 75/75 frozen grid rather than "on track", and board games' and anime's full-grid outcomes
are added since they postdate that document.

Target: a system that mines public sources and returns datasets satisfying, for >=3 of 5 learners,

```
Delta_Joint     = mean(all) - max(mean(no_text), mean(text_only)) > 0.001
Delta_Awareness = mean(ft)  - mean(all)                           > 0.001
```

---

## 0. Observed yield (use this to size the funnel)

| Stage | Count | Survival |
|---|---|---|
| T0 Kaggle search (30 queries) | 232 | -- |
| T1 typing profile | 232 read, 229 parsed | 99% |
| T1 "viable" per the gates | 120 | 52% |
| T1 after junk-column filter | 58 | 25% |
| after used-domain filter | 34 | 15% |
| **T2 Delta_Joint pass (1 model, 1 fold)** | **16** | **7%** |
| Full 5x5 grid attempted | 4 | -- |
| **Survived scrutiny** | **2** | **<1%** |

**Plan for roughly 1% end-to-end yield.** To deliver 5 accepted datasets, screen on the order of
500 candidates. The expensive tiers must be reached by very few.

---

## 1. Discovery (T0)

**R1.1 -- One query per API call.** Batching several queries into one search() lets strong terms
monopolise the merged result list; weak queries return zero on-topic hits. Two of four scouting
rounds were wasted this way.

**R1.2 -- Ban "reviews" as a query term.** It reliably returns single-column text corpora with no
structured block, failing the "structured column must survive" gate 100% of the time.

**R1.3 -- Domain exclusion, not just slug exclusion.** MulTaBench's 51 Kaggle slugs are not enough:
shrutimehta/zomato-restaurants-data and nikhilbhathi/data-scientist-salary-us-glassdoor both passed
slug filtering while duplicating existing MulTaBench domains. Maintain a domain blocklist: wine,
zomato/restaurants, used cars, video game sales, data scientist salary, job postings, book
price/readability, rotten tomatoes, anime, airbnb, Montgomery/Vancouver salaries, academic impact,
mercari, women's clothing reviews, product sentiment, US accidents, spotify genres, jigsaw
toxicity, kickstarter, chocolate/coffee/beer/ramen ratings, laptop prices, Amazon products.

**R1.4 -- Deprioritise proper-noun domains.** Olympics, universities, city/country listings pass the
uniqueness test but carry little signal E5 can exploit. Measured: top-women-chess-players reached
text_only R^2 of only 0.074 with Federation+Name as text.

**R1.5 -- Size window 0.1-60 MB** as a row-count proxy. Below ~50 KB there are too few rows for
stable 5-fold means; above ~60 MB usually means huge row counts (wasted -- training subsamples to
10k) or long documents (expensive to encode).

**R1.6 -- The Kaggle 2.x client uses snake_case** (total_bytes, download_count, vote_count,
usability_rating) and its server-side max_size filter is **not applied** -- filter client-side.

---

## 2. Ingestion

**R2.1 -- Parse fallbacks must be combinatorial, not one-axis-at-a-time.** fra_cleaned.csv needed
sep=";" AND encoding="latin-1" AND decimal="," **simultaneously**; each fix only exposed the next
error. A fallback varying one axis at a time declares such files unreadable. Our _read_any_csv had
this bug, so some read_failed rows in t1_batch.csv are false negatives.

**R2.2 -- Try latin-1/cp1252 before giving up.** melissamonfared/board-games fails as UTF-8.

**R2.3 -- Pick the file that PROFILES best, not the one that PARSES easiest.** In the fragrance
dataset, fra_perfumes.csv reads cleanly as UTF-8 (so an automated picker chooses it) yet profiles
with **zero numeric columns**: Rating Count arrives as object because of thousands separators, is
rejected as numeric, and its 2,913 distinct values then trip the text rule. The awkward
fra_cleaned.csv is the usable one.

**R2.4 -- Gate on duplicate rows.** mterzolo/lego-sets looks like 12,261 rows but is 744 products
replicated across 21 countries; near-identical rows would land in train and test in every fold.
Nothing else in the pipeline catches this.

---

## 3. Spec derivation -- where most false results are born

### 3.1 The typing rule fires in BOTH directions

multabench/preprocessing/feat_types.py::_is_text_feature calls a non-numeric column TEXT iff it has
**>=100 distinct values OR >=80% unique ratio**.

The documented risk is short free text silently becoming *categorical*. **The dominant failure in
practice is the opposite:** the >=100-distinct arm **promotes dates and IDs into TEXT**. Observed
false-positive text columns: date, Purchase Date, posting_date, PaymentDate, ReleaseDate, sku,
Item_Identifier, game_id, Menu_ID, drug_link, Sl No, CustomerID.

This is why 120 "viable" datasets became 58 after filtering -- **52% of T1 passes were false
positives.** It also blows the <=5 multimodal-column budget: Sephora's brand (324 distinct, only
3.5% unique) and category (143 distinct, 1.6% unique) both type as TEXT.

**R3.1** -- Filter candidate text columns by name pattern AND by content (parseable as a date? high
digit ratio? uniform length? monotone sequence?). Name matching alone misses camelCase (CustomerID)
and spaced abbreviations (Sl No).

### 3.2 THE BIGGEST TRAP: over-deleting structured columns MANUFACTURES Delta_Joint

**This invalidated one of the three recommended backup datasets — board games.**

A name-based junk filter dropping year|time|rank|id|count will delete **real structured
predictors**. For board games, Year Published and Play Time are not identifiers -- they are the two
strongest structured features. With them deleted, no_text is crippled and the text acts as a
**proxy for the deleted feature**, inventing a joint gain that vanishes when they return:

    Delta_Joint:  +0.0388  ->  +0.0229  ->  -0.0005
                  (as Year Published and Play Time are restored)

**R3.2 -- Never let a name-based filter remove a column from the STRUCTURED block.** Use name
patterns only to reject *text* candidates. Drop structured columns only on evidence: near-constant,
near-unique identifier, or leakage-correlated with the target.

**R3.3 -- Adversarially validate every positive Delta_Joint.** Re-run with the maximal structured
block (nothing dropped except proven leakage). If the delta collapses, the text was proxying for a
deleted feature and the result is an artifact. **Mandatory before reporting any dataset as passing.**

### 3.3 Target selection

Reject a candidate target if ANY of these hold -- each was observed:

| Failure | Example | Detection |
|---|---|---|
| Identifier | appid, CustomerID | name pattern + near-100% unique |
| Serial number | Sl No | monotone sequence, uniform step |
| **Sentinel-dominated** | Metacritic: **82% zeros** = missing-score | one value holds >20% of mass |
| Outlier-heavy | num_subscribers z=39.4, num_reviews z=49.2, watch price z=136.9 | abs(z)max > 5 |
| Uninterpretable | x33 | unusable in a writeup even if it passes |
| **Derived from its own text** | sentiment_polarity_score vs the review text | collapses to pure NLP -- the brief explicitly warns against this |

**R3.4** -- Sentinel detection is essential and cheap: if one value holds >20% of the target's mass,
treat it as missing and either drop those rows (verify >=2000 remain) or reject the target.
Metacritic's "regression" was really a has-a-score indicator.

**R3.5** -- Prefer a target with many distinct values, abs(z)max <= 5, and a plain-language meaning.

### 3.4 Leakage

Caught automatically by Spearman >= 0.95 against the target: avg_rating_recent / rating vs
avg_rating; original_Sentiment_Polarity; Seller Reviews. Needing a rank-aware check: BGG Rank
(monotone in the rating, not linear) -- also 100% unique, so it would both leak AND type as TEXT.
Also MRP (list price) beside a SellPrice target.

**R3.6 -- Use Spearman, not Pearson**, so monotone-but-nonlinear leaks (ranks) are caught.

---

## 4. Screening (T2) -- Delta_Joint

**R4.1 -- Exploit the criterion's shape to fail fast.** Since
Delta_Joint = all - max(no_text, text_only), if all <= no_text the dataset fails regardless of
text_only. Order runs **no_text (free, no encoder) -> all (the one expensive encode) -> text_only
(nearly free, cache warm)** and abort before text_only when already lost.

**R4.2 -- Require both unimodal baselines to be non-degenerate.** The single best predictor of a
real result:

- text_only ~ 0 means the text is inert; the "joint gain" is the structured model plus noise, and
  TAR has nothing to sharpen. (chess: 0.074)
- no_text >= ~0.95 means saturation, no headroom. (grocery 0.969, retail-store 0.964)
- Negative R^2 in any state means models are failing outright. (daraz perfumes: all = -0.009)

Both datasets that survived scrutiny have **balanced mid-range baselines**: Vietnam housing
(0.34 / 0.33 -> 0.63) and MTG cards (0.52 / 0.29 -> 0.59).

**R4.3 -- Noise calibration.** Measured on the paper's own data: per-state scores wander **+/-0.046**
across folds, while Delta_Joint reproduces within **+/-0.015 with no directional bias.** Therefore
**any Delta_Joint below ~0.02 is inside the noise band and carries no information.** Datasets
"passing" at +0.0005 (grocery) or +0.0043 (hotel) are noise.

**R4.4 -- A 1-model/1-fold screen is triage, not a verdict.** For Udemy the screen gave +0.2001 and
the full 5-fold LightGBM value was +0.209 (good agreement) -- but the **spread across the 5 models
was 0.07** (+0.136 to +0.208). A single-model screen says nothing about the 3-of-5 quorum.

**R4.5 -- Report significance, not a point estimate.** Over the 25 (model, fold) cells report mean,
std, positive-count and a one-sided t. Reference values from a strong pass (Vietnam housing):
mean=0.287, std=0.025, 25/25 positive, t=58.

---

## 5. TAR (T3) -- Delta_Awareness

**R5.1 -- THE screening-validity principle.** *A cheap screen is only valid where the cheapness does
not change the quantity being screened.*

- **True for Delta_Joint**: frozen encoders, so a subset of folds is an unbiased estimate.
- **FALSE for Delta_Awareness**: under-training the LoRA adapter leaves ft approximately equal to
  all, so the delta collapses to noise around zero **by construction**.

**R5.2 -- Never probe TAR at a token epoch budget.** A dataset was rejected at 1-of-5 using
epochs=2, then the identical cell was re-run at epochs=10:

| model, fold 0 | all | ft @ 2 ep | Delta @ 2 ep | ft @ 10 ep | Delta @ 10 ep |
|---|---|---|---|---|---|
| LightGBM | 0.4957 | 0.5056 | +0.0099 | 0.5279 | **+0.0322** |
| CatBoost | 0.5177 | 0.5282 | +0.0105 | 0.5299 | **+0.0122** |

A **fail at low epochs proves nothing**; only a pass is informative. This invalidated an entire
batch of TAR probes (Vietnam housing +0.0007, metacritic -0.0143, and others) -- they measured the
epoch budget, not the datasets. Use **epochs >= 10**, or the paper default of 50 with patience 3.

**R5.3 -- A single-fold Delta_Awareness screen is also invalid — same error, different axis.**
Board games' fold-0 screen (+0.0074 to +0.0163 across models) reversed on the full 5-fold grid
(-0.0003 to +0.0021, two models flipping sign); anime's fold-0 screen (+0.002 / +0.003) reversed to
0 of 5 on the full grid. The per-(model, fold) spread is sigma = 0.0063 over
[-0.0124, +0.0163] -- about 6x the delta threshold -- so one fold cannot resolve a criterion whose
threshold sits deep inside its noise band. Screening one fold cost two full grids. Screen
Delta_Awareness on all 5 folds, or at minimum 3.

**R5.4 -- TAR is where datasets actually die.** Delta_Joint is common (16 of 34 screened passed,
some enormously). Delta_Awareness is the discriminator: five of the eight candidates that reached a
Delta_Awareness screen in the second hunt round cleared Delta_Joint and then failed TAR. Budget
accordingly: cheap wide Delta_Joint screening, then deep TAR on very few — though see
[`03-methodological-findings.md`](03-methodological-findings.md) for why, now that GPU makes TAR
affordable, the ordering should be reversed.

---

## 6. Performance engineering (what makes CPU-only feasible)

| Optimization | Gain | Result-preserving? |
|---|---|---|
| Per-string frozen embedding cache | **~40x** (600s to 11s) | **Yes, bit-exact** (4 tests) |
| max_length cap, frozen encode | ~7x | **No** -- about 1e-7 drift |
| max_length cap, TAR training loop | **~40x** (333 s/step to 8 s/step) | Same 1e-7 caveat |
| TAR encoder sharing (25 to 10 fine-tunings) | 2.5x | Yes |

**R6.1 -- Cache embeddings per STRING, not per call.** The train split changes with the fold, so a
whole-list key misses every time. A per-string key lets all 5 folds and all 5 learners share one
encode of each unique text.

**R6.2 -- Compute max_length dynamically; never hardcode.** Upstream pads every text to 512 tokens.
Real caps measured: 40 (Udemy), 72 (product sentiment). Compute ceil(longest/8)*8 at runtime and
**assert zero truncation** -- a hardcoded cap would silently truncate a longer-texted dataset and
corrupt scores invisibly.

**R6.3 -- Capping is NOT bit-exact.** Padding is algebraically inert (masked out of the softmax), but
shrinking the padded sequence changes the matmul K dimension and BLAS reassociates its float32
sums, moving embeddings by about 1e-7. Fine for screening; not for numbers compared to the paper.
Compare with atol=1e-5, never array_equal.

**R6.4 -- The tuned encoder is a function of (x_train, y_train, e5_train_kwargs) -- NOT of the
tabular learner.** Fine-tuning happens only in the embedding step, never end-to-end (LightGBM and
CatBoost are not differentiable). Since split_to_val is called without a fold argument, the three
USE_VAL_SPLIT=True models share one x_train and the two TabPFNs share another: **2 distinct
fine-tunings per fold, not 5.** Key the cache on x_train **content**, never on the model grouping,
so it stays correct if upstream ever threads a real fold through.

**R6.5 -- TabPFN dominates cost on large datasets.** On MTG (10k train) an all-state TabPFNv2 cell
takes **about 36 minutes** on CPU vs seconds for LightGBM. Schedule TabPFN last; it is the binding
constraint on grid wall-clock.

---

## 7. Environment traps (all cost real debugging time)

- **pandas must be 2.3.3.** Under pandas 3.x, string columns get the new str dtype and
  tabstar.preprocessing.feat_types.is_numerical_feature raises "Unsupported dtype str", taking down
  all feature detection.
- **PYTHONIOENCODING=utf-8 always.** Model names contain emoji; a cp1255 console raises
  UnicodeEncodeError on a bare print.
- **wandb is an import-only dependency.** evaluate.py:17 imports multabench.utils.logging, which
  does "import wandb" at module level. The package is required; credentials are not, as long as
  wandb_run() is never called.
- **Python 3.11+ required** -- multabench/datasets/multimodal.py uses enum.StrEnum.
- **TabPFN-2.5's blocker was a Prior Labs API key, not HuggingFace gating** — see
  [`04-environment-and-performance.md`](04-environment-and-performance.md) for the full diagnosis
  and why the wrong explanation survived for weeks.
- **TAR asserts a GPU.** multabench/e5/e5_finetune.py:245 asserts CUDA_VISIBLE_DEVICES is in
  os.environ ("Single GPU only"). It checks only presence, so setting it plus passing an explicit
  torch.device("cpu") runs the path on CPU.
- **Register datasets by injecting into CURATIONS at runtime.** Do not add files to
  multabench/datasets/annotated/ -- it is auto-imported and **re-raises on failure**, so one bad
  generated file breaks dataset loading for everything.
- **HF Trainer on Windows**: dataloader_num_workers greater than 0 spawns workers by re-importing
  __main__, which dies with OSError(22) when the driver was piped in. Driver scripts must be real
  modules, not shell heredocs.

---

## 8. Reference architecture for the miner

    T0  discover     one query per call; domain blocklist; size window       free
    T1  profile      combinatorial parse fallbacks; duplicate-row gate;      seconds
                     CONTENT-based text/date/ID classification;
                     target screen (sentinel, outlier, ID, derived-from-text);
                     Spearman leakage; token-cost estimate
    T2  screen       no_text, then all, abort if all beats nothing,          minutes
                     then text_only; 1 model, 1-2 folds;
                     require non-degenerate baselines
    T2b adversarial  re-run with MAXIMAL structured block; discard if the    minutes
        validate     delta collapses.  ** MANDATORY **
    T3  grid         5 models x 3 frozen states x 5 folds; per-model         hours
                     Delta_Joint plus significance over 25 cells
    T4  TAR          epochs >= 10 ONLY; encoder sharing; TabPFN last         hours-days

**The single highest-value addition over a naive pipeline is T2b.** Without it a name-based junk
filter silently manufactures passing datasets -- exactly what happened with board games, caught
only because someone restored the deleted columns and watched the delta collapse to -0.0005.

---

## 9. Case file (ground truth for regression-testing a miner)

| dataset | Delta_Joint | Delta_Awareness | outcome |
|---|---|---|---|
| nguyentiennhan/vietnam-housing-dataset-2024 | +0.249..+0.324, t=58, 25/25 | +0.001..+0.015, 4-5 of 5 real margin | **ACCEPTED, 5 of 5** |
| mariahalshiekh/udemy-course-academy-teaching | +0.136..+0.209 | -0.007..+0.016, 3 of 5 | **ACCEPTED, 3 of 5** |
| douglascampospires/mtg-all-cards (log10 USD) | +0.050..+0.075, 75/75 cells | unmeasured | **NOT ACCEPTED — needs a TAR grid at epochs>=10, >=3 folds** |
| melissamonfared/board-games | +0.0388 becomes -0.0005 (junk-filter artifact); complete five-model grid +0.047..+0.059 Delta_Joint | 2 of 5 nominal, 1 of 5 counting the float knife-edge honestly | **REJECTED twice over** — artifact, then genuine Delta_Awareness failure. Completing the committee with TabPFN-2.5 (largest Delta_Joint at +0.059, Delta_Awareness exactly 0.000) did not rescue it. |
| douglascampospires-adjacent anime popularity | +0.031..+0.037 | -0.002..0.000, negative on 3 of 4 measured models | **REJECTED, 0 of 5** |
| thedevastator metacritic recommendations | one-fold +0.040 / -0.003 — but `all` is 1 fold against 5-fold baselines whose `no_text` spans 0.118-0.279, so CatBoost's sign is a fold-mismatch artifact, not a measurement | partial grid (52 rows, no `ft`) | **REJECTED at the target** — 82% sentinel zeros |
| tolstoyjustin/kerala-bevco-liquor-price-list | +0.136 | -- | invalid -- target was "Sl No" |
| muhammadaqeelkabir/steam-games-dataset | +0.024 | -- | invalid -- target was "appid" |
| neomatrix369/google-play-store-apps-extended | +0.028 | -- | rejected -- target derived from its own text |
| nomanmunir/daraz-perfumes | +0.037 | -- | rejected -- all-state R^2 negative |
| rrokon/global-grocery-nutrition-2025 | +0.0005 | -- | noise (baseline saturated at 0.969) |

A miner should reproduce every one of these verdicts, including the rejections.

---

## 10. POSITIVE signatures -- what a miner should SEEK

Sections 1-9 are mostly rejection rules. Those keep a miner honest but will not, on their own,
find anything. These are the measured properties of the datasets that actually worked, and they
are far more discriminative than Delta_Joint magnitude alone. Everything below carries over
unchanged from the original — it is measured, not inferred.

### 10.1 The measured signature of a real pass

| signal | Vietnam housing (strong) | MTG cards (solid) | interpretation |
|---|---|---|---|
| **Lift** = all / best baseline | **1.73x - 1.91x** | 1.11x - 1.16x | joint nearly doubles the better modality |
| **Baseline balance** = mean abs(no_text - text_only) | **0.016** | 0.230 | both modalities independently informative |
| **CV of delta** = std/mean over 25 cells | **0.086** | 0.143 | consistent across models AND folds |
| absolute R^2 range | 0.31 - 0.68 | 0.26 - 0.60 | mid-range: headroom, models working |
| positive cells | 25/25 | 13/13 so far | no model or fold dissents |

**R10.1 -- Baseline balance is the strongest single positive signal.** Vietnam housing has
no_text 0.31-0.36 and text_only 0.31-0.35: the two modalities are almost exactly equally
informative, differing by 0.016 on average. That is the fingerprint of two genuinely independent
information channels. Rank candidates by small abs(no_text - text_only) with both values in the
0.2-0.7 band.

**R10.2 -- Use lift, not raw delta.** all / max(baseline) is scale-free, so it compares across
datasets with different intrinsic difficulty. A raw delta of +0.04 means something very different
at baseline 0.05 than at baseline 0.95. Vietnam's 1.8x lift is a far stronger claim than its
+0.29 absolute delta alone conveys.

**R10.3 -- Low coefficient of variation proves it is not luck.** CV = std/mean over the 25
(model, fold) cells. Vietnam 0.086 means every model and every fold sees essentially the same
effect. A dataset whose delta is real but fragile shows a high CV even with a positive mean --
that is the signature to distrust.

### 10.2 The underlying mechanism -- and how to search for it directly

Both winners share one structural property: **the text encodes a causal channel that no structured
column captures.**

- **Vietnam housing**: `Address` encodes *location premium* (street, ward, district, city). The
  structured block is *physical attributes* (Area, Frontage, Floors, Bedrooms, Bathrooms, direction,
  legal status, furniture). Location and physical size are orthogonal drivers of price, and neither
  is derivable from the other. The screening agent verified explicitly that **no structured column
  duplicates the address**.
- **MTG cards**: `CARD_TEXT` and `TYPE` encode *what the card does* (abilities, interactions). The
  structured block is *cost and scarcity* (CMC, editions, power, toughness, first-edition year,
  rarity, colour). Playability and scarcity are independent price drivers.

**R10.4 -- Seek orthogonal channels; reject restatement.** The productive question at T1 is not
"does this dataset have text?" but **"does the text describe a different aspect of the entity than
the structured columns do?"** A product description reading "3 bedrooms, 2 bathrooms" beside
`bedrooms` and `bathrooms` columns is a *restatement*: it is redundant, Delta_Joint will be near
zero, and TAR has nothing to add. This is testable cheaply, before any model runs, by checking
whether structured column values appear verbatim inside the text.

**R10.5 -- A single rich text column beats several thin ones.** Both winners use 1-2 text columns.
More columns multiply encoding cost linearly while usually adding redundancy -- Rotten Tomatoes in
the paper pool has 13 text columns, i.e. 13x the encode cost for one dataset.

### 10.3 Practical ranking function

For candidates surviving T1, rank by (highest first):

    score = lift_estimate
            * (1 - abs(no_text - text_only))        # balance bonus
            * in_band(no_text, 0.2, 0.7)            # headroom, not saturated
            * in_band(text_only, 0.2, 0.7)          # text is genuinely informative
            * orthogonality(text, structured)       # 10.4, verbatim-overlap check
            / n_text_columns                        # encoding cost

Estimated from a single cheap LightGBM screen, this ranks Vietnam housing above every other
candidate found -- correctly, and before any expensive grid was run.

### 10.4 What the winners did NOT need

Worth recording, because effort was wasted assuming otherwise:

- **Not a large delta at screen time.** MTG's screen looked ordinary; its grid value held up.
- **Not many rows.** Vietnam has 30k but training subsamples to 10k; the extra rows were unused.
- **Not a novel domain.** Vietnam housing overlaps four existing MulTaBench housing datasets and
  still produced the cleanest result in the pool. Novelty matters for the *writeup*, not for
  whether the criterion is met -- keep the two judgements separate.
