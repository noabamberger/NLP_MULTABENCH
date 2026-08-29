# RESUME — pick up here

Written at a session boundary (token exhaustion). Branch: **`curation-lab`**.
Everything below is on disk and committed. Nothing needs to be reconstructed from chat.

## Read these first, in order

1. `CLAUDE.md` — repo guide + this machine's environment constraints
2. `docs/superpowers/specs/2026-08-28-curation-lab-design.md` — the approved design
3. `docs/superpowers/plans/2026-08-28-phase1-pipeline-lock.md` — the Phase 1 plan
4. `docs/superpowers/plans/phase1-findings.md` — **all measured results and corrections**
5. `RESEARCH_NOTES.md` — dataset scouting notes (written by an agent; unreviewed)

## Non-negotiable environment rules

- Use **`.venv/Scripts/python.exe`**. System python has pandas 3.0.3 which CRASHES this repo
  (`tabstar` rejects the pandas-3 `str` dtype). System python is deliberately left at 3.0.3
  because another project (`tap-text-tabular`) pins it — do not "fix" it.
- Always set **`PYTHONIOENCODING=utf-8`** (console is cp1255; emoji model names crash otherwise).
- **`multabench/` is READ-ONLY.** All new code lives in `curation_lab/`.
- Tests: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/curation_lab/ -q`

## Status: what is TRUSTWORTHY vs UNVERIFIED

### Trustworthy (verified, tests passing at time of writing)

- **Phase 1 criterion validation.** Shipped 56x10 `pass_matrix.csv` re-derives with **0 mismatched
  cells**. `verdict()` correctly accepts a known 5-of-5 and rejects a known 0-of-5.
- **Runner protocol fidelity.** `no_text` reproduces the paper EXACTLY (0.83454303717305) for every
  model. Delta_Joint agrees within 0.012 across 4 models, all signs matching.
- **Frozen embedding cache** — proven bit-exact, 4 tests. ~40x speedup (600s -> 11s warm).
- Anchor dataset validated values (`MUL_TEXT_PRODUCT_SENTIMENT`, LightGBM, fold 0):
  `no_text=0.83454303717305`, `text_only=0.7658517388790819`, `all=0.8618478948517495`.

### UNVERIFIED — was in flight when the session ended

Three agents were mid-task. Their code is committed but **was never reviewed or fully tested by
the parent session**. Treat as draft:

| File | Agent | State |
|---|---|---|
| `curation_lab/runner/cache.py` (max_length parts) | opt-maxlen | agent had not reported |
| `tests/curation_lab/test_max_length.py` | opt-maxlen | unreviewed |
| `curation_lab/runner/tar_cache.py` | tar-optimizer | **likely incomplete** |
| `RESEARCH_NOTES.md` | scout-datasets | unreviewed |

**First action on resume: run the test suite.** It tells you which of the above is sound.

## Known blockers

1. **TabPFN-2.5 cannot run.** Its weights are in a gated HuggingFace repo;
   `tabpfn/browser_auth.py::_poll_for_token` calls `select.select([sys.stdin], ...)`, which fails
   on Windows for non-socket handles (`OSError: WinError 10038`) in any non-interactive context.
   Needs a ONE-TIME human step: accept the TabPFN-2.5 licence on huggingface.co (Prior-Labs), with
   `HF_TOKEN` already set in `.env`. **Blocks any real 3-of-5 verdict** — it is one of the five
   committee models.
2. **TAR (`ft`) needs a GPU or the CPU bypass.** `multabench/e5/e5_finetune.py:245` asserts
   `CUDA_VISIBLE_DEVICES` is set. `run_one(..., cpu_ft=True)` / `--cpu-ft` bypasses it safely.

## Key finding: max_length capping is NOT bit-exact

I claimed it was result-neutral; that was **wrong**. Masked padding is algebraically inert, but
changing the padded length reassociates float32 matmuls and moves embeddings ~1e-7. `opt-maxlen`
correctly shipped it **OFF by default** (`--max-length-cap` to enable). Use `atol=1e-5`, never
`array_equal`, when comparing across different caps. Fine for screening; do not enable for numbers
compared against the paper.

## Performance economics (why TAR is feasible on CPU at all)

| Optimization | Speedup | Status |
|---|---|---|
| Frozen embedding cache | ~40x | DONE, bit-exact |
| `max_length` cap, frozen encode | ~7x | draft (off by default) |
| `max_length` cap, TAR training loop | ~7x | **NOT DONE** |
| TAR encoder sharing (25 -> 10 fine-tunings) | 2.5x | **NOT DONE** (draft in `tar_cache.py`) |

Full detail in `phase1-findings.md` -> "Performance economics". Combined, these take one dataset's
TAR sweep from ~37-200 h to roughly 3-4 h on this laptop's 8 cores.

**TAR sharing rationale (verify before trusting):** fine-tuning is ONLY in the embedding step, not
end-to-end. The tuned encoder is a function of `(x_train, y_train, e5_train_kwargs)` only.
`split_to_val` is called without a `fold` arg (defaults to -1, deterministic), and `USE_VAL_SPLIT`
is True for LightGBM/CatBoost/TabM, False for both TabPFNs — so 2 distinct fine-tunings per fold,
not 5. **Key the cache on x_train content, not on model group**, so it stays correct if upstream
ever threads a real fold through. Prove same-group equality empirically before relying on it.

## In-flight background job (will be dead on resume — restart it)

Batch T1 profiling of 232 Kaggle candidates. Partial results survive:

- `results/candidates/t1_batch.csv` — incremental, one row per dataset profiled
- `results/candidates/t1_batch.log` — full log

To resume, re-run and it will reuse the kagglehub cache for anything already downloaded:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
from curation_lab.discover.kaggle_search import search, rank
from curation_lab.screen.batch_profile import run
QUERIES = ['job postings salary','app store apps rating','steam games price','book price',
 'recipe rating','used car listings price','product price description','hotel listings price',
 'board games rating','news headlines category','startup funding','anime rating',
 'video game sales','course rating students','perfume price','watches price',
 'laptop specs price','restaurant menu price','movies budget revenue','songs popularity spotify',
 'podcast ratings','sneakers price resale','airline reviews','university ranking',
 'charity donations','museums visitors','wine beer price','grocery products nutrition',
 'medicine drug reviews','furniture price']
refs=[x.ref for x in rank(search(QUERIES, max_size_mb=60, min_size_mb=0.1, per_query=15))]
run(refs, 'results/candidates/t1_batch.csv')"
```

Consider skipping refs already present in the CSV to avoid redoing work.

## Leading dataset candidate

**`jilkothari/finance-accounting-courses-udemy-13k-course`** — passed T1, not yet screened.

- 13,608 rows; `title` is **99.67% unique** (safely above the 80% text-detection threshold)
- Token cost: mean **15.9**, max 72 -> cap 72 (2.4x cheaper than the anchor's 38)
- 1 text column after dropping `url`, `created`, `published_time`
- Target **`avg_rating`** (REG), |z|max = 3.8 (under the |z|>5 danger threshold)
- **Leakage to drop: `avg_rating_recent`, `rating`** (near-copies of the target), plus the
  `*_price_string` columns which duplicate the numeric price fields
- Domain is genuinely novel — MulTaBench has no course/MOOC dataset

**Not confirmed.** No Delta_Joint or Delta_Awareness has been measured on ANY candidate. 36 of the
paper's own 56 pool datasets were rejected, mostly for joint-signal failure.

## Domain novelty warning

Slug exclusion is NOT sufficient. These domains already exist in MulTaBench under other slugs —
treat as non-novel: wine, zomato/restaurants, used cars, video game sales, data scientist salary,
job postings, book price/readability, rotten tomatoes, anime, airbnb, Montgomery/Vancouver
salaries, academic impact, mercari, women's clothing reviews, product sentiment, US accidents,
spotify genres, jigsaw toxicity, kickstarter, chocolate/coffee/beer/ramen ratings, laptop prices,
Amazon products.

## Recommended next steps on resume

1. Run the test suite; triage the three agents' draft work.
2. Finish whatever `tar-optimizer` left incomplete (TAR training cap + encoder sharing).
3. Restart the batch T1 job; rank the 232 candidates on the real gates.
4. Register the chosen candidate via the spec-based ingest (design section 3) and run the **T2
   frozen screen** (LightGBM + CatBoost, 3 frozen states, folds 0-1) for a real Delta_Joint.
5. Resolve the TabPFN-2.5 licence blocker before attempting any full verdict.

## Remote GPU box

`student@nlpgpu2025s-1010.westus.cloudapp.azure.com`, key `~/.ssh/multabench_remote` (installed and
working). Tesla M60 (7.5 GB, sm_52), 12 cores, 110 GB RAM, Ubuntu 22.04, driver 535.
**Time-boxed allocation that does NOT renew** — was ~2 h remaining and is likely expired.
Repo + anchor dataset were copied to `/home/student/MulTaBench`. A uv-managed Python 3.11 venv was
being built at `/home/student/mtb311` (every preinstalled conda env is 3.10, but the repo needs
3.11 for `enum.StrEnum`; `conda create` is blocked by unaccepted channel ToS — use uv).
torch 2.7.1+cu126 is confirmed CUDA-capable on that card.
Setup script: `/home/student/remote_setup2.sh`, log `/home/student/setup2.log`.

**`remote_login.env` in the repo root still contains a plaintext password.** It is gitignored, and
key auth works, so it can be deleted.

---

# Scout findings (agent `scout-datasets`, completed)

Full detail in `RESEARCH_NOTES.md`. Top backups, all T1-PASS but **none T2-screened**:

1. **`melissamonfared/board-games`** (BGG, 20,343 rows) — cleanest shape found: exactly
   **2 TEXT** (`Name`, `Mechanics`) vs **8 numeric + 1 categorical**. `Mechanics` is semantically
   dense and not recoverable from the numerics; `Complexity Average` (|corr| 0.481) gives
   `no_text` a genuine non-degenerate baseline, so a positive Delta_Joint would be a real
   measurement rather than an artifact of an empty structured condition. Target
   `Rating Average`, only 7 rows over |z|>5.
   **CRITICAL: drop `BGG Rank`** — monotone function of the target AND 100% unique, so it would
   also type as TEXT and leak into every condition.
2. **`raghadalharbi/all-products-available-on-sephora-website`** (9,168 rows) — T1 PASS after
   dropping `id, URL, details, how_to_use, ingredients, options, value_price`. Leaves 4 TEXT /
   8 NUM / 1 CAT. **Cheapest to encode: every column caps at <=40 tokens, 386k tokens/pass,
   12.8x under a 512 cap.** Nearly free to screen.
3. **`olgagmiufana1/fragrantica-com-fragrance-dataset`**, file **`fra_cleaned.csv`** (24,063 rows)
   — 5 TEXT (name + Top/Middle/Base note pyramid + Brand) vs `mainaccord1..5` as five categorical
   accord words: an unusually clean "does full note text beat its coarse categorical summary"
   contrast. Risk: target std only 0.277 and structured |corr| maxes at 0.109, so all four states
   may sit near R^2=0 and absolute deltas could be tiny.

Rejected with reasons: luxury watches (|z|max 136.9 on raw price; log-price would need a
`PROCESSING_FUNC`), Google Play Store (4 columns needing separate parsers, 1,181 duplicate app
names, known corrupted `Rating` row), Drugs.com.

## Corrections to my stated assumptions

- **The typing rule fires in the OPPOSITE direction from what I warned about.** I emphasised
  "short free text silently becomes categorical". The dominant real failure is the reverse: the
  `>=100 distinct` arm **promotes low-cardinality columns into TEXT** and blows the <=5 multimodal
  budget. Sephora's `brand` (324 distinct, 3.5% unique) and `category` (143 distinct, 1.6% unique)
  both type as TEXT — that alone is why the raw file has 9 text columns.

## Two BUGS in my own tooling, found by the scout

1. **`batch_profile.py::_read_any_csv` varies one axis at a time and will wrongly declare files
   unreadable.** `fra_cleaned.csv` needed `sep=";"` AND `encoding="latin-1"` AND `decimal=","`
   *simultaneously*; each fix only exposed the next error. The fallback must try combinations, not
   a single axis. **The batch T1 run in progress is affected — some "read_failed" rows are false
   negatives.**
2. **`profile_frame` has no duplicate-row gate.** `mterzolo/lego-sets` looks like 12,261 rows but
   is 744 products replicated across 21 countries — near-identical rows would land in both train
   and test in every fold. Add a duplicate-key gate to T1; cheap, and nothing else caught it.

## Search-strategy lessons (feed these into `discover/rules.py`)

- **Batching several queries into one `search()` call is counterproductive** — strong terms
  monopolise the merged result list and weak queries return zero on-topic hits. Two of the four
  scout rounds were wasted this way. Search **one query per call**.
- **"reviews" as a query term is an anti-pattern** — reliably returns single-column text corpora
  with no structured block, failing gate 2.
- Proper-noun domains (olympics, universities) pass the uniqueness test while carrying no signal
  E5 can exploit.
