# Hypotheses: predicting pass/fail before paying for the grid

> **STATUS: HYPOTHESES, NOT FINDINGS.** Files `01`-`04` in this directory record things that were
> *measured*. Nothing in this file has been validated. Every entry states its falsification test,
> and any entry promoted to a finding must move out of this file with its evidence attached.

## Why this file exists

The deliverable being pursued is **a predictor, not a pile of accepted datasets**: a small set of
cheap, inspectable rules that say "this dataset will pass / will not pass" from observation, backed
by full-price curation runs used as *labels* rather than as acquisitions. Accepted datasets are then
a by-product of a working predictor.

## The label situation (read this before quoting any n)

Criterion-level labels — datasets where the actual pass rule ran on a complete grid:

| dataset | label | note |
|---|---|---|
| `REG_TEXT_HOUSES_VIETNAM_2024` | **pass** | 5 of 5 |
| `REG_TEXT_EDU_UDEMY_ACADEMY` | **pass** | 3 of 5, exactly at quorum |
| board games | **fail** | genuine Delta_Joint +0.047..+0.059 after restoring columns; died on TAR |
| anime popularity | **fail** | 80 cells, no gaps, one T4, `ft` at 10 epochs; 0 of 5 |

**n = 4.** MTG is **unlabelled** — Delta_Joint on 75/75 cells, Delta_Awareness never run. The other
seven entries in section 9 of `02-mining-method-rules.md` were rejected *before the criterion was
applied*, by the very guards a predictor would be tested against; counting them as correct negatives
assumes the conclusion, and several have no well-defined label at all (with an `Sl No` target,
"would this pass?" is not a question about the dataset).

This splits the work into two components that must never share a metric:

- **Validity guards** (sentinel, identifier, serial, Spearman leakage, derived-from-text, duplicate
  rows). Free, deterministic, a *precondition* rather than a classifier. The seven screen-time
  rejections are a genuine regression oracle. Report as "7 of 7 caught" — a test suite.
- **The pass predictor.** The research contribution. n = 4.

## The quantity worth predicting is Delta_Awareness, not Delta_Joint

Delta_Joint is measurable cheaply *and without bias*: frozen encoders make a subset of folds an
unbiased estimate (`03-methodological-findings.md`). A heuristic can therefore only approximate what
a short CPU run states exactly, so **predicting Delta_Joint is strictly worse than measuring it.**

Delta_Awareness is the opposite: R5.1 forbids cheapening it on either axis (epochs, folds), because
an under-trained adapter collapses the quantity toward zero *by construction*. There is no valid
cheap measurement, so a *correlate* measured at full fidelity is the only available shortcut.

## H1 — frozen-encoder headroom (the TAR heuristic)

**Claim.** Delta_Awareness is positive where the text carries target-relevant signal that frozen E5
fails to expose. Estimate that gap with two frozen, CPU-cheap, dimension-matched representations:

    gap = text_only(TF-IDF) - text_only(frozen E5)

Both are 30-dimensional: `--e5_model tf-idf` dispatches at
`multabench/baselines/preprocessing/text_embeddings.py:215` to `fit_text_encoders_skrub`, which
builds `skrub.StringEncoder(n_components=pca_components)`. Only the representation type differs.

**Mechanism.** `skrub.StringEncoder` defaults to `analyzer="char_wb"`, `ngram_range=(3, 4)` —
character 3-4 grams. So the contrast is *surface form* against *meaning*, not bag-of-words against
semantics. E5-small-v2 is trained with a contrastive retrieval objective whose job is to map
surface-different but semantically-similar strings to nearby vectors; collapsing surface identity is
that objective working as designed, not a bottleneck artifact. Any target driven by *which specific
entity this is* rather than *what this text means* has its signal actively compressed by E5 and
preserved by char n-grams.

Grounded in the two positives: Vietnam housing's `Address` carries district identity and its market
premium — E5 places "Quan 7, TP Ho Chi Minh" and "Quan 9, TP Ho Chi Minh" close together (both mean
"a district in HCMC") while the entire price signal lives in the 7-vs-9. Udemy's `course_instr` is
instructor identity: E5 encodes "a person's name"; the target cares which person.

**Why this predicts TAR specifically.** LoRA fine-tuning on target bins re-weights the encoder so
distinctions correlated with the target stop being collapsed. If price depends on district identity,
training on price bins pushes E5 to preserve Quan-7-vs-Quan-9. The diagnostic and the mechanism are
the same phenomenon measured two ways.

**Interpretation (one-sided — the asymmetry is part of the rule, not a caveat to it):**

| `text_only` (E5) | gap | reading | prediction |
|---|---|---|---|
| ~ 0 | any | text inert | ~ 0 (R4.2 rejects earlier) |
| mid | <= 0 | **ambiguous** — signal may be compositional and E5 already has it | none |
| mid | clearly > 0 | lexical identity signal E5 discards | **positive** |
| > 0.95 | any | saturated, no headroom | ~ 0 |

H1 nominates candidates for TAR success; it does not condemn low scorers.

**Why this does not violate R5.1.** H1 never measures Delta_Awareness. It measures a different
quantity at full fidelity for that quantity (frozen encoders, no shortcuts) and proposes it as a
correlate, validated against full-price labels. There is no self-fulfilling collapse.

**Falsification test — free, no GPU, runnable now.** The four labelled datasets already have
frozen-E5 `text_only` scores; only the TF-IDF half is missing, and it costs CPU seconds (no
tokenizer, no transformer). Compute `gap` for all four and check whether it ranks {Vietnam, Udemy}
above {board games, anime}. Run **both halves on the same machine** rather than differencing against
the stored grids — those came from mixed lanes (Udemy CPU-local, anime and board games Kaggle T4),
and cross-machine drift lands directly in the gap.

**Known risks.** (1) A positive gap may mean E5 is simply ill-suited to the text (non-English,
product codes, very short strings), and LoRA at rank 16 on 3 layers for 10 epochs may not fix that
either. (2) Pure restatement of structured columns also produces a positive gap — caught downstream,
since Delta_Joint is measured anyway.

## H2 — PCA bottleneck loss: REJECTED before testing

Proposed comparing target R^2 from the 30 PCA dims against the full 384-dim embedding, on the theory
that TAR helps by rotating discarded signal into the leading components. **Rejected**: the paper's
own PCA-dimension result is that the full embedding does not help much, so this diagnostic would
return ~0 for every candidate. A quantity with no variance across candidates cannot discriminate,
however sound its mechanism. It also has a weaker mechanistic link than H1 — it concerns what the
*downstream learner* can reach, while TAR changes the *encoder*.

## Delta_Joint: triage and validity, not prediction

Do not build a Delta_Joint predictor — it is cheap *and* unbiased when cheapened, so a heuristic can
only approximate what a short run states exactly. Build triage and a validity guard instead.

### The cost ladder (there is no tier between "no encoder" and "exact")

| tier | cost | what it buys |
|---|---|---|
| **0** | milliseconds, CSV only | cross-predictability (H3), verbatim overlap (R10.4), duplicate-row gate (R2.4), effective text cardinality (H5), target guards |
| **1** | seconds, **no encoder** | `no_text` alone: saturation guard (`no_text > 0.95`), degenerate-structured guard; TF-IDF `text_only` (H4) as a free by-product of H1's cheap arm |
| **2** | one frozen E5 encode, ~10 min CPU, ~40x cached after | `text_only`, `all`, **Delta_Joint exactly**, plus balance, lift, CV |

Once the encode is paid, `all` is nearly free — the per-string cache (R6.1) already holds every
embedding, which is why `hunt.py` orders `text_only` last as "nearly free, cache warm".

**Consequence, and it constrains the plan:** `R10.1` baseline balance and `R10.2` lift both need
`max(no_text, text_only)`, so they cost exactly as much as the answer. **They can never serve as
Delta_Joint triage.** They are Tier-2 by-products whose only possible job is predicting
Delta_Awareness — which also means they cannot be validated on the cheap high-n proving ground below
and must be spent against the scarce labels.

The saving from good triage is not "83 CPU-hours" for 500 candidates but ~4-5 days of **serial
wall-clock on the only local machine**, which blocks everything else.

### H3 — cross-predictability (build this first)

Delta_Joint > 0 mechanically requires **conditional** information in both directions:
`I(Y;T|S) > 0` *and* `I(Y;S|T) > 0`. If the text re-encodes the structured block, `all` ~ `no_text`
and the delta is zero by construction; if the structured block adds nothing over the text,
`all` ~ `text_only` and it is zero from the other side. Verbatim overlap is a weak proxy for the
first case only — it catches literal restatement, not templating or paraphrase.

The direct test is free and needs no encoder: TF-IDF the text, cluster into k groups, then fit a
small LightGBM predicting cluster membership from the structured columns. High accuracy means the
text is a re-encoding of the structured block — restatement, Delta_Joint ~ 0, and TAR has nothing to
sharpen either. Run it in reverse (predict structured columns from TF-IDF features) to catch the
redundant-structured case. Seconds per candidate. This is the quantitative form of R10.4 rather than
a correlate of it.

### H4 — TF-IDF `text_only` as a Tier-1 gate

Already built as H1's cheap arm, so it costs nothing extra, and it moves R4.2's inert-text gate from
Tier 2 to Tier 1. **One-sided**: TF-IDF ~ 0 does not prove E5 ~ 0, since E5 carries semantics
char n-grams cannot see. Hard-reject only on the extreme (negative R^2 — the daraz failure mode);
otherwise treat as ranking input, not a gate.

### H5 — effective text cardinality

Distinct text values after normalising case and whitespace. Boilerplate-dominated text is
categorical in disguise, `text_only` collapses, and nothing in the pipeline catches it today.
Related to R2.4 but applied to the text column rather than the row.

### The validity guard, which is nearly free and should be unconditional

Board games measured +0.0388 and the truth was -0.0005. The useful artifact is not a predictor of
Delta_Joint's value but a guard on the measurement. The auto spec and the maximal-structured spec
differ only in *structured* columns — same text columns, same target, therefore identical embeddings
and a warm cache — so the second Delta_Joint costs **one extra tabular refit, seconds**. Make
`build_spec` emit the pair, have every Tier-2 run emit both values, and treat disagreement as an
automatic reject. T2b stops being a stage anyone must remember.

### Evaluation: cost-recall, not regression

Score triage as a **cost-recall curve** — Delta_Joint-positives retained against Tier-2 runs avoided
— never as regression on Delta_Joint's value. Set the operating point asymmetrically: target
**recall >= 0.95** and report the spend saved there. A discarded good candidate is unrecoverable; a
passed-through bad one costs ten minutes.

### Why the cheap labels are worth gathering

Delta_Joint labels cost a fraction of a TAR label, so a triage rule can be tested at n ~ 50 rather
than n = 4. What that proves is **not** transfer to Delta_Awareness — the mechanisms differ — but
that this style of free CSV-derived feature carries real signal about downstream benchmark behaviour
at all, plus a dress rehearsal for the machinery (extraction, spec versioning, label discipline, the
rank test) before scarce labels are spent. The deliverable becomes two-tier: a **validated**
Delta_Joint triage rule at n ~ 50 with real statistics, and a **preliminary** Delta_Awareness
predictor at n ~ 18 with a rank test.

### On an LLM judge for R10.4

Deferred, not rejected. It does not leak (it never sees the label), but it is not *mechanical*, and
an API model is neither stable across runs nor reproducible for a grader — against the stated
criterion of logical, cheap, inspectable rules. H3 is the mechanical form and does not exist yet;
build it first. If H3 works the judge adds nothing; if H3 fails, there is a motivated reason to try
one and a genuine finding to report. If used, confine it to *description* — have it name the aspect
each column describes, then do the orthogonality arithmetic mechanically on top — pin the model,
version the prompt, log raw outputs, and admit it to the primary rule set only if it beats H3.

## Blocking defect: labels are undefined until the spec is frozen

Delta_Joint and Delta_Awareness are properties of **(dataset, spec)**, not of a dataset. Section 3.2
proves it: one CSV, one target, and the answer moves from +0.0388 to -0.0005 depending only on which
columns the spec drops. So "will this dataset pass?" is not well-posed — and not a well-defined
label — until the spec rule is frozen, and every later change to `auto_spec` silently relabels every
earlier candidate.

`curation_lab/screen/auto_spec.py:167` and `:176` apply `_is_junk` to the categorical and numeric
blocks, and `JUNK_TOKENS` contains `year` and `time` — i.e. board games' `Year Published` and
`Play Time`. **The mechanism section 3.2 identified is still live in the spec builder.** Line 166
applies the same filter to text candidates, which R3.2 explicitly permits; the two lines below it
are the violation.

Required order, before the first label is bought:

1. Correct `_is_junk` so it never touches the structured block.
2. Freeze and version `auto_spec`.
3. Stamp every label with the spec version; a later rule change invalidates labels taken under the
   old one.

## Evaluation design (for whenever labels start arriving)

At n ~ 18 an AUC is uninterpretable. Report a **one-sided exact Mann-Whitney U** on the heuristic
score, positives against negatives, plus the raw 2x2 written as fractions ("3 of 4 predicted-pass
actually passed"), never as percentages. With 3 positives among 18 a perfect ranking gives
p = 1/C(18,3) = 0.0012 and all-three-in-the-top-5 gives p = 10/816 = 0.012; with 2 positives among
16 the best achievable is 0.0083, and one misranking pushes past 0.02. **Target: >= 3 positives and
>= 12 negatives.**

Sample **enriched**, not randomly — at a ~1% base rate a random 16 returns zero positives. Then
state the consequence: an enriched sample supports "the heuristic discriminates", never "its
precision in the wild is X". Do not report a precision figure as a base-rate estimate.

Do not fit a model. Eight features on eighteen points is guaranteed overfitting; deliver
individually inspectable rules with a hand-specified ordering, each with its own 2x2, and say in the
writeup that this is a deliberate design choice.
