"""T1 typing probe + encoding-cost profile for a candidate CSV.

Runs multabench's OWN detectors rather than reimplementing the heuristics, because
the pass/fail boundary is decided by them:

  - tabstar.preprocessing.feat_types.detect_numerical_features
  - multabench.preprocessing.feat_types.classify_semantic_features
    (_is_text_feature: >=100 distinct values OR >=80% unique ratio)
  - multabench.baselines.preprocessing.feature_types.detect_image_features

A column that looks like free text to a human but falls under those thresholds is
classified CATEGORICAL, lands in the structured condition, and quietly destroys the
text_only/Delta_Joint comparison. That is the single most common silent killer, so
it is checked first.

Also reports the encoding cost model found in Phase 1:
    cost ~ n_rows x n_text_cols x mean_tokens
and the max_length cap (ceil(max_tokens/8)*8) that makes fine-tuning affordable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

MAX_MULTIMODAL = 5  # matches do_multabench_audit.MAX_MULTIMODAL and the no_pca guard


@dataclass
class Profile:
    name: str
    n_rows: int
    n_cols: int
    text_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    image_cols: list[str] = field(default_factory=list)
    token_stats: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def viable(self) -> bool:
        return not self.problems

    @property
    def structured_cols(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols


def profile_frame(df: pd.DataFrame, name: str = "candidate", target: str | None = None) -> Profile:
    """Classify columns exactly as the benchmark will, and flag disqualifiers."""
    from tabstar.preprocessing.feat_types import detect_numerical_features

    from multabench.baselines.preprocessing.feature_types import detect_image_features
    from multabench.preprocessing.feat_types import classify_semantic_features

    x = df.drop(columns=[target]) if target and target in df.columns else df
    image_cols = detect_image_features(x)
    x_no_dates = x.select_dtypes(exclude=["datetime", "datetimetz"])
    numeric = set(detect_numerical_features(x_no_dates))
    sem = classify_semantic_features(x=x, numerical_features=numeric | set(image_cols))

    p = Profile(
        name=name,
        n_rows=len(x),
        n_cols=len(x.columns),
        text_cols=sorted(c for c in sem.text_features if c not in image_cols),
        categorical_cols=sorted(sem.categorical_features),
        numeric_cols=sorted(numeric),
        image_cols=sorted(image_cols),
    )

    # Gate 1: at least one column must actually register as TEXT.
    if not p.text_cols:
        p.problems.append("no column detected as TEXT (all fell under the cardinality thresholds)")
    # Gate 2: something must survive as structured, or no_text is degenerate.
    if not p.structured_cols:
        p.problems.append("no structured column survives -> no_text state would be empty")
    # Gate 3: multimodal column budget.
    if len(p.text_cols) + len(p.image_cols) > MAX_MULTIMODAL:
        p.problems.append(f"{len(p.text_cols) + len(p.image_cols)} multimodal cols > {MAX_MULTIMODAL}")
    # Gate 4: enough rows for stable 5-fold estimates.
    if p.n_rows < 2000:
        p.problems.append(f"only {p.n_rows} rows (<2000 gives unstable 5-fold means)")
    return p


def token_profile(df: pd.DataFrame, text_cols: list[str], model_name: str = "intfloat/e5-small-v2") -> dict:
    """Token cost per text column, plus the safe max_length cap.

    Mirrors the exact string the pipeline builds: "passage: {col}: {val}".
    """
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    stats, total = {}, 0
    for col in text_cols:
        texts = ["passage: " + f"{col}: " + str(v).strip() for v in df[col].astype(str)]
        lens = [len(tok(t)["input_ids"]) for t in texts]
        s = pd.Series(lens)
        stats[col] = {
            "mean": round(float(s.mean()), 1),
            "p95": int(s.quantile(0.95)),
            "max": int(s.max()),
            "cap": max(8, math.ceil(int(s.max()) / 8) * 8),
        }
        total += float(s.sum())
    stats["_total_tokens_per_pass"] = int(total)
    stats["_speedup_vs_512"] = round(512 / max(1, max(v["cap"] for k, v in stats.items()
                                                      if isinstance(v, dict))), 1)
    return stats
