"""Compare the same curation cells measured on two machines.

Written to answer "why was Joint Frozen on the GPU so far from the local one?".
The short answer was that it was not far at all, and the question came from a bad
comparison (a five-fold mean read against a single fold). This script exists so
that claim is checkable rather than asserted.

The useful output is the breakdown BY STATE. `no_text` never touches E5, so it is a
control: if it matches exactly, the loader, the curation, the split, the seeding and
the learner are all identical across the two machines and any divergence in the
text-bearing states has to come from the encoder.

Usage:
    python -m curation_lab.kaggle.compare_environments \\
        --cpu results/candidates/dj_property.csv \\
        --gpu results/candidates/dj_property_tar_all_ft.csv \\
              results/candidates/dj_property_tar_frozen.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

# The CPU runner writes emoji MODEL_NAMEs; the Kaggle notebook writes SHORT_NAMEs.
EMOJI_TO_SHORT = {
    "LightGBM 💡": "light",
    "CatBoost 😸": "cat",
    "TabM Ⓜ️": "tabm",
    "TabPFN-v2 🤯": "tabpfnv2",
    "TabPFN-v2p5 🇩🇪": "tabpfnv2p5",
}


def load_cpu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["model"] = df["model"].map(lambda m: EMOJI_TO_SHORT.get(m, m))
    df = df.rename(columns={"multimodal_state": "state", "test_score": "cpu"})
    return df[["model", "fold", "state", "cpu"]]


def load_gpu(paths: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p, encoding="utf-8") for p in paths], ignore_index=True)
    df = df.rename(columns={"score": "gpu"})
    df = df.drop_duplicates(subset=["model", "state", "fold"], keep="last")
    return df[["model", "fold", "state", "gpu"]]


def compare(cpu: pd.DataFrame, gpu: pd.DataFrame) -> pd.DataFrame:
    j = cpu.merge(gpu, on=["model", "fold", "state"])
    if j.empty:
        raise SystemExit("no overlapping (model, state, fold) cells to compare")
    j["diff"] = j["gpu"] - j["cpu"]
    j["adiff"] = j["diff"].abs()
    return j


def report(j: pd.DataFrame) -> None:
    print(f"{len(j)} overlapping cells\n")

    print("by state (a state that never touches E5 is the control):")
    by_state = j.groupby("state")["adiff"].agg(["count", "mean", "max"]).round(5)
    print(by_state.to_string(), "\n")

    print("max |difference| per model x state:")
    print(j.pivot_table(index="model", columns="state", values="adiff",
                        aggfunc="max").round(4).to_string(), "\n")

    print("per-model means over folds:")
    mm = j.groupby(["model", "state"])[["cpu", "gpu"]].mean().round(4)
    mm["delta"] = (mm["gpu"] - mm["cpu"]).round(4)
    print(mm.to_string(), "\n")

    print(f"overall: mean signed diff {j['diff'].mean():+.5f}   "
          f"max |diff| {j['adiff'].max():.5f}")
    exact = j[j["adiff"] == 0]
    if not exact.empty:
        cells = sorted(set(zip(exact["model"], exact["state"])))
        print(f"bit-identical across machines: {cells}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", required=True, help="Results CSV from the local runner.")
    p.add_argument("--gpu", required=True, nargs="+", help="Results CSV(s) from the notebook.")
    args = p.parse_args()
    report(compare(load_cpu(args.cpu), load_gpu(args.gpu)))


if __name__ == "__main__":
    main()
