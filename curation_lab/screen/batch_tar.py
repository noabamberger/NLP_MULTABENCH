"""Batch TAR probes over T2 survivors, to find the 5 datasets the Outstanding track needs.

One ft run per candidate (LightGBM, fold 0) is enough to triage: a dataset whose
Delta_Awareness is negative on the cheapest model rarely recovers across the panel,
and each probe costs minutes rather than the hours a full sweep would.

Reads the hunt output so the `all` baseline is reused rather than recomputed.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
import warnings

import pandas as pd


def main() -> None:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--hunt-csv", default="results/candidates/hunt_full.csv")
    p.add_argument("--out", default="results/candidates/tar_probes.csv")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--min-dj", type=float, default=0.01)
    p.add_argument("--limit", type=int, default=8)
    args = p.parse_args()

    from curation_lab.discover.kaggle_search import _load_env
    _load_env()
    import kagglehub

    from curation_lab.screen.auto_spec import build_spec
    from curation_lab.screen.batch_profile import _read_any_csv
    from curation_lab.screen.t3_tar import tar_probe

    h = pd.read_csv(args.hunt_csv, encoding="utf-8")
    h = h[(h["verdict"] == "PASS") & (h["delta_joint"] >= args.min_dj)]
    h = h.sort_values("delta_joint", ascending=False).head(args.limit)
    print(f"TAR-probing {len(h)} T2 survivors (dj >= {args.min_dj})", flush=True)

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, encoding="utf-8")["ref"])

    rows = []
    for _, c in h.iterrows():
        ref = c["ref"]
        if ref in done:
            continue
        t0 = time.time()
        rec = {"ref": ref, "delta_joint": c["delta_joint"], "all": c["all"],
               "ft": None, "delta_awareness": None, "verdict": "", "secs": 0}
        try:
            d = kagglehub.dataset_download(ref)
            csv = max(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True),
                      key=os.path.getsize)
            df, enc, _ = _read_any_csv(csv)
            rk = {} if enc == "utf-8" else {"encoding": enc.split("/")[0]}
            name = "REG_TEXT_AUTO_" + "".join(ch if ch.isalnum() else "_"
                                              for ch in ref.upper())[:40]
            spec, why = build_spec(df, name=name, csv_path=csv, read_kwargs=rk)
            if spec is None:
                rec.update(verdict="skip")
            else:
                r = tar_probe(spec, all_score=float(c["all"]), fold=0, epochs=args.epochs)
                rec.update(ft=r["ft"], delta_awareness=r["delta_awareness"],
                           verdict="PASS" if r["delta_awareness"] > 0.001 else "FAIL")
        except Exception as e:
            rec.update(verdict="error", ft=f"{type(e).__name__}: {str(e)[:50]}")
        rec["secs"] = round(time.time() - t0, 1)
        rows.append(rec)
        print(f"  {rec['verdict']:<6} da={rec['delta_awareness']} dj={rec['delta_joint']} "
              f"{rec['secs']}s  {ref[:50]}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
