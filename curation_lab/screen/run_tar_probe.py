"""CLI for a single TAR probe. Must be a real module, not stdin.

HF Trainer uses dataloader_num_workers>0, and Windows spawns those workers by
re-importing __main__ -- which fails with OSError(22) on '<stdin>' if the driver
was piped in as a heredoc.
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings


def main() -> None:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--all-score", type=float, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()

    from curation_lab.discover.kaggle_search import _load_env
    _load_env()
    import kagglehub

    from curation_lab.screen.auto_spec import build_spec
    from curation_lab.screen.batch_profile import _read_any_csv
    from curation_lab.screen.t3_tar import tar_probe

    d = kagglehub.dataset_download(args.ref)
    csv = max(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True), key=os.path.getsize)
    df, enc, _ = _read_any_csv(csv)
    rk = {} if enc == "utf-8" else {"encoding": enc.split("/")[0]}
    spec, why = build_spec(df, name=args.name, csv_path=csv, read_kwargs=rk)
    if spec is None:
        print(f"SPEC FAILED: {why}")
        return
    print(f"rows={len(df)} | {why}", flush=True)

    r = tar_probe(spec, all_score=args.all_score, fold=args.fold, epochs=args.epochs)
    verdict = "PASS" if r["delta_awareness"] > 0.001 else "FAIL"
    print(f"\n=== TAR probe {args.name} ===")
    print(f"  all={r['all']:.4f}  ft={r['ft']:.4f}  Delta_Awareness={r['delta_awareness']}  "
          f"({r['secs']}s, {r['epochs']} epochs)  -> {verdict}")


if __name__ == "__main__":
    main()
