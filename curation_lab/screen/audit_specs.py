"""Audit the auto-derived spec for candidates that already passed the Delta_Joint screen.

No model is fitted here. The screen already told us Delta_Joint looks positive; what
it cannot tell us is whether the number came from a *sane* spec. Two failure modes
seen so far, both from RESUME.md:

- The auto target picker chose a serial number (`Sl No`) or an id (`CustomerID`).
  The JUNK regex uses word boundaries, so it misses camelCase and spaced
  abbreviations. A serial-number target makes every score meaningless.
- The structured baseline is degenerate (R^2 <= 0, or `no_text` already saturated),
  so a positive Delta_Joint is an artifact rather than complementarity.

Printing rows/target/text/numeric per candidate is seconds of work and decides where
hours of GPU go.

Usage:
    python -m curation_lab.screen.audit_specs --refs a/b c/d
    python -m curation_lab.screen.audit_specs --from-hunt results/candidates/hunt_full.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

from curation_lab.screen.auto_spec import build_spec
from curation_lab.screen.batch_profile import _read_any_csv

# Wider than auto_spec.JUNK on purpose: this catches what that regex misses, because
# here a false positive costs a human glance rather than a silently bogus target.
SUSPECT_TARGET = re.compile(
    r"(?:^|_|\b|(?<=[a-z]))(sl\s*no|s\s*no|serial|customer\s*id|cust\s*id|"
    r"[a-z]*id|no|num|number|count|rank|position|year)(?:$|_|\b|(?=[A-Z]))", re.I)


def audit_one(ref: str) -> dict:
    import kagglehub

    row = {"ref": ref, "rows": None, "target": None, "task": None, "transform": "",
           "n_text": None, "n_num": None, "n_cat": None,
           "text": None, "flags": "", "detail": ""}
    try:
        d = kagglehub.dataset_download(ref)
        csvs = [f for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
                if os.path.getsize(f) / 1e6 <= 200]
        if not csvs:
            row["flags"] = "no-csv"
            return row
        path = max(csvs, key=os.path.getsize)
        df, enc, err = _read_any_csv(path)
        if df is None:
            row["flags"] = "unreadable"
            row["detail"] = str(err)[:80]
            return row
        rk = {}
        if enc and enc != "utf-8":
            rk["encoding"] = enc.split("/")[0]
            if "sep=" in enc:
                rk["sep"] = enc.split("sep=")[1].strip("'\"")
        name = "REG_TEXT_AUDIT_" + "".join(c if c.isalnum() else "_" for c in ref.upper())[:40]
        spec, why = build_spec(df, name=name, csv_path=path, read_kwargs=rk)
        row["rows"] = len(df)
        row["detail"] = why[:110]
        if spec is None:
            row["flags"] = "no-spec"
            return row

        flags = []
        row.update(target=spec.target, task=spec.task, transform=spec.target_transform,
                   n_text=len(spec.text_cols), n_num=len(spec.numeric_cols),
                   n_cat=len(spec.categorical_cols), text=",".join(spec.text_cols)[:44])
        if SUSPECT_TARGET.search(str(spec.target)):
            flags.append("SUSPECT-TARGET")
        # A near-unique target is an identifier wearing a number's clothes.
        y = pd.to_numeric(df[spec.target], errors="coerce").dropna()
        if len(y) and y.nunique() / len(y) > 0.98:
            flags.append("TARGET-NEAR-UNIQUE")
        if len(df) < 1000:
            flags.append("SMALL")
        # Duplicate rows land in train and test together (the lego-sets trap).
        dup = float(df.duplicated().mean())
        if dup > 0.05:
            flags.append(f"DUPES-{dup:.0%}")
        if len(spec.text_cols) + len(spec.numeric_cols) + len(spec.categorical_cols) > 30:
            flags.append("WIDE")
        row["flags"] = " ".join(flags) or "ok"
    except Exception as e:
        row["flags"] = "error"
        row["detail"] = f"{type(e).__name__}: {str(e)[:70]}"
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refs", nargs="*", default=[])
    p.add_argument("--from-hunt", default=None,
                   help="hunt CSV; audits every row whose verdict contains PASS.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    refs = list(args.refs)
    if args.from_hunt:
        h = pd.read_csv(args.from_hunt, encoding="utf-8")
        h = h[h["verdict"].astype(str).str.contains("PASS", na=False)]
        refs += [r for r in h.sort_values("delta_joint", ascending=False)["ref"]
                 if r not in refs]
    if not refs:
        raise SystemExit("nothing to audit")

    rows = []
    for i, ref in enumerate(refs, 1):
        r = audit_one(ref)
        rows.append(r)
        print(f"[{i}/{len(refs)}] {r['flags']:<22} {str(r['target'])[:22]:<22} "
              f"rows={r['rows']} text={r['text']} :: {ref[:52]}", flush=True)
        if args.out:
            pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8")
    print()
    print(pd.DataFrame(rows)[["ref", "rows", "target", "transform", "n_text", "n_num", "n_cat",
                              "flags"]].to_string(index=False))


if __name__ == "__main__":
    main()
