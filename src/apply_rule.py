#!/usr/bin/env python3
"""
Applies the reconstructed-label rule to an existing submission.

Some movements were never matched to NM data. For those,
the airport filled BLOCK_TIME_UTC_mvt with the scheduled time, so that
the target is exactly MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt. Both
columns are available in the ranking file, so these rows are
recomputed instead of predicted.

The script touches nothing else: it reads a submission produced by
any pipeline and writes a corrected copy.

Examples:
  python apply_rule.py --ranking ./data/ranking.parquet --sub submission.parquet \\
      --out submission_hard.parquet --mode hard
  python apply_rule.py --ranking ./data/ranking.parquet --sub submission.parquet \\
      --out submission_soft.parquet --mode soft --p 0.731
  python apply_rule.py --ranking ./data/ranking.parquet --sub submission.parquet \\
      --dry-run
"""

import argparse
import sys

import numpy as np
import pandas as pd


def as_ns(s):
    v = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        v = v.dt.as_unit("ns")
    except (AttributeError, ValueError):
        pass
    return v


def main():
    ap = argparse.ArgumentParser(description="reconstructed-label rule")
    ap.add_argument("--ranking", default="./data/ranking.parquet")
    ap.add_argument("--sub", required=True, help="submission to correct")
    ap.add_argument("--out", help="output file (omitted with --dry-run)")
    ap.add_argument("--mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--p", type=float, default=0.731,
                    help="share of true defaults, used in soft mode")
    ap.add_argument("--thr", type=float, default=10800,
                    help="threshold on MVT_TIME - SCHED_TIME, in seconds")
    ap.add_argument("--airports", nargs="*", default=["LIRF"],
                    help="airports concerned; empty list for no filtering")
    ap.add_argument("--no-type-filter", action="store_true",
                    help="do not require an empty AIRCRAFT_TYPE_mvt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        sys.exit("--out is required unless --dry-run")

    rk = pd.read_parquet(args.ranking)
    dep = rk[rk["PHASE_mvt"].eq("DEP")].copy()
    mvt = as_ns(dep["MVT_TIME_UTC_mvt"])
    sch = as_ns(dep["SCHED_TIME_UTC_mvt"])
    dep["d_sched"] = (mvt - sch).dt.total_seconds()

    cond = dep["FLIGHT_ID_mvt"].isna() & (dep["d_sched"] > args.thr)
    if not args.no_type_filter:
        cond &= dep["AIRCRAFT_TYPE_mvt"].isna()
    if args.airports:
        cond &= dep["ADEP_mvt"].isin(args.airports)
    cand = dep[cond & dep["d_sched"].notna()]

    print("candidates: %d out of %d departures" % (len(cand), len(dep)))
    if len(cand):
        print("  d_sched: min %.0f s, median %.0f s, max %.0f s"
              % (cand["d_sched"].min(), cand["d_sched"].median(),
                 cand["d_sched"].max()))
        print("  per airport: %s" % dict(cand["ADEP_mvt"].value_counts()))

    sub = pd.read_parquet(args.sub)
    if "MVT_ID_mvt" not in sub.columns or "TAXITIME_SEC_mvt" not in sub.columns:
        sys.exit("the submission must contain MVT_ID_mvt and TAXITIME_SEC_mvt")
    before = sub["TAXITIME_SEC_mvt"].astype(float).copy()

    new_vals = pd.Series(cand["d_sched"].to_numpy(dtype=float),
                         index=cand["MVT_ID_mvt"].to_numpy())
    new_vals = new_vals[~new_vals.index.duplicated()]
    target = sub["MVT_ID_mvt"].map(new_vals)
    hit = target.notna()
    print("  of which present in the submission: %d" % int(hit.sum()))
    if int(hit.sum()) == 0:
        print("no row to correct, nothing to do")
        return

    if args.mode == "hard":
        updated = target[hit].astype(float)
    else:
        updated = args.p * target[hit].astype(float) + (1 - args.p) * before[hit]

    delta = (updated - before[hit]).abs()
    print("\nmode %s%s" % (args.mode, "" if args.mode == "hard" else " (p=%.3f)" % args.p))
    print("  before : median %9.0f s, max %10.0f s"
          % (before[hit].median(), before[hit].max()))
    print("  after  : median %9.0f s, max %10.0f s"
          % (updated.median(), updated.max()))
    print("  absolute change: median %9.0f s, max %10.0f s"
          % (delta.median(), delta.max()))

    if args.dry_run:
        print("\n--dry-run: no file written")
        return

    out = sub.copy()
    out.loc[hit, "TAXITIME_SEC_mvt"] = updated.to_numpy()
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].astype(float).clip(lower=0)
    if out["TAXITIME_SEC_mvt"].isna().any():
        sys.exit("missing values remain, output cancelled")
    if len(out) != len(sub):
        sys.exit("the number of rows changed, output cancelled")
    out.to_parquet(args.out, index=False)
    print("\nwritten: %s (%d rows, %d modified)" % (args.out, len(out), int(hit.sum())))


if __name__ == "__main__":
    main()
