#!/usr/bin/env python3
"""
Figure 5: distance between the airport off-block and the last filed
off-block (LOBT), 2025 departures matched to the Network Manager.

The airport block time never leaves the +/- 3606 s window around LOBT,
whereas the Network Manager actual off-block (AOBT_3) does. The figure is
written as a vector PDF because LaTeX includes it; use --show to display it
instead.

  python paper/fig5_lobt_window.py --data data --out paper/figures
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WIN = 3606.0


def load(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "training_*.parquet")))
    cols = ["PHASE_mvt", "BLOCK_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]
    if not files:  # local fallback: a departures-only feature cache
        files = sorted(glob.glob(os.path.join(data_dir, "dep_features_cache*.parquet")))[:1]
        cols = cols[1:]
    d = pd.concat((pd.read_parquet(f, columns=cols) for f in files), ignore_index=True)
    if "PHASE_mvt" in d:
        d = d[d["PHASE_mvt"].eq("DEP")]
    t = {c: pd.to_datetime(d[c], utc=True) for c in cols if c != "PHASE_mvt"}
    ok = t["LOBT_flt"].notna() & t["BLOCK_TIME_UTC_mvt"].notna()
    blk = (t["BLOCK_TIME_UTC_mvt"] - t["LOBT_flt"]).dt.total_seconds()[ok]
    aob = (t["AOBT_3_flt"] - t["LOBT_flt"]).dt.total_seconds()[ok].dropna()
    return blk.to_numpy(), aob.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    blk, aob = load(a.data)
    inside = np.mean(np.abs(blk) <= WIN)
    print("matched departures %d, |BLOCK - LOBT| <= %.0f s: %.4f %%" % (len(blk), WIN, 100 * inside))
    print("AOBT_3 outside the window: %d rows" % int(np.sum(np.abs(aob) > WIN)))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bins = np.arange(-180, 181, 2.0)
    ax.hist(np.clip(aob / 60, -179, 179), bins=bins, histtype="step", lw=1.2,
            color="#8a8f98", label="NM actual off-block (AOBT_3)")
    ax.hist(np.clip(blk / 60, -179, 179), bins=bins, histtype="stepfilled",
            alpha=0.55, color="#2f5496", label="Airport off-block (label)")
    for x in (-WIN / 60, WIN / 60):
        ax.axvline(x, color="#b03a2e", ls="--", lw=1)
    ax.text(WIN / 60 + 3, ax.get_ylim()[1] * 0.3, "\u00b13606 s window", color="#b03a2e")
    ax.set_yscale("log")
    ax.set_xlabel("Off-block minus LOBT (minutes)")
    ax.set_ylabel("Departures per 2-min bin")
    ax.set_xlim(-180, 180)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    if a.show:
        plt.show()
    else:
        os.makedirs(a.out, exist_ok=True)
        fig.savefig(os.path.join(a.out, "fig5_lobt_window.pdf"))
        print("written", os.path.join(a.out, "fig5_lobt_window.pdf"))


if __name__ == "__main__":
    main()
