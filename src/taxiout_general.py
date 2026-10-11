#!/usr/bin/env python3
# %% [markdown]
# # PRC Data Challenge 2026: generalist taxi-out model (motivated-ostrich)
#
# One self-contained pipeline, designed for the blind final phase. Every
# design choice is justified either by a structural property of the data or
# by out-of-time validation on 2025. Nothing is selected on the public
# leaderboard.
#
# Structure of the model
#
# 1. **Feasibility of the "block = scheduled" corruption.** In the 2025
#    training data, |BLOCK - LOBT| <= 3606 s holds for 100 % of matched
#    departures (2,062,251 rows): it is the window used to match airport
#    movements to NM flights. A substituted block (BLOCK = SCHED) is therefore
#    only possible when |SCHED - LOBT| <= 3606 s. Outside that window the
#    corruption probability is exactly zero.
# 2. **Two-regime expectation.** A classifier estimates p = P(BLOCK = SCHED)
#    on feasible rows; a regressor predicts the normal taxi as a residual over
#    the NM off-block (MVT - AOBT_3), gated to plausible values. Under squared
#    loss the optimal prediction is the expectation
#    p**g * (MVT - SCHED) + (1 - p**g) * mu, with g chosen out of time.
# 3. **Unmatched movements.** Rows without NM record and without aircraft type
#    whose MVT - SCHED exceeds 3 h are mostly substituted blocks. Their rate is
#    estimated per airport on the training data (shrunk to the global rate)
#    and used as the mixture weight.
# 4. **Projection on the matching window.** Since BLOCK lies within +/- 3606 s
#    of LOBT, the taxi time lies within [MVT - LOBT - 3606, MVT - LOBT + 3606].
#    Every prediction is projected on that interval, which can only reduce the
#    error.
#
# Features: plan and clock deltas, airport/runway traffic around the take-off,
# off-block anchored queue, airport-level climate of the hour and of the day
# (NM delays, arrival taxi-in, arrival block substitution rate), previous
# arrival at the same stand. Departure labels (BLOCK, TAXITIME) are masked
# before any contextual feature is built, so training features are computed
# under exactly the conditions of the ranking file.
#
# Validation: the selection fold (train without April and October, validate on
# April + October 2025) sets the number of trees and the mixture exponent; the
# control fold (train without January and July, validate on January + July
# 2025) is scored once with those settings. The final model is trained on the
# twelve months of 2025.
#
# Runs on Kaggle (GPU T4/P100, XGBoost with device="cuda") or on CPU.

# %%
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# %% [markdown]
# ## Configuration

# %%
TEAM = "motivated-ostrich"
TARGET = "TAXITIME_SEC_mvt"

LOBT_WIN_S = 3606.0          # matching window BLOCK vs LOBT (100 % of 2025)
CORRUPT_TOL_S = 10.0         # |BLOCK - SCHED| <= 10 s : substituted block
TAXI_MIN_S, TAXI_MAX_S = 30.0, 7200.0
ABERRANT_S = 20000.0         # reporting threshold only (clean-lane RMSE)
RULE_DSCHED_S = 10800.0      # unmatched rows: MVT - SCHED above 3 h
RULE_MATCH_S = 120.0         # y = MVT - SCHED within 2 min
RULE_PRIOR_N = 20.0          # shrinkage of the per-airport rate
GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

FOLDS = {
    "sel": dict(val=[4, 10]),   # selection: trees, gamma
    "ctl": dict(val=[1, 7]),    # control: scored once
}

TZ = {
    "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}

TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]

CAT_COLS = ["ADEP_mvt", "apt_rwy", "apt_stand", "AIRCRAFT_TYPE_mvt", "ADES_mvt",
            "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt",
            "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt"]

XGB_COMMON = dict(tree_method="hist", max_bin=256, max_cat_to_onehot=1,
                  max_cat_threshold=64)
XGB_REG = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.04,
               max_depth=9, min_child_weight=50, subsample=0.85,
               colsample_bytree=0.8, reg_lambda=5.0, **XGB_COMMON)
XGB_CLF = dict(objective="binary:logistic", eval_metric="logloss", eta=0.04,
               max_depth=8, min_child_weight=100, subsample=0.85,
               colsample_bytree=0.8, reg_lambda=5.0, **XGB_COMMON)
MAX_ROUNDS = dict(clf=3000, reg_a=4000, reg_b=3000, cat_a=6000, cat_b=4000, apt_a=4000)
CAT_REG = dict(loss_function="RMSE", learning_rate=0.06, depth=8, l2_leaf_reg=5.0,
               verbose=False, allow_writing_files=False)
MIN_GAIN_S = 0.5            # a blend must beat XGBoost alone by this on fold sel
APT_MIN_ROWS = int(os.environ.get("APT_MIN_ROWS", 40000))         # per-airport regressor only above this size
EARLY_STOP = 100


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# %% [markdown]
# ## Loading

# %%
def find_data_dir(hint=None):
    cands = [hint] if hint else []
    cands += ["data", "/kaggle/input"]
    for c in cands:
        if not c:
            continue
        hits = glob.glob(os.path.join(c, "**", "training_*.parquet"), recursive=True)
        if hits:
            return Path(os.path.dirname(sorted(hits)[0]))
    raise FileNotFoundError("no training_*.parquet found (looked in %s)" % cands)


def find_file(data_dir, names):
    roots = [data_dir, data_dir.parent, Path("/kaggle/input")]
    for n in names:
        for r in roots:
            hits = glob.glob(str(r / "**" / n), recursive=True)
            if hits:
                return Path(sorted(hits)[0])
    return None


def parse_times(df):
    for c in TIME_COLS:
        if c not in df.columns:
            df[c] = pd.NaT
        s = df[c]
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, utc=True, errors="coerce")
        elif s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        df[c] = s.astype("datetime64[ns, UTC]")
    return df


def load_frame(paths, raw_only=False):
    cols = None
    if raw_only:
        import pyarrow.parquet as pq
        cols = [c for c in pq.ParquetFile(paths[0]).schema_arrow.names
                if c.endswith(("_mvt", "_flt"))]
    df = pd.concat((pd.read_parquet(p, columns=cols) for p in paths), ignore_index=True)
    df = parse_times(df)
    for c in ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt"]:
        df[c] = df[c].astype("object") if c in df.columns else None
    df["airport"] = np.where(df["PHASE_mvt"].eq("DEP"), df["ADEP_mvt"], df["ADES_mvt"])
    return df


# %% [markdown]
# ## Feature engineering
#
# All window statistics use one vectorised primitive: events and queries are
# encoded as `key * 1e10 + seconds`, sorted once, and windows are answered by
# binary search, with cumulative sums for means.

# %%
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
BIG = np.int64(10_000_000_000)


def secs(s):
    return ((s - EPOCH).dt.total_seconds()).to_numpy(dtype=float)


class Index:
    """Sorted (key, time) events with optional values, for window queries."""

    def __init__(self, keys, t, vals=None, codes=None):
        keys = np.asarray(keys, dtype=object)
        t = np.asarray(t, dtype=float)
        ok = np.isfinite(t) & pd.notna(keys)
        if vals is not None:
            vals = np.asarray(vals, dtype=float)
            ok &= np.isfinite(vals)
        self.codes = codes if codes is not None else {}
        for x in pd.unique(keys[ok]):
            self.codes.setdefault(x, len(self.codes))
        k = pd.Series(keys[ok]).map(self.codes).to_numpy(dtype=np.int64)
        comp = k * BIG + np.round(t[ok]).astype(np.int64)
        o = np.argsort(comp, kind="mergesort")
        self.comp = comp[o]
        if vals is not None:
            self.csum = np.concatenate([[0.0], np.cumsum(vals[ok][o])])
        else:
            self.csum = None

    def _q(self, keys, t):
        keys = np.asarray(keys, dtype=object)
        t = np.asarray(t, dtype=float)
        k = pd.Series(keys).map(self.codes).fillna(-1).to_numpy(dtype=np.int64)
        ok = (k >= 0) & np.isfinite(t)
        return k, ok

    def window(self, keys, t_lo, t_hi, lo_open=False, hi_open=False):
        """Count (and mean of values) of events with same key and time in the
        interval [t_lo, t_hi] (bounds optionally open)."""
        t_lo = np.asarray(t_lo, dtype=float)
        t_hi = np.asarray(t_hi, dtype=float)
        k, ok = self._q(keys, t_lo)
        ok &= np.isfinite(t_hi)
        cnt = np.full(len(k), np.nan)
        mean = np.full(len(k), np.nan)
        if ok.any():
            lo = k[ok] * BIG + np.round(t_lo[ok]).astype(np.int64)
            hi = k[ok] * BIG + np.round(t_hi[ok]).astype(np.int64)
            l = np.searchsorted(self.comp, lo, side="right" if lo_open else "left")
            r = np.searchsorted(self.comp, hi, side="left" if hi_open else "right")
            n = np.maximum(r - l, 0)
            cnt[ok] = n
            if self.csum is not None:
                s = self.csum[np.maximum(r, l)] - self.csum[l]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean[ok] = np.where(n > 0, s / n, np.nan)
        return cnt, mean

    def gaps(self, keys, t):
        """Seconds to the previous and next event (strictly) with same key."""
        k, ok = self._q(keys, t)
        prev = np.full(len(k), np.nan)
        nxt = np.full(len(k), np.nan)
        if ok.any() and len(self.comp):
            q = k[ok] * BIG + np.round(np.asarray(t, float)[ok]).astype(np.int64)
            l = np.searchsorted(self.comp, q, side="left")
            r = np.searchsorted(self.comp, q, side="right")
            pv = self.comp[np.clip(l - 1, 0, len(self.comp) - 1)]
            nv = self.comp[np.clip(r, 0, len(self.comp) - 1)]
            same_p = (l > 0) & (pv // BIG == k[ok])
            same_n = (r < len(self.comp)) & (nv // BIG == k[ok])
            prev[ok] = np.where(same_p, q - pv, np.nan)
            nxt[ok] = np.where(same_n, nv - q, np.nan)
        return prev, nxt


def local_clock(airport, ts):
    """Local hour (float) and local operational day (string, 4 am cut)."""
    hour = np.full(len(ts), np.nan)
    day = np.full(len(ts), None, dtype=object)
    ap = np.asarray(airport, dtype=object)
    for a, tz in TZ.items():
        m = ap == a
        if not m.any():
            continue
        loc = ts[m].dt.tz_convert(tz)
        hour[m] = (loc.dt.hour + loc.dt.minute / 60.0).to_numpy(dtype=float)
        day[m] = (loc - pd.Timedelta(hours=4)).dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
    return hour, day


def build_features(raw):
    """Departure feature table from a frame holding DEP and ARR movements.
    Departure labels are masked before any contextual computation."""
    t0 = time.time()
    df = raw.copy()
    dep = df["PHASE_mvt"].eq("DEP").to_numpy()
    arr = df["PHASE_mvt"].eq("ARR").to_numpy()
    y = df[TARGET].to_numpy(dtype=float).copy()
    blk_dep = df["BLOCK_TIME_UTC_mvt"].copy()
    # mask departure labels
    df.loc[dep, TARGET] = np.nan
    df.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT

    ap = df["airport"].to_numpy(dtype=object)
    rwy = (df["airport"].astype(str) + "|" + df["RUNWAY_mvt"].astype(str)).to_numpy(dtype=object)
    rwy[pd.isna(df["RUNWAY_mvt"]).to_numpy()] = None
    t_mvt = secs(df["MVT_TIME_UTC_mvt"])
    t_sch = secs(df["SCHED_TIME_UTC_mvt"])
    t_aob = secs(df["AOBT_3_flt"])
    t_lob = secs(df["LOBT_flt"])
    t_blk = secs(df["BLOCK_TIME_UTC_mvt"])          # arrivals only after masking
    taxi_nm = t_mvt - t_aob
    aob_ok = dep & (taxi_nm >= 0) & (taxi_nm <= TAXI_MAX_S)
    arr_taxi = np.where(arr, np.clip(df[TARGET].to_numpy(dtype=float), 0, 3600), np.nan)
    arr_corr = np.where(arr & np.isfinite(t_blk) & np.isfinite(t_sch),
                        (np.abs(t_blk - t_sch) <= CORRUPT_TOL_S).astype(float), np.nan)
    nm_delay = np.where(dep, np.clip(t_aob - t_sch, -3600, 14400), np.nan)

    D = np.where(dep)[0]
    out = pd.DataFrame(index=df.index[D])
    q_ap, q_rwy, q_t = ap[D], rwy[D], t_mvt[D]
    q_a = np.where(aob_ok[D], t_aob[D], np.nan)

    # ---- plan and clock deltas -------------------------------------------
    def dsec(a, b):
        return a[D] - b[D]
    out["d_sched"] = dsec(t_mvt, t_sch)
    out["d_aobt3"] = dsec(t_mvt, t_aob)
    out["d_lobt"] = dsec(t_mvt, t_lob)
    out["d_eobt1"] = dsec(t_mvt, secs(df["EOBT_1_flt"]))
    out["d_iobt"] = dsec(t_mvt, secs(df["IOBT_flt"]))
    out["nm_delay"] = dsec(t_aob, t_sch)
    out["lobt_minus_sched"] = dsec(t_lob, t_sch)
    out["aobt_minus_lobt"] = dsec(t_aob, t_lob)
    out["eobt_minus_lobt"] = out["d_lobt"] - out["d_eobt1"]
    out["sched_minus_lobt_abs"] = np.abs(out["lobt_minus_sched"])
    out["corrupt_feasible"] = (~(out["sched_minus_lobt_abs"] > LOBT_WIN_S)).astype(float)
    out["day_gap_sched"] = np.round(out["d_sched"] / 86400.0)
    out["base_aobt"] = np.where(aob_ok[D], out["d_aobt3"], np.nan)
    out["no_flight_id"] = df["FLIGHT_ID_mvt"].isna().to_numpy()[D].astype(float) \
        if "FLIGHT_ID_mvt" in df.columns else 0.0
    out["no_type"] = df["AIRCRAFT_TYPE_mvt"].isna().to_numpy()[D].astype(float)
    out["no_lobt"] = (~np.isfinite(t_lob[D])).astype(float)
    if "AIRCRAFT_TYPE_flt" in df.columns:
        a1 = df["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()[D]
        a2 = df["AIRCRAFT_TYPE_flt"].astype(str).to_numpy()[D]
        ok2 = df["AIRCRAFT_TYPE_mvt"].notna().to_numpy()[D] & df["AIRCRAFT_TYPE_flt"].notna().to_numpy()[D]
        out["type_mismatch"] = (ok2 & (a1 != a2)).astype(float)

    mvt_ts = df["MVT_TIME_UTC_mvt"].iloc[D]
    lh, lday = local_clock(q_ap, mvt_ts)
    out["hour_local"] = lh
    out["dow"] = mvt_ts.dt.dayofweek.to_numpy()
    doy = mvt_ts.dt.dayofyear.to_numpy(dtype=float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    sh, _ = local_clock(q_ap, df["SCHED_TIME_UTC_mvt"].iloc[D])
    out["sched_hour_local"] = sh

    # ---- traffic around the take-off -------------------------------------
    idx_dep = Index(ap[dep], t_mvt[dep])
    idx_arr = Index(ap[arr], t_mvt[arr], codes=idx_dep.codes)
    idx_dep_r = Index(rwy[dep], t_mvt[dep])
    idx_arr_r = Index(rwy[arr], t_mvt[arr], codes=idx_dep_r.codes)
    out["dep_pm15"] = idx_dep.window(q_ap, q_t - 900, q_t + 900)[0] - 1
    out["arr_pm15"] = idx_arr.window(q_ap, q_t - 900, q_t + 900)[0]
    out["dep_rwy_b10"] = idx_dep_r.window(q_rwy, q_t - 600, q_t, hi_open=True)[0]
    out["dep_rwy_a5"] = idx_dep_r.window(q_rwy, q_t, q_t + 300, lo_open=True)[0]
    out["arr_rwy_b10"] = idx_arr_r.window(q_rwy, q_t - 600, q_t)[0]
    out["gap_prev_rwy"], out["gap_next_rwy"] = idx_dep_r.gaps(q_rwy, q_t)

    # ---- off-block anchored queue (plausible AOBT only) ------------------
    idx_aob = Index(ap[aob_ok], t_aob[aob_ok])
    idx_mvt_ok = Index(ap[aob_ok], t_mvt[aob_ok], codes=idx_aob.codes)
    n_out = idx_aob.window(q_ap, q_a - TAXI_MAX_S, q_a)[0]
    n_gone = idx_mvt_ok.window(q_ap, q_a - TAXI_MAX_S, q_a)[0]
    out["q_taxiing_apt"] = n_out - n_gone - 1
    out["q_ahead_apt"] = idx_dep.window(q_ap, q_a, q_t, lo_open=True, hi_open=True)[0]
    out["q_ahead_rwy"] = idx_dep_r.window(q_rwy, q_a, q_t, lo_open=True, hi_open=True)[0]
    out["arr_during_rwy"] = idx_arr_r.window(q_rwy, q_a, q_t)[0]
    idx_tx = Index(ap[aob_ok], t_mvt[aob_ok], vals=taxi_nm[aob_ok])
    idx_tx_r = Index(rwy[aob_ok], t_mvt[aob_ok], vals=taxi_nm[aob_ok])
    out["recent_taxi_n_apt"], out["recent_taxi_apt"] = idx_tx.window(q_ap, q_a - 1200, q_a)
    out["recent_taxi_n_rwy"], out["recent_taxi_rwy"] = idx_tx_r.window(q_rwy, q_a - 1200, q_a)

    # ---- airport climate of the hour and of the day ----------------------
    idx_tx_h = Index(ap[aob_ok], t_mvt[aob_ok], vals=taxi_nm[aob_ok])
    out["taxi_nm_mean_pm60"] = idx_tx_h.window(q_ap, q_t - 3600, q_t + 3600)[1]
    okd = dep & np.isfinite(nm_delay)
    idx_dl = Index(ap[okd], t_mvt[okd], vals=nm_delay[okd])
    out["nm_delay_mean_pm60"] = idx_dl.window(q_ap, q_t - 3600, q_t + 3600)[1]
    oka = arr & np.isfinite(arr_taxi)
    idx_ti = Index(ap[oka], t_mvt[oka], vals=arr_taxi[oka])
    out["arr_n_pm30"], out["arr_taxiin_pm30"] = idx_ti.window(q_ap, q_t - 1800, q_t + 1800)
    okc = arr & np.isfinite(arr_corr)
    idx_ac = Index(ap[okc], t_mvt[okc], vals=arr_corr[okc])
    out["arr_corrupt_pm60"] = idx_ac.window(q_ap, q_t - 3600, q_t + 3600)[1]

    _, day_all = local_clock(ap, df["MVT_TIME_UTC_mvt"])
    dkey = pd.Series(ap.astype(str)) + "|" + pd.Series(day_all).astype(str)
    day_stats = pd.DataFrame({
        "k": dkey.to_numpy(),
        "taxi_nm": np.where(aob_ok, taxi_nm, np.nan),
        "nm_delay": nm_delay,
        "arr_taxi": arr_taxi,
        "arr_corr": arr_corr,
        "is_dep": dep.astype(float),
    }).groupby("k").agg(day_taxi_nm=("taxi_nm", "mean"), day_nm_delay=("nm_delay", "mean"),
                        day_arr_taxiin=("arr_taxi", "mean"), day_arr_corrupt=("arr_corr", "mean"),
                        day_n_dep=("is_dep", "sum"))
    qk = (pd.Series(q_ap.astype(str)) + "|" + pd.Series(lday).astype(str)).to_numpy()
    ds = day_stats.reindex(qk)
    for c in ds.columns:
        out[c] = ds[c].to_numpy()
    # ratio of the day to the airport norm is learnt by the trees from ADEP

    # ---- previous arrival at the same stand -------------------------------
    stand = (df["airport"].astype(str) + "|" + df["STAND_mvt"].astype(str)).to_numpy(dtype=object)
    stand[pd.isna(df["STAND_mvt"]).to_numpy()] = None
    A = arr & np.isfinite(t_blk) & pd.notna(stand)
    arr_tab = pd.DataFrame({"stand": stand[A], "t": t_blk[A], "t_arr": t_blk[A],
                            "arr_corr": arr_corr[A],
                            "arr_taxi": arr_taxi[A],
                            "arr_type": df["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()[A]})
    dep_tab = pd.DataFrame({"row": np.arange(len(D)), "stand": stand[D], "t": t_sch[D],
                            "dep_type": df["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()[D]})
    dep_tab = dep_tab[dep_tab["stand"].notna() & np.isfinite(dep_tab["t"])]
    for c in ("turnaround", "prev_arr_corrupt", "prev_arr_taxiin", "prev_same_type"):
        out[c] = np.nan
    if len(arr_tab) and len(dep_tab):
        m = pd.merge_asof(dep_tab.sort_values("t"), arr_tab.sort_values("t"), on="t",
                          by="stand", direction="backward", tolerance=36 * 3600.0)
        rows = m["row"].to_numpy()
        out.iloc[rows, out.columns.get_loc("turnaround")] = m["t"].to_numpy() - m["t_arr"].to_numpy()
        out.iloc[rows, out.columns.get_loc("prev_arr_corrupt")] = m["arr_corr"].to_numpy()
        out.iloc[rows, out.columns.get_loc("prev_arr_taxiin")] = m["arr_taxi"].to_numpy()
        same = np.where(m["arr_type"].notna(),
                        (m["arr_type"] == m["dep_type"]).astype(float), np.nan)
        out.iloc[rows, out.columns.get_loc("prev_same_type")] = same

    # ---- categoricals and identifiers ------------------------------------
    sub = df.iloc[D]
    out["ADEP_mvt"] = sub["ADEP_mvt"].to_numpy(dtype=object)
    out["apt_rwy"] = q_rwy
    out["apt_stand"] = stand[D]
    for c in ["AIRCRAFT_TYPE_mvt", "ADES_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
              "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt"]:
        out[c] = sub[c].astype("object").to_numpy() if c in sub.columns else None
    out["MVT_ID_mvt"] = sub["MVT_ID_mvt"].to_numpy()
    out["month"] = mvt_ts.dt.month.to_numpy()
    out["y"] = y[D]
    blk = secs(blk_dep.iloc[D])
    out["is_corrupt"] = np.where(np.isfinite(blk) & np.isfinite(t_sch[D]),
                                 (np.abs(blk - t_sch[D]) <= CORRUPT_TOL_S).astype(float), np.nan)
    out = out.reset_index(drop=True)
    for c in out.columns:
        if c not in CAT_COLS and c not in ("MVT_ID_mvt", "y") and out[c].dtype.kind in "fb":
            out[c] = out[c].astype("float32")
    log("features: %d departures, %d columns, %.0f s"
        % (len(out), out.shape[1], time.time() - t0))
    return out


NUM_FEATS = None  # filled in main() from the built table


def feature_list(tab):
    excl = {"MVT_ID_mvt", "y", "is_corrupt", "month"}
    return [c for c in tab.columns if c not in excl]


# %% [markdown]
# ## Models

# %%
def device():
    try:
        b = xgb.build_info()
        if b.get("USE_CUDA"):
            import subprocess
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            if r.returncode == 0 and "GPU" in r.stdout:
                return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


DEVICE = "cpu"


class Encoder:
    """Categories fitted on training data, unseen levels mapped to NaN, plus a
    stand-runway reference of the NM-side taxi (target-free)."""

    def fit(self, tab):
        self.cats = {}
        for c in CAT_COLS:
            vc = tab[c].astype("object").value_counts()
            self.cats[c] = list(vc[vc >= 5].index)
        ok = tab["base_aobt"].notna()
        g = tab[ok].groupby(["apt_stand", "apt_rwy"])["base_aobt"]
        self.ref = g.agg(ref_p20=lambda s: s.quantile(0.2), ref_p50="median", ref_n="size")
        return self

    def transform(self, tab, feats):
        X = tab[feats].copy()
        ref = self.ref.reindex(pd.MultiIndex.from_arrays([tab["apt_stand"], tab["apt_rwy"]]))
        for c in ref.columns:
            X[c] = ref[c].to_numpy()
        for c in CAT_COLS:
            if c in X.columns:
                s = X[c].astype("object")
                X[c] = pd.Categorical(s.where(s.isin(set(self.cats[c]))),
                                      categories=self.cats[c])
        for c in X.columns:
            if c not in CAT_COLS:
                X[c] = X[c].astype("float32")
        return X


def train(params, X, y, rounds, Xv=None, yv=None, seed=0, w=None):
    p = dict(params, seed=seed, device=DEVICE)
    dtr = xgb.DMatrix(X, label=y, weight=w, enable_categorical=True)
    ev = []
    if Xv is not None:
        dva = xgb.DMatrix(Xv, label=yv, enable_categorical=True)
        ev = [(dva, "val")]
    bst = xgb.train(p, dtr, num_boost_round=rounds, evals=ev,
                    early_stopping_rounds=EARLY_STOP if ev else None,
                    verbose_eval=False)
    best = bst.best_iteration + 1 if ev else rounds
    return bst, best


def predict(bst, X, n=None):
    d = xgb.DMatrix(X, enable_categorical=True)
    it = (0, n) if n else (0, 0)
    return bst.predict(d, iteration_range=it)


def masks(tab):
    y = tab["y"].to_numpy(dtype=float)
    corr = tab["is_corrupt"].to_numpy() == 1
    feas = tab["corrupt_feasible"].to_numpy() == 1
    clean = (~corr) & (y >= TAXI_MIN_S) & (y <= TAXI_MAX_S)
    has_a = tab["base_aobt"].notna().to_numpy()
    return y, corr, feas, clean, has_a


def cat_frame(X):
    """CatBoost wants categoricals as strings."""
    X = X.copy()
    cats = [c for c in X.columns if c in CAT_COLS]
    for c in cats:
        X[c] = X[c].astype("object").where(X[c].notna(), "NA").astype(str)
    return X, cats


def train_cat(X, y, rounds, Xv=None, yv=None, seed=0):
    from catboost import CatBoostRegressor, Pool
    Xc, cats = cat_frame(X)
    params = dict(CAT_REG, iterations=rounds, random_seed=seed,
                  task_type="GPU" if DEVICE == "cuda" else "CPU")
    m = CatBoostRegressor(**params)
    if Xv is not None:
        Xvc, _ = cat_frame(Xv)
        m.fit(Pool(Xc, y, cat_features=cats), eval_set=Pool(Xvc, yv, cat_features=cats),
              early_stopping_rounds=EARLY_STOP, use_best_model=True)
        best = m.get_best_iteration() + 1
    else:
        m.fit(Pool(Xc, y, cat_features=cats))
        best = rounds
    return m, best


def predict_cat(m, X):
    Xc, _ = cat_frame(X)
    return m.predict(Xc)


def fit_stage(tr, va, feats, rounds=None, seeds=(0,), sample=None, comps=("xgb",)):
    """Fit the classifier and the regressor families.

    Always trained (XGBoost, global): classifier, regressor A (residual over
    AOBT_3) and regressor B (no plausible AOBT_3). Optional families for
    regressors A and B: "apt" (one XGBoost regressor A per airport, B global)
    and "cat" (CatBoost, global). With `va`, early stopping chooses the number
    of trees of every model."""
    if sample:
        tr = tr.sample(frac=sample, random_state=0)
    enc = Encoder().fit(tr)
    Xtr = enc.transform(tr, feats)
    y, corr, feas, clean, has_a = masks(tr)
    res_a = y - np.nan_to_num(tr["base_aobt"].to_numpy())
    if va is not None:
        Xva = enc.transform(va, feats)
        yv, corrv, feasv, cleanv, has_av = masks(va)
        res_av = yv - np.nan_to_num(va["base_aobt"].to_numpy())

    # (name, family, params, train mask, target, val mask, val target)
    jobs = [("clf", "xgb", XGB_CLF, feas & np.isfinite(y), corr.astype(float),
             (feasv & np.isfinite(yv)) if va is not None else None,
             corrv.astype(float) if va is not None else None),
            ("reg_a", "xgb", XGB_REG, clean & has_a, res_a,
             (cleanv & has_av) if va is not None else None, res_av if va is not None else None),
            ("reg_b", "xgb", XGB_REG, clean, y,
             cleanv if va is not None else None, yv if va is not None else None)]
    if "cat" in comps:
        jobs += [("cat_a", "cat", None, clean & has_a, res_a,
                  (cleanv & has_av) if va is not None else None, res_av if va is not None else None),
                 ("cat_b", "cat", None, clean, y,
                  cleanv if va is not None else None, yv if va is not None else None)]
    if "apt" in comps:
        ap_tr = tr["ADEP_mvt"].to_numpy()
        ap_va = va["ADEP_mvt"].to_numpy() if va is not None else None
        for a in sorted(pd.unique(ap_tr[clean & has_a])):
            m_a = clean & has_a & (ap_tr == a)
            if m_a.sum() < APT_MIN_ROWS:
                continue
            jobs.append(("apt_a|" + a, "xgb", XGB_REG, m_a, res_a,
                         (cleanv & has_av & (ap_va == a)) if va is not None else None,
                         res_av if va is not None else None))

    models, best = {}, {}
    for name, fam, params, m, target, mv, tv in jobs:
        if rounds and name in rounds:
            n_rounds = rounds[name]
        else:
            n_rounds = MAX_ROUNDS.get(name.split("|")[0], MAX_ROUNDS["reg_a"])
        use_val = va is not None and mv is not None and mv.sum() > 100
        models[name] = []
        for s in seeds:
            t0 = time.time()
            if fam == "cat":
                fit = train_cat(Xtr[m], target[m], n_rounds,
                                Xva[mv] if use_val else None, tv[mv] if use_val else None, seed=s)
            else:
                fit = train(params, Xtr[m], target[m], n_rounds,
                            Xva[mv] if use_val else None, tv[mv] if use_val else None, seed=s)
            models[name].append(fit)
            log("  %-10s seed %d: %d rows, %d trees, %.0f s"
                % (name, s, int(m.sum()), fit[1], time.time() - t0))
        best[name] = int(np.mean([b for _, b in models[name]]))
    return dict(models=models, enc=enc, feats=feats), best


def stage_outputs(st, tab):
    """Classifier probability and one normal-taxi prediction per family."""
    X = st["enc"].transform(tab, st["feats"])
    M = st["models"]

    def avg(name, rows=None):
        Xs = X if rows is None else X[rows]
        if name.startswith("cat"):
            return np.mean([predict_cat(b, Xs) for b, _ in M[name]], axis=0)
        return np.mean([predict(b, Xs, n) for b, n in M[name]], axis=0)

    p = avg("clf")
    p = np.where(tab["corrupt_feasible"].to_numpy() == 1, p, 0.0)
    base = tab["base_aobt"].to_numpy(dtype=float)
    has = np.isfinite(base)

    def mu_of(ra, rb):
        return np.clip(np.where(has, base + ra, rb), TAXI_MIN_S, TAXI_MAX_S)

    ra_x, rb_x = avg("reg_a"), avg("reg_b")
    mus = {"xgb": mu_of(ra_x, rb_x)}
    apt_models = [k for k in M if k.startswith("apt_a|")]
    if apt_models:
        ra_apt = ra_x.copy()
        ap = tab["ADEP_mvt"].to_numpy()
        for k in apt_models:
            rows = ap == k.split("|")[1]
            if rows.any():
                ra_apt[rows] = avg(k, rows)
        mus["apt"] = mu_of(ra_apt, rb_x)
    if "cat_a" in M:
        mus["cat"] = mu_of(avg("cat_a"), avg("cat_b"))
    return p, mus


def blend(mus, weights):
    return sum(weights[k] * mus[k] for k in weights if weights[k] > 0)


def weight_grid(names, step=0.25):
    n = int(round(1 / step))
    out = []
    def rec(i, left, cur):
        if i == len(names) - 1:
            out.append(dict(cur, **{names[i]: left * step}))
            return
        for k in range(left + 1):
            rec(i + 1, left - k, dict(cur, **{names[i]: k * step}))
    rec(0, n, {})
    return out


# %% [markdown]
# ## Combination: mixture, unmatched rule, projection

# %%
def rule_mask(tab):
    return ((tab["no_flight_id"].to_numpy() == 1) & (tab["no_type"].to_numpy() == 1)
            & (tab["d_sched"].to_numpy() > RULE_DSCHED_S))


def fit_rule_rates(tr):
    m = rule_mask(tr)
    y = tr["y"].to_numpy(dtype=float)
    hit = np.abs(y - tr["d_sched"].to_numpy()) <= RULE_MATCH_S
    sub = pd.DataFrame({"ap": tr["ADEP_mvt"].to_numpy()[m], "hit": hit[m]})
    q0 = float(sub["hit"].mean()) if len(sub) else 0.0
    g = sub.groupby("ap")["hit"].agg(["sum", "size"])
    rates = ((g["sum"] + RULE_PRIOR_N * q0) / (g["size"] + RULE_PRIOR_N)).to_dict()
    return dict(q0=q0, rates=rates, n=int(m.sum()))


def combine(tab, p, mu, gamma, rule):
    ds = tab["d_sched"].to_numpy(dtype=float)
    ok = np.isfinite(ds)
    pg = np.where(ok, p ** gamma, 0.0)
    pred = pg * np.where(ok, ds, 0.0) + (1 - pg) * mu
    if rule is not None:
        m = rule_mask(tab)
        q = np.array([rule["rates"].get(a, rule["q0"]) for a in tab["ADEP_mvt"].to_numpy()[m]])
        pred[m] = q * ds[m] + (1 - q) * mu[m]
    dl = tab["d_lobt"].to_numpy(dtype=float)
    lo = np.where(np.isfinite(dl), dl - LOBT_WIN_S, -np.inf)
    hi = np.where(np.isfinite(dl), dl + LOBT_WIN_S, np.inf)
    return np.maximum(np.clip(pred, lo, hi), 0.0)


def report(tab, pred, title):
    y = tab["y"].to_numpy(dtype=float)
    se = (pred - y) ** 2
    cl = y <= ABERRANT_S
    log("%s: RMSE %.2f s | clean lane %.2f s | median |err| %.0f s"
        % (title, np.sqrt(se.mean()), np.sqrt(se[cl].mean()), np.median(np.abs(pred - y))))
    rows = []
    for (a, mo), g in tab.assign(se=se, cl=cl).groupby(["ADEP_mvt", "month"]):
        rows.append((a, mo, len(g), np.sqrt(g["se"].mean()), np.sqrt(g.loc[g["cl"], "se"].mean())))
    r = pd.DataFrame(rows, columns=["airport", "month", "n", "rmse", "rmse_clean"])
    print(r.pivot(index="airport", columns="month", values="rmse_clean").round(1).to_string())
    return float(np.sqrt(se.mean())), float(np.sqrt(se[cl].mean()))


# %% [markdown]
# ## Main

# %%
def main(argv=None):
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "out")
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the final model")
    ap.add_argument("--cv-seeds", type=int, default=1)
    ap.add_argument("--sample", type=float, default=None, help="subsample training rows (tests)")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--target", choices=["final", "public"], default="final",
                    help="final: final_ranking + final_submitting (blind ranking); "
                         "public: ranking + submitting (leaderboard)")
    ap.add_argument("--rounds-json", default=None, help="reuse rounds/gamma from a previous run")
    ap.add_argument("--train-cache", default=None, help="parquet of departures only (local tests)")
    ap.add_argument("--eta", type=float, default=None, help="learning rate override (quick tests)")
    ap.add_argument("--folds", nargs="+", default=["sel", "ctl"], choices=list(FOLDS))
    ap.add_argument("--cat-lr", type=float, default=None, help="CatBoost learning rate")
    ap.add_argument("--cat-max", type=int, default=None,
                    help="maximum CatBoost iterations for regressors A and B")
    ap.add_argument("--components", nargs="+", default=["xgb", "apt", "cat"],
                    choices=["xgb", "apt", "cat"],
                    help="regressor families tried on the selection fold")
    args, _ = ap.parse_known_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    DEVICE = device()
    if args.eta:
        XGB_REG["eta"] = XGB_CLF["eta"] = args.eta
    if args.cat_lr:
        CAT_REG["learning_rate"] = args.cat_lr
    if args.cat_max:
        MAX_ROUNDS["cat_a"] = MAX_ROUNDS["cat_b"] = args.cat_max
    log("XGBoost %s on %s" % (xgb.__version__, DEVICE))

    # ---- training features ------------------------------------------------
    if args.train_cache:
        raw = load_frame([args.train_cache], raw_only=True)
        data_dir = None
    else:
        data_dir = find_data_dir(args.data)
        files = sorted(glob.glob(str(data_dir / "training_*.parquet")))
        log("training files: %d in %s" % (len(files), data_dir))
        raw = load_frame(files)
    tab = build_features(raw)
    del raw
    tab = tab[tab["y"].notna()].reset_index(drop=True)
    feats = feature_list(tab)
    log("%d labelled departures, %d features" % (len(tab), len(feats)))
    rule = fit_rule_rates(tab)
    log("unmatched rule: %d candidates in 2025, global rate %.3f, per airport %s"
        % (rule["n"], rule["q0"], {k: round(v, 3) for k, v in rule["rates"].items()}))

    settings = dict(rounds=None, gamma=1.0, weights={"xgb": 1.0})
    if args.rounds_json and os.path.exists(args.rounds_json):
        settings.update(json.load(open(args.rounds_json)))
        log("settings reused: %s" % {k: v for k, v in settings.items() if k != "rounds"})

    # ---- out-of-time validation --------------------------------------------
    comps = tuple(args.components)
    if not args.skip_cv:
        cv_seeds = tuple(range(args.cv_seeds))
        res = {}
        for fold in args.folds:
            val = tab["month"].isin(FOLDS[fold]["val"]).to_numpy()
            tr, va = tab[~val], tab[val].reset_index(drop=True)
            log("fold %s: train %d, validation %d (months %s)"
                % (fold, len(tr), len(va), FOLDS[fold]["val"]))
            fr = fit_rule_rates(tr)
            y_va = va["y"].to_numpy(dtype=float)

            def rmse_of(mu, g):
                return float(np.sqrt(np.mean((combine(va, p, mu, g, fr) - y_va) ** 2)))

            if fold == "sel" or settings["rounds"] is None:
                if fold != "sel":
                    log("  no selection fold run: early stopping on this fold (diagnostic only)")
                st, best = fit_stage(tr, va, feats, seeds=cv_seeds, sample=args.sample,
                                     comps=comps)
                settings["rounds"] = best
                p, mus = stage_outputs(st, va)
                # joint choice of family weights and mixture exponent
                grid = []
                for w in weight_grid(list(mus)):
                    mu_w = blend(mus, w)
                    for g in GAMMAS:
                        grid.append((rmse_of(mu_w, g), g, w))
                grid.sort(key=lambda t: t[0])
                xgb_only = min(t for t in grid if t[2].get("xgb", 0) == 1.0)
                for k in mus:
                    alone = min(t for t in grid if t[2].get(k, 0) == 1.0)
                    log("  family %-4s alone: %.2f s (gamma %.1f)" % (k, alone[0], alone[1]))
                best_t = grid[0]
                if xgb_only[0] - best_t[0] < MIN_GAIN_S:
                    log("  blend gain %.2f s below %.1f s: XGBoost alone is kept"
                        % (xgb_only[0] - best_t[0], MIN_GAIN_S))
                    best_t = xgb_only
                settings["gamma"], settings["weights"] = best_t[1], best_t[2]
                log("selected on fold sel: weights %s, gamma %.1f -> %.2f s"
                    % (settings["weights"], settings["gamma"], best_t[0]))
            else:
                st, _ = fit_stage(tr, None, feats, rounds=settings["rounds"],
                                  seeds=cv_seeds, sample=args.sample, comps=comps)
                p, mus = stage_outputs(st, va)
            mu = blend(mus, settings["weights"])
            pred = combine(va, p, mu, settings["gamma"], fr)
            res[fold] = report(va, pred, "fold %s" % fold)
            # ablations, for the documentation
            cl = y_va <= ABERRANT_S
            for k in mus:
                pk = combine(va, p, mus[k], settings["gamma"], fr)
                log("  ablation, family %-4s alone: %.2f s | clean lane %.2f s"
                    % (k, np.sqrt(np.mean((pk - y_va) ** 2)),
                       np.sqrt(np.mean((pk - y_va)[cl] ** 2))))
            pg = p ** settings["gamma"]
            no_proj = np.maximum(np.where(np.isfinite(va["d_sched"]),
                                          pg * va["d_sched"].fillna(0) + (1 - pg) * mu, mu), 0)
            log("  ablation without rule and projection: %.2f s"
                % np.sqrt(np.mean((no_proj - y_va) ** 2)))
            log("  ablation without rule: %.2f s"
                % np.sqrt(np.mean((combine(va, p, mu, settings["gamma"], None) - y_va) ** 2)))
            oof = pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"], "ADEP_mvt": va["ADEP_mvt"],
                                "month": va["month"], "y": y_va, "pred": pred, "p": p,
                                "mu": mu, "d_sched": va["d_sched"]})
            for k in mus:
                oof["mu_" + k] = mus[k]
            oof.to_parquet(out_dir / ("oof_%s.parquet" % fold))
            if fold == "sel":
                imp = st["models"]["reg_a"][0][0].get_score(importance_type="gain")
                top = sorted(imp.items(), key=lambda kv: -kv[1])[:25]
                log("top features (regressor A, gain): %s" % [k for k, _ in top])
            del st
        settings["cv"] = res
        json.dump(settings, open(out_dir / "settings.json", "w"), indent=1)

    if args.skip_final or args.train_cache:
        log("done (no final submission)")
        return

    # ---- final model on the twelve months -----------------------------------
    # ranking file and template are always chosen as a pair
    if args.target == "final":
        rk_path = find_file(data_dir, ["final_ranking.parquet"])
        sub_path = find_file(data_dir, ["final_submitting.parquet", "final_submission.parquet"])
    else:
        rk_path = find_file(data_dir, ["ranking.parquet"])
        sub_path = find_file(data_dir, ["submitting.parquet"])
    if rk_path is None or sub_path is None:
        raise FileNotFoundError("target %s: ranking %s, template %s"
                                % (args.target, rk_path, sub_path))
    log("prediction set: %s, template: %s" % (rk_path, sub_path))
    is_final = args.target == "final"
    tpl_ids = pd.read_parquet(sub_path, columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
    rk_dep = pd.read_parquet(rk_path, columns=["MVT_ID_mvt", "PHASE_mvt"])
    rk_dep = rk_dep.loc[rk_dep["PHASE_mvt"].eq("DEP"), "MVT_ID_mvt"]
    miss = int((~tpl_ids.isin(rk_dep)).sum())
    log("template: %d ids, ranking departures: %d, template ids absent from ranking: %d"
        % (len(tpl_ids), len(rk_dep), miss))
    if miss > 0.001 * len(tpl_ids):
        raise ValueError("template and ranking do not match: check the file pair")
    rk = build_features(load_frame([rk_path]))
    log("prediction set months: %s" % rk["month"].value_counts().sort_index().to_dict())
    rounds = {k: int(v * 1.1) for k, v in (settings["rounds"] or MAX_ROUNDS).items()}
    used = tuple(k for k, w in settings["weights"].items() if w > 0)
    log("final model: families %s, weights %s, gamma %.1f"
        % (used, settings["weights"], settings["gamma"]))
    st, _ = fit_stage(tab, None, feats, rounds=rounds, seeds=tuple(range(args.seeds)),
                      comps=used)
    p, mus = stage_outputs(st, rk)
    mu = blend(mus, settings["weights"])
    pred = combine(rk, p, mu, settings["gamma"], rule)

    tpl = pd.read_parquet(sub_path)[["MVT_ID_mvt"]]
    outp = tpl.merge(pd.DataFrame({"MVT_ID_mvt": rk["MVT_ID_mvt"], TARGET: pred}),
                     on="MVT_ID_mvt", how="left")
    n_na = int(outp[TARGET].isna().sum())
    if n_na:
        log("WARNING: %d ids without prediction, filled with the median" % n_na)
        outp[TARGET] = outp[TARGET].fillna(float(np.nanmedian(pred)))
    assert len(outp) == len(tpl) and outp["MVT_ID_mvt"].is_unique
    name = "%s_final.parquet" % TEAM if is_final else "%s_general.parquet" % TEAM
    outp.to_parquet(out_dir / name, index=False)
    s = outp[TARGET]
    log("written %s: %d rows, median %.0f s, mean %.0f s, max %.0f s"
        % (name, len(outp), s.median(), s.mean(), s.max()))
    by_m = rk.assign(pred=pred).groupby(["month", "ADEP_mvt"])["pred"].median().unstack()
    print("median prediction by month and airport:\n", by_m.round(0).to_string())
    feat_cov = pd.DataFrame({"train": tab[feats].notna().mean(), "ranking": rk[feats].notna().mean()})
    gap = feat_cov[(feat_cov["train"] - feat_cov["ranking"]).abs() > 0.05]
    if len(gap):
        log("features whose fill rate differs by more than 5 points:\n%s" % gap.round(3))
    json.dump(settings, open(out_dir / "settings.json", "w"), indent=1)


if __name__ == "__main__":
    main()
