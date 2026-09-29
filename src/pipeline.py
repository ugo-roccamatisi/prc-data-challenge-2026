"""
PRC Data Challenge 2026 - Taxi-out v16 (recalibrated mixture, without Rome)
======================================================================

Status before this version:
  - Regime architecture (v5.1 / ARCH 0) validated: the single model
    loses 100 s, the body/tail decomposition gains nothing.
  - Calibration of gamma_c on the leaderboard: the public curve is parallel to
    the local curve (constant offset ~32 s). The Jan+Jul 2025 validation
    is therefore a RELIABLE proxy, and we can iterate locally.
  - Public score 347.6 s, top of the leaderboard 246 s. The regressor on
    clean rows is at 235 s: the gap lies entirely in the 5 % of
    pathological rows, and it comes from the features, not the architecture.

Two sources of information never exploited, both present in
the ranking file (only the DEPARTURES are masked there, arrivals are
complete, block time and taxi-in included):

  1. ARRIVAL CONTEXT: the taxi-in of arrivals at the same airport in
     a window around the takeoff is a proxy of congestion and
     weather CONTEMPORARY with the target (snow day, runway incident,
     peak). Computed identically on 2025 and on Jan/Jul 2026.
       arr_taxi_mean_win      mean (clipped) taxi-in within +/- 30 min
       arr_taxi_frac_long_win share of taxi-ins > 1200 s in the window
       arr_taxi_day_mean      mean taxi-in over the airport day

  2. STAND CHAINING: the aircraft leaving stand X is almost
     always the one that arrived there just before. Its in-block is known
     in the ranking file and physically bounds the off-block of the departure.
       turnaround_s        MVT_TIME - in-block of the previous arrival at
                           the same stand (UPPER bound of the taxi-out)
       prev_arr_taxi_in    taxi-in of that arrival: proxy of the
                           stand <-> runway distance seen by the same aircraft
       prev_same_type      same aircraft type (confidence of the matching)
       prev_same_operator  same operator
       sched_before_inblock 1 if SCHED is earlier than the previous in-block
                           (block = sched would then be physically
                           impossible: a corruption signature)
       bound_minus_aobt    turnaround_s - delta_mvt_aobt3_s: margin between
                           the physical bound and what NM sees

Everything comes from the provided data, nothing from the future (arrival before
the takeoff), nothing external.

v9 result: 371.1 s (+8.3 s), gain spread over all airports, but
CDG (579) and Rome (867) almost unchanged and the corruption classifier
identical (recall 10.9 %). The v9 features help the normal regime, not
detection.

  3. REPORTING OUTAGE (v10): corruption is a system-level event.
     Checked on 2025: departure corruption rate of 4 % when
     no arrival of the hour is corrupted, 23 to 30 % when more than
     30 % are (correlation 0.34 per airport-hour). Arrivals in the
     ranking file have their block time: this signal is observable in 2026.
       arr_corrupt_frac_win / _wide / _day_frac
       arr_anom_frac_win / _wide   (aberrant taxi-ins, CDG/Rome regime)

v10 result: 368.2 s (+2.9 s), Rome -33 s but the other airports
slightly degraded. Classifier unchanged. Real but diluted signal.

  4. ROME BRANCH (v11): see the comment block above
     rome_subset_mask. Half a dozen rows at ~87,000 s of taxi
     (block stamped with the scheduled date) carry about half of the
     MSE. They are concentrated in an identifiable subset (Rome,
     unmatched, takeoff on the next day) where the taxi-out is almost
     never a real taxi-out. Expected-value prediction per delay
     bucket, checking that the gain is present in each validation
     month.

v11 result: 348.4 s (+19.8 s), gain in both months, Rome 834 -> 715.
Breakdown of the remaining MSE (121,400): 55 % background noise of the normal
regime, 25 % in TWO CDG rows at 84,240 and 58,206 s without any
signature (block time of the previous day), 8 % Rome residual, 5.6 % block=sched.
Without the two CDG rows: 303 s. No big lever is left.

  5. QUEUE PER RUNWAY (v12): congestion was counted at the airport
     level; the queue is per runway. Takeoffs on the same
     runway before / after, intervals between consecutive takeoffs,
     landings on the same runway, share of departures on this runway.
     Targets the normal regime (55 % of the MSE).

  6. ENSEMBLING (v12): 5 seeds for the classifier and the regressors,
     mean of the predictions. Rounds frozen per seed for retraining.

v12 result: 346.4 s (+2.0 s), gain from ensembling alone (residual 229 -> 226).
The runway features do not appear in the top 15.

  7. PER-AIRCRAFT AND PER-FLIGHT SIGNALS (v13). Diagnostics on 2025:
     - corruption of the previous arrival at the same stand: 14.6 % vs
       4.5 %; previous AND next corrupted: 38.5 % (3,000 flights/year)
     - target encoding by flight number: H1 -> H2 persistence of 0.815,
       37 numbers above 80 % (precision 94 %), but the numbers above
       30 % carry only 10 % of the corrupted rows. Real but bounded
       gain: corruption is mostly diffuse.
     Features : prev_arr_corrupt, next_arr_corrupt, both_arr_corrupt,
     te_adep_flight.

v13 result: 345.2 s (+1.2 s). te_adep_flight 9th in the classifier, AUC
unchanged. Diffuse corruption, confirmed.

  8. UNIMPEDED TIME (v14), adapting the "domain physics" of the
     2025 winning report (mass estimation): taxi-out = incompressible
     time tied to the stand-runway geometry + additional congestion
     time. Percentiles 10 and 50 per (airport, stand, runway), out-of-
     fold, with hierarchical fallback, as explicit numerical features;
     and target of regressor B = excess over p50. Targets the normal regime
     (55 % of the MSE) and the rows without AOBT_3.

v14 result: 345.9 s, neutral (noise). unimpeded_p50 8th in regressor A,
residual 226.1 -> 225.4. The geometry was already captured by stand_runway.

  9. QUEUE AT OFF-BLOCK (v15). Literature review (Idris 2001,
     Ravizza 2014, Wang 2021, Yin 2024): the most predictive variable
     is the queue size at push-back and the recent mean taxi-out.
     All our congestion features were anchored on the TAKEOFF.
     Rebuilt here with AOBT_3 as off-block (not masked in the
     ranking file): N (already taxiing), Q (overtaking), mean taxi-out of the
     previous 20 min, push-back rate, landings during my
     taxi; per runway and per airport.

v15 result: 345.0 s local, 338.02 public (with Rome). No q_* feature
in the top 15: congestion brings nothing when the NM-side duration
is known.

Leaderboard: v10 (v13 pipeline WITHOUT Rome) 337.95, v11 (WITH) 340.40.
The Rome branch loses 2.45 s on 2026 despite +19.8 s locally. Removed.

 10. RECALIBRATED MIXTURE (v16). y = n + w (s - n). The weight w = p^gamma is
     a uniform recalibration; the cost of an error on w grows as
     (s - n)^2. We learn w = f(p, log(s - n)) on the validation
     predictions, with a small model monotone in p, without airport. Transfer
     January <-> July: +7.6 / +1.3 s. Two submissions: reference
     gamma and recalibrated.

Pipeline: ARCH 0 unchanged (corruption classifier + probabilistic
mixture + AOBT_3 residual regressor), sweep of gamma_c,
retraining with frozen rounds, submissions named per the convention.

Usage: python taxiout_v16.py
"""

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import lightgbm as lgb

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # repository root
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "submissions"
OUT_DIR.mkdir(exist_ok=True)

# The v4 cache is suitable (same base columns); the target encoding is
# recomputed at each run because it depends on the split.
CACHE_PATH = DATA_DIR / "dep_features_cache_v15.parquet"
USE_CACHE = True

TAXI_MIN_S = 30
TAXI_MAX_S = 2 * 3600
CORRUPT_TOL_S = 10

# Body / tail split threshold (seconds). 3600 s: beyond that, a taxi
# is no longer an ordinary taxi. 0.35 % of departures at CDG, but 63 % of
# the variance of that airport.
TAIL_THR = 3600

CONGESTION_WINDOW_MIN = 15
VAL_MONTHS = [1, 7]

# Arrival context
ARR_WINDOW_MIN = 30        # +/- window around the takeoff
ARR_TAXI_CLIP_S = 3600     # taxi-in clipping before averaging
ARR_LONG_S = 1200          # "long taxi-in" threshold

# Stand chaining
STAND_LINK_MAX_H = 36      # beyond this, no arrival -> departure matching

# Queue per runway (v12)
RUNWAY_WINDOW_MIN = 10     # takeoffs on the same runway in the N min BEFORE
RUNWAY_WINDOW_AFTER_MIN = 5  # ... and AFTER (the queue that follows)

# Queue anchored on the off-block (v15), definitions from the literature
QUEUE_RECENT_MIN = 20      # window for the recent mean taxi-out (before my off-block)
QUEUE_MIN_RECENT_N = 2     # minimum number of departures to compute the mean

# Ensembling (v12): seeds for the classifier and the regressors
SEEDS = [42, 7, 123, 2026, 31337]

# Reporting-outage signal (v10): corruption of the ARRIVALS
ARR_WIDE_WINDOW_MIN = 120  # deuxieme fenetre, plus large
ARR_ANOM_TAXI_S = 3600     # aberrant taxi-in if > this threshold or < 0

# ---- Submission -------------------------------------------------------------
# Convention imposed by the challenge: <team-name>_v<integer>.parquet
TEAM = "motivated-ostrich"
START_V = 16                   # number of the next submission
GAMMAS = [0.75]                # optimum public

# Target encoding: number of folds and strength of the smoothing towards the prior
TE_FOLDS = 5
TE_SMOOTH = 50.0

CAT_FEATURES = [
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "stand_runway",
    "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
    "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt",
]

NUM_FEATURES = [
    "hour_utc", "dow", "month",
    "delta_mvt_sched_s", "delta_mvt_aobt3_s", "delta_mvt_eobt1_s",
    "delta_mvt_lobt_s", "delta_mvt_iobt_s",
    "dep_count_win", "arr_count_win", "mvt_count_win",
    "type_mismatch",
    "disagree_sched_aobt_s", "disagree_sched_eobt_s", "ratio_sched_aobt",
    "no_flight_id", "no_aircraft_type", "no_aobt",
    # --- v9: arrival context ---
    "arr_taxi_mean_win", "arr_taxi_frac_long_win", "arr_taxi_day_mean",
    "arr_count_ctx",
    # --- v10: reporting outage seen through the arrivals ---
    "arr_corrupt_frac_win",    # share of arrivals with in-block == sched, +/- 30 min
    "arr_corrupt_frac_wide",   # same, +/- 2 h
    "arr_corrupt_day_frac",    # same, airport day
    "arr_anom_frac_win",       # share of arrivals with an aberrant taxi-in, +/- 30 min
    "arr_anom_frac_wide",      # same, +/- 2 h
    # --- v15: queue anchored on the off-block (AOBT_3) ---
    "q_n_taxiing_rwy",       # departures on the same runway already taxiing at my off-block (N)
    "q_n_taxiing_apt",       # same, whole airport
    "q_ahead_dep_rwy",       # departures on the same runway taking off between my off-block and my takeoff (Q)
    "q_ahead_dep_apt",       # same, whole airport
    "q_ahead_arr_rwy",       # landings on the same runway during my taxi
    "q_recent_taxi_rwy",     # mean taxi-out (MVT - AOBT_3) of departures on the same runway, 20 min before my off-block
    "q_recent_taxi_apt",     # same, whole airport
    "q_recent_n_rwy",        # count behind the mean
    "q_pushback_rate_rwy",   # off-blocks on the same runway in the 10 min before mine
    # --- v12: queue per runway ---
    "rwy_dep_before",        # takeoffs on the same runway in the previous 10 min
    "rwy_dep_after",         # takeoffs on the same runway in the next 5 min
    "rwy_gap_prev_s",        # interval since the previous takeoff on the same runway
    "rwy_gap_next_s",        # interval until the next takeoff on the same runway
    "rwy_arr_before",        # landings on the same runway in the previous 10 min
    "rwy_share_win",         # share of the airport departures on this runway (+/- 15 min)
    # --- v9: stand chaining ---
    "turnaround_s", "prev_arr_taxi_in", "prev_same_type",
    "prev_same_operator", "sched_before_inblock", "bound_minus_aobt",
    "has_stand_link",
    # --- v13: corruption of the arrivals at the same stand (per-aircraft signal) ---
    "prev_arr_corrupt",      # in-block == sched on the previous arrival
    "next_arr_corrupt",      # same on the next arrival
    "both_arr_corrupt",      # both (38 % corruption in 2025)
]

# Target-encoding features (added dynamically to the classifier)
TE_SPECS = {
    "te_adep": ["ADEP_mvt"],
    "te_adep_flight": ["ADEP_mvt", "FLIGHT_mvt"],   # v13: flight number
    "te_adep_stand": ["ADEP_mvt", "STAND_mvt"],
    "te_operator": ["AIRCRAFT_OPERATOR_flt"],
    "te_adep_operator": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
    "te_adep_hour": ["ADEP_mvt", "hour_bin"],
}
TE_FEATURES = list(TE_SPECS.keys())

TARGET = "TAXITIME_SEC_mvt"
CORRUPT = "is_corrupt"

TIME_COLS = [
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt",
]

LGB_REG = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    lambda_l2=1.0, verbosity=-1, seed=42,
)

# Regressor B: robust loss (pathological rows no longer pull everything)
# v5.1: back to L2 (huber had degraded B: 3141 s vs 3047 s)
LGB_REG_ROBUST = dict(LGB_REG)

LGB_CLF = dict(
    objective="binary", metric="auc",
    learning_rate=0.02, num_leaves=255, min_data_in_leaf=200,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
    lambda_l2=2.0, verbosity=-1, seed=42,
)


# ----------------------------------------------------------------------------
# Loading / base features (identical to v4)
# ----------------------------------------------------------------------------

def load_training(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "training_*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No training_*.parquet file found in {data_dir.resolve()}")
    print(f"Loading {len(files)} training files...")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    print(f"  -> {len(df):,} movements")
    return df


def parse_times(df: pd.DataFrame) -> pd.DataFrame:
    for c in TIME_COLS:
        if c not in df.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        elif df[c].dt.tz is None:
            df[c] = df[c].dt.tz_localize("UTC")
        df[c] = df[c].astype("datetime64[ns, UTC]")
    return df


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    win = pd.Timedelta(minutes=CONGESTION_WINDOW_MIN)
    df["dep_count_win"] = np.nan
    df["arr_count_win"] = np.nan
    df["mvt_count_win"] = np.nan

    df["_airport"] = np.where(df["PHASE_mvt"] == "DEP",
                              df["ADEP_mvt"], df["ADES_mvt"])

    for _, grp in df.groupby("_airport"):
        t = grp["MVT_TIME_UTC_mvt"]
        valid = t.notna()
        if valid.sum() == 0:
            continue
        t_naive = t.dt.tz_localize(None)
        t_all = np.sort(t_naive[valid].values.astype("datetime64[ns]"))
        t_dep = np.sort(t_naive[(grp["PHASE_mvt"] == "DEP") & valid]
                        .values.astype("datetime64[ns]"))
        t_arr = np.sort(t_naive[(grp["PHASE_mvt"] == "ARR") & valid]
                        .values.astype("datetime64[ns]"))
        tv = t_naive[valid].values.astype("datetime64[ns]")
        lo, hi = tv - win.to_timedelta64(), tv + win.to_timedelta64()

        def cnt(s):
            if len(s) == 0:
                return np.zeros(len(tv))
            return (np.searchsorted(s, hi, side="right")
                    - np.searchsorted(s, lo, side="left"))

        idx = grp.index[valid]
        df.loc[idx, "mvt_count_win"] = cnt(t_all) - 1
        df.loc[idx, "dep_count_win"] = cnt(t_dep)
        df.loc[idx, "arr_count_win"] = cnt(t_arr)

    df = add_runway_queue_features(df)
    return df.drop(columns=["_airport"])


def add_runway_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Queue at the RUNWAY level (v12). Airport congestion does not
    tell which runway one waits for: here we count the takeoffs on
    the same runway just before and just after, and the intervals between
    consecutive takeoffs (a 60-90 s spacing reveals a dense queue,
    several minutes a fluid traffic). Everything is observable in
    the ranking file (MVT_TIME and RUNWAY of departures are not masked).
    """
    for c in ("rwy_dep_before", "rwy_dep_after", "rwy_gap_prev_s",
              "rwy_gap_next_s", "rwy_arr_before", "rwy_share_win"):
        df[c] = np.nan

    before = pd.Timedelta(minutes=RUNWAY_WINDOW_MIN).to_timedelta64()
    after = pd.Timedelta(minutes=RUNWAY_WINDOW_AFTER_MIN).to_timedelta64()
    share_win = pd.Timedelta(minutes=CONGESTION_WINDOW_MIN).to_timedelta64()

    rwy = df["RUNWAY_mvt"].astype(str)
    key = df["_airport"].astype(str) + "|" + rwy
    valid = df["MVT_TIME_UTC_mvt"].notna() & df["RUNWAY_mvt"].notna() & (rwy != "nan")
    is_dep = df["PHASE_mvt"] == "DEP"

    for k, grp in df[valid].groupby(key):
        t = grp["MVT_TIME_UTC_mvt"].dt.tz_localize(None).values.astype("datetime64[ns]")
        d = (grp["PHASE_mvt"] == "DEP").values
        t_dep = np.sort(t[d])
        t_arr = np.sort(t[~d])
        idx = grp.index

        if len(t_dep):
            n_before = (np.searchsorted(t_dep, t, side="left")
                        - np.searchsorted(t_dep, t - before, side="left"))
            n_after = (np.searchsorted(t_dep, t + after, side="right")
                       - np.searchsorted(t_dep, t, side="right"))
            pos = np.searchsorted(t_dep, t, side="left")
            prev_t = np.where(pos > 0, t_dep[np.maximum(pos - 1, 0)], np.datetime64("NaT"))
            pos_r = np.searchsorted(t_dep, t, side="right")
            next_t = np.where(pos_r < len(t_dep),
                              t_dep[np.minimum(pos_r, len(t_dep) - 1)],
                              np.datetime64("NaT"))
            gap_prev = (t - prev_t) / np.timedelta64(1, "s")
            gap_next = (next_t - t) / np.timedelta64(1, "s")
            df.loc[idx, "rwy_dep_before"] = n_before
            df.loc[idx, "rwy_dep_after"] = n_after
            df.loc[idx, "rwy_gap_prev_s"] = gap_prev
            df.loc[idx, "rwy_gap_next_s"] = gap_next
        if len(t_arr):
            df.loc[idx, "rwy_arr_before"] = (
                np.searchsorted(t_arr, t, side="left")
                - np.searchsorted(t_arr, t - before, side="left"))

    # Share of the airport departures that use this runway (+/- 15 min):
    # rwy_dep_win / dep_count_win, where rwy_dep_win counts the departures of the
    # same runway in the symmetric window.
    rwy_win = np.full(len(df), np.nan)
    for k, grp in df[valid & is_dep].groupby(key):
        t = grp["MVT_TIME_UTC_mvt"].dt.tz_localize(None).values.astype("datetime64[ns]")
        ts = np.sort(t)
        n = (np.searchsorted(ts, t + share_win, side="right")
             - np.searchsorted(ts, t - share_win, side="left"))
        rwy_win[df.index.get_indexer(grp.index)] = n
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rwy_share_win"] = np.where(
            df["dep_count_win"].values > 0,
            rwy_win / df["dep_count_win"].values, np.nan)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mvt = df["MVT_TIME_UTC_mvt"]

    df["hour_utc"] = mvt.dt.hour + mvt.dt.minute / 60.0
    df["dow"] = mvt.dt.dayofweek
    df["month"] = mvt.dt.month

    def delta_s(col):
        return (mvt - df[col]).dt.total_seconds() if col in df.columns else np.nan

    df["delta_mvt_sched_s"] = delta_s("SCHED_TIME_UTC_mvt")
    df["delta_mvt_aobt3_s"] = delta_s("AOBT_3_flt")
    df["delta_mvt_eobt1_s"] = delta_s("EOBT_1_flt")
    df["delta_mvt_lobt_s"] = delta_s("LOBT_flt")
    df["delta_mvt_iobt_s"] = delta_s("IOBT_flt")

    df["stand_runway"] = (df["ADEP_mvt"].astype(str) + "_"
                          + df["STAND_mvt"].astype(str) + "_"
                          + df["RUNWAY_mvt"].astype(str))

    if "AIRCRAFT_TYPE_flt" in df.columns:
        both = df["AIRCRAFT_TYPE_mvt"].notna() & df["AIRCRAFT_TYPE_flt"].notna()
        df["type_mismatch"] = (both & (df["AIRCRAFT_TYPE_mvt"].astype(str)
                                       != df["AIRCRAFT_TYPE_flt"].astype(str))
                               ).astype(float)
    else:
        df["type_mismatch"] = 0.0

    df["disagree_sched_aobt_s"] = df["delta_mvt_sched_s"] - df["delta_mvt_aobt3_s"]
    df["disagree_sched_eobt_s"] = df["delta_mvt_sched_s"] - df["delta_mvt_eobt1_s"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ratio_sched_aobt"] = (df["delta_mvt_sched_s"]
                                  / df["delta_mvt_aobt3_s"].replace(0, np.nan))
    df["no_flight_id"] = (df["FLIGHT_ID_mvt"].isna().astype(float)
                          if "FLIGHT_ID_mvt" in df.columns else 0.0)
    df["no_aircraft_type"] = df["AIRCRAFT_TYPE_mvt"].isna().astype(float)
    df["no_aobt"] = df["delta_mvt_aobt3_s"].isna().astype(float)

    # Key for the hourly target encoding
    df["hour_bin"] = mvt.dt.hour

    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


# ----------------------------------------------------------------------------
# Out-of-fold target encoding of the corruption rate
# ----------------------------------------------------------------------------

def _key(df: pd.DataFrame, cols: list) -> pd.Series:
    """Grouping key, as a string, robust to NaN and categories."""
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + "|" + df[c].astype(str)
    return out


def fit_te_maps(df: pd.DataFrame, label: np.ndarray) -> tuple[dict, float]:
    """Target-encoding statistics on a reference dataframe."""
    prior = float(np.mean(label))
    maps = {}
    for name, cols in TE_SPECS.items():
        k = _key(df, cols)
        g = pd.DataFrame({"k": k.values, "y": label}).groupby("k")["y"]
        agg = g.agg(["sum", "count"])
        # Bayesian smoothing towards the prior
        maps[name] = ((agg["sum"] + TE_SMOOTH * prior)
                      / (agg["count"] + TE_SMOOTH))
    return maps, prior


def apply_te(df: pd.DataFrame, maps: dict, prior: float) -> pd.DataFrame:
    """Applies target-encoding maps (val, ranking, or held-out fold)."""
    df = df.copy()
    for name, cols in TE_SPECS.items():
        k = _key(df, cols)
        df[name] = k.map(maps[name]).astype(float).fillna(prior)
    return df


def add_te_out_of_fold(train: pd.DataFrame, label_col: str,
                       n_folds: int = TE_FOLDS, seed: int = 42) -> pd.DataFrame:
    """Out-of-fold target encoding on the training set: each row receives an
    encoding computed WITHOUT it (on the other folds), hence no leakage.
    """
    train = train.copy()
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, n_folds, size=len(train))
    for name in TE_FEATURES:
        train[name] = np.nan

    y = train[label_col].values
    for f in range(n_folds):
        out = fold == f
        maps, prior = fit_te_maps(train[~out], y[~out])
        enc = apply_te(train[out], maps, prior)
        for name in TE_FEATURES:
            train.loc[out, name] = enc[name].values
    return train


# ----------------------------------------------------------------------------
# v14: unimpeded taxi time (PRU methodology) per stand-runway
# ----------------------------------------------------------------------------
#
# The taxi-out splits into an incompressible time that depends on the
# geometry (stand, runway, distance to cover) and an additional time
# that depends on congestion. The PRU estimates the former as a low
# percentile of the observed taxi per stand-runway pair ("unimpeded taxi
# time"). Here: percentiles 10 and 50 per (airport, stand, runway), with
# fallback to (airport, runway) then (airport) when the count is low.
# Computed OUT-OF-FOLD on the training set, on clean rows outside the
# tail, so as not to leak the target.
#
# Usage:
#   - explicit numerical features for A and B (the model saw
#     stand_runway as a category and had to relearn the geometry
#     by splitting, without generalizing to rare or new pairs)
#   - target of regressor B (without AOBT_3): EXCESS = taxi - p50, more
#     stable than the raw taxi, as the AOBT_3 residual is for A

UNIMP_MIN_N = 20
UNIMP_LEVELS = [["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
                ["ADEP_mvt", "RUNWAY_mvt"],
                ["ADEP_mvt"]]
UNIMP_FEATURES = ["unimpeded_p10", "unimpeded_p50", "unimpeded_n",
                  "unimpeded_level"]


def _unimp_stats_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows used for the statistics: clean and outside the tail."""
    ok = (df[TARGET].values <= TAIL_THR)
    if CORRUPT in df.columns:
        ok &= (df[CORRUPT].values == 0)
    return ok


def fit_unimpeded_maps(df: pd.DataFrame) -> list:
    """One table (p10, p50, n) per level of the hierarchy."""
    sub = df[_unimp_stats_mask(df)]
    maps = []
    for cols in UNIMP_LEVELS:
        k = _key(sub, cols)
        g = pd.DataFrame({"k": k.values, "y": sub[TARGET].values}).groupby("k")["y"]
        t = pd.DataFrame({"p10": g.quantile(0.10), "p50": g.quantile(0.50),
                          "n": g.size()})
        maps.append(t)
    return maps


def apply_unimpeded(df: pd.DataFrame, maps: list) -> pd.DataFrame:
    df = df.copy()
    p10 = np.full(len(df), np.nan)
    p50 = np.full(len(df), np.nan)
    n = np.full(len(df), np.nan)
    level = np.full(len(df), np.nan)
    todo = np.ones(len(df), dtype=bool)
    for lvl, (cols, t) in enumerate(zip(UNIMP_LEVELS, maps)):
        k = _key(df, cols)
        hit = k.map(t["n"]).astype(float).values
        use = todo & np.isfinite(hit) & (hit >= UNIMP_MIN_N)
        if use.any():
            p10[use] = k[use].map(t["p10"]).astype(float).values
            p50[use] = k[use].map(t["p50"]).astype(float).values
            n[use] = hit[use]
            level[use] = lvl
            todo &= ~use
    df["unimpeded_p10"] = p10
    df["unimpeded_p50"] = p50
    df["unimpeded_n"] = n
    df["unimpeded_level"] = level
    return df


def add_unimpeded_out_of_fold(train: pd.DataFrame, n_folds: int = TE_FOLDS,
                              seed: int = 43) -> pd.DataFrame:
    train = train.copy()
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, n_folds, size=len(train))
    for c in UNIMP_FEATURES:
        train[c] = np.nan
    for f in range(n_folds):
        out = fold == f
        maps = fit_unimpeded_maps(train[~out])
        enc = apply_unimpeded(train[out], maps)
        for c in UNIMP_FEATURES:
            train.loc[out, c] = enc[c].values
    return train


# ----------------------------------------------------------------------------
# Model utilities
# ----------------------------------------------------------------------------

def prepare_matrix(df, train_categories=None, extra=()):
    feats = [c for c in CAT_FEATURES + NUM_FEATURES + list(extra)
             if c in df.columns]
    X = df[feats].copy()
    if train_categories is not None:
        for c, cats in train_categories.items():
            if c in X.columns:
                s = X[c].astype("object")
                X[c] = pd.Categorical(s.where(s.isin(set(cats))), categories=cats)
    return X


def get_categories(df):
    return {c: df[c].cat.categories for c in CAT_FEATURES if c in df.columns}


def train_lgb(params, X, y, X_val=None, y_val=None, num_rounds=5000,
              stopping=200):
    dtrain = lgb.Dataset(X, label=y)
    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        return lgb.train(params, dtrain, num_boost_round=num_rounds,
                         valid_sets=[dval],
                         callbacks=[lgb.early_stopping(stopping),
                                    lgb.log_evaluation(250)])
    return lgb.train(params, dtrain, num_boost_round=num_rounds)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


class Ensemble:
    """Mean of LightGBM models trained with different seeds.
    Exposes predict() and best_iteration (mean of the kept rounds) to
    remain interchangeable with a single model."""

    def __init__(self, models, iters=None):
        self.models = models
        self.iters = iters or [None] * len(models)
        self.best_iteration = (int(np.mean([i for i in self.iters if i]))
                               if any(self.iters) else None)

    def predict(self, X, num_iteration=None):
        preds = [m.predict(X, num_iteration=(num_iteration if num_iteration
                                             is not None else it))
                 for m, it in zip(self.models, self.iters)]
        return np.mean(preds, axis=0)

    def feature_importance(self, importance_type="gain"):
        return np.mean([m.feature_importance(importance_type=importance_type)
                        for m in self.models], axis=0)


def train_ensemble(params, X, y, X_val=None, y_val=None, num_rounds=5000,
                   stopping=200, seeds=None, fixed_rounds=None, label=""):
    """Trains one model per seed. With X_val: early stopping per seed.
    With fixed_rounds: list (or integer) of rounds per seed, without
    validation (final retraining)."""
    seeds = seeds or SEEDS
    models, iters = [], []
    for k, sd in enumerate(seeds):
        p = dict(params, seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
        if fixed_rounds is not None:
            r = fixed_rounds[k] if isinstance(fixed_rounds, (list, tuple)) else fixed_rounds
            m = lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=int(r))
            models.append(m); iters.append(int(r))
        else:
            dtrain = lgb.Dataset(X, label=y)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            m = lgb.train(p, dtrain, num_boost_round=num_rounds, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(stopping, verbose=False)])
            models.append(m); iters.append(m.best_iteration)
            print(f"    {label} seed {sd:>5} : best iter {m.best_iteration:5d}, "
                  f"score {m.best_score['valid_0'][list(m.best_score['valid_0'])[0]]:.4f}")
    return Ensemble(models, iters)


# ----------------------------------------------------------------------------
# v9: contextual features (computed on DEP + ARR before filtering)
# ----------------------------------------------------------------------------

def _naive_ns(s: pd.Series) -> np.ndarray:
    """Serie datetime tz-aware -> tableau datetime64[ns] naif."""
    return s.dt.tz_localize(None).values.astype("datetime64[ns]")


def add_arrival_context(df: pd.DataFrame) -> pd.DataFrame:
    """Context provided by the ARRIVALS at the same airport around each
    movement. Everything is vectorized with cumulative sums over the sorted
    arrivals.

    v9  : taxi-in (contemporary congestion / weather).
    v10 : corruption of the arrivals (in-block == scheduled time) and
          taxi-in anomalies. Hypothesis checked on 2025: when more than
          30 % of the arrivals of an hour are corrupted, 23 to 30 % of the
          departures are too, against 4 % otherwise. An airport that does
          not report its off-blocks does not report its
          in-blocks either: it is a REPORTING OUTAGE signal, contemporary,
          observable in the ranking file (arrivals are complete there).
    """
    df = df.copy()
    win = pd.Timedelta(minutes=ARR_WINDOW_MIN).to_timedelta64()
    wide = pd.Timedelta(minutes=ARR_WIDE_WINDOW_MIN).to_timedelta64()

    new_cols = ("arr_taxi_mean_win", "arr_taxi_frac_long_win",
                "arr_taxi_day_mean", "arr_count_ctx",
                "arr_corrupt_frac_win", "arr_corrupt_frac_wide",
                "arr_corrupt_day_frac", "arr_anom_frac_win",
                "arr_anom_frac_wide")
    for c in new_cols:
        df[c] = np.nan

    df["_airport"] = np.where(df["PHASE_mvt"] == "DEP",
                              df["ADEP_mvt"], df["ADES_mvt"])

    is_arr = df["PHASE_mvt"] == "ARR"
    gap = (df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    df["_arr_corrupt"] = np.where(is_arr, (gap.abs() <= CORRUPT_TOL_S), False)
    df["_arr_anom"] = np.where(
        is_arr, (df[TARGET] > ARR_ANOM_TAXI_S) | (df[TARGET] < 0), False)

    for _, grp in df.groupby("_airport"):
        arr = grp[(grp["PHASE_mvt"] == "ARR") & grp["MVT_TIME_UTC_mvt"].notna()]
        if len(arr) == 0:
            continue
        t_arr = _naive_ns(arr["MVT_TIME_UTC_mvt"])
        order = np.argsort(t_arr)
        t_sorted = t_arr[order]

        taxi_raw = arr[TARGET].values[order]
        has_taxi = np.isfinite(taxi_raw)
        taxi = np.where(has_taxi, np.clip(taxi_raw, 0, ARR_TAXI_CLIP_S), 0.0)
        cs = np.concatenate([[0.0], np.cumsum(taxi)])
        cs_n = np.concatenate([[0.0], np.cumsum(has_taxi)])
        cs_long = np.concatenate([[0.0], np.cumsum(taxi > ARR_LONG_S)])
        cs_cor = np.concatenate([[0.0], np.cumsum(arr["_arr_corrupt"].values[order])])
        cs_anom = np.concatenate([[0.0], np.cumsum(arr["_arr_anom"].values[order])])

        valid = grp["MVT_TIME_UTC_mvt"].notna()
        tv = _naive_ns(grp.loc[valid, "MVT_TIME_UTC_mvt"])

        def window_stats(half):
            lo = np.searchsorted(t_sorted, tv - half, side="left")
            hi = np.searchsorted(t_sorted, tv + half, side="right")
            n_all = (hi - lo).astype(float)
            n_taxi = cs_n[hi] - cs_n[lo]
            with np.errstate(divide="ignore", invalid="ignore"):
                return dict(
                    n=n_all,
                    mean=np.where(n_taxi > 0, (cs[hi] - cs[lo]) / n_taxi, np.nan),
                    frac_long=np.where(n_taxi > 0,
                                       (cs_long[hi] - cs_long[lo]) / n_taxi, np.nan),
                    corrupt=np.where(n_all > 0, (cs_cor[hi] - cs_cor[lo]) / n_all, np.nan),
                    anom=np.where(n_all > 0, (cs_anom[hi] - cs_anom[lo]) / n_all, np.nan),
                )

        w = window_stats(win)
        ww = window_stats(wide)
        idx = grp.index[valid]
        df.loc[idx, "arr_taxi_mean_win"] = w["mean"]
        df.loc[idx, "arr_taxi_frac_long_win"] = w["frac_long"]
        df.loc[idx, "arr_count_ctx"] = w["n"]
        df.loc[idx, "arr_corrupt_frac_win"] = w["corrupt"]
        df.loc[idx, "arr_anom_frac_win"] = w["anom"]
        df.loc[idx, "arr_corrupt_frac_wide"] = ww["corrupt"]
        df.loc[idx, "arr_anom_frac_wide"] = ww["anom"]

    # Daily aggregates per airport (arrivals only)
    day = df["MVT_TIME_UTC_mvt"].dt.floor("D")
    key = df["_airport"].astype(str) + "|" + day.astype(str)
    arr_df = pd.DataFrame({
        "k": key[is_arr].values,
        "taxi": np.clip(df.loc[is_arr, TARGET].values, 0, ARR_TAXI_CLIP_S),
        "cor": df.loc[is_arr, "_arr_corrupt"].values.astype(float),
    })
    g = arr_df.groupby("k")
    df["arr_taxi_day_mean"] = key.map(g["taxi"].mean()).astype(float)
    df["arr_corrupt_day_frac"] = key.map(g["cor"].mean()).astype(float)

    return df.drop(columns=["_airport", "_arr_corrupt", "_arr_anom"])


def add_offblock_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Queue ANCHORED ON THE OFF-BLOCK (v15).

    All our congestion features were centered on the takeoff
    time. The literature (Idris 2001, Ravizza 2014, Wang 2021, review by
    Yin 2024) is unanimous: the most predictive variable is the size of
    the queue at push-back, and the mean taxi-out of recent
    departures. In the ranking file, BLOCK_TIME is masked for departures but
    AOBT_3 (NM off-block) is not, and neither is MVT_TIME: we
    therefore rebuild these quantities with AOBT_3 as the off-block.

      N  : departures already taxiing when I push back
           (their AOBT_3 < my AOBT_3 < their MVT_TIME)
      Q  : departures taking off between my AOBT_3 and my MVT_TIME
           (those that overtook me in the queue)
      recent : mean of (MVT_TIME - AOBT_3) of the departures whose takeoff
           precedes my off-block, within QUEUE_RECENT_MIN minutes
      pushback_rate : off-blocks in the 10 min before mine

    Computed per runway and per airport, with searchsorted / cumulative sums.
    Departures without AOBT_3 get NaN (handled by LightGBM).
    """
    cols = ["q_n_taxiing_rwy", "q_n_taxiing_apt", "q_ahead_dep_rwy",
            "q_ahead_dep_apt", "q_ahead_arr_rwy", "q_recent_taxi_rwy",
            "q_recent_taxi_apt", "q_recent_n_rwy", "q_pushback_rate_rwy"]
    for c in cols:
        df[c] = np.nan

    is_dep = (df["PHASE_mvt"] == "DEP").values
    has_ob = df["AOBT_3_flt"].notna().values & df["MVT_TIME_UTC_mvt"].notna().values
    airport = np.where(is_dep, df["ADEP_mvt"], df["ADES_mvt"]).astype(str)
    rwy = df["RUNWAY_mvt"].astype(str).values
    rwy_ok = df["RUNWAY_mvt"].notna().values & (rwy != "nan")

    t_mvt_all = df["MVT_TIME_UTC_mvt"].dt.tz_localize(None).values.astype("datetime64[ns]")
    t_ob_all = df["AOBT_3_flt"].dt.tz_localize(None).values.astype("datetime64[ns]")

    recent = pd.Timedelta(minutes=QUEUE_RECENT_MIN).to_timedelta64()
    push_win = pd.Timedelta(minutes=10).to_timedelta64()

    def queue_stats(t_ob_q, t_mvt_q, t_ob_ref, t_mvt_ref):
        """For queries (t_ob_q, t_mvt_q) against a population of reference
        departures (off-block, takeoff) sorted by takeoff."""
        order = np.argsort(t_mvt_ref)
        mvt_s = t_mvt_ref[order]
        ob_by_mvt = t_ob_ref[order]
        taxi_by_mvt = (mvt_s - ob_by_mvt) / np.timedelta64(1, "s")
        ob_s = np.sort(t_ob_ref)

        # N: off-block before mine AND takeoff after mine
        #   = (#ob < mine) - (#ob < mine and mvt < mine)
        #   The second term = number of refs entirely done before my ob:
        #   approximated by #mvt < my ob (a ref that took off before my ob
        #   necessarily pushed back before), exact.
        n_ob_before = np.searchsorted(ob_s, t_ob_q, side="left")
        n_mvt_before_ob = np.searchsorted(mvt_s, t_ob_q, side="left")
        n_taxiing = np.maximum(n_ob_before - n_mvt_before_ob, 0)

        # Q: takeoffs between my off-block and my takeoff (me excluded)
        lo = np.searchsorted(mvt_s, t_ob_q, side="left")
        hi = np.searchsorted(mvt_s, t_mvt_q, side="left")
        q_ahead = np.maximum(hi - lo, 0)

        # mean taxi-out of the departures that took off in [ob - recent, ob)
        cs = np.concatenate([[0.0], np.cumsum(taxi_by_mvt)])
        lo_r = np.searchsorted(mvt_s, t_ob_q - recent, side="left")
        hi_r = np.searchsorted(mvt_s, t_ob_q, side="left")
        n_r = (hi_r - lo_r).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            recent_mean = np.where(n_r >= QUEUE_MIN_RECENT_N,
                                   (cs[hi_r] - cs[lo_r]) / n_r, np.nan)

        # off-blocks in the 10 min before mine (me excluded)
        push = (np.searchsorted(ob_s, t_ob_q, side="left")
                - np.searchsorted(ob_s, t_ob_q - push_win, side="left"))
        return n_taxiing, q_ahead, recent_mean, n_r, push

    dep_idx = np.where(is_dep & has_ob)[0]
    # ---- per airport ----
    for apt in np.unique(airport[dep_idx]):
        sel = dep_idx[airport[dep_idx] == apt]
        if len(sel) == 0:
            continue
        n_t, q_a, r_m, n_r, push = queue_stats(
            t_ob_all[sel], t_mvt_all[sel], t_ob_all[sel], t_mvt_all[sel])
        df.iloc[sel, df.columns.get_loc("q_n_taxiing_apt")] = n_t
        df.iloc[sel, df.columns.get_loc("q_ahead_dep_apt")] = q_a
        df.iloc[sel, df.columns.get_loc("q_recent_taxi_apt")] = r_m

        # ---- per runway ----
        sel_r = sel[rwy_ok[sel]]
        for r in np.unique(rwy[sel_r]):
            s_r = sel_r[rwy[sel_r] == r]
            n_t, q_a, r_m, n_r, push = queue_stats(
                t_ob_all[s_r], t_mvt_all[s_r], t_ob_all[s_r], t_mvt_all[s_r])
            df.iloc[s_r, df.columns.get_loc("q_n_taxiing_rwy")] = n_t
            df.iloc[s_r, df.columns.get_loc("q_ahead_dep_rwy")] = q_a
            df.iloc[s_r, df.columns.get_loc("q_recent_taxi_rwy")] = r_m
            df.iloc[s_r, df.columns.get_loc("q_recent_n_rwy")] = n_r
            df.iloc[s_r, df.columns.get_loc("q_pushback_rate_rwy")] = push

            # landings on the same runway during my taxi
            arr_sel = np.where((~is_dep) & (airport == apt) & rwy_ok & (rwy == r)
                               & ~np.isnat(t_mvt_all))[0]
            if len(arr_sel):
                t_arr = np.sort(t_mvt_all[arr_sel])
                lo = np.searchsorted(t_arr, t_ob_all[s_r], side="left")
                hi = np.searchsorted(t_arr, t_mvt_all[s_r], side="left")
                df.iloc[s_r, df.columns.get_loc("q_ahead_arr_rwy")] = np.maximum(hi - lo, 0)
            else:
                df.iloc[s_r, df.columns.get_loc("q_ahead_arr_rwy")] = 0.0
    return df


def add_stand_linking(df: pd.DataFrame) -> pd.DataFrame:
    """Matches each departure to the last arrival at the same stand of the same
    airport whose in-block precedes the takeoff (backward merge_asof).
    """
    df = df.copy()
    for c in ("turnaround_s", "prev_arr_taxi_in", "prev_same_type",
              "prev_same_operator", "sched_before_inblock",
              "bound_minus_aobt", "prev_arr_corrupt", "next_arr_corrupt",
              "both_arr_corrupt"):
        df[c] = np.nan
    df["has_stand_link"] = 0.0

    # Corruption of the arrivals (in-block == scheduled time)
    gap_arr = (df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    arr_corrupt_all = (gap_arr.abs() <= CORRUPT_TOL_S).astype(float)

    stand = df["STAND_mvt"].astype(str)
    stand_ok = df["STAND_mvt"].notna() & (stand != "nan") & (stand != "")

    dep_mask = (df["PHASE_mvt"] == "DEP") & stand_ok & df["MVT_TIME_UTC_mvt"].notna()
    arr_mask = ((df["PHASE_mvt"] == "ARR") & stand_ok
                & df["BLOCK_TIME_UTC_mvt"].notna())

    dep = pd.DataFrame({
        "_idx": df.index[dep_mask],
        "_key": (df.loc[dep_mask, "ADEP_mvt"].astype(str) + "|"
                 + stand[dep_mask]).values,
        "_t": _naive_ns(df.loc[dep_mask, "MVT_TIME_UTC_mvt"]),
    })
    arr = pd.DataFrame({
        "_key": (df.loc[arr_mask, "ADES_mvt"].astype(str) + "|"
                 + stand[arr_mask]).values,
        "_t": _naive_ns(df.loc[arr_mask, "BLOCK_TIME_UTC_mvt"]),
        "_arr_taxi": df.loc[arr_mask, TARGET].values,
        "_arr_type": df.loc[arr_mask, "AIRCRAFT_TYPE_mvt"].astype(str).values,
        "_arr_op": (df.loc[arr_mask, "AIRCRAFT_OPERATOR_flt"].astype(str).values
                    if "AIRCRAFT_OPERATOR_flt" in df.columns
                    else np.full(arr_mask.sum(), "nan")),
        "_arr_corrupt": arr_corrupt_all[arr_mask].values,
    })
    if len(dep) == 0 or len(arr) == 0:
        return df

    dep = dep.sort_values("_t")
    arr = arr.sort_values("_t")
    linked = pd.merge_asof(
        dep, arr, on="_t", by="_key", direction="backward",
        tolerance=pd.Timedelta(hours=STAND_LINK_MAX_H),
        suffixes=("", "_arr"),
    )
    # merge_asof does not keep the right-hand time key: we recover it
    # with a second asof on the time column alone.
    arr_t = arr[["_key", "_t"]].rename(columns={"_t": "_t_inblock"})
    arr_t["_t"] = arr_t["_t_inblock"]
    linked = pd.merge_asof(
        linked.sort_values("_t"), arr_t.sort_values("_t"),
        on="_t", by="_key", direction="backward",
        tolerance=pd.Timedelta(hours=STAND_LINK_MAX_H),
    )

    has = linked["_t_inblock"].notna().values
    idx = linked["_idx"].values
    turnaround = ((linked["_t"] - linked["_t_inblock"])
                  .dt.total_seconds().values)

    df.loc[idx, "has_stand_link"] = has.astype(float)
    df.loc[idx[has], "turnaround_s"] = turnaround[has]
    df.loc[idx[has], "prev_arr_taxi_in"] = linked["_arr_taxi"].values[has]

    dep_type = df.loc[idx, "AIRCRAFT_TYPE_mvt"].astype(str).values
    dep_op = (df.loc[idx, "AIRCRAFT_OPERATOR_flt"].astype(str).values
              if "AIRCRAFT_OPERATOR_flt" in df.columns
              else np.full(len(idx), "nan"))
    same_type = (dep_type == linked["_arr_type"].values) & has
    same_op = (dep_op == linked["_arr_op"].values) & has
    df.loc[idx[has], "prev_same_type"] = same_type[has].astype(float)
    df.loc[idx[has], "prev_same_operator"] = same_op[has].astype(float)
    df.loc[idx[has], "prev_arr_corrupt"] = linked["_arr_corrupt"].values[has]

    # NEXT arrival at the same stand (non-causal signal, but observable
    # in the ranking file: arrivals are complete there; allowed by the
    # organizers, to be documented in the repository)
    nxt = pd.merge_asof(
        dep.sort_values("_t"), arr[["_key", "_t", "_arr_corrupt"]].sort_values("_t"),
        on="_t", by="_key", direction="forward",
        tolerance=pd.Timedelta(hours=STAND_LINK_MAX_H))
    has_n = nxt["_arr_corrupt"].notna().values
    idx_n = nxt["_idx"].values
    df.loc[idx_n[has_n], "next_arr_corrupt"] = nxt["_arr_corrupt"].values[has_n]

    pa = df["prev_arr_corrupt"]
    na = df["next_arr_corrupt"]
    df["both_arr_corrupt"] = np.where(pa.notna() & na.notna(), pa * na, np.nan)
    return df


def add_stand_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Linking derivatives that depend on the deltas (after build_features)."""
    df = df.copy()
    has = df["turnaround_s"].notna()
    # SCHED earlier than the in-block of the previous arrival at the same stand:
    # a block = sched would then be physically impossible.
    df["sched_before_inblock"] = np.where(
        has, (df["delta_mvt_sched_s"] > df["turnaround_s"]).astype(float), np.nan)
    df["bound_minus_aobt"] = np.where(
        has, df["turnaround_s"] - df["delta_mvt_aobt3_s"], np.nan)
    return df


# ----------------------------------------------------------------------------
# v11: dedicated branch for the "Rome, unmatched, takeoff on the next day" regime
# ----------------------------------------------------------------------------
#
# Diagnostic on 2025 (168 unmatched LIRF flights taking off the day after
# the scheduled date, local time):
#
#   delta_sched            +1 day    block=sched   normal
#   3 h  - 12 h              0-1         106          33
#   > 12 h                    11          16           0
#
# On this subset, the "taxi-out" is almost never a taxi-out:
# it is either delta_sched (block time filled with the scheduled time), or
# 86,400 + taxi (block stamped with the scheduled date and the actual
# time of the next day). Half a dozen of these rows per validation
# fold carry about half of the total MSE, because their
# errors amount to tens of thousands of seconds.
#
# Expected-value prediction, per delay bucket, with rates estimated on
# the training set (Laplace smoothing):
#
#   pred = r_plus1 * (86400 + mu_normal) + r_bs * delta_sched
#          + r_normal * mu_normal
#
# The 12 "+1 day" cases are spread over the whole of 2025: a structural
# bug of the Fiumicino system, presumed present in 2026.

ROME_AIRPORT = "LIRF"
ROME_TZ = "Europe/Rome"
ROME_BUCKETS = [0, 3600, 10800, 43200, np.inf]
ROME_LAPLACE = 1.0
DAY_S = 86400.0


def rome_subset_mask(df: pd.DataFrame) -> np.ndarray:
    """LIRF, not matched to NM, takeoff on the day after (local date) the scheduled
    day. Everything is observable in the ranking file."""
    is_rome = df["ADEP_mvt"].astype(str).values == ROME_AIRPORT
    unmatched = df["no_flight_id"].values == 1
    mvt_d = df["MVT_TIME_UTC_mvt"].dt.tz_convert(ROME_TZ).dt.floor("D")
    sch_d = df["SCHED_TIME_UTC_mvt"].dt.tz_convert(ROME_TZ).dt.floor("D")
    next_day = (mvt_d > sch_d).values
    return is_rome & unmatched & next_day & np.isfinite(df["delta_mvt_sched_s"].values)


def rome_bucket(delta_sched: np.ndarray) -> np.ndarray:
    return np.digitize(delta_sched, ROME_BUCKETS[1:-1])   # 0..3


def rome_categorize(df: pd.DataFrame) -> np.ndarray:
    """0 = +1 day, 1 = block = sched, 2 = normal (priority in this order)."""
    y = df[TARGET].values
    ds = df["delta_mvt_sched_s"].values
    plus1 = (y - DAY_S >= TAXI_MIN_S) & (y - DAY_S <= TAXI_MAX_S)
    bs = np.abs(y - ds) <= 60
    return np.where(plus1, 0, np.where(bs, 1, 2))


def fit_rome_rates(train_df: pd.DataFrame) -> np.ndarray:
    """Rates (3 categories) per delay bucket, smoothed. Shape (4, 3)."""
    m = rome_subset_mask(train_df)
    sub = train_df[m]
    b = rome_bucket(sub["delta_mvt_sched_s"].values)
    cat = rome_categorize(sub)
    rates = np.zeros((4, 3))
    for k in range(4):
        cnt = np.array([(cat[b == k] == c).sum() for c in range(3)], dtype=float)
        rates[k] = (cnt + ROME_LAPLACE) / (cnt.sum() + 3 * ROME_LAPLACE)
    return rates


def apply_rome_branch(pred: np.ndarray, df: pd.DataFrame, mu_normal: np.ndarray,
                      rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replaces the prediction on the subset with the expectation of the
    three-regime mixture. Returns (modified pred, mask)."""
    m = rome_subset_mask(df)
    if not m.any():
        return pred, m
    out = pred.copy()
    ds = df["delta_mvt_sched_s"].values[m]
    mu = mu_normal[m]
    r = rates[rome_bucket(ds)]
    out[m] = (r[:, 0] * (DAY_S + mu) + r[:, 1] * ds + r[:, 2] * mu)
    return out, m


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def build_all(raw: pd.DataFrame) -> pd.DataFrame:
    """Full feature chain on a DEP + ARR table. Used identically
    for training and for the ranking file."""
    raw = parse_times(raw)
    print("  congestion...")
    raw = add_congestion_features(raw)
    print("  arrival context...")
    raw = add_arrival_context(raw)
    print("  stand chaining...")
    raw = add_stand_linking(raw)
    print("  queue at off-block (AOBT_3)...")
    raw = add_offblock_queue_features(raw)
    dep = raw[raw["PHASE_mvt"] == "DEP"].copy()
    dep = build_features(dep)
    dep = add_stand_derived(dep)
    return dep


def load_or_build(cache_path: Path) -> pd.DataFrame:
    if USE_CACHE and cache_path.exists():
        print(f"Loading cache {cache_path.name}...")
        dep = pd.read_parquet(cache_path)
        print(f"  -> {len(dep):,} departures")
        return dep

    raw = load_training(DATA_DIR)
    print("Computing features (training)...")
    dep = build_all(raw)
    del raw
    gap = (dep["BLOCK_TIME_UTC_mvt"]
           - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    dep[CORRUPT] = (gap.abs() <= CORRUPT_TOL_S).astype(int)
    n0 = len(dep)
    dep = dep[dep[TARGET].notna() & (dep[TARGET] >= TAXI_MIN_S)]
    print(f"Minimal filtering: {n0 - len(dep):,} removed, {len(dep):,} left")
    dep.to_parquet(cache_path)
    print(f"Cache written: {cache_path.name}")
    return dep


def main():
    dep = load_or_build(CACHE_PATH)
    print(f"Corrupted rows: {dep[CORRUPT].mean():.2%}")
    print(f"Stand matching available: {dep['has_stand_link'].mean():.1%} "
          f"(same aircraft type: {dep['prev_same_type'].mean():.1%})")
    print(f"Arrival context available: "
          f"{dep['arr_taxi_mean_win'].notna().mean():.1%}")
    print(f"Queue at off-block: Q available {dep['q_ahead_dep_rwy'].notna().mean():.1%}, "
          f"N median {dep['q_n_taxiing_rwy'].median():.0f}, Q median {dep['q_ahead_dep_rwy'].median():.0f}, "
          f"recent runway taxi median {dep['q_recent_taxi_rwy'].median():.0f} s")
    hi = dep["arr_corrupt_frac_win"] > 0.3
    print(f"Departures in a window with >30 % corrupted arrivals: "
          f"{hi.mean():.2%}, of which corrupted: {dep.loc[hi, CORRUPT].mean():.1%} "
          f"(base : {dep[CORRUPT].mean():.1%})")

    is_val = dep["month"].isin(VAL_MONTHS)
    tr_all, va_all = dep[~is_val].copy(), dep[is_val].copy()
    print(f"Temporal split: train {len(tr_all):,} / val {len(va_all):,}")

    print(f"\nOut-of-fold target encoding ({TE_FOLDS} folds)...")
    tr_all = add_te_out_of_fold(tr_all, CORRUPT)
    maps_tr, prior_tr = fit_te_maps(tr_all, tr_all[CORRUPT].values)
    va_all = apply_te(va_all, maps_tr, prior_tr)

    print("Out-of-fold unimpeded time (p10 / p50 per stand-runway)...")
    tr_all = add_unimpeded_out_of_fold(tr_all)
    umaps_tr = fit_unimpeded_maps(tr_all)
    va_all = apply_unimpeded(va_all, umaps_tr)
    lv = va_all["unimpeded_level"]
    print(f"  stand-runway level: {(lv == 0).mean():.1%}, runway: {(lv == 1).mean():.1%}, "
          f"airport: {(lv == 2).mean():.1%}")
    cl = va_all[(va_all[CORRUPT] == 0) & (va_all[TARGET] <= TAIL_THR)]
    print(f"  naive p50-only RMSE on clean rows outside the tail: "
          f"{rmse(cl['unimpeded_p50'], cl[TARGET]):.1f} s "
          f"(global median: {rmse(np.full(len(cl), cl[TARGET].median()), cl[TARGET]):.1f} s)")

    y_va = va_all[TARGET].values
    sched = va_all["delta_mvt_sched_s"].values
    ok = np.isfinite(sched)
    sched_safe = np.where(ok, sched, 0.0)

    catsAll = get_categories(tr_all)
    X_tr = prepare_matrix(tr_all, extra=TE_FEATURES + UNIMP_FEATURES)
    X_va = prepare_matrix(va_all, catsAll, extra=TE_FEATURES + UNIMP_FEATURES)

    # ---- Corruption classifier -------------------------------------------
    print("\n-- Corruption classifier --")
    clf_c = train_ensemble(LGB_CLF, X_tr, tr_all[CORRUPT].values,
                           X_va, va_all[CORRUPT].values, num_rounds=3000,
                           stopping=150, label="clf")
    bestC = clf_c.iters
    p_c = clf_c.predict(X_va)
    y_corr = va_all[CORRUPT].values
    print(f"True rate {y_corr.mean():.2%} | mean p {p_c.mean():.2%}")
    for thr in (0.3, 0.5, 0.7, 0.9):
        sel = p_c > thr
        if sel.sum():
            print(f"  p > {thr}: {sel.sum():6,} rows, "
                  f"precision {y_corr[sel].mean():5.1%}, "
                  f"recall {y_corr[sel].sum() / max(y_corr.sum(), 1):5.1%}"
                  f"  (v14 at p>0.5: 71.8 % / 11.1 %)")
    print("\nTop 12 classifier features (gain):")
    print(pd.Series(clf_c.feature_importance(importance_type="gain"),
                    index=X_tr.columns).sort_values(ascending=False)
          .head(12).to_string())

    clean_tr = tr_all[tr_all[CORRUPT] == 0]
    clean_va = va_all[va_all[CORRUPT] == 0]
    UNIMP_FALLBACK = float(clean_tr[TARGET].median())

    def fit_pair(train_df, val_df):
        ha_t = train_df["delta_mvt_aobt3_s"].notna()
        ha_v = val_df["delta_mvt_aobt3_s"].notna()
        tA, vA = train_df[ha_t].copy(), val_df[ha_v].copy()
        tA["residual"] = tA[TARGET] - tA["delta_mvt_aobt3_s"]
        vA["residual"] = vA[TARGET] - vA["delta_mvt_aobt3_s"]
        cA = get_categories(tA)
        mA_ = train_ensemble(LGB_REG, prepare_matrix(tA, extra=TE_FEATURES + UNIMP_FEATURES),
                             tA["residual"].values,
                             prepare_matrix(vA, cA, extra=TE_FEATURES + UNIMP_FEATURES),
                             vA["residual"].values, label="regA")
        # Regressor B: target = EXCESS over the unimpeded time (p50)
        cB = get_categories(train_df)
        yB = (train_df[TARGET] - train_df["unimpeded_p50"].fillna(
            train_df[TARGET].median())).values
        vB = val_df[~ha_v]
        if len(vB):
            yvB = (vB[TARGET] - vB["unimpeded_p50"].fillna(
                train_df[TARGET].median())).values
            mB_ = train_ensemble(LGB_REG, prepare_matrix(train_df, extra=TE_FEATURES + UNIMP_FEATURES),
                                 yB,
                                 prepare_matrix(vB, cB, extra=TE_FEATURES + UNIMP_FEATURES),
                                 yvB, label="regB")
            iB = mB_.iters
        else:
            iB = [max(mA_.best_iteration // 2, 200)] * len(SEEDS)
            mB_ = train_ensemble(LGB_REG, prepare_matrix(train_df, extra=TE_FEATURES + UNIMP_FEATURES),
                                 yB, fixed_rounds=iB)
        return mA_, cA, mA_.iters, mB_, cB, iB

    def fit_pair_fixed(train_df, rounds_a, rounds_b):
        """Final retraining: rounds frozen PER SEED (the validated ones)."""
        ha_t = train_df["delta_mvt_aobt3_s"].notna()
        tA = train_df[ha_t].copy()
        tA["residual"] = tA[TARGET] - tA["delta_mvt_aobt3_s"]
        cA = get_categories(tA)
        mA_ = train_ensemble(LGB_REG, prepare_matrix(tA, extra=TE_FEATURES + UNIMP_FEATURES),
                             tA["residual"].values, fixed_rounds=rounds_a)
        cB = get_categories(train_df)
        yB = (train_df[TARGET] - train_df["unimpeded_p50"].fillna(
            train_df[TARGET].median())).values
        mB_ = train_ensemble(LGB_REG, prepare_matrix(train_df, extra=TE_FEATURES + UNIMP_FEATURES),
                             yB, fixed_rounds=rounds_b)
        return mA_, cA, mB_, cB

    def predict_normal(df, ca, cb, ma, mb, ia=None, ib=None):
        out = np.full(len(df), np.nan)
        ha = df["delta_mvt_aobt3_s"].notna().values
        if ha.any():
            out[ha] = (df.loc[ha, "delta_mvt_aobt3_s"].values
                       + ma.predict(prepare_matrix(df[ha], ca, extra=TE_FEATURES + UNIMP_FEATURES)))
        if (~ha).any():
            base = df.loc[~ha, "unimpeded_p50"].fillna(UNIMP_FALLBACK).values
            out[~ha] = base + mb.predict(
                prepare_matrix(df[~ha], cb, extra=TE_FEATURES + UNIMP_FEATURES))
        return np.clip(out, TAXI_MIN_S, TAXI_MAX_S)

    # ---- Regressors ------------------------------------------------------------
    print("\n-- Regressor A (AOBT_3 residual, clean rows) --")
    mA, cA, iA, mB, cB, iB = fit_pair(clean_tr, clean_va)
    ha_v = clean_va["delta_mvt_aobt3_s"].notna()
    vA_ = clean_va[ha_v]
    r_ens = rmse(mA.predict(prepare_matrix(vA_, cA, extra=TE_FEATURES + UNIMP_FEATURES)),
                 (vA_[TARGET] - vA_["delta_mvt_aobt3_s"]).values)
    print(f"   residual RMSE, ensemble of {len(SEEDS)} seeds: {r_ens:.1f} s "
          f"(v14 : 225.4 s)")
    print("\nTop 15 features of regressor A (gain):")
    print(pd.Series(mA.feature_importance(importance_type="gain"),
                    index=prepare_matrix(clean_tr.head(1), extra=TE_FEATURES + UNIMP_FEATURES).columns)
          .sort_values(ascending=False).head(15).to_string())

    mu_normal = predict_normal(va_all, cA, cB, mA, mB, iA, iB)
    nb = clean_va["delta_mvt_aobt3_s"].isna().values
    if nb.any():
        idx_nb = va_all.index.get_indexer(clean_va.index[nb])
        print(f"   RMSE of B without AOBT_3 (as taxi, excess + p50): "
              f"{rmse(mu_normal[idx_nb], clean_va.loc[nb, TARGET]):.1f} s  (v14 : 3038 s)")

    # ---- Mixture and sweep ------------------------------------------------
    print("\n" + "=" * 68)
    print("MIXTURE (v14 reference: 369.3 s mixture alone, 345.9 s with Rome)")
    print("=" * 68)
    print(f"RMSE without mixture: {rmse(mu_normal, y_va):.1f} s  (v14: 531.6)")
    g0, best_r = None, np.inf
    for g in (0.5, 0.6, 0.75, 1.0):
        pg = np.where(ok, p_c ** g, 0.0)
        r = rmse(np.maximum(pg * sched_safe + (1 - pg) * mu_normal,
                            TAXI_MIN_S), y_va)
        mark = ""
        if r < best_r:
            best_r, g0, mark = r, g, "  <--"
        print(f"    p^{g:<5} : {r:8.1f} s{mark}")
    print(f"\nRMSE mixture alone: {best_r:.1f} s (p^{g0})   [v14: 369.3 s]")

    final_val = np.maximum(np.where(ok, p_c ** g0, 0.0) * sched_safe
                           + (1 - np.where(ok, p_c ** g0, 0.0)) * mu_normal,
                           TAXI_MIN_S)
    # ---- v11: Rome branch --------------------------------------------------
    print("\n" + "=" * 68)
    print("ROME BRANCH (unmatched, takeoff on the next day)")
    print("=" * 68)
    rates_tr = fit_rome_rates(tr_all)
    print("Rates estimated on the training set (per delta_sched bucket):")
    labels = ["0-1 h", "1-3 h", "3-12 h", "> 12 h"]
    for k in range(4):
        print(f"  {labels[k]:7s}: +1 day {rates_tr[k, 0]:5.1%}  "
              f"block=sched {rates_tr[k, 1]:5.1%}  normal {rates_tr[k, 2]:5.1%}")

    rmse_before = rmse(final_val, y_va)
    final_rome, m_rome = apply_rome_branch(final_val, va_all, mu_normal, rates_tr)
    rmse_after = rmse(final_rome, y_va)
    print(f"\nValidation subset: {m_rome.sum()} flights")
    if m_rome.any():
        cat_va = rome_categorize(va_all[m_rome])
        print(f"  actual composition: +1 day {(cat_va == 0).sum()}, "
              f"block=sched {(cat_va == 1).sum()}, normal {(cat_va == 2).sum()}")
        e_before = final_val[m_rome] - y_va[m_rome]
        e_after = final_rome[m_rome] - y_va[m_rome]
        n_va = len(y_va)
        print(f"  MSE contribution: before {np.sum(e_before**2)/n_va:,.0f}, "
              f"after {np.sum(e_after**2)/n_va:,.0f}  "
              f"(total MSE before: {rmse_before**2:,.0f})")
    print(f"\nRMSE without Rome branch: {rmse_before:.1f} s")
    print(f"RMSE with Rome branch   : {rmse_after:.1f} s   "
          f"(gain {rmse_before - rmse_after:+.1f} s)   [v14: 345.9 s]")

    # Transfer check: rates fitted on one month, evaluated on the other.
    # Here the rates are estimated on the 10 training months, so we
    # only check that the gain is present in EACH of the two months.
    print("\nGain per validation month:")
    month = va_all["month"].values
    gains = []
    for mo in VAL_MONTHS:
        mm = month == mo
        rb, ra = rmse(final_val[mm], y_va[mm]), rmse(final_rome[mm], y_va[mm])
        gains.append(rb - ra)
        print(f"  month {mo}: {rb:.1f} -> {ra:.1f} s  (gain {rb - ra:+.1f} s, "
              f"{(m_rome & mm).sum()} flights concerned)")
    use_rome_local = all(g > -1.0 for g in gains) and (rmse_before - rmse_after) > 0
    print("\nRome branch: local gain", "YES" if use_rome_local else "NO",
          "| DISABLED in production (v16): +19.8 s local, -2.45 s public "
          "(v10 337.95 without, v11 340.40 with). The composition of the subset "
          "changed in 2026.")
    use_rome = False

    # ---- v16: recalibration of the mixture weight ----------------------------
    # The mixture reads y = n + w * (s - n). With a calibrated p, the L2 optimum is
    # w = p; the exponent gamma is only a uniform recalibration. But the
    # cost of an error on w is proportional to (s - n)^2, which the
    # log-loss of the classifier ignores. We therefore recalibrate w = f(p, log(s - n))
    # with a deliberately small model, monotone in p, without airport
    # (checked: the gain does not come from memorizing airports).
    # Transfer January <-> July on val_preds_v15: +7.6 / +1.3 s.
    # Trained on the full validation set (predictions of 10-month models),
    # applied to the ranking file.
    print("\n" + "=" * 68)
    print("MIXTURE RECALIBRATION w = f(p, log(s - n))")
    print("=" * 68)
    gap_va = sched - mu_normal
    Xcal = pd.DataFrame({"p_c": p_c,
                         "log_gap": np.sign(gap_va) * np.log1p(np.abs(gap_va))})
    CAL_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=7,
                      min_data_in_leaf=500, verbosity=-1, seed=42,
                      monotone_constraints=[1, 0])
    CAL_ROUNDS = 150
    month = va_all["month"].values
    y_corr = va_all[CORRUPT].values
    ref_val = np.maximum(mu_normal + np.where(ok, p_c ** g0, 0.0) * (sched_safe - mu_normal),
                         TAXI_MIN_S)
    for fit_m, test_m in ((1, 7), (7, 1)):
        trm, tem = month == fit_m, month == test_m
        cal = lgb.train(CAL_PARAMS, lgb.Dataset(Xcal[trm], y_corr[trm]),
                        num_boost_round=CAL_ROUNDS)
        w = np.where(ok[tem], cal.predict(Xcal[tem]), 0.0)
        pred_cal = np.maximum(mu_normal[tem] + w * (sched_safe[tem] - mu_normal[tem]),
                              TAXI_MIN_S)
        print(f"  fitted on month {fit_m}, tested on month {test_m}: "
              f"{rmse(ref_val[tem], y_va[tem]):.1f} -> {rmse(pred_cal, y_va[tem]):.1f} s")
    calibrator = lgb.train(CAL_PARAMS, lgb.Dataset(Xcal, y_corr),
                           num_boost_round=CAL_ROUNDS)
    w_all = np.where(ok, calibrator.predict(Xcal), 0.0)
    final_val = np.maximum(mu_normal + w_all * (sched_safe - mu_normal), TAXI_MIN_S)
    print(f"  (in-sample, indicative) reference {rmse(ref_val, y_va):.1f} s, "
          f"recalibrated {rmse(final_val, y_va):.1f} s")

    # Save the validation predictions for the MSE breakdown
    pd.DataFrame({
        "MVT_ID_mvt": va_all["MVT_ID_mvt"].values,
        "ADEP_mvt": va_all["ADEP_mvt"].astype(str).values,
        "month": month,
        "y": y_va, "pred": final_val, "pred_ref_gamma": ref_val,
        "p_c": p_c, "mu_normal": mu_normal, "delta_sched": sched,
        "is_corrupt": va_all[CORRUPT].values, "rome_branch": m_rome.astype(int),
    }).to_parquet(DATA_DIR / "val_preds_v15.parquet", index=False)
    print(f"Validation predictions saved: val_preds_v15.parquet")

    print("\nRMSE per airport (v14 in parentheses):")
    ref = {"EDDF": 201.0, "EDDM": 191.4, "EGLL": 279.2, "EHAM": 197.8,
           "LEBL": 239.7, "LEMD": 186.8, "LFPG": 578.8, "LIRF": 713.5,
           "LSZH": 214.1, "LTFM": 265.4}
    apt = va_all["ADEP_mvt"].astype(str).values
    for a in np.unique(apt):
        m = apt == a
        r = rmse(final_val[m], y_va[m])
        print(f"  {a} : {r:8.1f} s  ({ref.get(a, float('nan')):6.1f})  "
              f"{r - ref.get(a, r):+7.1f}")

    # ---- Retraining on all of 2025 ------------------------------------------
    print("\n" + "=" * 68)
    print("Retraining on the whole of 2025")
    print("=" * 68)
    print(f"Rounds frozen per seed: classifier {bestC}, regressor A {iA}, B {iB}")
    dep_te = add_te_out_of_fold(dep, CORRUPT)
    maps_f, prior_f = fit_te_maps(dep_te, dep_te[CORRUPT].values)
    dep_te = add_unimpeded_out_of_fold(dep_te)
    umaps_f = fit_unimpeded_maps(dep_te)
    catsC_f = get_categories(dep_te)
    clf_c_f = train_ensemble(LGB_CLF, prepare_matrix(dep_te, extra=TE_FEATURES + UNIMP_FEATURES),
                             dep_te[CORRUPT].values, fixed_rounds=bestC)
    clean_f = dep_te[dep_te[CORRUPT] == 0]
    mA_f, cA_f, mB_f, cB_f = fit_pair_fixed(clean_f, iA, iB)
    rates_f = fit_rome_rates(dep_te) if use_rome else None

    # ---- Ranking ------------------------------------------------------------
    rank_path = DATA_DIR / "ranking.parquet"
    subm_path = DATA_DIR / "submitting.parquet"
    if not rank_path.exists() or not subm_path.exists():
        print("\nRanking files not found: submission skipped.")
        return

    print("\nLoading the ranking dataset...")
    rank = pd.read_parquet(rank_path)
    print("Computing features (ranking)...")
    rdep = build_all(rank)
    rdep = apply_te(rdep, maps_f, prior_f)
    rdep = apply_unimpeded(rdep, umaps_f)
    lvr = rdep["unimpeded_level"]
    print(f"Ranking, unimpeded level: stand-runway {(lvr == 0).mean():.1%}, "
          f"runway {(lvr == 1).mean():.1%}, airport {(lvr == 2).mean():.1%}")
    print(f"Ranking: {len(rdep):,} departures | stand matching "
          f"{rdep['has_stand_link'].mean():.1%} | arrival context "
          f"{rdep['arr_taxi_mean_win'].notna().mean():.1%}")

    p_c_r = clf_c_f.predict(prepare_matrix(rdep, catsC_f, extra=TE_FEATURES + UNIMP_FEATURES))
    sched_r = rdep["delta_mvt_sched_s"].values
    ok_r = np.isfinite(sched_r)
    sched_r_safe = np.where(ok_r, sched_r, 0.0)
    mu_r = predict_normal(rdep, cA_f, cB_f, mA_f, mB_f)
    print(f"Mean P(corrupted): {p_c_r.mean():.2%}")

    subm_tpl = pd.read_parquet(subm_path).drop(columns=[TARGET], errors="ignore")
    print("\nSubmissions:")
    # Two files per gamma: WITHOUT and WITH the Rome branch. The public score
    # of v7 (341.3) only recovered 6 s of the 31 s gained locally since
    # v3: something does not transfer to 2026. Submitting both
    # variants isolates the effect of the Rome branch on the leaderboard.
    n_written = 0
    for i, g in enumerate(GAMMAS):
        pg = np.where(ok_r, p_c_r ** g, 0.0)
        preds_g = np.maximum(pg * sched_r_safe + (1 - pg) * mu_r, TAXI_MIN_S)

        gap_r = sched_r - mu_r
        Xcal_r = pd.DataFrame({"p_c": p_c_r,
                               "log_gap": np.sign(gap_r) * np.log1p(np.abs(gap_r))})
        w_r = np.where(ok_r, calibrator.predict(Xcal_r), 0.0)
        preds_cal = np.maximum(mu_r + w_r * (sched_r_safe - mu_r), TAXI_MIN_S)
        print(f"  recalibrated weight on the ranking set: mean {w_r.mean():.3f} "
              f"(mean p^gamma {pg.mean():.3f})")

        variants = [("refGamma", preds_g), ("recalibrated", preds_cal)]

        for tag, pv in variants:
            s = subm_tpl.merge(
                pd.DataFrame({"MVT_ID_mvt": rdep["MVT_ID_mvt"].values, "pred": pv}),
                on="MVT_ID_mvt", how="left")
            if s["pred"].isna().any():
                print(f"  Warning: {s['pred'].isna().sum()} without prediction.")
                s["pred"] = s["pred"].fillna(float(np.median(dep[TARGET])))
            s[TARGET] = s["pred"]
            s = s[["MVT_ID_mvt", TARGET]]
            name = f"{TEAM}_v{START_V + n_written}.parquet"
            n_written += 1
            s.to_parquet(OUT_DIR / name, index=False)
            print(f"  {name:38s} {tag:10s} "
                  f"median {np.median(pv):5.0f} s  "
                  f"mean {pv.mean():5.0f} s  max {pv.max():7.0f} s")
            print(f"    .\\mc.exe cp submissions\\{name} dc26/prc-2026-{TEAM}/")


if __name__ == "__main__":
    main()
