"""
PRC Data Challenge 2026 - Taxi-out v19: single script
======================================================

Self-contained merge of taxiout_v16.py (features, encodings, first stage) and
of taxiout_v17.py (out-of-fold residual corrector), plus the demand-queue
features added in v19. A single file, with no dependency on
taxiout_v16.py or taxiout_v17.py.

New in v19 compared with v17:
  q_sched_apt, q_sched_rwy   demand queue counted on the scheduled time
  q_eobt_apt,  q_eobt_rwy    and on the EOBT, hence also available for
                             rows not matched to NM, unlike the
                             v15 queues that require the flight's own AOBT
  rwy_burst_pos              consecutive takeoffs less than 2 min apart
  rwy_gap_mean5_s            mean of the last five intervals on the runway
  wake_pair                  wake pair with the previous takeoff

Options (see --help for the full list):
  python taxiout_v19.py                runs the full chain
  python taxiout_v19.py --no-month     removes "month" from the features (season
                                       proxy, suspected of degrading July)
  python taxiout_v19.py --demand       enables the v19 demand features, to measure
                                       their contribution at equal configuration

The feature cache has a distinct name for each configuration, so there is
nothing to delete between two runs.
"""

import argparse



from pathlib import Path
import glob

import numpy as np
import pandas as pd
import lightgbm as lgb

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
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
    if USE_DEMAND:
        print("  demand queue (SCHED / EOBT)...")
        raw = add_demand_queue_features(raw)
    if USE_DEGRADED:
        print("  degraded day (taxi-in relative to its norm)...")
        raw = add_degraded_day_features(raw)
    if USE_FOLLOWING:
        print("  contexte posterieur (decollages suivants)...")
        raw = add_following_features(raw)
    if USE_NEIGHBOR:
        print("  stand, airport and arrival neighborhood...")
        raw = add_neighbor_features(raw)
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


def main_v16():
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
                  f"  (v14 a p>0.5 : 71.8 % / 11.1 %)")
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




# ----------------------------------------------------------------------------
# v19: demand queue and takeoff sequence
# ----------------------------------------------------------------------------

DEMAND_NUM = ["q_sched_apt", "q_sched_rwy", "q_eobt_apt", "q_eobt_rwy",
              "rwy_burst_pos", "rwy_gap_mean5_s"]
DEMAND_CAT = ["wake_pair"]
NUM_FEATURES = NUM_FEATURES + DEMAND_NUM
CAT_FEATURES = CAT_FEATURES + DEMAND_CAT
USE_DEMAND = True


def add_demand_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Demand queue and takeoff sequence, without depending on the flight's own
    AOBT. To be called on the DEP + ARR table, before filtering
    on departures."""
    n = len(df)
    out = {k: np.full(n, np.nan) for k in
           ("q_sched_apt", "q_sched_rwy", "q_eobt_apt", "q_eobt_rwy",
            "rwy_burst_pos", "rwy_gap_mean5_s")}
    wake_pair = np.full(n, "NA", dtype=object)

    airport = np.where(df["PHASE_mvt"].values == "DEP",
                       df["ADEP_mvt"].astype(str).values,
                       df["ADES_mvt"].astype(str).values)
    rwy = df["RUNWAY_mvt"].astype(str).values
    is_dep = (df["PHASE_mvt"].values == "DEP")

    def secs(col):
        """Datetime series -> seconds. The unit is forced to nanoseconds before
        the integer conversion: the challenge parquet files are in
        microseconds, and dividing by 1e9 without this gives a factor of 1000."""
        if col not in df.columns:
            return np.full(n, np.nan)
        v = pd.to_datetime(df[col], errors="coerce", utc=True)
        try:
            v = v.dt.as_unit("ns")
        except (AttributeError, ValueError):
            pass
        out_s = v.astype("int64").astype("float64") / 1e9
        out_s[v.isna()] = np.nan
        return out_s.to_numpy()

    t = secs("MVT_TIME_UTC_mvt")
    sched = secs("SCHED_TIME_UTC_mvt")
    eobt = secs("EOBT_1_flt")
    wake = (df["WK_TBL_CAT_flt"].astype(str).values
            if "WK_TBL_CAT_flt" in df.columns else np.full(n, "NA", dtype=object))

    # (airport, month) partitions: the ranking file only contains
    # January and July in isolation, whereas training is continuous. A
    # count straddling month boundaries would thus be truncated on one side and
    # not on the other, which shifts the distribution between 2025 and 2026.
    # Partitioning on both sides evens out this bias.
    ym = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce", utc=True)
    part = (airport.astype(object) + "|"
            + ym.dt.year.astype("Int64").astype(str) + "-"
            + ym.dt.month.astype("Int64").astype(str)).to_numpy()

    pos_all = np.arange(n)
    for a in pd.unique(part):
        sel = (part == a) & np.isfinite(t)
        idx = pos_all[sel]
        if idx.size == 0:
            continue
        tt = t[idx]
        d_idx = idx[is_dep[idx]]
        dep_t_sorted = np.sort(t[d_idx])

        for src, name in ((sched, "q_sched_apt"), (eobt, "q_eobt_apt")):
            v = src[d_idx]
            v = np.sort(v[np.isfinite(v)])
            if v.size == 0:
                continue
            planned = np.searchsorted(v, tt, side="right")
            gone = np.searchsorted(dep_t_sorted, tt, side="right")
            out[name][idx] = np.maximum(planned - gone, 0)

        for r in pd.unique(rwy[idx]):
            sub = idx[rwy[idx] == r]
            if sub.size == 0:
                continue
            ts = t[sub]
            d_sub = sub[is_dep[sub]]
            if d_sub.size == 0:
                continue
            d_sub = d_sub[np.argsort(t[d_sub])]
            dr = t[d_sub]

            for src, name in ((sched, "q_sched_rwy"), (eobt, "q_eobt_rwy")):
                v = src[d_sub]
                v = np.sort(v[np.isfinite(v)])
                if v.size == 0:
                    continue
                out[name][sub] = np.maximum(
                    np.searchsorted(v, ts, side="right")
                    - np.searchsorted(dr, ts, side="right"), 0)

            if dr.size < 2:
                continue
            gaps = np.diff(dr)
            prev = np.searchsorted(dr, ts, side="left") - 1

            # mean of the last five intervals
            csum = np.concatenate([[0.0], np.cumsum(gaps)])
            lo = np.maximum(prev - 5, 0)
            cnt = prev - lo
            g5 = np.full(ts.shape, np.nan)
            ok = (prev >= 1) & (cnt > 0)
            g5[ok] = (csum[prev[ok]] - csum[lo[ok]]) / cnt[ok]
            out["rwy_gap_mean5_s"][sub] = g5

            # burst length
            burst = np.zeros(dr.size)
            for i in range(1, dr.size):
                burst[i] = burst[i - 1] + 1 if gaps[i - 1] < 120 else 0.0
            b = np.full(ts.shape, np.nan)
            has = prev >= 0
            b[has] = burst[prev[has]]
            out["rwy_burst_pos"][sub] = b

            # wake pair with the previous takeoff
            pw = np.full(ts.shape, "NA", dtype=object)
            pw[has] = wake[d_sub][prev[has]]
            wake_pair[sub] = np.char.add(np.char.add(pw.astype(str), ">"),
                                         wake[sub].astype(str))

    for k, v in out.items():
        df[k] = v
    df["wake_pair"] = pd.Series(wake_pair, index=df.index).astype("category")
    return df


# ----------------------------------------------------------------------------
# v19: degraded-day detection (de-icing, LVP, surface incident)
# ----------------------------------------------------------------------------
#
# The arrivals' taxi-in is NOT masked in the ranking file: we
# therefore have, for January and July 2026, the actual state of the surface
# hour by hour. What v16 lacks is not the absolute value of the
# recent taxi-in (arr_taxi_mean_win already gives it) but its RATIO TO THE
# NORM: a mean taxi-in of 700 s is ordinary at CDG at 9 am and signals a
# degraded day at 5 am.
#
# The norm is computed per (airport, month, hour) with a median, hence
# robust to a few exceptional days, and within the same partition on
# both sides to avoid the month-boundary problem.

DEGRADED_NUM = ["arr_taxi_norm_h", "arr_taxi_ratio_1h",
                "arr_taxi_ratio_3h", "arr_taxi_ratio_6h"]
USE_DEGRADED = False


def add_degraded_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of the arrivals' recent taxi-in to its hourly norm. Causal:
    only arrivals before the time considered are used."""
    n = len(df)
    out = {k: np.full(n, np.nan) for k in DEGRADED_NUM}

    airport = np.where(df["PHASE_mvt"].values == "DEP",
                       df["ADEP_mvt"].astype(str).values,
                       df["ADES_mvt"].astype(str).values)
    is_arr = (df["PHASE_mvt"].values == "ARR")
    taxi = pd.to_numeric(df.get("TAXITIME_SEC_mvt"), errors="coerce").to_numpy(dtype=float)

    def _secs(col):
        if col not in df.columns:
            return np.full(n, np.nan)
        v = pd.to_datetime(df[col], errors="coerce", utc=True)
        try:
            v = v.dt.as_unit("ns")
        except (AttributeError, ValueError):
            pass
        s = v.astype("int64").astype("float64") / 1e9
        s[v.isna()] = np.nan
        return s.to_numpy()

    ts = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce", utc=True)
    t = _secs("MVT_TIME_UTC_mvt")
    t_block = _secs("BLOCK_TIME_UTC_mvt")
    hour = ts.dt.hour.to_numpy(dtype="float64")
    part = (airport.astype(object) + "|"
            + ts.dt.year.astype("Int64").astype(str) + "-"
            + ts.dt.month.astype("Int64").astype(str)).to_numpy()

    def _causal_mean(ev_t, ev_v, q, window):
        order = np.argsort(ev_t)
        et, ev = ev_t[order], ev_v[order]
        ok = np.isfinite(ev)
        csum = np.concatenate([[0.0], np.cumsum(np.where(ok, ev, 0.0))])
        ccnt = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
        hi = np.searchsorted(et, q, side="right")
        lo = np.searchsorted(et, q - window, side="left")
        s, c = csum[hi] - csum[lo], ccnt[hi] - ccnt[lo]
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)

    pos_all = np.arange(n)
    for p in pd.unique(part):
        sel = (part == p) & np.isfinite(t)
        idx = pos_all[sel]
        if idx.size == 0:
            continue
        a_idx = idx[is_arr[idx] & np.isfinite(taxi[idx])]
        if a_idx.size < 50:
            continue

        # hourly norm, robust median over the partition
        norm = (pd.DataFrame({"h": hour[a_idx], "v": taxi[a_idx]})
                .groupby("h")["v"].median())
        base = pd.Series(hour[idx]).map(norm).to_numpy(dtype=float)
        out["arr_taxi_norm_h"][idx] = base

        # the taxi-in of an arrival is known at its in-block time
        ev_t = np.where(np.isfinite(t_block[a_idx]), t_block[a_idx], t[a_idx])
        ev_v = taxi[a_idx]
        for win, name in ((3600.0, "arr_taxi_ratio_1h"),
                          (10800.0, "arr_taxi_ratio_3h"),
                          (21600.0, "arr_taxi_ratio_6h")):
            recent = _causal_mean(ev_t, ev_v, t[idx], win)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[name][idx] = np.where(np.isfinite(base) & (base > 0),
                                          recent / base, np.nan)

    for k, v in out.items():
        df[k] = v
    return df


NUM_FEATURES = NUM_FEATURES + DEGRADED_NUM


# ----------------------------------------------------------------------------
# v19: dense posterior context (non-causal features)
# ----------------------------------------------------------------------------
#
# The task is a retrospective reconstruction: the ranking file
# contains all the movements of January and July 2026, including those
# taking off after the target row. The following takeoffs, and above all
# their NM off-block times, directly inform on the queue in
# which the flight was.
#
# None of these columns uses BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt
# of departures, which are masked in the ranking file.
#
# These features have no operational meaning: a controller does not know
# future takeoffs. To be stated explicitly in any publication.

FOLLOWING_NUM = [
    # intervals to the following takeoffs on the runway
    "gap_next_1_rwy", "gap_next_2_rwy", "gap_next_3_rwy",
    "gap_next_5_rwy", "gap_next_10_rwy", "gap_next_20_rwy",
    "rate_next_5_rwy", "rate_next_10_rwy", "rate_next_20_rwy",
    # upcoming density, per runway and per airport
    "dep_ahead_5_rwy", "dep_ahead_15_rwy", "dep_ahead_30_rwy", "dep_ahead_60_rwy",
    "dep_ahead_5_apt", "dep_ahead_15_apt", "dep_ahead_30_apt", "dep_ahead_60_apt",
    # symmetric density: the queue around the flight
    "dep_sym_15_rwy", "dep_sym_30_rwy", "dep_sym_30_apt",
    "ahead_share_15_rwy", "ahead_share_30_apt",
    # rafales
    "burst_ahead_rwy", "pos_in_burst_rwy", "burst_len_rwy",
    "burst_time_ahead_rwy", "burst_time_since_rwy", "burst_share_done",
    # symmetry and regime of the runway
    "gap_ratio_next_prev", "rate_ahead_vs_norm_rwy", "gap_next_vs_norm_rwy",
    # NM clocks of the following flights: who was already taxiing behind me
    "next_aobt_minus_t", "next_eobt_minus_t",
    "n_behind_taxiing_5", "n_behind_taxiing_10", "behind_aobt_mean_5",
    "next_sched_slack", "next_slack_mean_5",
    # composition of the upcoming flow
    "n_heavy_next_5", "next_same_operator", "rwy_switch_next",
]
USE_FOLLOWING = False


def add_following_features(df: pd.DataFrame) -> pd.DataFrame:
    """Context of the later takeoffs, per (airport, month, runway)."""
    n = len(df)
    out = {k: np.full(n, np.nan) for k in FOLLOWING_NUM}

    airport = np.where(df["PHASE_mvt"].values == "DEP",
                       df["ADEP_mvt"].astype(str).values,
                       df["ADES_mvt"].astype(str).values)
    rwy = df["RUNWAY_mvt"].astype(str).values
    is_dep = (df["PHASE_mvt"].values == "DEP")

    def _secs(col):
        if col not in df.columns:
            return np.full(n, np.nan)
        v = pd.to_datetime(df[col], errors="coerce", utc=True)
        try:
            v = v.dt.as_unit("ns")
        except (AttributeError, ValueError):
            pass
        s = v.astype("int64").astype("float64") / 1e9
        s[v.isna()] = np.nan
        return s.to_numpy()

    ts_dt = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce", utc=True)
    t = _secs("MVT_TIME_UTC_mvt")
    sched = _secs("SCHED_TIME_UTC_mvt")
    aobt = _secs("AOBT_3_flt")
    eobt = _secs("EOBT_1_flt")
    wake = (df["WK_TBL_CAT_flt"].astype(str).values
            if "WK_TBL_CAT_flt" in df.columns else np.full(n, "NA", dtype=object))
    heavy = np.isin(wake, ("H", "J")).astype(float)
    oper = (df["AIRCRAFT_OPERATOR_flt"].astype(str).values
            if "AIRCRAFT_OPERATOR_flt" in df.columns else np.full(n, "NA", dtype=object))

    part = (airport.astype(object) + "|"
            + ts_dt.dt.year.astype("Int64").astype(str) + "-"
            + ts_dt.dt.month.astype("Int64").astype(str)).to_numpy()

    def _ahead(arr, q, w, start):
        return (np.searchsorted(arr, q + w, side="right") - start).astype(float)

    pos_all = np.arange(n)
    for p in pd.unique(part):
        sel = (part == p) & np.isfinite(t)
        idx = pos_all[sel]
        if idx.size == 0:
            continue
        tt = t[idx]
        d_idx = idx[is_dep[idx]]
        if d_idx.size == 0:
            continue
        dep_apt = np.sort(t[d_idx])
        nxt_apt = np.searchsorted(dep_apt, tt, side="right")
        for w, name in ((300, "dep_ahead_5_apt"), (900, "dep_ahead_15_apt"),
                        (1800, "dep_ahead_30_apt"), (3600, "dep_ahead_60_apt")):
            out[name][idx] = _ahead(dep_apt, tt, w, nxt_apt)
        sym30 = (np.searchsorted(dep_apt, tt + 1800, side="right")
                 - np.searchsorted(dep_apt, tt - 1800, side="left")).astype(float)
        out["dep_sym_30_apt"][idx] = sym30
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ahead_share_30_apt"][idx] = np.where(
                sym30 > 0, out["dep_ahead_30_apt"][idx] / sym30, np.nan)

        for r in pd.unique(rwy[idx]):
            sub = idx[rwy[idx] == r]
            d_sub = sub[is_dep[sub]]
            if d_sub.size < 3:
                continue
            order = np.argsort(t[d_sub])
            d_sub = d_sub[order]
            dr = t[d_sub]
            a_sorted = aobt[d_sub]
            e_sorted = eobt[d_sub]
            s_sorted = sched[d_sub]
            h_sorted = heavy[d_sub]
            o_sorted = oper[d_sub]
            ts_sub = t[sub]
            m = dr.size

            nxt = np.searchsorted(dr, ts_sub, side="right")
            prev = nxt - 1
            has = prev >= 0
            gaps = np.diff(dr)

            def _pick(arr, k, default=np.nan):
                """value of the k-th following takeoff, NaN beyond the end"""
                v = np.full(ts_sub.shape, default, dtype=float)
                j = prev + k
                ok = has & (j < m) & (j >= 0)
                v[ok] = arr[j[ok]]
                return v

            base = np.full(ts_sub.shape, np.nan)
            base[has] = dr[prev[has]]
            for k, name in ((1, "gap_next_1_rwy"), (2, "gap_next_2_rwy"),
                            (3, "gap_next_3_rwy"), (5, "gap_next_5_rwy"),
                            (10, "gap_next_10_rwy"), (20, "gap_next_20_rwy")):
                out[name][sub] = _pick(dr, k) - base
            for k, name in ((5, "rate_next_5_rwy"), (10, "rate_next_10_rwy"),
                            (20, "rate_next_20_rwy")):
                out[name][sub] = out["gap_next_%d_rwy" % k][sub] / float(k)

            for w, name in ((300, "dep_ahead_5_rwy"), (900, "dep_ahead_15_rwy"),
                            (1800, "dep_ahead_30_rwy"), (3600, "dep_ahead_60_rwy")):
                out[name][sub] = _ahead(dr, ts_sub, w, nxt)
            for w, name in ((900, "dep_sym_15_rwy"), (1800, "dep_sym_30_rwy")):
                out[name][sub] = (np.searchsorted(dr, ts_sub + w, side="right")
                                  - np.searchsorted(dr, ts_sub - w, side="left")).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["ahead_share_15_rwy"][sub] = np.where(
                    out["dep_sym_15_rwy"][sub] > 0,
                    out["dep_ahead_15_rwy"][sub] / out["dep_sym_15_rwy"][sub], np.nan)

            # bursts: sequences of takeoffs less than two minutes apart
            tight = gaps < 120
            start = np.zeros(m, dtype=np.int64)
            for i in range(1, m):
                start[i] = start[i - 1] if tight[i - 1] else i
            end = np.arange(m)
            for i in range(m - 2, -1, -1):
                end[i] = end[i + 1] if tight[i] else i
            length = (end - start + 1).astype(float)
            ahead_n = (end - np.arange(m)).astype(float)
            inpos = (np.arange(m) - start).astype(float)
            t_ahead = dr[end] - dr
            t_since = dr - dr[start]
            for name, arr in (("burst_ahead_rwy", ahead_n),
                              ("pos_in_burst_rwy", inpos),
                              ("burst_len_rwy", length),
                              ("burst_time_ahead_rwy", t_ahead),
                              ("burst_time_since_rwy", t_since)):
                v = np.full(ts_sub.shape, np.nan)
                v[has] = arr[prev[has]]
                out[name][sub] = v
            with np.errstate(invalid="ignore", divide="ignore"):
                out["burst_share_done"][sub] = np.where(
                    out["burst_len_rwy"][sub] > 0,
                    (out["pos_in_burst_rwy"][sub] + 1) / out["burst_len_rwy"][sub], np.nan)

            gp = np.full(ts_sub.shape, np.nan)
            okp = prev >= 1
            gp[okp] = gaps[prev[okp] - 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                out["gap_ratio_next_prev"][sub] = np.where(
                    np.isfinite(gp) & (gp > 0), out["gap_next_1_rwy"][sub] / gp, np.nan)

            # runway regime relative to its norm over the partition
            norm_gap = float(np.nanmedian(gaps)) if gaps.size else np.nan
            norm_ahead = float(np.nanmedian(out["dep_ahead_15_rwy"][sub]))
            with np.errstate(invalid="ignore", divide="ignore"):
                if np.isfinite(norm_gap) and norm_gap > 0:
                    out["gap_next_vs_norm_rwy"][sub] = out["gap_next_1_rwy"][sub] / norm_gap
                if np.isfinite(norm_ahead) and norm_ahead > 0:
                    out["rate_ahead_vs_norm_rwy"][sub] = (
                        out["dep_ahead_15_rwy"][sub] / norm_ahead)

            # NM clocks of the following flights
            out["next_aobt_minus_t"][sub] = _pick(a_sorted, 1) - ts_sub
            out["next_eobt_minus_t"][sub] = _pick(e_sorted, 1) - ts_sub
            out["next_sched_slack"][sub] = _pick(dr, 1) - _pick(s_sorted, 1)

            # how many of the following flights had ALREADY pushed back when I took off:
            # they were therefore taxiing behind me, this is the queue itself
            cnt5 = np.zeros(ts_sub.shape)
            cnt10 = np.zeros(ts_sub.shape)
            acc = np.zeros(ts_sub.shape)
            accn = np.zeros(ts_sub.shape)
            slack_acc = np.zeros(ts_sub.shape)
            slack_n = np.zeros(ts_sub.shape)
            heavy_acc = np.zeros(ts_sub.shape)
            for k in range(1, 11):
                ak = _pick(a_sorted, k)
                behind = np.isfinite(ak) & (ak < ts_sub)
                cnt10 += behind
                if k <= 5:
                    cnt5 += behind
                    d = ak - ts_sub
                    ok = np.isfinite(d)
                    acc += np.where(ok, d, 0.0)
                    accn += ok
                    sk = _pick(dr, k) - _pick(s_sorted, k)
                    oks = np.isfinite(sk)
                    slack_acc += np.where(oks, sk, 0.0)
                    slack_n += oks
                    hk = _pick(h_sorted, k)
                    heavy_acc += np.where(np.isfinite(hk), hk, 0.0)
            out["n_behind_taxiing_5"][sub] = cnt5
            out["n_behind_taxiing_10"][sub] = cnt10
            with np.errstate(invalid="ignore", divide="ignore"):
                out["behind_aobt_mean_5"][sub] = np.where(accn > 0, acc / np.maximum(accn, 1), np.nan)
                out["next_slack_mean_5"][sub] = np.where(slack_n > 0, slack_acc / np.maximum(slack_n, 1), np.nan)
            out["n_heavy_next_5"][sub] = heavy_acc

            nxt_op = np.full(ts_sub.shape, np.nan)
            j = prev + 1
            okj = has & (j < m)
            nxt_op[okj] = (o_sorted[j[okj]] == oper[sub][okj]).astype(float)
            out["next_same_operator"][sub] = nxt_op

            sw = np.full(ts_sub.shape, np.nan)
            k_apt = np.searchsorted(dep_apt, ts_sub, side="right")
            oka = k_apt < dep_apt.size
            nxt_rwy_t = _pick(dr, 1)
            sw[oka] = (dep_apt[k_apt[oka]] != nxt_rwy_t[oka]).astype(float)
            out["rwy_switch_next"][sub] = sw

    for k, v in out.items():
        df[k] = v
    return df


NUM_FEATURES = NUM_FEATURES + FOLLOWING_NUM


# ----------------------------------------------------------------------------
# v19: context of the neighbors, of the whole airport and of the arrivals
# ----------------------------------------------------------------------------
#
# Three families, all bidirectional (past and future), in line with
# the posterior context:
#
#   1. Stand neighborhood: the taxi-out depends on who clutters the same
#      taxiway. The prefix of STAND_mvt approximates the satellite
#      or pier (C03 and C07 are neighbors, 826 and 828 too).
#   2. Whole airport rather than runway: departure waves cross
#      runways, and a flight can be delayed by a queue that is not
#      its own.
#   3. Upcoming arrivals: an aircraft landing while you taxi can
#      cross your route, especially in mixed mode.
#
# No masked column is used.

NEIGHBOR_NUM = [
    # stand neighborhood
    "stand_zone_sym_15", "stand_zone_sym_30", "stand_zone_ahead_15",
    "stand_zone_gap_prev", "stand_zone_gap_next", "stand_zone_share_ahead",
    "same_stand_gap_prev", "same_stand_gap_next",
    "stand_zone_arr_sym_30", "stand_zone_load",
    # whole airport
    "gap_next_1_apt", "gap_next_5_apt", "gap_next_10_apt", "gap_next_20_apt",
    "rate_next_5_apt", "rate_next_10_apt", "rate_next_20_apt",
    "gap_prev_1_apt", "gap_ratio_next_prev_apt",
    "burst_len_apt", "pos_in_burst_apt", "burst_ahead_apt",
    "n_behind_taxiing_5_apt", "n_behind_taxiing_10_apt",
    "rate_ahead_vs_norm_apt",
    # arrivals
    "arr_ahead_15_apt", "arr_ahead_30_apt", "arr_sym_30_apt",
    "arr_ahead_15_rwy", "arr_sym_30_rwy",
    "arr_dep_ratio_ahead_30", "next_arr_gap_rwy", "next_arr_gap_apt",
    "arr_inblock_ahead_30",
]
USE_NEIGHBOR = False


def _nb_secs(df, col, n):
    if col not in df.columns:
        return np.full(n, np.nan)
    v = pd.to_datetime(df[col], errors="coerce", utc=True)
    try:
        v = v.dt.as_unit("ns")
    except (AttributeError, ValueError):
        pass
    s = v.astype("int64").astype("float64") / 1e9
    s[v.isna()] = np.nan
    return s.to_numpy()


def _burst_arrays(arr, tight_s=120.0):
    """start, end, length of each burst for a sorted sequence of times"""
    m = arr.size
    if m == 0:
        return (np.zeros(0, dtype=np.int64),) * 2 + (np.zeros(0),) * 3
    gaps = np.diff(arr)
    tight = gaps < tight_s
    start = np.zeros(m, dtype=np.int64)
    for i in range(1, m):
        start[i] = start[i - 1] if tight[i - 1] else i
    end = np.arange(m)
    for i in range(m - 2, -1, -1):
        end[i] = end[i + 1] if tight[i] else i
    return start, end, (end - start + 1).astype(float), \
        (end - np.arange(m)).astype(float), (np.arange(m) - start).astype(float)


def add_neighbor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stand neighborhood, airport context and arrival context."""
    n = len(df)
    out = {k: np.full(n, np.nan) for k in NEIGHBOR_NUM}

    airport = np.where(df["PHASE_mvt"].values == "DEP",
                       df["ADEP_mvt"].astype(str).values,
                       df["ADES_mvt"].astype(str).values)
    rwy = df["RUNWAY_mvt"].astype(str).values
    is_dep = (df["PHASE_mvt"].values == "DEP")
    stand = pd.Series(df["STAND_mvt"].astype(str).values).fillna("NA").to_numpy()
    zone = np.array([s[:1] if len(s) else "NA" for s in stand], dtype=object)

    ts_dt = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce", utc=True)
    t = _nb_secs(df, "MVT_TIME_UTC_mvt", n)
    t_block = _nb_secs(df, "BLOCK_TIME_UTC_mvt", n)
    aobt = _nb_secs(df, "AOBT_3_flt", n)

    part = (airport.astype(object) + "|"
            + ts_dt.dt.year.astype("Int64").astype(str) + "-"
            + ts_dt.dt.month.astype("Int64").astype(str)).to_numpy()

    pos_all = np.arange(n)
    for p in pd.unique(part):
        sel = (part == p) & np.isfinite(t)
        idx = pos_all[sel]
        if idx.size == 0:
            continue
        tt = t[idx]
        d_idx = idx[is_dep[idx]]
        a_idx = idx[~is_dep[idx]]
        if d_idx.size < 3:
            continue

        # --- whole airport ---------------------------------------------------
        order = np.argsort(t[d_idx])
        d_sorted = d_idx[order]
        dr = t[d_sorted]
        a_of = aobt[d_sorted]
        m = dr.size
        nxt = np.searchsorted(dr, tt, side="right")
        prev = nxt - 1
        has = prev >= 0

        def _pick_apt(arr, k):
            v = np.full(tt.shape, np.nan)
            j = prev + k
            ok = has & (j < m) & (j >= 0)
            v[ok] = arr[j[ok]]
            return v

        base = np.full(tt.shape, np.nan)
        base[has] = dr[prev[has]]
        for k, name in ((1, "gap_next_1_apt"), (5, "gap_next_5_apt"),
                        (10, "gap_next_10_apt"), (20, "gap_next_20_apt")):
            out[name][idx] = _pick_apt(dr, k) - base
        for k, name in ((5, "rate_next_5_apt"), (10, "rate_next_10_apt"),
                        (20, "rate_next_20_apt")):
            out[name][idx] = out["gap_next_%d_apt" % k][idx] / float(k)
        gp = np.full(tt.shape, np.nan)
        okp = prev >= 1
        gp[okp] = dr[prev[okp]] - dr[prev[okp] - 1]
        out["gap_prev_1_apt"][idx] = gp
        with np.errstate(invalid="ignore", divide="ignore"):
            out["gap_ratio_next_prev_apt"][idx] = np.where(
                np.isfinite(gp) & (gp > 0), out["gap_next_1_apt"][idx] / gp, np.nan)

        _, _, blen, bahead, bpos = _burst_arrays(dr, 60.0)
        for name, arr in (("burst_len_apt", blen), ("burst_ahead_apt", bahead),
                          ("pos_in_burst_apt", bpos)):
            v = np.full(tt.shape, np.nan)
            v[has] = arr[prev[has]]
            out[name][idx] = v

        c5 = np.zeros(tt.shape)
        c10 = np.zeros(tt.shape)
        for k in range(1, 11):
            ak = _pick_apt(a_of, k)
            behind = np.isfinite(ak) & (ak < tt)
            c10 += behind
            if k <= 5:
                c5 += behind
        out["n_behind_taxiing_5_apt"][idx] = c5
        out["n_behind_taxiing_10_apt"][idx] = c10

        ahead15 = (np.searchsorted(dr, tt + 900, side="right") - nxt).astype(float)
        norm = float(np.nanmedian(ahead15))
        if np.isfinite(norm) and norm > 0:
            out["rate_ahead_vs_norm_apt"][idx] = ahead15 / norm

        # --- arrivals --------------------------------------------------------
        if a_idx.size:
            ar = np.sort(t[a_idx])
            nxt_a = np.searchsorted(ar, tt, side="right")
            out["arr_ahead_15_apt"][idx] = (
                np.searchsorted(ar, tt + 900, side="right") - nxt_a).astype(float)
            out["arr_ahead_30_apt"][idx] = (
                np.searchsorted(ar, tt + 1800, side="right") - nxt_a).astype(float)
            out["arr_sym_30_apt"][idx] = (
                np.searchsorted(ar, tt + 1800, side="right")
                - np.searchsorted(ar, tt - 1800, side="left")).astype(float)
            g = np.full(tt.shape, np.nan)
            ok = nxt_a < ar.size
            g[ok] = ar[nxt_a[ok]] - tt[ok]
            out["next_arr_gap_apt"][idx] = g
            dep_ahead30 = (np.searchsorted(dr, tt + 1800, side="right") - nxt).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["arr_dep_ratio_ahead_30"][idx] = np.where(
                    dep_ahead30 > 0, out["arr_ahead_30_apt"][idx] / dep_ahead30, np.nan)
            # upcoming in-blocks: arrivals that free or occupy the surface
            ab = t_block[a_idx]
            ab = np.sort(ab[np.isfinite(ab)])
            if ab.size:
                out["arr_inblock_ahead_30"][idx] = (
                    np.searchsorted(ab, tt + 1800, side="right")
                    - np.searchsorted(ab, tt, side="left")).astype(float)

        # --- per runway: arrivals crossing the route -------------------------
        for r in pd.unique(rwy[idx]):
            sub = idx[rwy[idx] == r]
            ts_sub = t[sub]
            ar_r = np.sort(t[a_idx[rwy[a_idx] == r]]) if a_idx.size else np.zeros(0)
            if ar_r.size:
                k0 = np.searchsorted(ar_r, ts_sub, side="right")
                out["arr_ahead_15_rwy"][sub] = (
                    np.searchsorted(ar_r, ts_sub + 900, side="right") - k0).astype(float)
                out["arr_sym_30_rwy"][sub] = (
                    np.searchsorted(ar_r, ts_sub + 1800, side="right")
                    - np.searchsorted(ar_r, ts_sub - 1800, side="left")).astype(float)
                g = np.full(ts_sub.shape, np.nan)
                ok = k0 < ar_r.size
                g[ok] = ar_r[k0[ok]] - ts_sub[ok]
                out["next_arr_gap_rwy"][sub] = g

        # --- stand neighborhood ----------------------------------------------
        for z in pd.unique(zone[idx]):
            sub = idx[zone[idx] == z]
            if sub.size < 3:
                continue
            ts_sub = t[sub]
            dz = np.sort(t[sub[is_dep[sub]]])
            az = np.sort(t[sub[~is_dep[sub]]])
            if dz.size:
                k0 = np.searchsorted(dz, ts_sub, side="right")
                sym15 = (np.searchsorted(dz, ts_sub + 900, side="right")
                         - np.searchsorted(dz, ts_sub - 900, side="left")).astype(float)
                ahead = (np.searchsorted(dz, ts_sub + 900, side="right") - k0).astype(float)
                out["stand_zone_sym_15"][sub] = sym15
                out["stand_zone_sym_30"][sub] = (
                    np.searchsorted(dz, ts_sub + 1800, side="right")
                    - np.searchsorted(dz, ts_sub - 1800, side="left")).astype(float)
                out["stand_zone_ahead_15"][sub] = ahead
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["stand_zone_share_ahead"][sub] = np.where(
                        sym15 > 0, ahead / sym15, np.nan)
                gnext = np.full(ts_sub.shape, np.nan)
                ok = k0 < dz.size
                gnext[ok] = dz[k0[ok]] - ts_sub[ok]
                out["stand_zone_gap_next"][sub] = gnext
                gprev = np.full(ts_sub.shape, np.nan)
                okp2 = (k0 - 1) >= 1
                gprev[okp2] = ts_sub[okp2] - dz[k0[okp2] - 2]
                out["stand_zone_gap_prev"][sub] = gprev
                # load of the area relative to its norm over the partition
                nz = float(np.nanmedian(sym15))
                if np.isfinite(nz) and nz > 0:
                    out["stand_zone_load"][sub] = sym15 / nz
            if az.size:
                out["stand_zone_arr_sym_30"][sub] = (
                    np.searchsorted(az, ts_sub + 1800, side="right")
                    - np.searchsorted(az, ts_sub - 1800, side="left")).astype(float)

        # --- exactly the same stand: rotation ---------------------------------
        for st in pd.unique(stand[idx]):
            sub = idx[stand[idx] == st]
            if sub.size < 2:
                continue
            ts_sub = t[sub]
            allt = np.sort(t[sub])
            k0 = np.searchsorted(allt, ts_sub, side="right")
            gn = np.full(ts_sub.shape, np.nan)
            ok = k0 < allt.size
            gn[ok] = allt[k0[ok]] - ts_sub[ok]
            out["same_stand_gap_next"][sub] = gn
            gpv = np.full(ts_sub.shape, np.nan)
            okp3 = (k0 - 2) >= 0
            gpv[okp3] = ts_sub[okp3] - allt[k0[okp3] - 2]
            out["same_stand_gap_prev"][sub] = gpv

    for k, v in out.items():
        df[k] = v
    return df


NUM_FEATURES = NUM_FEATURES + NEIGHBOR_NUM


# ----------------------------------------------------------------------------
# v19: per-airport models and CatBoost ensemble
# ----------------------------------------------------------------------------

USE_PER_AIRPORT = False
CLF_GLOBAL = False
USE_CATBOOST = False
PER_AIRPORT_MIN = 40000     # minimum number of rows to deserve a dedicated model
CAT_ROUNDS_A = 1200
CAT_ROUNDS_B = 300
CAT_WEIGHT = 0.5
PER_RUNWAY = False


def train_cat(X, y, rounds, seed):
    """CatBoost on the same matrix as LightGBM. Categorical columns
    are converted to strings: CatBoost handles them natively and its
    ordered encoding copes better with high cardinalities such as STAND_mvt."""
    from catboost import CatBoostRegressor, Pool
    Xc = X.copy()
    cats = [c for c in Xc.columns if str(Xc[c].dtype) == "category"]
    for c in cats:
        Xc[c] = Xc[c].astype("object").where(Xc[c].notna(), "NA").astype(str)
    pool = Pool(Xc, y, cat_features=cats)
    m = CatBoostRegressor(iterations=rounds, learning_rate=0.05, depth=8,
                          loss_function="RMSE", random_seed=seed,
                          verbose=False, allow_writing_files=False)
    m.fit(pool)
    return m, cats


def predict_cat(model_cats, X):
    m, cats = model_cats
    Xc = X.copy()
    for c in cats:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype("object").where(Xc[c].notna(), "NA").astype(str)
    missing = [c for c in cats if c not in Xc.columns]
    for c in missing:
        Xc[c] = "NA"
    return m.predict(Xc[[c for c in m.feature_names_]])


def fit_stage1_grouped(train_df):
    """One model per sufficiently represented airport, plus a fallback
    model for the others. Rome does not behave like Munich, and a
    single model compromises between the two instead of serving both."""
    if not USE_PER_AIRPORT:
        return {"__global__": fit_stage1(train_df)}
    key = train_df["ADEP_mvt"].astype(str)
    if PER_RUNWAY:
        key = key + "|" + train_df["RUNWAY_mvt"].astype(str)
    train_df = train_df.assign(_grp=key.values)
    counts = key.value_counts()
    big = [a for a, c in counts.items() if c >= PER_AIRPORT_MIN]
    models = {}
    print("    per-airport models: %d dedicated, 1 fallback" % len(big))
    for a in big:
        part = train_df[train_df["_grp"] == a]
        models[a] = fit_stage1(part)
    models["__global__"] = fit_stage1(train_df)
    return models


def predict_stage1_grouped(models, df):
    """Per-airport regressors, classifier of choice.

    A classifier dedicated to Rome sees its fifty corrupted labels in a
    much higher proportion than the global classifier, which drowns them in
    1.7 million rows, and it overfits them. CLF_GLOBAL therefore keeps the
    p_c of the global model while taking the mu of the local model.
    """
    if "__global__" in models and len(models) == 1:
        return predict_stage1(models["__global__"], df)
    apt = df["ADEP_mvt"].astype(str)
    if PER_RUNWAY:
        apt = apt + "|" + df["RUNWAY_mvt"].astype(str)
    apt = apt.values
    parts = []
    for a in pd.unique(apt):
        sub = df[apt == a]
        mdl = models.get(a, models["__global__"])
        loc = predict_stage1(mdl, sub)
        if CLF_GLOBAL and a in models:
            glob = predict_stage1(models["__global__"], sub)
            p_c = glob["p_c"].values
            mu = loc["mu_normal"].values
            sched = loc["delta_mvt_sched_s"].values
            ok = np.isfinite(sched)
            pg = np.where(ok, p_c ** GAMMA, 0.0)
            loc = loc.copy()
            loc["p_c"] = p_c
            loc["mix"] = np.maximum(pg * np.where(ok, sched, 0.0) + (1 - pg) * mu,
                                    TAXI_MIN_S)
            gap = np.where(ok, sched - mu, 0.0)
            loc["gap"] = gap
            loc["log_gap"] = np.sign(gap) * np.log1p(np.abs(gap))
        parts.append(loc)
    return pd.concat(parts).loc[df.index]


# ----------------------------------------------------------------------------
# v19: lean corruption classifier
# ----------------------------------------------------------------------------
#
# Two independent trials showed that adding context features
# improves the clean lane but degrades Rome once the corrupted labels are
# included: the p_c classifier gets more and more material to
# wrongly justify high predictions. All the context is therefore removed
# from it, while the regressors keep using it fully.

CLF_DROP = set(DEMAND_NUM + DEMAND_CAT + DEGRADED_NUM + FOLLOWING_NUM + NEIGHBOR_NUM)
CLF_LEAN = False


def prepare_matrix_clf(df, train_categories=None, extra=()):
    X = prepare_matrix(df, train_categories, extra=extra)
    if CLF_LEAN:
        drop = [c for c in X.columns if c in CLF_DROP]
        if drop:
            X = X.drop(columns=drop)
    return X


# ---- Configuration -----------------------------------------------------------
SEED = 42
GAMMA = 0.75
FOLDS = [[2, 3], [4, 5], [6, 8], [9, 10], [11, 12]]   # folds over the training months
INNER_HOLDOUT = 4                                       # index of the selection fold (11-12)
VAL_MONTHS = [1, 7]

# Frozen rounds of the first stage (typical values observed in v13-v15)
ROUNDS_CLF, ROUNDS_A, ROUNDS_B = 210, 580, 100

# Corrector
CLIP_RESID_S = 3600.0
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
LGB_CORR = dict(objective="huber", alpha=600.0, metric="rmse",
                learning_rate=0.03, num_leaves=63, min_data_in_leaf=300,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=5.0, verbosity=-1, seed=SEED)
CORR_MAX_ROUNDS = 1500
STAGE1_OUT = ["p_c", "mu_normal", "mu_raw", "mix", "gap", "log_gap"]

TEAM_V17, START_V_V17 = "motivated-ostrich", 19


# ---- First stage (compact version, one seed, frozen rounds) --------------

def fit_stage1(train_df: pd.DataFrame) -> dict:
    """Classifier + regressors A/B, target encodings and unimpeded time,
    all fitted on train_df. Frozen rounds (no internal validation)."""
    tr = add_te_out_of_fold(train_df, CORRUPT)
    maps, prior = fit_te_maps(tr, tr[CORRUPT].values)
    tr = add_unimpeded_out_of_fold(tr)
    umaps = fit_unimpeded_maps(tr)
    extra = TE_FEATURES + UNIMP_FEATURES

    catsC = get_categories(tr)
    clf = train_lgb(dict(LGB_CLF, seed=SEED), prepare_matrix_clf(tr, extra=extra),
                        tr[CORRUPT].values, num_rounds=ROUNDS_CLF)

    clean = tr[tr[CORRUPT] == 0]
    ha = clean["delta_mvt_aobt3_s"].notna()
    tA = clean[ha].copy()
    tA["residual"] = tA[TARGET] - tA["delta_mvt_aobt3_s"]
    cA = get_categories(tA)
    XA = prepare_matrix(tA, extra=extra)
    mA = train_lgb(dict(LGB_REG, seed=SEED), XA, tA["residual"].values, num_rounds=ROUNDS_A)
    cbA = train_cat(XA, tA["residual"].values, CAT_ROUNDS_A, SEED) if USE_CATBOOST else None
    cB = get_categories(clean)
    med = float(clean[TARGET].median())
    yB = (clean[TARGET] - clean["unimpeded_p50"].fillna(med)).values
    XB = prepare_matrix(clean, extra=extra)
    mB = train_lgb(dict(LGB_REG, seed=SEED), XB, yB, num_rounds=ROUNDS_B)
    cbB = train_cat(XB, yB, CAT_ROUNDS_B, SEED) if USE_CATBOOST else None
    return dict(maps=maps, prior=prior, umaps=umaps, catsC=catsC, clf=clf,
                cA=cA, mA=mA, cB=cB, mB=mB, med=med, cbA=cbA, cbB=cbB)


def predict_stage1(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Applies the first stage to df (with the encodings in models). Returns df
    enriched with the STAGE1_OUT outputs."""
    d = apply_te(df, models["maps"], models["prior"])
    d = apply_unimpeded(d, models["umaps"])
    extra = TE_FEATURES + UNIMP_FEATURES

    p_c = models["clf"].predict(prepare_matrix_clf(d, models["catsC"], extra=extra))
    mu = np.full(len(d), np.nan)
    mu_lgb = np.full(len(d), np.nan)
    mu_cat = np.full(len(d), np.nan)
    ha = d["delta_mvt_aobt3_s"].notna().values
    if ha.any():
        XhA = prepare_matrix(d[ha], models["cA"], extra=extra)
        rA = models["mA"].predict(XhA)
        base_a = d.loc[ha, "delta_mvt_aobt3_s"].values
        mu_lgb[ha] = base_a + rA
        if models.get("cbA") is not None:
            rA_cat = predict_cat(models["cbA"], XhA)
            mu_cat[ha] = base_a + rA_cat
            rA = (1 - CAT_WEIGHT) * rA + CAT_WEIGHT * rA_cat
        mu[ha] = base_a + rA
    if (~ha).any():
        base = d.loc[~ha, "unimpeded_p50"].fillna(models["med"]).values
        XhB = prepare_matrix(d[~ha], models["cB"], extra=extra)
        rB = models["mB"].predict(XhB)
        mu_lgb[~ha] = base + rB
        if models.get("cbB") is not None:
            rB_cat = predict_cat(models["cbB"], XhB)
            mu_cat[~ha] = base + rB_cat
            rB = (1 - CAT_WEIGHT) * rB + CAT_WEIGHT * rB_cat
        mu[~ha] = base + rB
    mu_raw = mu.copy()
    mu = np.clip(mu, TAXI_MIN_S, TAXI_MAX_S)

    sched = d["delta_mvt_sched_s"].values
    ok = np.isfinite(sched)
    pg = np.where(ok, p_c ** GAMMA, 0.0)
    mix = np.maximum(pg * np.where(ok, sched, 0.0) + (1 - pg) * mu, TAXI_MIN_S)
    gap = np.where(ok, sched - mu, 0.0)

    d["p_c"], d["mu_normal"], d["mu_raw"], d["mix"] = p_c, mu, mu_raw, mix
    d["mu_lgb"] = np.clip(mu_lgb, TAXI_MIN_S, TAXI_MAX_S)
    d["mu_cat"] = np.clip(mu_cat, TAXI_MIN_S, TAXI_MAX_S)
    d["gap"] = gap
    d["log_gap"] = np.sign(gap) * np.log1p(np.abs(gap))
    return d



N_SEEDS = 1
SEED_LIST = [42, 7, 123, 2024, 31337]
MODEL_CACHE = False
MODEL_TAG = "v19"


def _model_key(scope, seed, n_rows):
    import hashlib
    sig = "|".join(str(x) for x in [
        MODEL_TAG, scope, seed, n_rows, ROUNDS_CLF, ROUNDS_A, ROUNDS_B,
        USE_PER_AIRPORT, CLF_GLOBAL, USE_CATBOOST, CLF_LEAN, CAT_ROUNDS_A, CAT_ROUNDS_B,
        len(NUM_FEATURES), len(CAT_FEATURES)])
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def fit_stage1_cached(train_df, scope, seed):
    """Reuses a first stage already trained for the same configuration.

    The key covers the scope, the seed, the number of rows, the capacities and
    the architecture flags, as well as the number of features: an obsolete
    cache can therefore not be served by mistake.
    """
    if not MODEL_CACHE:
        return fit_stage1_grouped(train_df)
    import pickle
    d = DATA_DIR / "model_cache"
    d.mkdir(exist_ok=True)
    f = d / ("stage1_%s.pkl" % _model_key(scope, seed, len(train_df)))
    if f.exists():
        print("    models reloaded from the cache (%s)" % f.name, flush=True)
        with open(f, "rb") as fh:
            return pickle.load(fh)
    mdl = fit_stage1_grouped(train_df)
    try:
        with open(f, "wb") as fh:
            pickle.dump(mdl, fh, protocol=4)
        print("    modeles mis en cache (%s, %.0f Mo)"
              % (f.name, f.stat().st_size / 1e6), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("    mise en cache impossible : %s" % exc)
    return mdl


def fit_predict_seeds(train_df, target_df, label=""):
    """First stage averaged over several seeds, equal weights."""
    global SEED
    seeds = SEED_LIST[:max(1, N_SEEDS)]
    mixes, first = [], None
    for k, sd in enumerate(seeds):
        SEED = sd
        if len(seeds) > 1:
            print("    %s seed %d/%d (seed=%d)..." % (label, k + 1, len(seeds), sd),
                  flush=True)
        mdl = fit_stage1_cached(train_df, label, sd)
        out = predict_stage1_grouped(mdl, target_df)
        mixes.append(out["mix"].values)
        if k == 0:
            first = out
    SEED = seeds[0]
    first = first.copy()
    first["mix"] = np.mean(mixes, axis=0)
    return first


def corrector_matrix(d: pd.DataFrame):
    return prepare_matrix(d, None, extra=TE_FEATURES + UNIMP_FEATURES + STAGE1_OUT)


def align(X: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = X.copy()
    for c, cs in cats.items():
        if c in X.columns:
            s = X[c].astype("object")
            X[c] = pd.Categorical(s.where(s.isin(set(cs))), categories=cs)
    return X


# ---- Pipeline ----------------------------------------------------------------

def main_v17():
    dep = load_or_build(CACHE_PATH)
    is_val = dep["month"].isin(VAL_MONTHS)
    train, val = dep[~is_val].copy(), dep[is_val].copy()
    print(f"Train {len(train):,} / val {len(val):,}")

    # ---- 1. Out-of-fold predictions on the training set -------------------
    print("\n" + "=" * 68)
    print("STEP 1: out-of-fold predictions of the first stage (5 folds, 1 seed)")
    print("=" * 68)
    oof_parts = []
    for k, months in enumerate(FOLDS):
        out = train["month"].isin(months)
        print(f"  fold {k + 1}/5: months {months} predicted by the others "
              f"({(~out).sum():,} training rows)...", flush=True)
        models = fit_stage1(train[~out])
        part = predict_stage1(models, train[out])
        part["_fold"] = k
        oof_parts.append(part)
        r = rmse(part["mix"].values, part[TARGET].values)
        print(f"      out-of-fold mixture RMSE on this fold: {r:.1f} s")
    oof = pd.concat(oof_parts)
    oof["resid"] = oof[TARGET].values - oof["mix"].values
    print(f"\nOut-of-fold mixture RMSE, 10 months: {rmse(oof['mix'], oof[TARGET]):.1f} s")
    print(f"Residual: median {oof['resid'].median():.0f} s, std "
          f"{oof['resid'].std():.0f} s, share |resid| > {CLIP_RESID_S:.0f}: "
          f"{(oof['resid'].abs() > CLIP_RESID_S).mean():.3%}")

    # ---- 2. Corrector: selection on the inner fold -----------------------
    print("\n" + "=" * 68)
    print("STEP 2: corrector (Huber, clipped residual), selection on fold 5")
    print("=" * 68)
    inner = oof["_fold"].values == INNER_HOLDOUT
    y_res = np.clip(oof["resid"].values, -CLIP_RESID_S, CLIP_RESID_S)
    X_all = corrector_matrix(oof)
    cats_corr = {c: X_all[c].cat.categories for c in X_all.columns
                 if str(X_all[c].dtype) == "category"}
    X_fit, X_in = X_all[~inner], align(X_all[inner], cats_corr)

    corr = lgb.train(LGB_CORR, lgb.Dataset(X_fit, label=y_res[~inner]),
                     num_boost_round=CORR_MAX_ROUNDS,
                     valid_sets=[lgb.Dataset(X_in, label=y_res[inner], reference=None)],
                     callbacks=[lgb.early_stopping(150, verbose=False),
                                lgb.log_evaluation(300)])
    best_rounds = corr.best_iteration
    print(f"Iterations kept: {best_rounds}")

    # Sweep of lambda on the inner fold, on the TRUE (unclipped) residual
    c_in = corr.predict(X_in, num_iteration=best_rounds)
    mix_in, y_in = oof.loc[inner, "mix"].values, oof.loc[inner, TARGET].values
    best_l, best_r = 0.0, np.inf
    print("\n  lambda   RMSE fold 5")
    for lam in LAMBDAS:
        r = rmse(np.maximum(mix_in + lam * c_in, TAXI_MIN_S), y_in)
        mark = ""
        if r < best_r:
            best_r, best_l, mark = r, lam, "  <--"
        print(f"  {lam:5.2f}   {r:8.1f} s{mark}")
    print(f"\nFrozen configuration: lambda = {best_l}, {best_rounds} iterations")
    if best_l == 0.0:
        print("  lambda = 0: the corrector brings nothing on the inner fold.")

    print("\nTop 15 corrector features (gain):")
    print(pd.Series(corr.feature_importance(importance_type="gain"),
                    index=X_fit.columns).sort_values(ascending=False).head(15).to_string())

    # Final corrector on the 5 folds (frozen rounds)
    corr_full = lgb.train(LGB_CORR, lgb.Dataset(align(X_all, cats_corr), label=y_res),
                          num_boost_round=best_rounds)

    # ---- 3. Single evaluation on January + July -------------------------
    print("\n" + "=" * 68)
    print("STEP 3: evaluation on January + July 2025 (only once)")
    print("=" * 68)
    v = fit_predict_seeds(train, val, label='evaluation')
    c_val = corr_full.predict(align(corrector_matrix(v), cats_corr))
    ref = v["mix"].values
    cor = np.maximum(ref + best_l * c_val, TAXI_MIN_S)
    y = v[TARGET].values
    print(f"Reference RMSE (mixture p^{GAMMA}): {rmse(ref, y):.1f} s")
    print(f"Corrected RMSE (lambda = {best_l})  : {rmse(cor, y):.1f} s   "
          f"gain {rmse(ref, y) - rmse(cor, y):+.1f} s")
    gains = []
    for mo in VAL_MONTHS:
        m = v["month"].values == mo
        g = rmse(ref[m], y[m]) - rmse(cor[m], y[m])
        gains.append(g)
        print(f"  month {mo}: {rmse(ref[m], y[m]):.1f} -> {rmse(cor[m], y[m]):.1f} s "
              f"(gain {g:+.1f})")
    print("\nPer airport (reference -> corrected):")
    apt = v["ADEP_mvt"].astype(str).values
    for a in np.unique(apt):
        m = apt == a
        print(f"  {a} : {rmse(ref[m], y[m]):7.1f} -> {rmse(cor[m], y[m]):7.1f} s "
              f"({rmse(ref[m], y[m]) - rmse(cor[m], y[m]):+.1f})")
    transfers = all(g > 0 for g in gains)
    print("\nPositive gain in both months:", "YES" if transfers else "NO",
          "" if transfers else "(corrected submission written anyway, read with caution)")

    pd.DataFrame({"MVT_ID_mvt": v["MVT_ID_mvt"].values, "y": y, "ref": ref,
                  "corr": c_val, "final": cor, "month": v["month"].values,
                  "ADEP_mvt": apt}).to_parquet(DATA_DIR / "val_preds_v17.parquet", index=False)

    # ---- 4. Ranking ------------------------------------------------------
    rank_path, subm_path = DATA_DIR / "ranking.parquet", DATA_DIR / "submitting.parquet"
    if not rank_path.exists():
        print("\nranking.parquet not found: submission skipped.")
        return
    print("\n" + "=" * 68)
    print("STEP 4: ranking (first stage on 12 months, corrector as is)")
    print("=" * 68)
    # first stage on 12 months, mean over seeds
    rank = pd.read_parquet(rank_path)
    rdep = build_all(rank)
    r = fit_predict_seeds(dep, rdep, label='ranking')
    c_r = corr_full.predict(align(corrector_matrix(r), cats_corr))
    ref_r = r["mix"].values
    cor_r = np.maximum(ref_r + best_l * c_r, TAXI_MIN_S)
    print(f"Correction on the ranking set: mean {c_r.mean():+.0f} s, "
          f"|c| > 600 s on {(np.abs(c_r) > 600).mean():.2%} of the rows")

    subm_tpl = pd.read_parquet(subm_path).drop(columns=[TARGET], errors="ignore")
    for j, (tag, pv) in enumerate([("ref", ref_r), (f"corr_l{best_l}", cor_r)]):
        s = subm_tpl.merge(pd.DataFrame({"MVT_ID_mvt": r["MVT_ID_mvt"].values, "pred": pv}),
                           on="MVT_ID_mvt", how="left")
        if s["pred"].isna().any():
            s["pred"] = s["pred"].fillna(float(np.median(dep[TARGET])))
        s[TARGET] = s["pred"]
        s = s[["MVT_ID_mvt", TARGET]]
        name = f"{TEAM_V17}_v{START_V_V17 + j}.parquet"
        s.to_parquet(OUT_DIR / name, index=False)
        print(f"  {name:38s} {tag:12s} median {np.median(pv):5.0f} s  "
              f"mean {pv.mean():5.0f} s  max {pv.max():7.0f} s")
        print(f"    .\\mc.exe cp submissions\\{name} dc26/prc-2026-{TEAM_V17}/")




# ----------------------------------------------------------------------------
# Fast mode: a single training of the first stage, no corrector,
# no ranking. To compare feature sets without paying for seven fits.
# ----------------------------------------------------------------------------

ABERRANT_S = 20000.0


def main_fast(sample=None, rounds_scale=1.0, n_seeds=1):
    global ROUNDS_CLF, ROUNDS_A, ROUNDS_B, SEED
    if rounds_scale != 1.0:
        ROUNDS_CLF = int(ROUNDS_CLF * rounds_scale)
        ROUNDS_A = int(ROUNDS_A * rounds_scale)
        ROUNDS_B = int(ROUNDS_B * rounds_scale)
        print("capacite x%.1f : clf %d, regA %d, regB %d"
              % (rounds_scale, ROUNDS_CLF, ROUNDS_A, ROUNDS_B))
    dep = load_or_build(CACHE_PATH)
    is_val = dep["month"].isin(VAL_MONTHS)
    train, val = dep[~is_val].copy(), dep[is_val].copy()
    if sample:
        train = train.sample(frac=sample, random_state=SEED)
        print("sampling: %.0f %% of the training set, %d rows" % (100 * sample, len(train)))
    print("Train %d / val %d" % (len(train), len(val)))

    seeds = [SEED, 7, 123, 2024, 31337][:max(1, n_seeds)]
    preds = []
    for k, sd in enumerate(seeds):
        SEED = sd
        if len(seeds) > 1:
            print("  seed %d/%d (seed=%d)..." % (k + 1, len(seeds), sd), flush=True)
        mdl = fit_stage1_grouped(train)
        vv = predict_stage1_grouped(mdl, val)
        preds.append(vv["mix"].values)
        if k == 0:
            v = vv
    if len(preds) > 1:
        for k, pk in enumerate(preds):
            print("    seed %d alone: RMSE %.2f s"
                  % (k + 1, np.sqrt(np.mean((pk - v[TARGET].values) ** 2))))
    pred = np.mean(preds, axis=0)
    y = v[TARGET].values
    sq = (pred - y) ** 2
    clean = y <= ABERRANT_S

    print("")
    print("=" * 68)
    print("FAST EVALUATION on January + July 2025")
    print("=" * 68)
    print("RMSE global        : %.2f s" % np.sqrt(sq.mean()))
    print("clean-lane RMSE    : %.2f s  (%d aberrant rows excluded)"
          % (np.sqrt(sq[clean].mean()), int((~clean).sum())))
    for mo in VAL_MONTHS:
        m = v["month"].values == mo
        print("  month %-2d: global %8.2f s   clean %8.2f s"
              % (mo, np.sqrt(sq[m].mean()), np.sqrt(sq[m & clean].mean())))

    print("")
    print("%-6s %9s %11s %12s %9s" % ("", "n", "RMSE", "clean RMSE", "MSE share"))
    apt = v["ADEP_mvt"].astype(str).values
    rows = []
    for a in np.unique(apt):
        m = apt == a
        c = clean[m]
        rows.append((a, int(m.sum()), float(np.sqrt(sq[m].mean())),
                     float(np.sqrt(sq[m & clean].mean())) if c.any() else float("nan"),
                     100 * float(sq[m & ~clean].sum()) / float(sq[m].sum())
                     if sq[m].sum() else 0.0))
    for a, n, r, rc, sh in sorted(rows, key=lambda z: -z[3]):
        print("%-6s %9d %9.1f s %10.1f s %7.1f %%" % (a, n, r, rc, sh))

    # airport x month breakdown, on the clean lane
    print("")
    print("clean lane, RMSE per airport and per month:")
    print("%-6s %9s %10s %10s %9s %9s"
          % ("", "n Jan", "January", "July", "gap", "MSE share"))
    mo_v = v["month"].values
    tot_c = float(sq[clean].sum())
    crossed = []
    for a in np.unique(apt):
        m = (apt == a) & clean
        m1, m7 = m & (mo_v == 1), m & (mo_v == 7)
        if m1.sum() == 0 or m7.sum() == 0:
            continue
        r1 = float(np.sqrt(sq[m1].mean()))
        r7 = float(np.sqrt(sq[m7].mean()))
        share = 100 * float(sq[m].sum()) / tot_c if tot_c else 0.0
        crossed.append((a, int(m1.sum()), r1, r7, r7 - r1, share))
    for a, n1, r1, r7, d, share in sorted(crossed, key=lambda z: -z[4]):
        print("%-6s %9d %8.1f s %8.1f s %+8.1f s %7.1f %%" % (a, n1, r1, r7, d, share))
    print("  the gap is the degradation of July compared with January;")
    print("  the MSE share tells how much each airport weighs in the clean error.")

    if USE_CATBOOST and "mu_cat" in v.columns and v["mu_cat"].notna().any():
        print("")
        print("sweep of the CatBoost weight (no retraining):")
        sched_v = v["delta_mvt_sched_s"].values
        okv = np.isfinite(sched_v)
        pgv = np.where(okv, v["p_c"].values ** GAMMA, 0.0)
        for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
            muw = np.where(np.isfinite(v["mu_cat"].values),
                           (1 - w) * v["mu_lgb"].values + w * v["mu_cat"].values,
                           v["mu_lgb"].values)
            mixw = np.maximum(pgv * np.where(okv, sched_v, 0.0) + (1 - pgv) * muw,
                              TAXI_MIN_S)
            sqw = (mixw - y) ** 2
            print("  weight %.2f: global %8.2f s   clean %8.2f s"
                  % (w, np.sqrt(sqw.mean()), np.sqrt(sqw[clean].mean())))

    sq_c = np.sort(sq[clean])[::-1]
    print("")
    print("concentration of the error on the clean lane:")
    for frac in (0.0001, 0.001, 0.01, 0.1):
        k = max(int(len(sq_c) * frac), 1)
        print("  the %6.2f %% worst (%6d rows) carry %5.1f %% of the clean MSE"
              % (100 * frac, k, 100 * sq_c[:k].sum() / sq_c.sum()))
    err = np.abs(pred[clean] - y[clean])
    print("  absolute error: median %5.0f s, p90 %5.0f s, p99 %6.0f s, max %7.0f s"
          % (np.median(err), np.percentile(err, 90), np.percentile(err, 99), err.max()))



def main_submit():
    """Goes directly to the ranking: no out-of-fold folds, no corrector,
    no evaluation. To produce a submission when the configuration has
    already been measured in fast mode."""
    dep = load_or_build(CACHE_PATH)
    rank_path = DATA_DIR / "ranking.parquet"
    subm_path = DATA_DIR / "submitting.parquet"
    if not rank_path.exists():
        sys.exit("ranking.parquet introuvable")
    print("first stage on the 12 months, %d departures" % len(dep))
    rdep = build_all(pd.read_parquet(rank_path))
    r = fit_predict_seeds(dep, rdep, label="ranking")
    pv = r["mix"].values
    print("predictions: median %.0f s, mean %.0f s, max %.0f s"
          % (np.median(pv), pv.mean(), pv.max()))
    tpl = pd.read_parquet(subm_path).drop(columns=[TARGET], errors="ignore")
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": r["MVT_ID_mvt"].values, "pred": pv}),
                    on="MVT_ID_mvt", how="left")
    if out["pred"].isna().any():
        n_na = int(out["pred"].isna().sum())
        print("warning: %d identifiers without prediction, filled with the median" % n_na)
        out["pred"] = out["pred"].fillna(float(np.median(dep[TARGET])))
    out[TARGET] = out["pred"]
    name = "%s_v%d.parquet" % (TEAM_V17, START_V_V17)
    out[["MVT_ID_mvt", TARGET]].to_parquet(OUT_DIR / name, index=False)
    print("written: %s (%d rows)" % (name, len(out)))
    print("  mc.exe cp submissions/%s dc26/prc-2026-%s/" % (name, TEAM_V17))


# ----------------------------------------------------------------------------
# Point d'entree unique
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PRC 2026 - taxi-out v19")
    ap.add_argument("--fast", action="store_true",
                    help="a single training run, no corrector and no submission")
    ap.add_argument("--sample", type=float, default=None,
                    help="fraction of the training set to sample in fast mode, e.g. 0.3")
    ap.add_argument("--no-month", action="store_true",
                    help='removes "month" from the numerical features')
    ap.add_argument("--demand", action="store_true",
                    help="enables the demand-queue features (refuted on the 2025 holdout)")
    ap.add_argument("--degraded", action="store_true",
                    help="enables the degraded-day features")
    ap.add_argument("--following", action="store_true",
                    help="enables the posterior context (following takeoffs)")
    ap.add_argument("--neighbor", action="store_true",
                    help="enables the stand, airport and arrival neighborhood")
    ap.add_argument("--all-context", action="store_true",
                    help="equivalent to --following --neighbor")
    ap.add_argument("--rounds-scale", type=float, default=1.0,
                    help="multiplier of the number of trees of the first stage, e.g. 4")
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of seeds averaged with equal weights (fast and full modes)")
    ap.add_argument("--submit-only", action="store_true",
                    help="goes directly to the ranking, without folds or evaluation")
    ap.add_argument("--model-cache", action="store_true",
                    help="reuses the models already trained for the same configuration")
    ap.add_argument("--per-airport", action="store_true",
                    help="one model per sufficiently represented airport")
    ap.add_argument("--cat-rounds", type=int, default=1200,
                    help="CatBoost iterations of the main regressor")
    ap.add_argument("--cat-weight", type=float, default=0.5,
                    help="weight of CatBoost in the ensemble, 0 = LightGBM only")
    ap.add_argument("--per-runway", action="store_true",
                    help="with --per-airport: one model per airport-runway pair")
    ap.add_argument("--clf-global", action="store_true",
                    help="with --per-airport: keeps the global classifier, "
                         "per-airport regressors")
    ap.add_argument("--catboost", action="store_true",
                    help="CatBoost + LightGBM ensemble with equal weights")
    ap.add_argument("--clf-lean", action="store_true",
                    help="removes the context features from the corruption classifier")
    args = ap.parse_args()

    global USE_DEMAND, USE_DEGRADED, USE_FOLLOWING, USE_NEIGHBOR, CLF_LEAN, CACHE_PATH, NUM_FEATURES, CAT_FEATURES
    USE_DEMAND = args.demand
    USE_DEGRADED = args.degraded
    USE_FOLLOWING = args.following or args.all_context
    USE_NEIGHBOR = args.neighbor or args.all_context
    CLF_LEAN = args.clf_lean
    global USE_PER_AIRPORT, USE_CATBOOST
    global MODEL_CACHE
    MODEL_CACHE = args.model_cache
    global CLF_GLOBAL
    USE_PER_AIRPORT = args.per_airport
    CLF_GLOBAL = args.clf_global
    global CAT_ROUNDS_A, CAT_ROUNDS_B, CAT_WEIGHT, PER_RUNWAY
    CAT_ROUNDS_A = args.cat_rounds
    CAT_ROUNDS_B = max(100, args.cat_rounds // 4)
    CAT_WEIGHT = args.cat_weight
    PER_RUNWAY = args.per_runway
    USE_CATBOOST = args.catboost
    global N_SEEDS, ROUNDS_CLF, ROUNDS_A, ROUNDS_B
    N_SEEDS = max(1, args.seeds)
    if not args.fast and args.rounds_scale != 1.0:
        ROUNDS_CLF = int(ROUNDS_CLF * args.rounds_scale)
        ROUNDS_A = int(ROUNDS_A * args.rounds_scale)
        ROUNDS_B = int(ROUNDS_B * args.rounds_scale)
        print("capacite x%.1f : clf %d, regA %d, regB %d"
              % (args.rounds_scale, ROUNDS_CLF, ROUNDS_A, ROUNDS_B))

    if args.no_month and "month" in NUM_FEATURES:
        NUM_FEATURES = [c for c in NUM_FEATURES if c != "month"]
    if not USE_DEMAND:
        NUM_FEATURES = [c for c in NUM_FEATURES if c not in DEMAND_NUM]
        CAT_FEATURES = [c for c in CAT_FEATURES if c not in DEMAND_CAT]
    if not USE_DEGRADED:
        NUM_FEATURES = [c for c in NUM_FEATURES if c not in DEGRADED_NUM]
    if not USE_FOLLOWING:
        NUM_FEATURES = [c for c in NUM_FEATURES if c not in FOLLOWING_NUM]
    if not USE_NEIGHBOR:
        NUM_FEATURES = [c for c in NUM_FEATURES if c not in NEIGHBOR_NUM]

    tag = ("v19" + ("_nomonth" if args.no_month else "")
           + ("_demand" if USE_DEMAND else "") + ("_degraded" if USE_DEGRADED else ""))
    CACHE_PATH = DATA_DIR / ("dep_features_cache_%s.parquet" % tag)
    print("configuration: %s%s" % (tag, "  [fast mode]" if args.fast else ""))
    print('  "month": %s' % ("no" if args.no_month else "yes"))
    print("  demand queue: %s" % ("yes" if USE_DEMAND else "no"))
    print("  degraded day: %s" % ("yes" if USE_DEGRADED else "no"))
    print("  posterior context: %s" % ("yes" if USE_FOLLOWING else "no"))
    print("  neighborhood/airport/arrivals: %s" % ("yes" if USE_NEIGHBOR else "no"))
    print("  lean classifier: %s" % ("yes" if CLF_LEAN else "no"))
    print("  per-airport models: %s%s"
          % ("yes" if USE_PER_AIRPORT else "no",
             " (global classifier)" if (USE_PER_AIRPORT and CLF_GLOBAL) else ""))
    print("  CatBoost ensemble: %s" % ("yes" if USE_CATBOOST else "no"))
    print("  cache : %s" % CACHE_PATH.name)

    if args.fast:
        main_fast(args.sample, args.rounds_scale, args.seeds)
        return
    if args.submit_only:
        main_submit()
        return
    main_v17()

    if USE_DEGRADED:
        print("")
        print("Fill-rate check of the degraded-day features:")
        dd = pd.read_parquet(CACHE_PATH, columns=[c for c in DEGRADED_NUM])
        for c in DEGRADED_NUM:
            print("  %-22s filled for %5.1f %%   median %6.2f"
                  % (c, 100 * dd[c].notna().mean(), dd[c].median()))


if __name__ == "__main__":
    main()
