"""
PRC Data Challenge 2026 - Taxi-out v19 : script unique
======================================================

Fusion autonome de taxiout_v16.py (features, encodages, premier etage) et
de taxiout_v17.py (correcteur de residu hors-fold), plus les features de
file de demande ajoutees en v19. Un seul fichier, aucune dependance a
taxiout_v16.py ni a taxiout_v17.py.

Nouveautes v19 par rapport a v17 :
  q_sched_apt, q_sched_rwy   file de demande comptee sur l'heure programmee
  q_eobt_apt,  q_eobt_rwy    et sur l'EOBT, donc disponible aussi sur les
                             lignes non appariees a NM, contrairement aux
                             files v15 qui exigent l'AOBT du vol lui-meme
  rwy_burst_pos              decollages consecutifs espaces de moins de 2 min
  rwy_gap_mean5_s            moyenne des cinq derniers intervalles sur la piste
  wake_pair                  couple de sillage avec le decollage precedent

Options :
  python taxiout_v19.py                lance la chaine complete
  python taxiout_v19.py --no-month     retire "month" des features (proxy de
                                       saison, suspecte de degrader juillet)
  python taxiout_v19.py --no-demand    desactive les features v19, pour mesurer
                                       leur apport a configuration egale

Le cache de features porte un nom distinct par configuration, il n'y a donc
rien a supprimer entre deux essais.
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

# Le cache v4 convient (memes colonnes de base) ; le target encoding est
# recalcule a chaque run car il depend du split.
CACHE_PATH = DATA_DIR / "dep_features_cache_v15.parquet"
USE_CACHE = True

TAXI_MIN_S = 30
TAXI_MAX_S = 2 * 3600
CORRUPT_TOL_S = 10

# Seuil de separation corps / queue (secondes). 3600 s : au dela, un taxi
# n'est plus un roulage ordinaire. 0.35 % des departs a CDG, mais 63 % de
# la variance de cet aeroport.
TAIL_THR = 3600

CONGESTION_WINDOW_MIN = 15
VAL_MONTHS = [1, 7]

# Contexte arrivees
ARR_WINDOW_MIN = 30        # fenetre +/- autour du decollage
ARR_TAXI_CLIP_S = 3600     # ecretage du taxi-in avant moyenne
ARR_LONG_S = 1200          # seuil "taxi-in long"

# Enchainement par poste
STAND_LINK_MAX_H = 36      # au dela, pas d'appariement arrivee -> depart

# File d'attente par piste (v12)
RUNWAY_WINDOW_MIN = 10     # decollages sur la meme piste dans les N min AVANT
RUNWAY_WINDOW_AFTER_MIN = 5  # ... et APRES (la file qui suit)

# File d'attente ancree sur l'off-block (v15), definitions de la litterature
QUEUE_RECENT_MIN = 20      # fenetre pour le taxi-out moyen recent (avant mon off-block)
QUEUE_MIN_RECENT_N = 2     # nombre minimal de departs pour calculer la moyenne

# Ensembling (v12) : seeds pour classifieur et regresseurs
SEEDS = [42, 7, 123, 2026, 31337]

# Signal de panne de reporting (v10) : corruption des ARRIVEES
ARR_WIDE_WINDOW_MIN = 120  # deuxieme fenetre, plus large
ARR_ANOM_TAXI_S = 3600     # taxi-in aberrant si > ce seuil ou < 0

# ---- Soumission ------------------------------------------------------------
# Convention imposee par le challenge : <team-name>_v<entier>.parquet
TEAM = "motivated-ostrich"
START_V = 16                   # numero de la prochaine soumission
GAMMAS = [0.75]                # optimum public

# Target encoding : nombre de folds et force du lissage vers le prior
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
    # --- v9 : contexte arrivees ---
    "arr_taxi_mean_win", "arr_taxi_frac_long_win", "arr_taxi_day_mean",
    "arr_count_ctx",
    # --- v10 : panne de reporting vue par les arrivees ---
    "arr_corrupt_frac_win",    # part d'arrivees a in-block == sched, +/- 30 min
    "arr_corrupt_frac_wide",   # idem, +/- 2 h
    "arr_corrupt_day_frac",    # idem, journee aeroport
    "arr_anom_frac_win",       # part d'arrivees a taxi-in aberrant, +/- 30 min
    "arr_anom_frac_wide",      # idem, +/- 2 h
    # --- v15 : file d'attente ancree sur l'off-block (AOBT_3) ---
    "q_n_taxiing_rwy",       # departs meme piste deja en roulage a mon off-block (N)
    "q_n_taxiing_apt",       # idem, tout l'aeroport
    "q_ahead_dep_rwy",       # departs meme piste qui decollent entre mon off-block et mon decollage (Q)
    "q_ahead_dep_apt",       # idem, tout l'aeroport
    "q_ahead_arr_rwy",       # atterrissages meme piste pendant mon roulage
    "q_recent_taxi_rwy",     # taxi-out moyen (MVT - AOBT_3) des departs meme piste, 20 min avant mon off-block
    "q_recent_taxi_apt",     # idem, tout l'aeroport
    "q_recent_n_rwy",        # effectif derriere la moyenne
    "q_pushback_rate_rwy",   # off-blocks meme piste dans les 10 min avant le mien
    # --- v12 : file d'attente par piste ---
    "rwy_dep_before",        # decollages meme piste dans les 10 min avant
    "rwy_dep_after",         # decollages meme piste dans les 5 min apres
    "rwy_gap_prev_s",        # intervalle depuis le decollage precedent meme piste
    "rwy_gap_next_s",        # intervalle jusqu'au decollage suivant meme piste
    "rwy_arr_before",        # atterrissages meme piste dans les 10 min avant
    "rwy_share_win",         # part des departs de l'aeroport sur cette piste (+/- 15 min)
    # --- v9 : enchainement par poste ---
    "turnaround_s", "prev_arr_taxi_in", "prev_same_type",
    "prev_same_operator", "sched_before_inblock", "bound_minus_aobt",
    "has_stand_link",
    # --- v13 : corruption des arrivees au meme poste (signal par avion) ---
    "prev_arr_corrupt",      # in-block == sched sur l'arrivee precedente
    "next_arr_corrupt",      # idem sur l'arrivee suivante
    "both_arr_corrupt",      # les deux (38 % de corruption sur 2025)
]

# Features de target encoding (ajoutees dynamiquement au classifieur)
TE_SPECS = {
    "te_adep": ["ADEP_mvt"],
    "te_adep_flight": ["ADEP_mvt", "FLIGHT_mvt"],   # v13 : numero de vol
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

# Regresseur B : loss robuste (les pathologiques ne tirent plus tout)
# v5.1 : retour a la L2 (huber avait degrade B : 3141 s contre 3047 s)
LGB_REG_ROBUST = dict(LGB_REG)

LGB_CLF = dict(
    objective="binary", metric="auc",
    learning_rate=0.02, num_leaves=255, min_data_in_leaf=200,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
    lambda_l2=2.0, verbosity=-1, seed=42,
)


# ----------------------------------------------------------------------------
# Chargement / features de base (identiques v4)
# ----------------------------------------------------------------------------

def load_training(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "training_*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier training_*.parquet trouve dans {data_dir.resolve()}")
    print(f"Chargement de {len(files)} fichiers d'entrainement...")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    print(f"  -> {len(df):,} mouvements")
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
    """File d'attente au niveau de la PISTE (v12). La congestion aeroport ne
    dit pas sur quelle piste on attend : ici on compte les decollages sur
    la meme piste juste avant et juste apres, et les intervalles entre
    decollages consecutifs (un espacement de 60-90 s trahit une file
    dense, plusieurs minutes un trafic fluide). Tout est observable dans
    le ranking (MVT_TIME et RUNWAY des departs ne sont pas masques).
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

    # Part des departs de l'aeroport qui utilisent cette piste (+/- 15 min) :
    # rwy_dep_win / dep_count_win, ou rwy_dep_win compte les departs de la
    # meme piste dans la fenetre symetrique.
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

    # Cle pour le target encoding horaire
    df["hour_bin"] = mvt.dt.hour

    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


# ----------------------------------------------------------------------------
# Target encoding hors-fold du taux de corruption
# ----------------------------------------------------------------------------

def _key(df: pd.DataFrame, cols: list) -> pd.Series:
    """Cle de groupement, en chaine, robuste aux NaN et aux categories."""
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + "|" + df[c].astype(str)
    return out


def fit_te_maps(df: pd.DataFrame, label: np.ndarray) -> tuple[dict, float]:
    """Statistiques de target encoding sur un dataframe de reference."""
    prior = float(np.mean(label))
    maps = {}
    for name, cols in TE_SPECS.items():
        k = _key(df, cols)
        g = pd.DataFrame({"k": k.values, "y": label}).groupby("k")["y"]
        agg = g.agg(["sum", "count"])
        # Lissage bayesien vers le prior
        maps[name] = ((agg["sum"] + TE_SMOOTH * prior)
                      / (agg["count"] + TE_SMOOTH))
    return maps, prior


def apply_te(df: pd.DataFrame, maps: dict, prior: float) -> pd.DataFrame:
    """Applique des maps de target encoding (val, ranking, ou fold sortant)."""
    df = df.copy()
    for name, cols in TE_SPECS.items():
        k = _key(df, cols)
        df[name] = k.map(maps[name]).astype(float).fillna(prior)
    return df


def add_te_out_of_fold(train: pd.DataFrame, label_col: str,
                       n_folds: int = TE_FOLDS, seed: int = 42) -> pd.DataFrame:
    """Target encoding hors-fold sur le train : chaque ligne recoit un
    encodage calcule SANS elle (sur les autres folds), donc pas de fuite.
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
# v14 : temps de roulage non entrave (methodologie PRU) par poste-piste
# ----------------------------------------------------------------------------
#
# Le taxi-out se decompose en un temps incompressible qui depend de la
# geometrie (poste, piste, distance a parcourir) et un temps additionnel
# qui depend de la congestion. La PRU estime le premier comme un bas
# percentile du taxi observe par couple poste-piste ("unimpeded taxi
# time"). Ici : percentiles 10 et 50 par (aeroport, poste, piste), avec
# repli sur (aeroport, piste) puis (aeroport) quand l'effectif est faible.
# Calcule HORS-FOLD sur l'entrainement, sur les lignes saines et hors
# queue, pour ne pas fuir la cible.
#
# Utilisation :
#   - features numeriques explicites pour A et B (le modele voyait
#     stand_runway comme une categorie et devait reapprendre la geometrie
#     par decoupage, sans generaliser aux couples rares ou nouveaux)
#   - cible du regresseur B (sans AOBT_3) : EXCEDENT = taxi - p50, plus
#     stable que le taxi brut, comme le residu AOBT_3 l'est pour A

UNIMP_MIN_N = 20
UNIMP_LEVELS = [["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
                ["ADEP_mvt", "RUNWAY_mvt"],
                ["ADEP_mvt"]]
UNIMP_FEATURES = ["unimpeded_p10", "unimpeded_p50", "unimpeded_n",
                  "unimpeded_level"]


def _unimp_stats_mask(df: pd.DataFrame) -> np.ndarray:
    """Lignes utilisees pour les statistiques : saines et hors queue."""
    ok = (df[TARGET].values <= TAIL_THR)
    if CORRUPT in df.columns:
        ok &= (df[CORRUPT].values == 0)
    return ok


def fit_unimpeded_maps(df: pd.DataFrame) -> list:
    """Une table (p10, p50, n) par niveau de la hierarchie."""
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
# Utilitaires modele
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
    """Moyenne de modeles LightGBM entraines avec des seeds differentes.
    Expose predict() et best_iteration (moyenne des rounds retenus) pour
    rester interchangeable avec un modele unique."""

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
    """Entraine un modele par seed. Avec X_val : early stopping par seed.
    Avec fixed_rounds : liste (ou entier) de rounds par seed, sans
    validation (reentrainement final)."""
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
# v9 : features contextuelles (calculees sur DEP + ARR avant filtrage)
# ----------------------------------------------------------------------------

def _naive_ns(s: pd.Series) -> np.ndarray:
    """Serie datetime tz-aware -> tableau datetime64[ns] naif."""
    return s.dt.tz_localize(None).values.astype("datetime64[ns]")


def add_arrival_context(df: pd.DataFrame) -> pd.DataFrame:
    """Contexte fourni par les ARRIVEES au meme aeroport autour de chaque
    mouvement. Tout est vectorise par sommes cumulees sur les arrivees
    triees.

    v9  : taxi-in (congestion / meteo contemporaines).
    v10 : corruption des arrivees (in-block == heure programmee) et
          anomalies de taxi-in. Hypothese verifiee sur 2025 : quand plus de
          30 % des arrivees d'une heure sont corrompues, 23 a 30 % des
          departs le sont aussi, contre 4 % sinon. Un aeroport qui ne
          reporte pas ses off-blocks ne reporte pas non plus ses
          in-blocks : c'est un signal de PANNE DE REPORTING, contemporain,
          observable dans le ranking (les arrivees y sont completes).
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

    # Agregats journaliers par aeroport (arrivees seulement)
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
    """File d'attente ANCREE SUR L'OFF-BLOCK (v15).

    Toutes nos features de congestion etaient centrees sur l'heure de
    decollage. La litterature (Idris 2001, Ravizza 2014, Wang 2021, revue
    Yin 2024) est unanime : la variable la plus predictive est la taille de
    la file au moment du push-back, et le taxi-out moyen des departs
    recents. Dans le ranking, BLOCK_TIME est masque pour les departs mais
    AOBT_3 (off-block NM) ne l'est pas, et MVT_TIME non plus : on
    reconstruit donc ces quantites avec AOBT_3 comme off-block.

      N  : departs deja en roulage quand je pousse
           (leur AOBT_3 < mon AOBT_3 < leur MVT_TIME)
      Q  : departs qui decollent entre mon AOBT_3 et mon MVT_TIME
           (ceux qui sont passes devant moi dans la file)
      recent : moyenne de (MVT_TIME - AOBT_3) des departs dont le decollage
           precede mon off-block, dans les QUEUE_RECENT_MIN minutes
      pushback_rate : off-blocks dans les 10 min avant le mien

    Calcule par piste et par aeroport, avec searchsorted / sommes cumulees.
    Les departs sans AOBT_3 recoivent NaN (LightGBM les gere).
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
        """Pour des requetes (t_ob_q, t_mvt_q) contre une population de
        departs de reference (off-block, decollage) triee par decollage."""
        order = np.argsort(t_mvt_ref)
        mvt_s = t_mvt_ref[order]
        ob_by_mvt = t_ob_ref[order]
        taxi_by_mvt = (mvt_s - ob_by_mvt) / np.timedelta64(1, "s")
        ob_s = np.sort(t_ob_ref)

        # N : off-block avant le mien ET decollage apres le mien
        #   = (#ob < mine) - (#ob < mine et mvt < mine)
        #   Le second terme = nombre de ref entierement passes avant mon ob :
        #   approx par #mvt < mon ob (un ref decolle avant mon ob a
        #   forcement pousse avant), exact.
        n_ob_before = np.searchsorted(ob_s, t_ob_q, side="left")
        n_mvt_before_ob = np.searchsorted(mvt_s, t_ob_q, side="left")
        n_taxiing = np.maximum(n_ob_before - n_mvt_before_ob, 0)

        # Q : decollages entre mon off-block et mon decollage (exclus moi)
        lo = np.searchsorted(mvt_s, t_ob_q, side="left")
        hi = np.searchsorted(mvt_s, t_mvt_q, side="left")
        q_ahead = np.maximum(hi - lo, 0)

        # taxi-out moyen des departs qui ont decolle dans [ob - recent, ob)
        cs = np.concatenate([[0.0], np.cumsum(taxi_by_mvt)])
        lo_r = np.searchsorted(mvt_s, t_ob_q - recent, side="left")
        hi_r = np.searchsorted(mvt_s, t_ob_q, side="left")
        n_r = (hi_r - lo_r).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            recent_mean = np.where(n_r >= QUEUE_MIN_RECENT_N,
                                   (cs[hi_r] - cs[lo_r]) / n_r, np.nan)

        # off-blocks dans les 10 min avant le mien (moi exclu)
        push = (np.searchsorted(ob_s, t_ob_q, side="left")
                - np.searchsorted(ob_s, t_ob_q - push_win, side="left"))
        return n_taxiing, q_ahead, recent_mean, n_r, push

    dep_idx = np.where(is_dep & has_ob)[0]
    # ---- par aeroport ----
    for apt in np.unique(airport[dep_idx]):
        sel = dep_idx[airport[dep_idx] == apt]
        if len(sel) == 0:
            continue
        n_t, q_a, r_m, n_r, push = queue_stats(
            t_ob_all[sel], t_mvt_all[sel], t_ob_all[sel], t_mvt_all[sel])
        df.iloc[sel, df.columns.get_loc("q_n_taxiing_apt")] = n_t
        df.iloc[sel, df.columns.get_loc("q_ahead_dep_apt")] = q_a
        df.iloc[sel, df.columns.get_loc("q_recent_taxi_apt")] = r_m

        # ---- par piste ----
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

            # atterrissages sur la meme piste pendant mon roulage
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
    """Appaire chaque depart a la derniere arrivee au meme poste du meme
    aeroport dont l'in-block precede le decollage (merge_asof backward).
    """
    df = df.copy()
    for c in ("turnaround_s", "prev_arr_taxi_in", "prev_same_type",
              "prev_same_operator", "sched_before_inblock",
              "bound_minus_aobt", "prev_arr_corrupt", "next_arr_corrupt",
              "both_arr_corrupt"):
        df[c] = np.nan
    df["has_stand_link"] = 0.0

    # Corruption des arrivees (in-block == heure programmee)
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
    # merge_asof ne garde pas la cle temporelle de droite : on la recupere
    # via un second asof sur la seule colonne temps.
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

    # Arrivee SUIVANTE au meme poste (signal non causal, mais observable
    # dans le ranking : les arrivees y sont completes ; autorise par les
    # organisateurs, a documenter dans le depot)
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
    """Derives du linking qui dependent des deltas (apres build_features)."""
    df = df.copy()
    has = df["turnaround_s"].notna()
    # SCHED anterieur a l'in-block de l'arrivee precedente au meme poste :
    # un block = sched serait alors physiquement impossible.
    df["sched_before_inblock"] = np.where(
        has, (df["delta_mvt_sched_s"] > df["turnaround_s"]).astype(float), np.nan)
    df["bound_minus_aobt"] = np.where(
        has, df["turnaround_s"] - df["delta_mvt_aobt3_s"], np.nan)
    return df


# ----------------------------------------------------------------------------
# v11 : branche dediee au regime "Rome, non apparie, decollage le lendemain"
# ----------------------------------------------------------------------------
#
# Diagnostic sur 2025 (168 vols LIRF non apparies decollant le lendemain de
# la date programmee, heure locale) :
#
#   delta_sched            +1 jour   block=sched   normal
#   3 h  - 12 h              0-1         106          33
#   > 12 h                    11          16           0
#
# Sur ce sous-ensemble, le "taxi-out" n'est presque jamais un taxi-out :
# c'est soit delta_sched (block time rempli avec l'heure programmee), soit
# 86 400 + roulage (block estampille a la date programmee avec l'heure
# reelle du lendemain). Une demi-douzaine de ces lignes par fold de
# validation portent environ la moitie du MSE total, parce que leurs
# erreurs se comptent en dizaines de milliers de secondes.
#
# Prediction en esperance, par tranche de retard, avec taux estimes sur
# l'entrainement (lissage de Laplace) :
#
#   pred = r_plus1 * (86400 + mu_normal) + r_bs * delta_sched
#          + r_normal * mu_normal
#
# Les 12 cas "+1 jour" sont repartis sur toute l'annee 2025 : bug
# structurel du systeme de Fiumicino, presume present en 2026.

ROME_AIRPORT = "LIRF"
ROME_TZ = "Europe/Rome"
ROME_BUCKETS = [0, 3600, 10800, 43200, np.inf]
ROME_LAPLACE = 1.0
DAY_S = 86400.0


def rome_subset_mask(df: pd.DataFrame) -> np.ndarray:
    """LIRF, non apparie NM, decollage le lendemain (date locale) du jour
    programme. Tout est observable dans le ranking."""
    is_rome = df["ADEP_mvt"].astype(str).values == ROME_AIRPORT
    unmatched = df["no_flight_id"].values == 1
    mvt_d = df["MVT_TIME_UTC_mvt"].dt.tz_convert(ROME_TZ).dt.floor("D")
    sch_d = df["SCHED_TIME_UTC_mvt"].dt.tz_convert(ROME_TZ).dt.floor("D")
    next_day = (mvt_d > sch_d).values
    return is_rome & unmatched & next_day & np.isfinite(df["delta_mvt_sched_s"].values)


def rome_bucket(delta_sched: np.ndarray) -> np.ndarray:
    return np.digitize(delta_sched, ROME_BUCKETS[1:-1])   # 0..3


def rome_categorize(df: pd.DataFrame) -> np.ndarray:
    """0 = +1 jour, 1 = block = sched, 2 = normal (priorite dans cet ordre)."""
    y = df[TARGET].values
    ds = df["delta_mvt_sched_s"].values
    plus1 = (y - DAY_S >= TAXI_MIN_S) & (y - DAY_S <= TAXI_MAX_S)
    bs = np.abs(y - ds) <= 60
    return np.where(plus1, 0, np.where(bs, 1, 2))


def fit_rome_rates(train_df: pd.DataFrame) -> np.ndarray:
    """Taux (3 categories) par tranche de retard, lisses. Forme (4, 3)."""
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
    """Remplace la prediction sur le sous-ensemble par l'esperance du
    melange a trois regimes. Retourne (pred modifiee, masque)."""
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
    """Chaine complete de features sur un tableau DEP + ARR. Utilisee a
    l'identique pour l'entrainement et pour le ranking."""
    raw = parse_times(raw)
    print("  congestion...")
    raw = add_congestion_features(raw)
    print("  contexte arrivees...")
    raw = add_arrival_context(raw)
    print("  enchainement par poste...")
    raw = add_stand_linking(raw)
    print("  file d'attente a l'off-block (AOBT_3)...")
    raw = add_offblock_queue_features(raw)
    if USE_DEMAND:
        print("  file de demande (SCHED / EOBT)...")
        raw = add_demand_queue_features(raw)
    if USE_DEGRADED:
        print("  journee degradee (taxi-in rapporte a la norme)...")
        raw = add_degraded_day_features(raw)
    if USE_FOLLOWING:
        print("  contexte posterieur (decollages suivants)...")
        raw = add_following_features(raw)
    if USE_NEIGHBOR:
        print("  voisinage de poste, aeroport et arrivees...")
        raw = add_neighbor_features(raw)
    dep = raw[raw["PHASE_mvt"] == "DEP"].copy()
    dep = build_features(dep)
    dep = add_stand_derived(dep)
    return dep


def load_or_build(cache_path: Path) -> pd.DataFrame:
    if USE_CACHE and cache_path.exists():
        print(f"Chargement du cache {cache_path.name}...")
        dep = pd.read_parquet(cache_path)
        print(f"  -> {len(dep):,} departs")
        return dep

    raw = load_training(DATA_DIR)
    print("Calcul des features (entrainement)...")
    dep = build_all(raw)
    del raw
    gap = (dep["BLOCK_TIME_UTC_mvt"]
           - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    dep[CORRUPT] = (gap.abs() <= CORRUPT_TOL_S).astype(int)
    n0 = len(dep)
    dep = dep[dep[TARGET].notna() & (dep[TARGET] >= TAXI_MIN_S)]
    print(f"Filtrage minimal : {n0 - len(dep):,} retirees, reste {len(dep):,}")
    dep.to_parquet(cache_path)
    print(f"Cache ecrit : {cache_path.name}")
    return dep


def main_v16():
    dep = load_or_build(CACHE_PATH)
    print(f"Lignes corrompues : {dep[CORRUPT].mean():.2%}")
    print(f"Appariement poste disponible : {dep['has_stand_link'].mean():.1%} "
          f"(meme type avion : {dep['prev_same_type'].mean():.1%})")
    print(f"Contexte arrivees disponible : "
          f"{dep['arr_taxi_mean_win'].notna().mean():.1%}")
    print(f"File a l'off-block : Q dispo {dep['q_ahead_dep_rwy'].notna().mean():.1%}, "
          f"N median {dep['q_n_taxiing_rwy'].median():.0f}, Q median {dep['q_ahead_dep_rwy'].median():.0f}, "
          f"taxi recent piste median {dep['q_recent_taxi_rwy'].median():.0f} s")
    hi = dep["arr_corrupt_frac_win"] > 0.3
    print(f"Departs dans une fenetre a >30 % d'arrivees corrompues : "
          f"{hi.mean():.2%}, dont corrompus : {dep.loc[hi, CORRUPT].mean():.1%} "
          f"(base : {dep[CORRUPT].mean():.1%})")

    is_val = dep["month"].isin(VAL_MONTHS)
    tr_all, va_all = dep[~is_val].copy(), dep[is_val].copy()
    print(f"Split temporel : train {len(tr_all):,} / val {len(va_all):,}")

    print(f"\nTarget encoding hors-fold ({TE_FOLDS} folds)...")
    tr_all = add_te_out_of_fold(tr_all, CORRUPT)
    maps_tr, prior_tr = fit_te_maps(tr_all, tr_all[CORRUPT].values)
    va_all = apply_te(va_all, maps_tr, prior_tr)

    print("Temps non entrave hors-fold (p10 / p50 par poste-piste)...")
    tr_all = add_unimpeded_out_of_fold(tr_all)
    umaps_tr = fit_unimpeded_maps(tr_all)
    va_all = apply_unimpeded(va_all, umaps_tr)
    lv = va_all["unimpeded_level"]
    print(f"  niveau poste-piste : {(lv == 0).mean():.1%}, piste : {(lv == 1).mean():.1%}, "
          f"aeroport : {(lv == 2).mean():.1%}")
    cl = va_all[(va_all[CORRUPT] == 0) & (va_all[TARGET] <= TAIL_THR)]
    print(f"  RMSE naif p50 seul sur les saines hors queue : "
          f"{rmse(cl['unimpeded_p50'], cl[TARGET]):.1f} s "
          f"(mediane globale : {rmse(np.full(len(cl), cl[TARGET].median()), cl[TARGET]):.1f} s)")

    y_va = va_all[TARGET].values
    sched = va_all["delta_mvt_sched_s"].values
    ok = np.isfinite(sched)
    sched_safe = np.where(ok, sched, 0.0)

    catsAll = get_categories(tr_all)
    X_tr = prepare_matrix(tr_all, extra=TE_FEATURES + UNIMP_FEATURES)
    X_va = prepare_matrix(va_all, catsAll, extra=TE_FEATURES + UNIMP_FEATURES)

    # ---- Classifieur de corruption ----------------------------------------
    print("\n-- Classifieur de corruption --")
    clf_c = train_ensemble(LGB_CLF, X_tr, tr_all[CORRUPT].values,
                           X_va, va_all[CORRUPT].values, num_rounds=3000,
                           stopping=150, label="clf")
    bestC = clf_c.iters
    p_c = clf_c.predict(X_va)
    y_corr = va_all[CORRUPT].values
    print(f"Taux reel {y_corr.mean():.2%} | p moyen {p_c.mean():.2%}")
    for thr in (0.3, 0.5, 0.7, 0.9):
        sel = p_c > thr
        if sel.sum():
            print(f"  p > {thr} : {sel.sum():6,} lignes, "
                  f"precision {y_corr[sel].mean():5.1%}, "
                  f"rappel {y_corr[sel].sum() / max(y_corr.sum(), 1):5.1%}"
                  f"  (v14 a p>0.5 : 71.8 % / 11.1 %)")
    print("\nTop 12 features du classifieur (gain) :")
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
        # Regresseur B : cible = EXCEDENT sur le temps non entrave (p50)
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
        """Reentrainement final : rounds figes PAR SEED (ceux valides)."""
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

    # ---- Regresseurs ---------------------------------------------------------
    print("\n-- Regresseur A (residu AOBT_3, lignes saines) --")
    mA, cA, iA, mB, cB, iB = fit_pair(clean_tr, clean_va)
    ha_v = clean_va["delta_mvt_aobt3_s"].notna()
    vA_ = clean_va[ha_v]
    r_ens = rmse(mA.predict(prepare_matrix(vA_, cA, extra=TE_FEATURES + UNIMP_FEATURES)),
                 (vA_[TARGET] - vA_["delta_mvt_aobt3_s"]).values)
    print(f"   RMSE residu, ensemble {len(SEEDS)} seeds : {r_ens:.1f} s "
          f"(v14 : 225.4 s)")
    print("\nTop 15 features du regresseur A (gain) :")
    print(pd.Series(mA.feature_importance(importance_type="gain"),
                    index=prepare_matrix(clean_tr.head(1), extra=TE_FEATURES + UNIMP_FEATURES).columns)
          .sort_values(ascending=False).head(15).to_string())

    mu_normal = predict_normal(va_all, cA, cB, mA, mB, iA, iB)
    nb = clean_va["delta_mvt_aobt3_s"].isna().values
    if nb.any():
        idx_nb = va_all.index.get_indexer(clean_va.index[nb])
        print(f"   RMSE B sans AOBT_3 (en taxi, excedent + p50) : "
              f"{rmse(mu_normal[idx_nb], clean_va.loc[nb, TARGET]):.1f} s  (v14 : 3038 s)")

    # ---- Melange et balayage ---------------------------------------------
    print("\n" + "=" * 68)
    print("MELANGE (reference v14 : 369.3 s melange seul, 345.9 s avec Rome)")
    print("=" * 68)
    print(f"RMSE sans melange : {rmse(mu_normal, y_va):.1f} s  (v14 : 531.6)")
    g0, best_r = None, np.inf
    for g in (0.5, 0.6, 0.75, 1.0):
        pg = np.where(ok, p_c ** g, 0.0)
        r = rmse(np.maximum(pg * sched_safe + (1 - pg) * mu_normal,
                            TAXI_MIN_S), y_va)
        mark = ""
        if r < best_r:
            best_r, g0, mark = r, g, "  <--"
        print(f"    p^{g:<5} : {r:8.1f} s{mark}")
    print(f"\nRMSE melange seul : {best_r:.1f} s (p^{g0})   [v14 : 369.3 s]")

    final_val = np.maximum(np.where(ok, p_c ** g0, 0.0) * sched_safe
                           + (1 - np.where(ok, p_c ** g0, 0.0)) * mu_normal,
                           TAXI_MIN_S)
    # ---- v11 : branche Rome ------------------------------------------------
    print("\n" + "=" * 68)
    print("BRANCHE ROME (non apparie, decollage le lendemain)")
    print("=" * 68)
    rates_tr = fit_rome_rates(tr_all)
    print("Taux estimes sur l'entrainement (par tranche de delta_sched) :")
    labels = ["0-1 h", "1-3 h", "3-12 h", "> 12 h"]
    for k in range(4):
        print(f"  {labels[k]:7s} : +1 jour {rates_tr[k, 0]:5.1%}  "
              f"block=sched {rates_tr[k, 1]:5.1%}  normal {rates_tr[k, 2]:5.1%}")

    rmse_before = rmse(final_val, y_va)
    final_rome, m_rome = apply_rome_branch(final_val, va_all, mu_normal, rates_tr)
    rmse_after = rmse(final_rome, y_va)
    print(f"\nSous-ensemble en validation : {m_rome.sum()} vols")
    if m_rome.any():
        cat_va = rome_categorize(va_all[m_rome])
        print(f"  composition reelle : +1 jour {(cat_va == 0).sum()}, "
              f"block=sched {(cat_va == 1).sum()}, normal {(cat_va == 2).sum()}")
        e_before = final_val[m_rome] - y_va[m_rome]
        e_after = final_rome[m_rome] - y_va[m_rome]
        n_va = len(y_va)
        print(f"  contribution au MSE : avant {np.sum(e_before**2)/n_va:,.0f}, "
              f"apres {np.sum(e_after**2)/n_va:,.0f}  "
              f"(MSE total avant : {rmse_before**2:,.0f})")
    print(f"\nRMSE sans branche Rome : {rmse_before:.1f} s")
    print(f"RMSE avec branche Rome : {rmse_after:.1f} s   "
          f"(gain {rmse_before - rmse_after:+.1f} s)   [v14 : 345.9 s]")

    # Controle de transfert : taux ajustes sur un mois, evalues sur l'autre.
    # Ici les taux sont estimes sur les 10 mois d'entrainement, donc on
    # verifie seulement que le gain est present sur CHACUN des deux mois.
    print("\nGain par mois de validation :")
    month = va_all["month"].values
    gains = []
    for mo in VAL_MONTHS:
        mm = month == mo
        rb, ra = rmse(final_val[mm], y_va[mm]), rmse(final_rome[mm], y_va[mm])
        gains.append(rb - ra)
        print(f"  mois {mo} : {rb:.1f} -> {ra:.1f} s  (gain {rb - ra:+.1f} s, "
              f"{(m_rome & mm).sum()} vols concernes)")
    use_rome_local = all(g > -1.0 for g in gains) and (rmse_before - rmse_after) > 0
    print("\nBranche Rome : gain local", "OUI" if use_rome_local else "NON",
          "| DESACTIVEE en production (v16) : +19.8 s local, -2.45 s public "
          "(v10 337.95 sans, v11 340.40 avec). La composition du sous-ensemble "
          "a change en 2026.")
    use_rome = False

    # ---- v16 : recalibrage du poids du melange ----------------------------
    # Le melange s'ecrit y = n + w * (s - n). Avec p calibre, l'optimum L2 est
    # w = p ; l'exposant gamma n'est qu'une recalibration uniforme. Or le
    # cout d'une erreur sur w est proportionnel a (s - n)^2, ce que la
    # log-loss du classifieur ignore. On recalibre donc w = f(p, log(s - n))
    # avec un modele volontairement petit, monotone en p, sans aeroport
    # (verifie : le gain ne vient pas d'une memorisation par aeroport).
    # Transfert janvier <-> juillet sur val_preds_v15 : +7.6 / +1.3 s.
    # Entraine sur la validation complete (predictions de modeles a 10 mois),
    # applique au ranking.
    print("\n" + "=" * 68)
    print("RECALIBRAGE DU MELANGE w = f(p, log(s - n))")
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
        print(f"  ajuste sur mois {fit_m}, teste sur mois {test_m} : "
              f"{rmse(ref_val[tem], y_va[tem]):.1f} -> {rmse(pred_cal, y_va[tem]):.1f} s")
    calibrator = lgb.train(CAL_PARAMS, lgb.Dataset(Xcal, y_corr),
                           num_boost_round=CAL_ROUNDS)
    w_all = np.where(ok, calibrator.predict(Xcal), 0.0)
    final_val = np.maximum(mu_normal + w_all * (sched_safe - mu_normal), TAXI_MIN_S)
    print(f"  (en echantillon, indicatif) reference {rmse(ref_val, y_va):.1f} s, "
          f"recalibre {rmse(final_val, y_va):.1f} s")

    # Sauvegarde des predictions de validation pour decomposition du MSE
    pd.DataFrame({
        "MVT_ID_mvt": va_all["MVT_ID_mvt"].values,
        "ADEP_mvt": va_all["ADEP_mvt"].astype(str).values,
        "month": month,
        "y": y_va, "pred": final_val, "pred_ref_gamma": ref_val,
        "p_c": p_c, "mu_normal": mu_normal, "delta_sched": sched,
        "is_corrupt": va_all[CORRUPT].values, "rome_branch": m_rome.astype(int),
    }).to_parquet(DATA_DIR / "val_preds_v15.parquet", index=False)
    print(f"Predictions de validation sauvees : val_preds_v15.parquet")

    print("\nRMSE par aeroport (v14 entre parentheses) :")
    ref = {"EDDF": 201.0, "EDDM": 191.4, "EGLL": 279.2, "EHAM": 197.8,
           "LEBL": 239.7, "LEMD": 186.8, "LFPG": 578.8, "LIRF": 713.5,
           "LSZH": 214.1, "LTFM": 265.4}
    apt = va_all["ADEP_mvt"].astype(str).values
    for a in np.unique(apt):
        m = apt == a
        r = rmse(final_val[m], y_va[m])
        print(f"  {a} : {r:8.1f} s  ({ref.get(a, float('nan')):6.1f})  "
              f"{r - ref.get(a, r):+7.1f}")

    # ---- Reentrainement sur tout 2025 --------------------------------------
    print("\n" + "=" * 68)
    print("Reentrainement sur l'ensemble de 2025")
    print("=" * 68)
    print(f"Rounds figes par seed : classifieur {bestC}, regresseur A {iA}, B {iB}")
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
        print("\nFichiers de ranking introuvables : soumission ignoree.")
        return

    print("\nChargement du ranking dataset...")
    rank = pd.read_parquet(rank_path)
    print("Calcul des features (ranking)...")
    rdep = build_all(rank)
    rdep = apply_te(rdep, maps_f, prior_f)
    rdep = apply_unimpeded(rdep, umaps_f)
    lvr = rdep["unimpeded_level"]
    print(f"Ranking, niveau non entrave : poste-piste {(lvr == 0).mean():.1%}, "
          f"piste {(lvr == 1).mean():.1%}, aeroport {(lvr == 2).mean():.1%}")
    print(f"Ranking : {len(rdep):,} departs | appariement poste "
          f"{rdep['has_stand_link'].mean():.1%} | contexte arrivees "
          f"{rdep['arr_taxi_mean_win'].notna().mean():.1%}")

    p_c_r = clf_c_f.predict(prepare_matrix(rdep, catsC_f, extra=TE_FEATURES + UNIMP_FEATURES))
    sched_r = rdep["delta_mvt_sched_s"].values
    ok_r = np.isfinite(sched_r)
    sched_r_safe = np.where(ok_r, sched_r, 0.0)
    mu_r = predict_normal(rdep, cA_f, cB_f, mA_f, mB_f)
    print(f"P(corrompue) moyenne : {p_c_r.mean():.2%}")

    subm_tpl = pd.read_parquet(subm_path).drop(columns=[TARGET], errors="ignore")
    print("\nSoumissions :")
    # Deux fichiers par gamma : SANS et AVEC branche Rome. Le score public
    # de v7 (341.3) n'a repris que 6 s des 31 s gagnees en local depuis
    # v3 : quelque chose ne se transpose pas a 2026. Soumettre les deux
    # variantes isole l'effet de la branche Rome au leaderboard.
    n_written = 0
    for i, g in enumerate(GAMMAS):
        pg = np.where(ok_r, p_c_r ** g, 0.0)
        preds_g = np.maximum(pg * sched_r_safe + (1 - pg) * mu_r, TAXI_MIN_S)

        gap_r = sched_r - mu_r
        Xcal_r = pd.DataFrame({"p_c": p_c_r,
                               "log_gap": np.sign(gap_r) * np.log1p(np.abs(gap_r))})
        w_r = np.where(ok_r, calibrator.predict(Xcal_r), 0.0)
        preds_cal = np.maximum(mu_r + w_r * (sched_r_safe - mu_r), TAXI_MIN_S)
        print(f"  poids recalibre sur le ranking : moyenne {w_r.mean():.3f} "
              f"(p^gamma moyen {pg.mean():.3f})")

        variants = [("refGamma", preds_g), ("recalibre", preds_cal)]

        for tag, pv in variants:
            s = subm_tpl.merge(
                pd.DataFrame({"MVT_ID_mvt": rdep["MVT_ID_mvt"].values, "pred": pv}),
                on="MVT_ID_mvt", how="left")
            if s["pred"].isna().any():
                print(f"  Attention : {s['pred'].isna().sum()} sans prediction.")
                s["pred"] = s["pred"].fillna(float(np.median(dep[TARGET])))
            s[TARGET] = s["pred"]
            s = s[["MVT_ID_mvt", TARGET]]
            name = f"{TEAM}_v{START_V + n_written}.parquet"
            n_written += 1
            s.to_parquet(OUT_DIR / name, index=False)
            print(f"  {name:38s} {tag:10s} "
                  f"mediane {np.median(pv):5.0f} s  "
                  f"moyenne {pv.mean():5.0f} s  max {pv.max():7.0f} s")
            print(f"    .\\mc.exe cp submissions\\{name} dc26/prc-2026-{TEAM}/")




# ----------------------------------------------------------------------------
# v19 : file de demande et sequence de decollage
# ----------------------------------------------------------------------------

DEMAND_NUM = ["q_sched_apt", "q_sched_rwy", "q_eobt_apt", "q_eobt_rwy",
              "rwy_burst_pos", "rwy_gap_mean5_s"]
DEMAND_CAT = ["wake_pair"]
NUM_FEATURES = NUM_FEATURES + DEMAND_NUM
CAT_FEATURES = CAT_FEATURES + DEMAND_CAT
USE_DEMAND = True


def add_demand_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """File de demande et sequence de decollage, sans dependance a l'AOBT
    du vol lui-meme. A appeler sur le tableau DEP + ARR, avant le filtrage
    sur les departs."""
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
        """Serie datetime -> secondes. On force l'unite en nanosecondes avant
        la conversion en entier : les parquet du challenge sont en
        microsecondes, et diviser par 1e9 sans cela donne un facteur 1000."""
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

    # Partitions (aeroport, mois) : le fichier de ranking ne contient que
    # janvier et juillet isolement, alors que l'entrainement est continu. Un
    # comptage a cheval sur les bords de mois serait donc tronque d'un cote et
    # pas de l'autre, ce qui decale la distribution entre 2025 et 2026.
    # Partitionner des deux cotes egalise ce biais.
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

            # moyenne des cinq derniers intervalles
            csum = np.concatenate([[0.0], np.cumsum(gaps)])
            lo = np.maximum(prev - 5, 0)
            cnt = prev - lo
            g5 = np.full(ts.shape, np.nan)
            ok = (prev >= 1) & (cnt > 0)
            g5[ok] = (csum[prev[ok]] - csum[lo[ok]]) / cnt[ok]
            out["rwy_gap_mean5_s"][sub] = g5

            # longueur de rafale
            burst = np.zeros(dr.size)
            for i in range(1, dr.size):
                burst[i] = burst[i - 1] + 1 if gaps[i - 1] < 120 else 0.0
            b = np.full(ts.shape, np.nan)
            has = prev >= 0
            b[has] = burst[prev[has]]
            out["rwy_burst_pos"][sub] = b

            # couple de sillage avec le decollage precedent
            pw = np.full(ts.shape, "NA", dtype=object)
            pw[has] = wake[d_sub][prev[has]]
            wake_pair[sub] = np.char.add(np.char.add(pw.astype(str), ">"),
                                         wake[sub].astype(str))

    for k, v in out.items():
        df[k] = v
    df["wake_pair"] = pd.Series(wake_pair, index=df.index).astype("category")
    return df


# ----------------------------------------------------------------------------
# v19 : detection de journee degradee (degivrage, LVP, incident de surface)
# ----------------------------------------------------------------------------
#
# Le taxi-in des arrivees n'est PAS masque dans le fichier de ranking : on
# dispose donc, pour janvier et juillet 2026, de l'etat reel de la surface
# heure par heure. Ce qui manque a v16, ce n'est pas la valeur absolue du
# taxi-in recent (arr_taxi_mean_win la donne deja) mais son RAPPORT A LA
# NORME : un taxi-in moyen de 700 s est banal a CDG a 9 h et signale une
# journee degradee a 5 h du matin.
#
# La norme est calculee par (aeroport, mois, heure) avec une mediane, donc
# robuste a quelques journees exceptionnelles, et dans la meme partition des
# deux cotes pour eviter le probleme de bord de mois.

DEGRADED_NUM = ["arr_taxi_norm_h", "arr_taxi_ratio_1h",
                "arr_taxi_ratio_3h", "arr_taxi_ratio_6h"]
USE_DEGRADED = False


def add_degraded_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rapport du taxi-in recent des arrivees a sa norme horaire. Causal :
    seules les arrivees anterieures a l'instant considere sont utilisees."""
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

        # norme horaire, mediane robuste sur la partition
        norm = (pd.DataFrame({"h": hour[a_idx], "v": taxi[a_idx]})
                .groupby("h")["v"].median())
        base = pd.Series(hour[idx]).map(norm).to_numpy(dtype=float)
        out["arr_taxi_norm_h"][idx] = base

        # le taxi-in d'une arrivee est connu a son heure de mise en bloc
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
# v19 : contexte posterieur dense (features non causales)
# ----------------------------------------------------------------------------
#
# La tache est une reconstruction retrospective : le fichier de ranking
# contient tous les mouvements de janvier et juillet 2026, y compris ceux
# qui decollent apres la ligne cible. Les decollages suivants, et surtout
# leurs heures d'off-block NM, renseignent directement sur la file dans
# laquelle le vol se trouvait.
#
# Aucune de ces colonnes n'utilise BLOCK_TIME_UTC_mvt ni TAXITIME_SEC_mvt
# des departs, qui sont masques au ranking.
#
# Ces features n'ont aucun sens operationnel : un controleur ne connait pas
# les decollages futurs. A signaler explicitement dans toute publication.

FOLLOWING_NUM = [
    # intervalles vers les decollages suivants sur la piste
    "gap_next_1_rwy", "gap_next_2_rwy", "gap_next_3_rwy",
    "gap_next_5_rwy", "gap_next_10_rwy", "gap_next_20_rwy",
    "rate_next_5_rwy", "rate_next_10_rwy", "rate_next_20_rwy",
    # densite a venir, par piste et par aeroport
    "dep_ahead_5_rwy", "dep_ahead_15_rwy", "dep_ahead_30_rwy", "dep_ahead_60_rwy",
    "dep_ahead_5_apt", "dep_ahead_15_apt", "dep_ahead_30_apt", "dep_ahead_60_apt",
    # densite symetrique : la file autour du vol
    "dep_sym_15_rwy", "dep_sym_30_rwy", "dep_sym_30_apt",
    "ahead_share_15_rwy", "ahead_share_30_apt",
    # rafales
    "burst_ahead_rwy", "pos_in_burst_rwy", "burst_len_rwy",
    "burst_time_ahead_rwy", "burst_time_since_rwy", "burst_share_done",
    # symetrie et regime de la piste
    "gap_ratio_next_prev", "rate_ahead_vs_norm_rwy", "gap_next_vs_norm_rwy",
    # horloges NM des vols suivants : qui roulait deja derriere moi
    "next_aobt_minus_t", "next_eobt_minus_t",
    "n_behind_taxiing_5", "n_behind_taxiing_10", "behind_aobt_mean_5",
    "next_sched_slack", "next_slack_mean_5",
    # composition du flux a venir
    "n_heavy_next_5", "next_same_operator", "rwy_switch_next",
]
USE_FOLLOWING = False


def add_following_features(df: pd.DataFrame) -> pd.DataFrame:
    """Contexte des decollages posterieurs, par (aeroport, mois, piste)."""
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
                """valeur du k-ieme decollage suivant, NaN au dela de la fin"""
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

            # rafales : suites de decollages espaces de moins de deux minutes
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

            # regime de la piste rapporte a sa norme sur la partition
            norm_gap = float(np.nanmedian(gaps)) if gaps.size else np.nan
            norm_ahead = float(np.nanmedian(out["dep_ahead_15_rwy"][sub]))
            with np.errstate(invalid="ignore", divide="ignore"):
                if np.isfinite(norm_gap) and norm_gap > 0:
                    out["gap_next_vs_norm_rwy"][sub] = out["gap_next_1_rwy"][sub] / norm_gap
                if np.isfinite(norm_ahead) and norm_ahead > 0:
                    out["rate_ahead_vs_norm_rwy"][sub] = (
                        out["dep_ahead_15_rwy"][sub] / norm_ahead)

            # horloges NM des vols suivants
            out["next_aobt_minus_t"][sub] = _pick(a_sorted, 1) - ts_sub
            out["next_eobt_minus_t"][sub] = _pick(e_sorted, 1) - ts_sub
            out["next_sched_slack"][sub] = _pick(dr, 1) - _pick(s_sorted, 1)

            # combien des vols suivants avaient DEJA repousse quand j'ai decolle :
            # ils roulaient donc derriere moi, c'est la file elle-meme
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
# v19 : contexte des voisins, de l'aeroport entier et des arrivees
# ----------------------------------------------------------------------------
#
# Trois familles, toutes bidirectionnelles (passe et futur), dans la lignee
# du contexte posterieur :
#
#   1. Voisinage de poste : le taxi-out depend de qui encombre la meme
#      bretelle. Le prefixe de STAND_mvt sert d'approximation du satellite
#      ou de la jetee (C03 et C07 sont voisins, 826 et 828 aussi).
#   2. Aeroport entier plutot que piste : les vagues de depart traversent
#      les pistes, et un vol peut etre retarde par une file qui n'est pas
#      la sienne.
#   3. Arrivees a venir : un avion qui se pose pendant que vous roulez peut
#      couper votre itineraire, surtout en mode mixte.
#
# Aucune colonne masquee n'est utilisee.

NEIGHBOR_NUM = [
    # voisinage de poste
    "stand_zone_sym_15", "stand_zone_sym_30", "stand_zone_ahead_15",
    "stand_zone_gap_prev", "stand_zone_gap_next", "stand_zone_share_ahead",
    "same_stand_gap_prev", "same_stand_gap_next",
    "stand_zone_arr_sym_30", "stand_zone_load",
    # aeroport entier
    "gap_next_1_apt", "gap_next_5_apt", "gap_next_10_apt", "gap_next_20_apt",
    "rate_next_5_apt", "rate_next_10_apt", "rate_next_20_apt",
    "gap_prev_1_apt", "gap_ratio_next_prev_apt",
    "burst_len_apt", "pos_in_burst_apt", "burst_ahead_apt",
    "n_behind_taxiing_5_apt", "n_behind_taxiing_10_apt",
    "rate_ahead_vs_norm_apt",
    # arrivees
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
    """debut, fin, longueur de chaque rafale pour une suite d'heures triees"""
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
    """Voisinage de poste, contexte aeroport et contexte des arrivees."""
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

        # --- aeroport entier -------------------------------------------------
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

        # --- arrivees --------------------------------------------------------
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
            # mises en bloc a venir : les arrivees qui liberent ou occupent la surface
            ab = t_block[a_idx]
            ab = np.sort(ab[np.isfinite(ab)])
            if ab.size:
                out["arr_inblock_ahead_30"][idx] = (
                    np.searchsorted(ab, tt + 1800, side="right")
                    - np.searchsorted(ab, tt, side="left")).astype(float)

        # --- par piste : arrivees croisant l'itineraire -----------------------
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

        # --- voisinage de poste ----------------------------------------------
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
                # charge de la zone rapportee a sa norme sur la partition
                nz = float(np.nanmedian(sym15))
                if np.isfinite(nz) and nz > 0:
                    out["stand_zone_load"][sub] = sym15 / nz
            if az.size:
                out["stand_zone_arr_sym_30"][sub] = (
                    np.searchsorted(az, ts_sub + 1800, side="right")
                    - np.searchsorted(az, ts_sub - 1800, side="left")).astype(float)

        # --- meme poste exactement : rotation ---------------------------------
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
# v19 : modeles par aeroport et ensemble CatBoost
# ----------------------------------------------------------------------------

USE_PER_AIRPORT = False
CLF_GLOBAL = False
USE_CATBOOST = False
PER_AIRPORT_MIN = 40000     # lignes minimales pour meriter son propre modele
CAT_ROUNDS_A = 1200
CAT_ROUNDS_B = 300
CAT_WEIGHT = 0.5
PER_RUNWAY = False


def train_cat(X, y, rounds, seed):
    """CatBoost sur la meme matrice que LightGBM. Les colonnes categorielles
    sont converties en chaines : CatBoost les traite nativement et son
    encodage ordonne gere mieux les fortes cardinalites comme STAND_mvt."""
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
    """Un modele par aeroport suffisamment represente, plus un modele de
    repli pour les autres. Rome ne se comporte pas comme Munich, et un
    modele unique arbitre entre les deux au lieu de les servir."""
    if not USE_PER_AIRPORT:
        return {"__global__": fit_stage1(train_df)}
    key = train_df["ADEP_mvt"].astype(str)
    if PER_RUNWAY:
        key = key + "|" + train_df["RUNWAY_mvt"].astype(str)
    train_df = train_df.assign(_grp=key.values)
    counts = key.value_counts()
    big = [a for a, c in counts.items() if c >= PER_AIRPORT_MIN]
    models = {}
    print("    modeles par aeroport : %d dedies, 1 de repli" % len(big))
    for a in big:
        part = train_df[train_df["_grp"] == a]
        models[a] = fit_stage1(part)
    models["__global__"] = fit_stage1(train_df)
    return models


def predict_stage1_grouped(models, df):
    """Regresseurs par aeroport, classifieur au choix.

    Un classifieur dedie a Rome voit ses cinquante labels corrompus en
    proportion bien plus forte que le classifieur global, qui les noie dans
    1,7 million de lignes, et il les surapprend. CLF_GLOBAL garde donc le
    p_c du modele global tout en prenant le mu du modele local.
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
# v19 : classifieur de corruption allege
# ----------------------------------------------------------------------------
#
# Deux essais independants ont montre que l'ajout de features de contexte
# ameliore la voie propre mais degrade Rome une fois les labels corrompus
# inclus : le classifieur p_c dispose de plus en plus de matiere pour
# justifier des predictions elevees a tort. On lui retire donc tout le
# contexte, que les regresseurs continuent d'utiliser pleinement.

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
FOLDS = [[2, 3], [4, 5], [6, 8], [9, 10], [11, 12]]   # plis sur les mois de train
INNER_HOLDOUT = 4                                       # indice du pli de selection (11-12)
VAL_MONTHS = [1, 7]

# Rounds figes du premier etage (valeurs typiques observees v13-v15)
ROUNDS_CLF, ROUNDS_A, ROUNDS_B = 210, 580, 100

# Correcteur
CLIP_RESID_S = 3600.0
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
LGB_CORR = dict(objective="huber", alpha=600.0, metric="rmse",
                learning_rate=0.03, num_leaves=63, min_data_in_leaf=300,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=5.0, verbosity=-1, seed=SEED)
CORR_MAX_ROUNDS = 1500
STAGE1_OUT = ["p_c", "mu_normal", "mu_raw", "mix", "gap", "log_gap"]

TEAM_V17, START_V_V17 = "motivated-ostrich", 19


# ---- Premier etage (version compacte, une seed, rounds figes) -------------

def fit_stage1(train_df: pd.DataFrame) -> dict:
    """Classifieur + regresseurs A/B, target encodings et temps non entrave,
    tout ajuste sur train_df. Rounds figes (pas de validation interne)."""
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
    """Applique le premier etage a df (avec encodages de models). Retourne df
    enrichi des sorties STAGE1_OUT."""
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
    """Reutilise un premier etage deja entraine pour la meme configuration.

    La cle couvre la portee, la graine, le nombre de lignes, les capacites et
    les drapeaux d'architecture, ainsi que le nombre de features : un cache
    obsolete ne peut donc pas etre servi par erreur.
    """
    if not MODEL_CACHE:
        return fit_stage1_grouped(train_df)
    import pickle
    d = DATA_DIR / "model_cache"
    d.mkdir(exist_ok=True)
    f = d / ("stage1_%s.pkl" % _model_key(scope, seed, len(train_df)))
    if f.exists():
        print("    modeles relus du cache (%s)" % f.name, flush=True)
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
    """Premier etage moyenne sur plusieurs graines, poids egaux."""
    global SEED
    seeds = SEED_LIST[:max(1, N_SEEDS)]
    mixes, first = [], None
    for k, sd in enumerate(seeds):
        SEED = sd
        if len(seeds) > 1:
            print("    %s graine %d/%d (seed=%d)..." % (label, k + 1, len(seeds), sd),
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

    # ---- 1. Predictions hors-fold sur le train --------------------------
    print("\n" + "=" * 68)
    print("ETAPE 1 : predictions hors-fold du premier etage (5 plis, 1 seed)")
    print("=" * 68)
    oof_parts = []
    for k, months in enumerate(FOLDS):
        out = train["month"].isin(months)
        print(f"  pli {k + 1}/5 : mois {months} predits par les autres "
              f"({(~out).sum():,} lignes d'entrainement)...", flush=True)
        models = fit_stage1(train[~out])
        part = predict_stage1(models, train[out])
        part["_fold"] = k
        oof_parts.append(part)
        r = rmse(part["mix"].values, part[TARGET].values)
        print(f"      RMSE melange hors-fold sur ce pli : {r:.1f} s")
    oof = pd.concat(oof_parts)
    oof["resid"] = oof[TARGET].values - oof["mix"].values
    print(f"\nRMSE melange hors-fold, 10 mois : {rmse(oof['mix'], oof[TARGET]):.1f} s")
    print(f"Residu : median {oof['resid'].median():.0f} s, ecart-type "
          f"{oof['resid'].std():.0f} s, part |resid| > {CLIP_RESID_S:.0f} : "
          f"{(oof['resid'].abs() > CLIP_RESID_S).mean():.3%}")

    # ---- 2. Correcteur : selection sur le pli interne -------------------
    print("\n" + "=" * 68)
    print("ETAPE 2 : correcteur (Huber, residu ecrete), selection sur le pli 5")
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
    print(f"Iterations retenues : {best_rounds}")

    # Balayage de lambda sur le pli interne, sur le VRAI residu (non ecrete)
    c_in = corr.predict(X_in, num_iteration=best_rounds)
    mix_in, y_in = oof.loc[inner, "mix"].values, oof.loc[inner, TARGET].values
    best_l, best_r = 0.0, np.inf
    print("\n  lambda   RMSE pli 5")
    for lam in LAMBDAS:
        r = rmse(np.maximum(mix_in + lam * c_in, TAXI_MIN_S), y_in)
        mark = ""
        if r < best_r:
            best_r, best_l, mark = r, lam, "  <--"
        print(f"  {lam:5.2f}   {r:8.1f} s{mark}")
    print(f"\nConfiguration figee : lambda = {best_l}, {best_rounds} iterations")
    if best_l == 0.0:
        print("  lambda = 0 : le correcteur n'apporte rien sur le pli interne.")

    print("\nTop 15 features du correcteur (gain) :")
    print(pd.Series(corr.feature_importance(importance_type="gain"),
                    index=X_fit.columns).sort_values(ascending=False).head(15).to_string())

    # Correcteur final sur les 5 plis (rounds figes)
    corr_full = lgb.train(LGB_CORR, lgb.Dataset(align(X_all, cats_corr), label=y_res),
                          num_boost_round=best_rounds)

    # ---- 3. Evaluation unique sur janvier + juillet ---------------------
    print("\n" + "=" * 68)
    print("ETAPE 3 : evaluation sur janvier + juillet 2025 (une seule fois)")
    print("=" * 68)
    v = fit_predict_seeds(train, val, label='evaluation')
    c_val = corr_full.predict(align(corrector_matrix(v), cats_corr))
    ref = v["mix"].values
    cor = np.maximum(ref + best_l * c_val, TAXI_MIN_S)
    y = v[TARGET].values
    print(f"RMSE reference (melange p^{GAMMA}) : {rmse(ref, y):.1f} s")
    print(f"RMSE corrige (lambda = {best_l})     : {rmse(cor, y):.1f} s   "
          f"gain {rmse(ref, y) - rmse(cor, y):+.1f} s")
    gains = []
    for mo in VAL_MONTHS:
        m = v["month"].values == mo
        g = rmse(ref[m], y[m]) - rmse(cor[m], y[m])
        gains.append(g)
        print(f"  mois {mo} : {rmse(ref[m], y[m]):.1f} -> {rmse(cor[m], y[m]):.1f} s "
              f"(gain {g:+.1f})")
    print("\nPar aeroport (reference -> corrige) :")
    apt = v["ADEP_mvt"].astype(str).values
    for a in np.unique(apt):
        m = apt == a
        print(f"  {a} : {rmse(ref[m], y[m]):7.1f} -> {rmse(cor[m], y[m]):7.1f} s "
              f"({rmse(ref[m], y[m]) - rmse(cor[m], y[m]):+.1f})")
    transfers = all(g > 0 for g in gains)
    print("\nGain positif sur les deux mois :", "OUI" if transfers else "NON",
          "" if transfers else "(soumission corrigee ecrite quand meme, a lire avec prudence)")

    pd.DataFrame({"MVT_ID_mvt": v["MVT_ID_mvt"].values, "y": y, "ref": ref,
                  "corr": c_val, "final": cor, "month": v["month"].values,
                  "ADEP_mvt": apt}).to_parquet(DATA_DIR / "val_preds_v17.parquet", index=False)

    # ---- 4. Ranking ------------------------------------------------------
    rank_path, subm_path = DATA_DIR / "ranking.parquet", DATA_DIR / "submitting.parquet"
    if not rank_path.exists():
        print("\nranking.parquet introuvable : soumission ignoree.")
        return
    print("\n" + "=" * 68)
    print("ETAPE 4 : ranking (premier etage sur 12 mois, correcteur tel quel)")
    print("=" * 68)
    # premier etage sur 12 mois, moyenne de graines
    rank = pd.read_parquet(rank_path)
    rdep = build_all(rank)
    r = fit_predict_seeds(dep, rdep, label='ranking')
    c_r = corr_full.predict(align(corrector_matrix(r), cats_corr))
    ref_r = r["mix"].values
    cor_r = np.maximum(ref_r + best_l * c_r, TAXI_MIN_S)
    print(f"Correction sur le ranking : moyenne {c_r.mean():+.0f} s, "
          f"|c| > 600 s sur {(np.abs(c_r) > 600).mean():.2%} des lignes")

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
        print(f"  {name:38s} {tag:12s} mediane {np.median(pv):5.0f} s  "
              f"moyenne {pv.mean():5.0f} s  max {pv.max():7.0f} s")
        print(f"    .\\mc.exe cp submissions\\{name} dc26/prc-2026-{TEAM_V17}/")




# ----------------------------------------------------------------------------
# Mode rapide : un seul entrainement du premier etage, pas de correcteur,
# pas de ranking. Pour comparer des jeux de features sans payer sept fits.
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
        print("echantillonnage : %.0f %% du train, %d lignes" % (100 * sample, len(train)))
    print("Train %d / val %d" % (len(train), len(val)))

    seeds = [SEED, 7, 123, 2024, 31337][:max(1, n_seeds)]
    preds = []
    for k, sd in enumerate(seeds):
        SEED = sd
        if len(seeds) > 1:
            print("  graine %d/%d (seed=%d)..." % (k + 1, len(seeds), sd), flush=True)
        mdl = fit_stage1_grouped(train)
        vv = predict_stage1_grouped(mdl, val)
        preds.append(vv["mix"].values)
        if k == 0:
            v = vv
    if len(preds) > 1:
        for k, pk in enumerate(preds):
            print("    graine %d seule : RMSE %.2f s"
                  % (k + 1, np.sqrt(np.mean((pk - v[TARGET].values) ** 2))))
    pred = np.mean(preds, axis=0)
    y = v[TARGET].values
    sq = (pred - y) ** 2
    clean = y <= ABERRANT_S

    print("")
    print("=" * 68)
    print("EVALUATION RAPIDE sur janvier + juillet 2025")
    print("=" * 68)
    print("RMSE global        : %.2f s" % np.sqrt(sq.mean()))
    print("RMSE voie propre   : %.2f s  (%d lignes aberrantes exclues)"
          % (np.sqrt(sq[clean].mean()), int((~clean).sum())))
    for mo in VAL_MONTHS:
        m = v["month"].values == mo
        print("  mois %-2d : global %8.2f s   propre %8.2f s"
              % (mo, np.sqrt(sq[m].mean()), np.sqrt(sq[m & clean].mean())))

    print("")
    print("%-6s %9s %11s %12s %9s" % ("", "n", "RMSE", "RMSE propre", "part MSE"))
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

    # ventilation croisee aeroport x mois, en voie propre
    print("")
    print("voie propre, RMSE par aeroport et par mois :")
    print("%-6s %9s %10s %10s %9s %9s"
          % ("", "n janv", "janvier", "juillet", "ecart", "part MSE"))
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
    print("  l'ecart est la degradation de juillet par rapport a janvier ;")
    print("  la part de MSE dit combien chaque aeroport pese dans l'erreur propre.")

    if USE_CATBOOST and "mu_cat" in v.columns and v["mu_cat"].notna().any():
        print("")
        print("balayage du poids CatBoost (sans reentrainement) :")
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
            print("  poids %.2f : global %8.2f s   propre %8.2f s"
                  % (w, np.sqrt(sqw.mean()), np.sqrt(sqw[clean].mean())))

    sq_c = np.sort(sq[clean])[::-1]
    print("")
    print("concentration de l'erreur sur la voie propre :")
    for frac in (0.0001, 0.001, 0.01, 0.1):
        k = max(int(len(sq_c) * frac), 1)
        print("  les %6.2f %% pires (%6d lignes) portent %5.1f %% du MSE propre"
              % (100 * frac, k, 100 * sq_c[:k].sum() / sq_c.sum()))
    err = np.abs(pred[clean] - y[clean])
    print("  erreur absolue : mediane %5.0f s, p90 %5.0f s, p99 %6.0f s, max %7.0f s"
          % (np.median(err), np.percentile(err, 90), np.percentile(err, 99), err.max()))



def main_submit():
    """Va directement au ranking : pas de plis hors-fold, pas de correcteur,
    pas d'evaluation. Pour produire une soumission quand la configuration a
    deja ete mesuree en mode rapide."""
    dep = load_or_build(CACHE_PATH)
    rank_path = DATA_DIR / "ranking.parquet"
    subm_path = DATA_DIR / "submitting.parquet"
    if not rank_path.exists():
        sys.exit("ranking.parquet introuvable")
    print("premier etage sur les 12 mois, %d departs" % len(dep))
    rdep = build_all(pd.read_parquet(rank_path))
    r = fit_predict_seeds(dep, rdep, label="ranking")
    pv = r["mix"].values
    print("predictions : mediane %.0f s, moyenne %.0f s, max %.0f s"
          % (np.median(pv), pv.mean(), pv.max()))
    tpl = pd.read_parquet(subm_path).drop(columns=[TARGET], errors="ignore")
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": r["MVT_ID_mvt"].values, "pred": pv}),
                    on="MVT_ID_mvt", how="left")
    if out["pred"].isna().any():
        n_na = int(out["pred"].isna().sum())
        print("attention : %d identifiants sans prediction, remplis par la mediane" % n_na)
        out["pred"] = out["pred"].fillna(float(np.median(dep[TARGET])))
    out[TARGET] = out["pred"]
    name = "%s_v%d.parquet" % (TEAM_V17, START_V_V17)
    out[["MVT_ID_mvt", TARGET]].to_parquet(OUT_DIR / name, index=False)
    print("ecrit : %s (%d lignes)" % (name, len(out)))
    print("  mc.exe cp submissions/%s dc26/prc-2026-%s/" % (name, TEAM_V17))


# ----------------------------------------------------------------------------
# Point d'entree unique
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PRC 2026 - taxi-out v19")
    ap.add_argument("--fast", action="store_true",
                    help="un seul entrainement, pas de correcteur ni de soumission")
    ap.add_argument("--sample", type=float, default=None,
                    help="fraction du train a echantillonner en mode rapide, ex 0.3")
    ap.add_argument("--no-month", action="store_true",
                    help='retire "month" des features numeriques')
    ap.add_argument("--demand", action="store_true",
                    help="active les features de file de demande (refutees en holdout 2025)")
    ap.add_argument("--degraded", action="store_true",
                    help="active les features de journee degradee")
    ap.add_argument("--following", action="store_true",
                    help="active le contexte posterieur (decollages suivants)")
    ap.add_argument("--neighbor", action="store_true",
                    help="active le voisinage de poste, aeroport et arrivees")
    ap.add_argument("--all-context", action="store_true",
                    help="equivaut a --following --neighbor")
    ap.add_argument("--rounds-scale", type=float, default=1.0,
                    help="multiplicateur du nombre d'arbres du premier etage, ex 4")
    ap.add_argument("--seeds", type=int, default=1,
                    help="nombre de graines moyennees a poids egaux (mode rapide et complet)")
    ap.add_argument("--submit-only", action="store_true",
                    help="va directement au ranking, sans plis ni evaluation")
    ap.add_argument("--model-cache", action="store_true",
                    help="reutilise les modeles deja entraines pour la meme configuration")
    ap.add_argument("--per-airport", action="store_true",
                    help="un modele par aeroport suffisamment represente")
    ap.add_argument("--cat-rounds", type=int, default=1200,
                    help="iterations CatBoost du regresseur principal")
    ap.add_argument("--cat-weight", type=float, default=0.5,
                    help="poids de CatBoost dans l'ensemble, 0 = LightGBM seul")
    ap.add_argument("--per-runway", action="store_true",
                    help="avec --per-airport : un modele par couple aeroport-piste")
    ap.add_argument("--clf-global", action="store_true",
                    help="avec --per-airport : garde le classifieur global, "
                         "regresseurs par aeroport")
    ap.add_argument("--catboost", action="store_true",
                    help="ensemble CatBoost + LightGBM a poids egaux")
    ap.add_argument("--clf-lean", action="store_true",
                    help="retire les features de contexte du classifieur de corruption")
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
    print("configuration : %s%s" % (tag, "  [mode rapide]" if args.fast else ""))
    print('  "month" : %s' % ("non" if args.no_month else "oui"))
    print("  file de demande : %s" % ("oui" if USE_DEMAND else "non"))
    print("  journee degradee : %s" % ("oui" if USE_DEGRADED else "non"))
    print("  contexte posterieur : %s" % ("oui" if USE_FOLLOWING else "non"))
    print("  voisinage/aeroport/arrivees : %s" % ("oui" if USE_NEIGHBOR else "non"))
    print("  classifieur allege : %s" % ("oui" if CLF_LEAN else "non"))
    print("  modeles par aeroport : %s%s"
          % ("oui" if USE_PER_AIRPORT else "non",
             " (classifieur global)" if (USE_PER_AIRPORT and CLF_GLOBAL) else ""))
    print("  ensemble CatBoost : %s" % ("oui" if USE_CATBOOST else "non"))
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
        print("Controle de remplissage des features de journee degradee :")
        dd = pd.read_parquet(CACHE_PATH, columns=[c for c in DEGRADED_NUM])
        for c in DEGRADED_NUM:
            print("  %-22s renseignee sur %5.1f %%   mediane %6.2f"
                  % (c, 100 * dd[c].notna().mean(), dd[c].median()))


if __name__ == "__main__":
    main()
