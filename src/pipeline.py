"""
PRC Data Challenge 2026 - Taxi-out v16 (melange recalibre, sans Rome)
======================================================================

Bilan avant cette version :
  - Architecture a regimes (v5.1 / ARCH 0) validee : le modele unique
    perd 100 s, la decomposition corps/queue ne gagne rien.
  - Calibration de gamma_c au leaderboard : courbe publique parallele a
    la courbe locale (decalage constant ~32 s). La validation jan+juil
    2025 est donc un proxy FIABLE, on peut iterer en local.
  - Score public 347.6 s, tete de classement 246 s. Le regresseur sur
    lignes saines est a 235 s : l'ecart est entierement dans les 5 % de
    lignes pathologiques, et il vient des features, pas de l'architecture.

Deux sources d'information jamais exploitees, toutes deux presentes dans
le ranking (seuls les DEPARTS y sont masques, les arrivees sont
completes, block time et taxi-in compris) :

  1. CONTEXTE ARRIVEES : le taxi-in des arrivees au meme aeroport dans
     une fenetre autour du decollage est un proxy de congestion et de
     meteo CONTEMPORAIN de la cible (journee de neige, incident piste,
     pointe). Calcule a l'identique sur 2025 et sur jan/juil 2026.
       arr_taxi_mean_win      moyenne du taxi-in (ecrete) dans +/- 30 min
       arr_taxi_frac_long_win part des taxi-in > 1200 s dans la fenetre
       arr_taxi_day_mean      moyenne du taxi-in sur la journee aeroport

  2. ENCHAINEMENT PAR POSTE : l'avion qui part du poste X est presque
     toujours celui qui y est arrive juste avant. Son in-block est connu
     dans le ranking et borne physiquement l'off-block du depart.
       turnaround_s        MVT_TIME - in-block de l'arrivee precedente au
                           meme poste (borne SUPERIEURE du taxi-out)
       prev_arr_taxi_in    taxi-in de cette arrivee : proxy de la distance
                           poste <-> piste vue par le meme avion
       prev_same_type      meme type avion (confiance de l'appariement)
       prev_same_operator  meme operateur
       sched_before_inblock 1 si SCHED est anterieur a l'in-block precedent
                           (block = sched serait alors physiquement
                           impossible : signature de corruption)
       bound_minus_aobt    turnaround_s - delta_mvt_aobt3_s : marge entre
                           la borne physique et ce que NM voit

Tout est dans les donnees fournies, rien de futur (arrivee anterieure au
decollage), rien d'externe.

Resultat v9 : 371.1 s (+8.3 s), gain etale sur tous les aeroports, mais
CDG (579) et Rome (867) presque inchanges et classifieur de corruption
identique (rappel 10.9 %). Les features v9 aident le regime normal, pas
la detection.

  3. PANNE DE REPORTING (v10) : la corruption est un evenement systeme.
     Verifie sur 2025 : taux de corruption des departs de 4 % quand
     aucune arrivee de l'heure n'est corrompue, 23 a 30 % quand plus de
     30 % le sont (correlation 0.34 par aeroport-heure). Les arrivees du
     ranking ont leur block time : ce signal est observable en 2026.
       arr_corrupt_frac_win / _wide / _day_frac
       arr_anom_frac_win / _wide   (taxi-in aberrants, regime CDG/Rome)

Resultat v10 : 368.2 s (+2.9 s), Rome -33 s mais les autres aeroports
legerement degrades. Classifieur inchange. Signal reel mais dilue.

  4. BRANCHE ROME (v11) : voir le bloc de commentaires au dessus de
     rome_subset_mask. Une demi-douzaine de lignes a ~87 000 s de taxi
     (block estampille a la date programmee) portent environ la moitie du
     MSE. Elles sont concentrees sur un sous-ensemble identifiable (Rome,
     non apparie, decollage le lendemain) ou le taxi-out n'est presque
     jamais un vrai taxi-out. Prediction en esperance par tranche de
     retard, avec controle de presence du gain sur chaque mois de
     validation.

Resultat v11 : 348.4 s (+19.8 s), gain sur les deux mois, Rome 834 -> 715.
Decomposition du MSE restant (121 400) : 55 % bruit de fond du regime
normal, 25 % dans DEUX lignes de CDG a 84 240 et 58 206 s sans aucune
signature (block time de la veille), 8 % residu Rome, 5.6 % block=sched.
Sans les deux lignes de CDG : 303 s. Il ne reste pas de gros levier.

  5. FILE D'ATTENTE PAR PISTE (v12) : la congestion etait comptee au niveau
     de l'aeroport ; la file, c'est par piste. Decollages sur la meme
     piste avant / apres, intervalles entre decollages consecutifs,
     atterrissages sur la meme piste, part des departs sur cette piste.
     Vise le regime normal (55 % du MSE).

  6. ENSEMBLING (v12) : 5 seeds pour le classifieur et les regresseurs,
     moyenne des predictions. Rounds figes par seed au reentrainement.

Resultat v12 : 346.4 s (+2.0 s), gain de l'ensembling seul (residu 229 -> 226).
Les features de piste n'apparaissent pas dans le top 15.

  7. SIGNAUX PAR AVION ET PAR VOL (v13). Diagnostics sur 2025 :
     - corruption de l'arrivee precedente au meme poste : 14.6 % contre
       4.5 % ; precedente ET suivante corrompues : 38.5 % (3 000 vols/an)
     - target encoding par numero de vol : persistance H1 -> H2 de 0.815,
       37 numeros a plus de 80 % (precision 94 %), mais les numeros a plus
       de 30 % ne portent que 10 % des lignes corrompues. Gain reel et
       borne : la corruption est majoritairement diffuse.
     Features : prev_arr_corrupt, next_arr_corrupt, both_arr_corrupt,
     te_adep_flight.

Resultat v13 : 345.2 s (+1.2 s). te_adep_flight 9e du classifieur, AUC
inchangee. Corruption diffuse, confirme.

  8. TEMPS NON ENTRAVE (v14), transposition de la "physique du domaine" du
     rapport gagnant 2025 (estimation de masse) : le taxi-out = temps
     incompressible lie a la geometrie poste-piste + temps additionnel de
     congestion. Percentiles 10 et 50 par (aeroport, poste, piste) hors-
     fold, avec repli hierarchique, en features numeriques explicites ;
     et cible du regresseur B = excedent sur le p50. Vise le regime normal
     (55 % du MSE) et les lignes sans AOBT_3.

Resultat v14 : 345.9 s, neutre (bruit). unimpeded_p50 8e du regresseur A,
residu 226.1 -> 225.4. La geometrie etait deja extraite de stand_runway.

  9. FILE D'ATTENTE A L'OFF-BLOCK (v15). Revue de litterature (Idris 2001,
     Ravizza 2014, Wang 2021, Yin 2024) : la variable la plus predictive
     est la taille de la file au push-back et le taxi-out moyen recent.
     Toutes nos features de congestion etaient ancrees sur le DECOLLAGE.
     Reconstruites ici avec AOBT_3 comme off-block (non masque dans le
     ranking) : N (deja en roulage), Q (passes devant), taxi-out moyen des
     20 min precedentes, taux de push-back, atterrissages pendant mon
     roulage ; par piste et par aeroport.

Resultat v15 : 345.0 s local, 338.02 public (avec Rome). Aucune feature
q_* dans le top 15 : la congestion n'apporte rien quand la duree cote NM
est connue.

Leaderboard : v10 (pipeline v13 SANS Rome) 337.95, v11 (AVEC) 340.40.
La branche Rome perd 2.45 s sur 2026 malgre +19.8 s en local. Retiree.

 10. MELANGE RECALIBRE (v16). y = n + w (s - n). Le poids w = p^gamma est
     une recalibration uniforme ; le cout d'une erreur sur w croit en
     (s - n)^2. On apprend w = f(p, log(s - n)) sur les predictions de
     validation, petit modele monotone en p, sans aeroport. Transfert
     janvier <-> juillet : +7.6 / +1.3 s. Deux soumissions : reference
     gamma et recalibree.

Pipeline : ARCH 0 inchange (classifieur de corruption + melange
probabiliste + regresseur residu AOBT_3), balayage de gamma_c,
reentrainement a rounds figes, soumissions nommees selon la convention.

Usage : python taxiout_v16.py
"""

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import lightgbm as lgb

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # racine du depot
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


def main():
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


if __name__ == "__main__":
    main()
