#!/usr/bin/env python3
"""
PRC Data Challenge 2026 - prediction du taxi-out.

Trois sous-commandes :

  diagnose  verifie les colonnes potentiellement "quasi-cible" (AOBT_3_flt, LOBT_flt)
            sur 2025 et leur taux de remplissage sur le fichier de ranking,
            et calcule des baselines triviales pour situer le RMSE du modele.

  metar     telecharge les METAR horaires des 10 aeroports (archive ASOS Iowa State)
            pour 2025 + janvier/juillet 2026 et les met en cache en CSV.

  run       construit les features, entraine LightGBM avec validation sur
            janvier 2025 et juillet 2025, reentraine sur tout 2025 et ecrit
            la soumission au format attendu.

Exemples :
  python prc_taxiout.py diagnose --data ./data
  python prc_taxiout.py metar   --data ./data
  python prc_taxiout.py run     --data ./data --out submission.parquet --with-metar

Arborescence attendue sous --data :
  training_2025-01-01_2025-02-01.parquet ... training_2025-12-01_2026-01-01.parquet
  ranking.parquet
  submitting.parquet
  metar/  (cree par la sous-commande metar)
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]

# reseaux ASOS d'Iowa State pour le telechargement des METAR
METAR_NETWORK = {
    "EDDF": "DE__ASOS", "EDDM": "DE__ASOS", "EGLL": "GB__ASOS", "EHAM": "NL__ASOS",
    "LEBL": "ES__ASOS", "LEMD": "ES__ASOS", "LFPG": "FR__ASOS", "LIRF": "IT__ASOS",
    "LTFM": "TR__ASOS", "LSZH": "CH__ASOS",
}

TIME_COLS = [
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
]

CAT_COLS = [
    "AIRPORT", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt",
    "ADES_mvt", "arr_config", "dep_config", "wake_pair",
]


# ----------------------------------------------------------------------------- io

def load_training(data_dir, months=None):
    files = sorted(glob.glob(os.path.join(data_dir, "training_*.parquet")))
    if not files:
        sys.exit("aucun fichier training_*.parquet trouve dans %s" % data_dir)
    if months:
        files = [f for f in files if any(m in os.path.basename(f) for m in months)]
    frames = []
    for f in files:
        print("  lecture %s" % os.path.basename(f))
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True)
    return normalise(df)


def load_ranking(data_dir):
    path = os.path.join(data_dir, "ranking.parquet")
    if not os.path.exists(path):
        sys.exit("ranking.parquet introuvable dans %s" % data_dir)
    return normalise(pd.read_parquet(path))


def normalise(df):
    """Colonnes derivees communes : aeroport de report et temps en secondes."""
    for c in TIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    if "PHASE_mvt" not in df.columns and "PHASE" in df.columns:
        df = df.rename(columns={"PHASE": "PHASE_mvt"})
    df["AIRPORT"] = np.where(df["PHASE_mvt"].eq("DEP"), df["ADEP_mvt"], df["ADES_mvt"])
    df["t"] = to_epoch(df["MVT_TIME_UTC_mvt"])
    df["t_block"] = to_epoch(df["BLOCK_TIME_UTC_mvt"])
    return df


def to_epoch(s):
    """Serie datetime -> secondes float, NaT devient NaN.

    Les parquet peuvent etre en microsecondes ou en nanosecondes : on force
    l'unite avant la conversion en entier, sinon le facteur est faux.
    """
    v = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        v = v.dt.as_unit("ns")
    except (AttributeError, ValueError):
        pass
    out = v.astype("int64").astype("float64") / 1e9
    out[v.isna()] = np.nan
    return out


# --------------------------------------------------------------------- diagnostics

def cmd_diagnose(args):
    print("lecture de deux mois de training pour les diagnostics")
    tr = load_training(args.data, months=["2025-01-01", "2025-07-01"])
    dep = tr[tr["PHASE_mvt"].eq("DEP")].copy()
    dep = dep[dep["TAXITIME_SEC_mvt"].notna()]
    print("\ndepart 2025 (janv + juil) : %d lignes" % len(dep))

    y = dep["TAXITIME_SEC_mvt"].astype(float)
    print("taxi-out : moyenne %.0f s, ecart-type %.0f s, mediane %.0f s"
          % (y.mean(), y.std(), y.median()))
    for q in [0.001, 0.01, 0.99, 0.999]:
        print("  quantile %.3f : %.0f s" % (q, y.quantile(q)))
    print("part < 60 s   : %.3f %%" % (100 * (y < 60).mean()))
    print("part > 3600 s : %.3f %%" % (100 * (y > 3600).mean()))

    # colonnes potentiellement quasi-cible
    print("\n--- colonnes candidates a une reconstruction directe de la cible ---")
    for col in ["AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt"]:
        if col not in dep.columns:
            print("%s : absente" % col)
            continue
        est = dep["t"] - to_epoch(dep[col])
        err = est - y
        fill = 100 * est.notna().mean()
        print("%s : renseignee sur %.1f %% des departs 2025" % (col, fill))
        if est.notna().sum() > 1000:
            e = err.dropna()
            print("    erreur vs cible : mediane %.0f s, |err|<60s sur %.1f %%, "
                  "RMSE %.1f s" % (e.median(), 100 * (e.abs() < 60).mean(),
                                   np.sqrt((e ** 2).mean())))

    print("\n--- memes colonnes sur le fichier de ranking (janv + juil 2026) ---")
    rk = load_ranking(args.data)
    rdep = rk[rk["PHASE_mvt"].eq("DEP")]
    print("departs a predire : %d" % len(rdep))
    for col in ["MVT_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "IOBT_flt",
                "EOBT_1_flt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt"]:
        if col in rdep.columns:
            print("  %-22s renseignee sur %5.1f %%" % (col, 100 * rdep[col].notna().mean()))
    rarr = rk[rk["PHASE_mvt"].eq("ARR")]
    print("  arrivees presentes : %d, dont taxi-in renseigne %.1f %%"
          % (len(rarr), 100 * rarr["TAXITIME_SEC_mvt"].notna().mean()))

    # baselines triviales, pour situer le RMSE du modele
    print("\n--- baselines (validation croisee grossiere janv/juil) ---")
    dep["hour"] = dep["MVT_TIME_UTC_mvt"].dt.hour
    tr_part = dep[dep["MVT_TIME_UTC_mvt"].dt.month == 1]
    va_part = dep[dep["MVT_TIME_UTC_mvt"].dt.month == 7]
    baselines = {
        "moyenne globale": [],
        "moyenne par aeroport": ["AIRPORT"],
        "moyenne aeroport x piste": ["AIRPORT", "RUNWAY_mvt"],
        "moyenne aeroport x stand x piste": ["AIRPORT", "STAND_mvt", "RUNWAY_mvt"],
        "moyenne aeroport x stand x piste x heure": ["AIRPORT", "STAND_mvt", "RUNWAY_mvt", "hour"],
    }
    gmean = tr_part["TAXITIME_SEC_mvt"].mean()
    for name, keys in baselines.items():
        if not keys:
            pred = np.full(len(va_part), gmean)
        else:
            m = tr_part.groupby(keys, dropna=False)["TAXITIME_SEC_mvt"].mean()
            pred = va_part.set_index(keys).index.map(m).to_numpy(dtype=float)
            pred = np.where(np.isnan(pred), gmean, pred)
        rmse = np.sqrt(np.mean((pred - va_part["TAXITIME_SEC_mvt"].to_numpy()) ** 2))
        print("  %-42s RMSE %.1f s" % (name, rmse))
    print("\nnote : ces baselines sont calibrees janvier -> juillet, donc pessimistes. "
          "Elles donnent malgre tout l'ordre de grandeur a battre.")


# --------------------------------------------------------------------------- metar

def cmd_metar(args):
    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")
    outdir = os.path.join(args.data, "metar")
    os.makedirs(outdir, exist_ok=True)
    base = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    for icao in AIRPORTS:
        dest = os.path.join(outdir, "%s.csv" % icao)
        if os.path.exists(dest) and not args.force:
            print("  %s deja en cache" % icao)
            continue
        params = {
            "station": icao, "network": METAR_NETWORK[icao],
            "data": ["tmpf", "dwpf", "relh", "sknt", "drct", "vsby", "p01i", "wxcodes", "skyc1", "skyl1"],
            "year1": 2025, "month1": 1, "day1": 1,
            "year2": 2026, "month2": 8, "day2": 1,
            "tz": "UTC", "format": "onlycomma", "latlon": "no",
            "missing": "empty", "trace": "0.0001", "direct": "no", "report_type": [3, 4],
        }
        print("  telechargement %s" % icao)
        for attempt in range(3):
            try:
                r = requests.get(base, params=params, timeout=180)
                r.raise_for_status()
                with open(dest, "w") as fh:
                    fh.write(r.text)
                break
            except Exception as exc:  # noqa: BLE001
                print("    echec (%s), nouvelle tentative" % exc)
                time.sleep(5)
        else:
            print("    abandon pour %s" % icao)
    print("METAR en cache dans %s" % outdir)


def load_metar(data_dir):
    outdir = os.path.join(data_dir, "metar")
    files = sorted(glob.glob(os.path.join(outdir, "*.csv")))
    if not files:
        return None
    frames = []
    for f in files:
        icao = os.path.splitext(os.path.basename(f))[0]
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if "valid" not in d.columns:
            continue
        d["AIRPORT"] = icao
        frames.append(d)
    if not frames:
        return None
    m = pd.concat(frames, ignore_index=True)
    m["valid"] = pd.to_datetime(m["valid"], errors="coerce", utc=True)
    m = m.dropna(subset=["valid"])
    for c in ["tmpf", "dwpf", "relh", "sknt", "drct", "vsby", "p01i", "skyl1"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m["temp_c"] = (m["tmpf"] - 32) * 5 / 9 if "tmpf" in m else np.nan
    m["dew_c"] = (m["dwpf"] - 32) * 5 / 9 if "dwpf" in m else np.nan
    wx = m["wxcodes"].fillna("").astype(str) if "wxcodes" in m else pd.Series("", index=m.index)
    m["wx_snow"] = wx.str.contains("SN|SG|PL|GS|IC", regex=True).astype(np.int8)
    m["wx_freezing"] = wx.str.contains("FZ").astype(np.int8)
    m["wx_rain"] = wx.str.contains("RA|DZ", regex=True).astype(np.int8)
    m["wx_ts"] = wx.str.contains("TS").astype(np.int8)
    m["wx_fog"] = wx.str.contains("FG|BR", regex=True).astype(np.int8)
    m["lvp"] = ((m.get("vsby", pd.Series(np.nan, index=m.index)) < 0.5)
                | (m.get("skyl1", pd.Series(np.nan, index=m.index)) < 200)).astype(np.int8)
    # potentiel de degivrage : temperature basse et humidite ou precipitation
    m["deice_risk"] = ((m["temp_c"] < 3) & ((m["wx_snow"] == 1) | (m["wx_freezing"] == 1)
                                            | (m["wx_rain"] == 1)
                                            | (m["temp_c"] - m["dew_c"] < 3))).astype(np.int8)
    keep = ["AIRPORT", "valid", "temp_c", "dew_c", "sknt", "vsby", "p01i",
            "wx_snow", "wx_freezing", "wx_rain", "wx_ts", "wx_fog", "lvp", "deice_risk"]
    keep = [c for c in keep if c in m.columns]
    m = m[keep].sort_values(["AIRPORT", "valid"]).reset_index(drop=True)
    return m


def as_ns(s):
    """Force une serie datetime en datetime64[ns, UTC].

    Les parquet du challenge sont en microsecondes et les METAR parses en
    nanosecondes : merge_asof refuse deux unites differentes.
    """
    v = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        v = v.dt.as_unit("ns")
    except (AttributeError, ValueError):
        pass
    return v


def merge_metar(df, metar):
    """Jointure asof (dernier releve connu) par aeroport."""
    if metar is None:
        return df
    df = df.copy()
    df["_mt"] = as_ns(df["MVT_TIME_UTC_mvt"])
    metar = metar.copy()
    metar["valid"] = as_ns(metar["valid"])
    metar = metar.dropna(subset=["valid"]).sort_values("valid")
    df = df.sort_values("_mt", kind="mergesort")
    known = df["_mt"].notna()
    out = []
    if (~known).any():
        out.append(df[~known])
    for ap, chunk in df[known].groupby("AIRPORT", sort=False):
        sub = metar[metar["AIRPORT"] == ap].drop(columns=["AIRPORT"])
        if sub.empty:
            out.append(chunk)
            continue
        merged = pd.merge_asof(
            chunk.sort_values("_mt", kind="mergesort"), sub,
            left_on="_mt", right_on="valid",
            direction="backward", tolerance=pd.Timedelta("2h"),
        )
        out.append(merged.drop(columns=["valid"], errors="ignore"))
    res = pd.concat(out, ignore_index=True)
    return res.drop(columns=["_mt"], errors="ignore")


# ------------------------------------------------------------------ features temps

def window_count(events, queries, before_s, after_s):
    """Nombre d'evenements dans [q - before, q + after]. events doit etre trie."""
    lo = np.searchsorted(events, queries - before_s, side="left")
    hi = np.searchsorted(events, queries + after_s, side="right")
    return (hi - lo).astype(np.float32)


def causal_mean(event_t, event_v, queries, window_s):
    """Moyenne des valeurs d'evenements dans ]q - window, q]."""
    order = np.argsort(event_t)
    t = event_t[order]
    v = event_v[order]
    ok = np.isfinite(v)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    hi = np.searchsorted(t, queries, side="right")
    lo = np.searchsorted(t, queries - window_s, side="left")
    s = csum[hi] - csum[lo]
    c = ccnt[hi] - ccnt[lo]
    return np.where(c > 0, s / np.maximum(c, 1), np.nan).astype(np.float32)


def prev_gap(events, queries):
    """Ecart au dernier evenement strictement anterieur."""
    idx = np.searchsorted(events, queries, side="left") - 1
    out = np.full(len(queries), np.nan, dtype=np.float32)
    ok = idx >= 0
    out[ok] = (queries[ok] - events[idx[ok]]).astype(np.float32)
    return out


def add_congestion(df):
    """Features de congestion ancrees sur l'heure de decollage (connue au test)."""
    n = len(df)
    cols = {k: np.full(n, np.nan, dtype=np.float32) for k in [
        "dep_cnt_15", "dep_cnt_30", "dep_cnt_60", "dep_cnt_ahead_15",
        "dep_rwy_cnt_15", "dep_rwy_cnt_30", "dep_rwy_rank_15", "gap_prev_dep_rwy",
        "arr_cnt_30", "arr_cnt_60", "taxiin_mean_60", "taxiin_mean_180",
        "dep_arr_ratio_60",
    ]}
    is_dep = df["PHASE_mvt"].eq("DEP").to_numpy()
    t = df["t"].to_numpy(dtype=float)
    ap = df["AIRPORT"].to_numpy()
    rwy = df["RUNWAY_mvt"].astype(str).to_numpy()
    taxi = pd.to_numeric(df.get("TAXITIME_SEC_mvt"), errors="coerce").to_numpy(dtype=float)
    t_block = df["t_block"].to_numpy(dtype=float)

    for a in pd.unique(ap):
        sel = (ap == a) & np.isfinite(t)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        tt = t[idx]
        dep_t = np.sort(tt[is_dep[idx]])
        arr_t = np.sort(tt[~is_dep[idx]])
        cols["dep_cnt_15"][idx] = window_count(dep_t, tt, 900, 900)
        cols["dep_cnt_30"][idx] = window_count(dep_t, tt, 1800, 1800)
        cols["dep_cnt_60"][idx] = window_count(dep_t, tt, 3600, 3600)
        cols["dep_cnt_ahead_15"][idx] = window_count(dep_t, tt, 0, 900)
        cols["arr_cnt_30"][idx] = window_count(arr_t, tt, 1800, 1800)
        cols["arr_cnt_60"][idx] = window_count(arr_t, tt, 3600, 3600)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols["dep_arr_ratio_60"][idx] = (cols["dep_cnt_60"][idx]
                                             / np.maximum(cols["arr_cnt_60"][idx], 1))
        # taxi-in observe des arrivees : proxy direct de l'etat de la surface
        arr_idx = idx[~is_dep[idx]]
        if arr_idx.size:
            ev_t = np.where(np.isfinite(t_block[arr_idx]), t_block[arr_idx], t[arr_idx])
            ev_v = taxi[arr_idx]
            cols["taxiin_mean_60"][idx] = causal_mean(ev_t, ev_v, tt, 3600)
            cols["taxiin_mean_180"][idx] = causal_mean(ev_t, ev_v, tt, 10800)
        # par piste
        for r in pd.unique(rwy[idx]):
            sub = idx[rwy[idx] == r]
            if sub.size == 0:
                continue
            ts = t[sub]
            dep_r = np.sort(t[sub[is_dep[sub]]])
            cols["dep_rwy_cnt_15"][sub] = window_count(dep_r, ts, 900, 900)
            cols["dep_rwy_cnt_30"][sub] = window_count(dep_r, ts, 1800, 1800)
            cols["dep_rwy_rank_15"][sub] = window_count(dep_r, ts, 900, 0)
            cols["gap_prev_dep_rwy"][sub] = prev_gap(dep_r, ts)

    for k, v in cols.items():
        df[k] = v
    return df


def add_queue(df):
    """File d'attente au depart et sequence de decollage.

    Tout est ancre sur des colonnes disponibles en 2026 : l'heure de decollage,
    l'heure programmee et l'EOBT. On ne se sert jamais du block time, masque
    pour les departs du fichier de ranking.
    """
    n = len(df)
    keys = ["q_eobt", "q_sched", "q_eobt_rwy", "rate_rwy_15", "gap_mean_5_rwy",
            "gap_prev_1_rwy", "burst_pos", "arr_same_rwy_30", "wake_prev_gap"]
    cols = {k: np.full(n, np.nan, dtype=np.float32) for k in keys}
    wake_pair = np.full(n, "NA", dtype=object)

    is_dep = df["PHASE_mvt"].eq("DEP").to_numpy()
    t = df["t"].to_numpy(dtype=float)
    ap = df["AIRPORT"].to_numpy()
    rwy = df["RUNWAY_mvt"].astype(str).to_numpy()
    wake = df["WK_TBL_CAT_flt"].astype(str).to_numpy() if "WK_TBL_CAT_flt" in df.columns \
        else np.full(n, "NA", dtype=object)
    eobt = (to_epoch(df["EOBT_1_flt"]).to_numpy(dtype=float)
            if "EOBT_1_flt" in df.columns else np.full(n, np.nan))
    sched = (to_epoch(df["SCHED_TIME_UTC_mvt"]).to_numpy(dtype=float)
             if "SCHED_TIME_UTC_mvt" in df.columns else np.full(n, np.nan))

    for a in pd.unique(ap):
        sel = (ap == a) & np.isfinite(t)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        tt = t[idx]
        d_idx = idx[is_dep[idx]]
        a_idx = idx[~is_dep[idx]]
        dep_t = np.sort(t[d_idx])

        # nombre de vols dont le repoussage etait prevu avant t et qui n'ont pas
        # encore decolle : une mesure directe de la file, calculable au test
        for src, name in [(eobt, "q_eobt"), (sched, "q_sched")]:
            v = np.sort(src[d_idx][np.isfinite(src[d_idx])])
            if v.size == 0:
                continue
            planned = np.searchsorted(v, tt, side="right")
            gone = np.searchsorted(dep_t, tt, side="right")
            cols[name][idx] = np.maximum(planned - gone, 0).astype(np.float32)

        for r in pd.unique(rwy[idx]):
            sub = idx[rwy[idx] == r]
            if sub.size == 0:
                continue
            ts = t[sub]
            d_sub = sub[is_dep[sub]]
            order = np.argsort(t[d_sub])
            d_sub = d_sub[order]
            dr = t[d_sub]

            # file par piste
            v = np.sort(eobt[d_sub][np.isfinite(eobt[d_sub])])
            if v.size:
                cols["q_eobt_rwy"][sub] = np.maximum(
                    np.searchsorted(v, ts, side="right")
                    - np.searchsorted(dr, ts, side="right"), 0).astype(np.float32)

            # debit observe de la piste
            cnt15 = (np.searchsorted(dr, ts, side="right")
                     - np.searchsorted(dr, ts - 900, side="left"))
            cols["rate_rwy_15"][sub] = (cnt15 / 15.0).astype(np.float32)

            if dr.size >= 2:
                gaps = np.diff(dr)
                # ecart au decollage precedent et moyenne des cinq derniers ecarts
                pos = np.searchsorted(dr, ts, side="left") - 1
                ok = pos >= 1
                g1 = np.full(ts.shape, np.nan, dtype=np.float32)
                g1[ok] = gaps[pos[ok] - 1]
                cols["gap_prev_1_rwy"][sub] = g1
                csum = np.concatenate([[0.0], np.cumsum(gaps)])
                lo = np.maximum(pos - 5, 0)
                g5 = np.full(ts.shape, np.nan, dtype=np.float32)
                cnt = pos - lo
                good = ok & (cnt > 0)
                g5[good] = ((csum[pos[good]] - csum[lo[good]]) / cnt[good])
                cols["gap_mean_5_rwy"][sub] = g5

                # longueur de la rafale : decollages consecutifs espaces de moins
                # de deux minutes, ce qui caracterise une file qui se vide
                burst = np.zeros(dr.size, dtype=np.float32)
                for i in range(1, dr.size):
                    burst[i] = burst[i - 1] + 1 if gaps[i - 1] < 120 else 0.0
                bpos = np.full(ts.shape, np.nan, dtype=np.float32)
                bpos[pos >= 0] = burst[pos[pos >= 0]]
                cols["burst_pos"][sub] = bpos

                # categorie de sillage du decollage precedent
                pw = np.full(ts.shape, "NA", dtype=object)
                pw[pos >= 0] = wake[d_sub][pos[pos >= 0]]
                wake_pair[sub] = pd.Series(pw).astype(str) + ">" + pd.Series(wake[sub]).astype(str)
                cols["wake_prev_gap"][sub] = g1

            # arrivees sur la meme piste : mode mixte
            ar = np.sort(t[a_idx[rwy[a_idx] == r]]) if a_idx.size else np.array([])
            if ar.size:
                cols["arr_same_rwy_30"][sub] = window_count(ar, ts, 1800, 1800)

    for k, v in cols.items():
        df[k] = v
    df["wake_pair"] = wake_pair
    return df


def add_config(df):
    """Configuration de pistes deduite par tranche de 30 min, arrivees et departs."""
    bucket = np.floor(df["t"].to_numpy(dtype=float) / 1800.0)
    df["_bucket"] = bucket
    out = df[["AIRPORT", "_bucket"]].copy()
    for phase, name in [("ARR", "arr_config"), ("DEP", "dep_config")]:
        sub = df[df["PHASE_mvt"].eq(phase)]
        if sub.empty:
            df[name] = "NA"
            continue
        cnt = (sub.groupby(["AIRPORT", "_bucket", "RUNWAY_mvt"], dropna=False)
               .size().rename("n").reset_index())
        cnt = cnt.sort_values("n", ascending=False)
        dom = cnt.drop_duplicates(subset=["AIRPORT", "_bucket"])[
            ["AIRPORT", "_bucket", "RUNWAY_mvt"]].rename(columns={"RUNWAY_mvt": name})
        out = out.merge(dom, on=["AIRPORT", "_bucket"], how="left")
    df["arr_config"] = out["arr_config"].astype(str).to_numpy()
    df["dep_config"] = out["dep_config"].astype(str).to_numpy()
    df = df.drop(columns=["_bucket"])
    return df


def add_calendar_and_plan(df):
    ts = df["MVT_TIME_UTC_mvt"]
    df["hour"] = ts.dt.hour.astype("float32")
    df["minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute).astype("float32")
    df["dow"] = ts.dt.dayofweek.astype("float32")
    df["month"] = ts.dt.month.astype("float32")
    df["doy"] = ts.dt.dayofyear.astype("float32")
    df["is_weekend"] = (df["dow"] >= 5).astype("float32")
    df["tod_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440.0).astype("float32")
    df["tod_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440.0).astype("float32")

    t = df["t"].to_numpy(dtype=float)
    for col, name in [("SCHED_TIME_UTC_mvt", "d_sched"), ("EOBT_1_flt", "d_eobt"),
                      ("IOBT_flt", "d_iobt"), ("LOBT_flt", "d_lobt"),
                      ("AOBT_3_flt", "d_aobt3")]:
        if col in df.columns:
            df[name] = (t - to_epoch(df[col]).to_numpy(dtype=float)).astype("float32")
    if "EOBT_1_flt" in df.columns and "SCHED_TIME_UTC_mvt" in df.columns:
        df["eobt_vs_sched"] = (to_epoch(df["EOBT_1_flt"]).to_numpy(dtype=float)
                               - to_epoch(df["SCHED_TIME_UTC_mvt"]).to_numpy(dtype=float)
                               ).astype("float32")
    if "EOBT_1_flt" in df.columns and "IOBT_flt" in df.columns:
        df["eobt_vs_iobt"] = (to_epoch(df["EOBT_1_flt"]).to_numpy(dtype=float)
                              - to_epoch(df["IOBT_flt"]).to_numpy(dtype=float)).astype("float32")
    if "ARVT_1_flt" in df.columns and "EOBT_1_flt" in df.columns:
        df["planned_block"] = (to_epoch(df["ARVT_1_flt"]).to_numpy(dtype=float)
                               - to_epoch(df["EOBT_1_flt"]).to_numpy(dtype=float)).astype("float32")
    df["nm_non_apparie"] = (df["FLIGHT_ID_mvt"].isna().astype("float32")
                            if "FLIGHT_ID_mvt" in df.columns else np.float32(0))
    df["nm_sans_aobt"] = (df["AOBT_3_flt"].isna().astype("float32")
                          if "AOBT_3_flt" in df.columns else np.float32(1))
    if "ADES_FILED_flt" in df.columns and "ADES_flt" in df.columns:
        df["deroute"] = (df["ADES_FILED_flt"].astype(str)
                         != df["ADES_flt"].astype(str)).astype("float32")
    return df

def fit_target_stats(train_dep, keys, prior_mean, prior_weight=50):
    g = train_dep.groupby(keys, dropna=False)["TAXITIME_SEC_mvt"]
    stats = g.agg(["mean", "count", "median"])
    stats["q20"] = g.quantile(0.20)
    stats["q80"] = g.quantile(0.80)
    stats["smooth"] = ((stats["mean"] * stats["count"] + prior_mean * prior_weight)
                       / (stats["count"] + prior_weight))
    return stats


def apply_target_stats(df, stats, keys, prefix, prior_mean):
    idx = df.set_index(keys).index
    for src, name in [("smooth", "te"), ("median", "med"), ("q20", "q20"),
                      ("q80", "q80"), ("count", "n")]:
        vals = idx.map(stats[src]).to_numpy(dtype=float)
        if src in ("smooth", "median", "q20", "q80"):
            vals = np.where(np.isnan(vals), prior_mean, vals)
        else:
            vals = np.where(np.isnan(vals), 0.0, vals)
        df["%s_%s" % (prefix, name)] = vals.astype("float32")
    return df


# --------------------------------------------------------------------- construction

def build(df, metar=None):
    print("  calendrier et plan de vol")
    df = add_calendar_and_plan(df)
    print("  configuration de pistes")
    df = add_config(df)
    print("  congestion")
    df = add_congestion(df)
    print("  file d'attente et sequence de decollage")
    df = add_queue(df)
    if metar is not None:
        print("  meteo")
        df = merge_metar(df, metar)
    return df


def is_datetime(col):
    return pd.api.types.is_datetime64_any_dtype(col)


def prepare_matrix(df, cat_maps=None):
    feats = [c for c in df.columns if c not in {
        "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "TAXITIME_SEC_mvt",
        "t", "t_block", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt",
        "AIRCRAFT_TYPE_flt", "FLIGHT_RULE_flt",
    } and not is_datetime(df[c])]
    X = df[feats].copy()
    maps = {} if cat_maps is None else cat_maps
    for c in feats:
        if c in CAT_COLS or X[c].dtype == object:
            s = X[c].astype("object").where(X[c].notna(), "NA").astype(str)
            if cat_maps is None:
                cats = pd.Index(sorted(s.unique()))
                maps[c] = cats
            else:
                cats = maps.get(c, pd.Index([]))
            X[c] = pd.Categorical(s, categories=cats)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X, maps, feats


EXTREME_THR = 20000.0  # au-dela, le label est un defaut d'horaire, pas un roulage


def match_state(df):
    """Trois populations distinctes, que le mouvement soit apparie ou non a NM.

    La jointure a gauche laisse toutes les colonnes _flt nulles pour un
    mouvement jamais apparie : ce n'est pas la meme chose qu'un vol apparie
    dont l'heure d'off-block reelle manque.
    """
    unmatched = df["FLIGHT_ID_mvt"].isna() if "FLIGHT_ID_mvt" in df.columns \
        else pd.Series(False, index=df.index)
    no_aobt = df["AOBT_3_flt"].isna() if "AOBT_3_flt" in df.columns \
        else pd.Series(True, index=df.index)
    out = pd.Series("apparie avec AOBT", index=df.index, dtype=object)
    out[no_aobt & ~unmatched] = "apparie sans AOBT"
    out[unmatched] = "non apparie NM"
    return out


def lane_mask(df):
    """Voie sans heure d'off-block NM : c'est la ou vivent les labels par defaut."""
    if "d_aobt3" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["d_aobt3"].isna()


def fit_mixture(fit_df, y_fit, lane_fit, prior_weight=30):
    """Par aeroport, sur la voie sans off-block NM : proportion de labels par
    defaut et moyenne de chaque composante. Renvoie un dict aeroport -> params."""
    out = {}
    idx = np.flatnonzero(lane_fit)
    if idx.size == 0:
        return out
    yl = y_fit[idx]
    ap = fit_df["AIRPORT"].to_numpy()[idx]
    hi_all = yl > EXTREME_THR
    p_global = float(hi_all.mean())
    m_hi_global = float(yl[hi_all].mean()) if hi_all.any() else np.nan
    m_lo_global = float(yl[~hi_all].mean()) if (~hi_all).any() else np.nan
    out["__global__"] = (p_global, m_hi_global, m_lo_global, int(idx.size))
    for a in pd.unique(ap):
        m = ap == a
        ya = yl[m]
        hi = ya > EXTREME_THR
        n = int(m.sum())
        p = (hi.sum() + p_global * prior_weight) / (n + prior_weight)
        m_hi = float(ya[hi].mean()) if hi.any() else m_hi_global
        m_lo = float(ya[~hi].mean()) if (~hi).any() else m_lo_global
        out[a] = (float(p), m_hi, m_lo, n)
    return out


def apply_mixture(pred, df, lane, mix):
    """Sur la voie concernee, remplace la prediction par la moyenne du melange."""
    if not mix:
        return pred, 0
    pred = pred.copy()
    ap = df["AIRPORT"].to_numpy()
    gp, ghi, glo, _ = mix.get("__global__", (0.0, np.nan, np.nan, 0))
    idx = np.flatnonzero(np.asarray(lane))
    changed = 0
    for i in idx:
        p, m_hi, m_lo, _ = mix.get(ap[i], (gp, ghi, glo, 0))
        if not np.isfinite(m_hi) or p <= 0:
            continue
        base = min(pred[i], EXTREME_THR) if np.isfinite(pred[i]) else m_lo
        if not np.isfinite(base):
            base = m_lo
        pred[i] = p * m_hi + (1 - p) * base
        changed += 1
    return pred, changed


def cmd_outliers(args):
    """Analyse un aeroport a forte erreur : d'ou vient la variance de la cible."""
    ap = args.airport
    print("lecture du training complet")
    tr = load_training(args.data)
    dep = tr[tr["PHASE_mvt"].eq("DEP") & tr["TAXITIME_SEC_mvt"].notna()].copy()
    dep["TAXITIME_SEC_mvt"] = dep["TAXITIME_SEC_mvt"].astype(float)

    print("\n--- distribution de la cible par aeroport (2025) ---")
    g = dep.groupby("AIRPORT")["TAXITIME_SEC_mvt"]
    tab = pd.DataFrame({
        "n": g.size(), "moy": g.mean(), "med": g.median(), "ecart_type": g.std(),
        "p95": g.quantile(0.95), "p99": g.quantile(0.99), "max": g.max(),
        "pct_sup_3600": 100 * dep.assign(x=dep["TAXITIME_SEC_mvt"] > 3600)
                        .groupby("AIRPORT")["x"].mean(),
        "pct_inf_120": 100 * dep.assign(x=dep["TAXITIME_SEC_mvt"] < 120)
                       .groupby("AIRPORT")["x"].mean(),
    })
    print(tab.sort_values("ecart_type", ascending=False).round(1).to_string())

    sub = dep[dep["AIRPORT"].eq(ap)].copy()
    if sub.empty:
        sys.exit("aucun depart pour %s" % ap)
    print("\n--- %s : %d departs ---" % (ap, len(sub)))
    sub["mois"] = sub["MVT_TIME_UTC_mvt"].dt.month
    sub["heure"] = sub["MVT_TIME_UTC_mvt"].dt.hour
    sub["extreme"] = (sub["TAXITIME_SEC_mvt"] > 3600).astype(float)

    for by, label in [("mois", "mois"), ("heure", "heure")]:
        gg = sub.groupby(by)["TAXITIME_SEC_mvt"]
        t = pd.DataFrame({"n": gg.size(), "moy": gg.mean(), "med": gg.median(),
                          "ecart_type": gg.std(),
                          "pct_sup_3600": 100 * sub.groupby(by)["extreme"].mean()})
        print("\npar %s :" % label)
        print(t.round(1).to_string())

    print("\npistes les plus dispersees :")
    gr = sub.groupby("RUNWAY_mvt")["TAXITIME_SEC_mvt"]
    tr_tab = pd.DataFrame({"n": gr.size(), "moy": gr.mean(), "ecart_type": gr.std()})
    print(tr_tab[tr_tab["n"] > 200].sort_values("ecart_type", ascending=False)
          .head(10).round(1).to_string())

    # les arrivees 2026 ne sont pas masquees : meme pathologie l'annee suivante ?
    rk = load_ranking(args.data)
    arr26 = rk[rk["PHASE_mvt"].eq("ARR") & rk["AIRPORT"].eq(ap)]["TAXITIME_SEC_mvt"].astype(float)
    arr25 = tr[tr["PHASE_mvt"].eq("ARR") & tr["AIRPORT"].eq(ap)]["TAXITIME_SEC_mvt"].astype(float)
    print("\n--- taxi-in a %s, 2025 vs 2026 (les arrivees ne sont pas masquees) ---" % ap)
    for name, s in [("2025", arr25.dropna()), ("2026 janv+juil", arr26.dropna())]:
        if len(s):
            print("  %-16s n=%7d  moy %6.0f  med %6.0f  ecart-type %6.0f  "
                  "p99 %6.0f  pct>3600 %.2f %%"
                  % (name, len(s), s.mean(), s.median(), s.std(), s.quantile(0.99),
                     100 * (s > 3600).mean()))

    # un estimateur trivial fait-il mieux que le modele sur cet aeroport ?
    print("\n--- estimateurs simples sur %s (entraine hors janv/juil, teste sur janv+juil) ---" % ap)
    m = sub["mois"]
    fit = sub[~m.isin([1, 7])]
    test = sub[m.isin([1, 7])]
    if len(fit) < 500 or len(test) < 500:
        print("  pas assez de donnees")
        return
    yt = test["TAXITIME_SEC_mvt"].to_numpy()

    def rmse(p):
        return float(np.sqrt(np.mean((np.asarray(p, dtype=float) - yt) ** 2)))

    gmean = fit["TAXITIME_SEC_mvt"].mean()
    gmed = fit["TAXITIME_SEC_mvt"].median()
    wins = fit["TAXITIME_SEC_mvt"].clip(upper=fit["TAXITIME_SEC_mvt"].quantile(0.99)).mean()
    print("  moyenne globale            RMSE %8.1f s  (pred %.0f)" % (rmse(np.full(len(test), gmean)), gmean))
    print("  mediane globale            RMSE %8.1f s  (pred %.0f)" % (rmse(np.full(len(test), gmed)), gmed))
    print("  moyenne winsorisee p99     RMSE %8.1f s  (pred %.0f)" % (rmse(np.full(len(test), wins)), wins))
    for keys, name in [(["STAND_mvt", "RUNWAY_mvt"], "moyenne stand x piste"),
                       (["RUNWAY_mvt", "heure"], "moyenne piste x heure"),
                       (["STAND_mvt", "RUNWAY_mvt", "heure"], "moyenne stand x piste x heure")]:
        mp = fit.groupby(keys, dropna=False)["TAXITIME_SEC_mvt"].mean()
        p = test.set_index(keys).index.map(mp).to_numpy(dtype=float)
        p = np.where(np.isnan(p), gmean, p)
        print("  %-26s RMSE %8.1f s" % (name, rmse(p)))
    print("\nlecture : si la moyenne globale approche les estimateurs structures, la cible")
    print("de cet aeroport est majoritairement du bruit et il vaut mieux y predire une")
    print("moyenne conditionnelle prudente que chercher a expliquer la variance.")


def rule_candidates(df, thr, require_no_type=False, airports=None,
                    require_unmatched=False):
    """Lignes suspectees d'avoir un label recalcule sur l'heure programmee."""
    ds = df["d_sched"] if "d_sched" in df.columns else pd.Series(np.nan, index=df.index)
    cand = lane_mask(df) & (ds > thr)
    if require_no_type and "AIRCRAFT_TYPE_mvt" in df.columns:
        cand = cand & df["AIRCRAFT_TYPE_mvt"].isna()
    if require_unmatched and "FLIGHT_ID_mvt" in df.columns:
        cand = cand & df["FLIGHT_ID_mvt"].isna()
    if airports:
        cand = cand & df["AIRPORT"].isin(airports)
    return cand.to_numpy()


RULE_VARIANTS = [
    ("base", dict()),
    ("+ type avion vide", dict(require_no_type=True)),
    ("+ non apparie NM", dict(require_no_type=True, require_unmatched=True)),
    ("+ Rome seul", dict(require_no_type=True, require_unmatched=True,
                         airports=["LIRF"])),
]


def cmd_dayshift(args):
    """Famille des labels decales d'un jour : sont-ils reconstructibles ?

    La regle y = MVT_TIME - SCHED_TIME couvre 76 % des labels aberrants.
    Le reste se concentre autour de 86 400 s : le block time semble valoir
    une heure de reference diminuee d'un jour entier. Si le decalage est
    exactement 86 400 s, ces lignes se recalculent aussi.
    """
    print("lecture du training complet")
    tr = load_training(args.data)
    dep = tr[tr["PHASE_mvt"].eq("DEP") & tr["TAXITIME_SEC_mvt"].notna()].copy()
    dep["TAXITIME_SEC_mvt"] = dep["TAXITIME_SEC_mvt"].astype(float)
    dep = add_calendar_and_plan(dep)
    y = dep["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    ds = dep["d_sched"].to_numpy(dtype=float)

    ext = y > args.thr
    by_rule = np.abs(y - ds) < 120
    rest = ext & ~by_rule
    print("\nlabels > %d s : %d" % (args.thr, int(ext.sum())))
    print("  couverts par y = MVT_TIME - SCHED_TIME : %d" % int((ext & by_rule).sum()))
    print("  restants : %d" % int(rest.sum()))
    if rest.sum() == 0:
        return
    sub = dep[rest].copy()

    print("\ndistribution des restants :")
    v = sub["TAXITIME_SEC_mvt"]
    print("  min %.0f s, mediane %.0f s, max %.0f s" % (v.min(), v.median(), v.max()))
    print("  dans [80000, 95000] : %d / %d"
          % (int(((v >= 80000) & (v <= 95000)).sum()), len(v)))
    print("  par aeroport : %s" % dict(sub["AIRPORT"].value_counts()))

    # y - 86400 vaut-il un delta planifie connu ?
    print("\ny - 86400 compare a chaque delta planifie :")
    resid = sub["TAXITIME_SEC_mvt"] - 86400.0
    print("  y - 86400 : mediane %.0f s, min %.0f s, max %.0f s"
          % (resid.median(), resid.min(), resid.max()))
    for c in ["d_sched", "d_eobt", "d_iobt", "d_lobt", "d_aobt3"]:
        if c not in sub.columns:
            continue
        d = (resid - sub[c]).dropna()
        if len(d) < 2:
            continue
        print("  (y - 86400) - %-9s n=%3d  mediane %8.0f s  |ecart|<120 s sur %3d"
              % (c, len(d), d.median(), int((d.abs() < 120).sum())))

    # le decalage est-il exactement un nombre entier de jours ?
    print("\ndecalage en jours entiers :")
    for k in (1, 2, 3):
        cand = sub["TAXITIME_SEC_mvt"] - 86400.0 * k
        for c in ["d_sched", "d_eobt"]:
            if c not in sub.columns:
                continue
            d = (cand - sub[c]).dropna()
            hit = int((d.abs() < 120).sum())
            if hit:
                print("  y - %d jour(s) = %-9s sur %d / %d lignes" % (k, c, hit, len(d)))

    cols = ["AIRPORT", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt", "d_sched", "d_eobt", "STAND_mvt", "RUNWAY_mvt"]
    cols = [c for c in cols if c in sub.columns]
    print("\nles lignes restantes :")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(sub[cols].to_string())

    print("\ncandidats dans le ranking 2026 : lignes non appariees NM, type avion vide,")
    print("dont l'heure programmee est anterieure de plus de 20 h au decollage")
    rk = load_ranking(args.data)
    rdep = add_calendar_and_plan(rk[rk["PHASE_mvt"].eq("DEP")].copy())
    c = (rdep["FLIGHT_ID_mvt"].isna() & rdep["AIRCRAFT_TYPE_mvt"].isna()
         & (rdep["d_sched"] > 72000))
    print("  %d lignes" % int(c.sum()))
    if int(c.sum()):
        print("  par aeroport : %s" % dict(rdep.loc[c, "AIRPORT"].value_counts()))


def cmd_rule(args):
    """Mesure precision et rappel de la regle y = MVT_TIME - SCHED_TIME."""
    print("lecture du training complet")
    tr = load_training(args.data)
    dep = tr[tr["PHASE_mvt"].eq("DEP") & tr["TAXITIME_SEC_mvt"].notna()].copy()
    dep["TAXITIME_SEC_mvt"] = dep["TAXITIME_SEC_mvt"].astype(float)
    dep = add_calendar_and_plan(dep)
    y = dep["TAXITIME_SEC_mvt"].to_numpy()
    ds = dep["d_sched"].to_numpy(dtype=float)

    # verite : le label vaut l'ecart a l'heure programmee, et il est aberrant
    is_default = (np.abs(y - ds) < args.tol) & (y > args.thr_extreme)
    print("\nlabels reconstructibles par y = MVT_TIME - SCHED_TIME : %d"
          % int(is_default.sum()))
    print("labels aberrants au total (> %d s) : %d"
          % (args.thr_extreme, int((y > args.thr_extreme).sum())))
    print("part des aberrants couverte par la regle : %.1f %%"
          % (100 * is_default.sum() / max((y > args.thr_extreme).sum(), 1)))

    normal_pred = float(np.median(y[y < 3600]))
    print("\nvaleur de repli utilisee pour la comparaison : %.0f s" % normal_pred)
    print("\n%-9s %8s %8s %8s %10s %12s" % ("seuil", "cand.", "vrais", "precision",
                                            "rappel", "gain RMSE"))
    n_total = len(y)
    for thr in args.grid:
        cand = rule_candidates(dep, thr)
        n_c = int(cand.sum())
        if n_c == 0:
            print("%-9d %8d %8s %8s %10s %12s" % (thr, 0, "-", "-", "-", "-"))
            continue
        n_true = int((cand & is_default).sum())
        prec = n_true / n_c
        rec = n_true / max(is_default.sum(), 1)
        # erreur si on applique la regle, versus si on predit le repli
        sq_rule = float(np.sum((ds[cand] - y[cand]) ** 2))
        sq_base = float(np.sum((normal_pred - y[cand]) ** 2))
        gain = (np.sqrt(sq_base / n_total) ** 2 - np.sqrt(sq_rule / n_total) ** 2)
        gain_rmse = np.sqrt(max(sq_base, 0) / n_total) - np.sqrt(max(sq_rule, 0) / n_total)
        print("%-9d %8d %8d %8.3f %10.3f %10.1f s"
              % (thr, n_c, n_true, prec, rec, gain_rmse))
    print("\nlecture : le gain indique la reduction de RMSE apportee par la regle sur")
    print("ces lignes seules, rapportee a l'ensemble des departs. Un seuil utile combine")
    print("une precision elevee et un gain positif. Une precision faible veut dire que")
    print("la regle ecrase des lignes normales et coute plus qu'elle ne rapporte.")

    # variante douce : predire l'esperance au lieu de trancher.
    # p est estime hors janvier et juillet, le gain est mesure sur janvier et juillet.
    mois = dep["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    fit_m = ~np.isin(mois, [1, 7])
    ev_m = np.isin(mois, [1, 7])
    n_ev = int(ev_m.sum())
    print("\nvariante douce : prediction = p x d_sched + (1-p) x repli")
    print("p estime hors janv/juil, gain mesure sur janv+juil (%d departs)" % n_ev)
    print("le repli imite le modele : moyenne des labels normaux du meme aeroport")
    fallback = dep.groupby("AIRPORT")["TAXITIME_SEC_mvt"].transform(
        lambda s: s[s < 3600].mean()).to_numpy(dtype=float)
    fallback = np.where(np.isfinite(fallback), fallback, normal_pred)
    for vname, kw in RULE_VARIANTS:
        print("\n  variante %s" % vname)
        print("  %-9s %8s %8s %10s %12s %12s"
              % ("seuil", "p", "cand.ev", "vrais.ev", "gain dur", "gain doux"))
        for thr in args.grid:
            cand = rule_candidates(dep, thr, **kw)
            c_fit = cand & fit_m
            c_ev = cand & ev_m
            if c_fit.sum() < 3 or c_ev.sum() == 0:
                continue
            p = float((c_fit & is_default).sum()) / float(c_fit.sum())
            soft = p * ds[c_ev] + (1 - p) * fallback[c_ev]
            sq_soft = float(np.sum((soft - y[c_ev]) ** 2))
            sq_hard = float(np.sum((ds[c_ev] - y[c_ev]) ** 2))
            sq_base = float(np.sum((fallback[c_ev] - y[c_ev]) ** 2))
            g_soft = np.sqrt(sq_base / n_ev) - np.sqrt(sq_soft / n_ev)
            g_hard = np.sqrt(sq_base / n_ev) - np.sqrt(sq_hard / n_ev)
            print("  %-9d %8.3f %8d %10d %10.1f s %10.1f s"
                  % (thr, p, int(c_ev.sum()), int((c_ev & is_default).sum()),
                     g_hard, g_soft))

    # qu'est-ce qui separe les vrais defauts des faux candidats ?
    thr_sep = args.sep_thr
    cand = rule_candidates(dep, thr_sep)
    vrais = dep[cand & is_default]
    faux = dep[cand & ~is_default]
    print("\n--- separation vrais / faux candidats au seuil %d (%d vs %d) ---"
          % (thr_sep, len(vrais), len(faux)))
    if len(vrais) >= 5 and len(faux) >= 5:
        st_v = match_state(vrais).value_counts(normalize=True)
        st_f = match_state(faux).value_counts(normalize=True)
        print("  etat d'appariement NM :")
        for k in ["non apparie NM", "apparie sans AOBT", "apparie avec AOBT"]:
            print("    %-20s vrais %5.1f %%   faux %5.1f %%"
                  % (k, 100 * st_v.get(k, 0.0), 100 * st_f.get(k, 0.0)))
        # un decalage d'un jour entier laisse d_sched proche d'un multiple de 86400
        for name, sset in [("vrais", vrais), ("faux", faux)]:
            r = (sset["d_sched"] % 86400.0)
            near = ((r < 1800) | (r > 84600)).mean()
            print("  %-6s d_sched proche d'un multiple de 24 h : %.1f %%"
                  % (name, 100 * near))
        for c in ["hour", "dow", "d_sched"]:
            if c in dep.columns:
                print("  %-22s vrais med %10.0f   faux med %10.0f"
                      % (c, vrais[c].median(), faux[c].median()))
        for c in ["AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt",
                  "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
                  "AIRCRAFT_TYPE_mvt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"]:
            if c not in dep.columns:
                continue
            print("  %-22s vide : vrais %5.1f %%   faux %5.1f %%"
                  % (c, 100 * vrais[c].isna().mean(), 100 * faux[c].isna().mean()))
        for c in ["AIRPORT", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"]:
            if c not in dep.columns:
                continue
            a = vrais[c].astype(str).value_counts(normalize=True).head(4)
            b = faux[c].astype(str).value_counts(normalize=True).head(4)
            print("  %s : vrais %s | faux %s"
                  % (c, dict(a.round(2)), dict(b.round(2))))
        sc = pd.to_datetime(dep["SCHED_TIME_UTC_mvt"], errors="coerce", utc=True)
        print("  heure programmee a la minute ronde : vrais %.1f %%   faux %.1f %%"
              % (100 * (sc[cand & is_default].dt.second == 0).mean(),
                 100 * (sc[cand & ~is_default].dt.second == 0).mean()))
    else:
        print("  effectifs insuffisants pour comparer")

    # la meme population existe-t-elle en 2026 ?
    rk = load_ranking(args.data)
    rdep = rk[rk["PHASE_mvt"].eq("DEP")].copy()
    rdep = add_calendar_and_plan(rdep)
    print("\ncandidats dans le fichier de ranking 2026 :")
    for vname, kw in RULE_VARIANTS:
        print("  variante %s" % vname)
        for thr in args.grid:
            cand = rule_candidates(rdep, thr, **kw)
            n_c = int(cand.sum())
            if n_c:
                v = rdep.loc[cand, "d_sched"]
                print("    seuil %6d : %5d lignes, mediane %8.0f s, max %9.0f s, "
                      "dont LIRF %d" % (thr, n_c, v.median(), v.max(),
                                        int(rdep.loc[cand, "AIRPORT"].eq("LIRF").sum())))
            else:
                print("    seuil %6d : aucune ligne" % thr)
    print("\nsi les effectifs 2026 sont du meme ordre que 2025 rapportes au nombre de mois,")
    print("la regle transfere. S'ils sont tres differents, le defaut a ete corrige et il")
    print("ne faut rien appliquer.")


def cmd_defaults(args):
    """Inspecte les labels aberrants : sont-ils reconstructibles ?"""
    print("lecture du training complet")
    tr = load_training(args.data)
    dep = tr[tr["PHASE_mvt"].eq("DEP") & tr["TAXITIME_SEC_mvt"].notna()].copy()
    dep["TAXITIME_SEC_mvt"] = dep["TAXITIME_SEC_mvt"].astype(float)
    dep = add_calendar_and_plan(dep)
    y = dep["TAXITIME_SEC_mvt"]
    ext = dep[y > args.thr].copy()
    print("\n%d labels > %d s sur %d departs (%.4f %%)"
          % (len(ext), args.thr, len(dep), 100 * len(ext) / len(dep)))
    if ext.empty:
        return

    print("\npar aeroport :")
    print(ext.groupby("AIRPORT")["TAXITIME_SEC_mvt"]
          .agg(["size", "min", "median", "max"]).round(0).to_string())
    print("\npar mois :")
    print(ext.groupby(ext["MVT_TIME_UTC_mvt"].dt.month).size().to_string())

    lane = lane_mask(ext)
    print("\nsans off-block NM : %d / %d" % (int(lane.sum()), len(ext)))

    # les valeurs se repetent-elles ? un defaut deterministe laisse des doublons
    vals = ext["TAXITIME_SEC_mvt"].round(0)
    vc = vals.value_counts()
    print("\nvaleurs les plus frequentes (repetition = defaut deterministe) :")
    print(vc.head(10).to_string())
    print("valeurs distinctes : %d pour %d lignes" % (vals.nunique(), len(vals)))

    # le block time implicite tombe-t-il sur une heure ronde ?
    blk = ext["BLOCK_TIME_UTC_mvt"]
    if blk.notna().any():
        print("\nblock time implicite : minute == 0 sur %.1f %%, seconde == 0 sur %.1f %%"
              % (100 * (blk.dt.minute == 0).mean(), 100 * (blk.dt.second == 0).mean()))
        print("heures les plus frequentes :")
        print(blk.dt.hour.value_counts().head(6).to_string())

    # y est-il une fonction simple d'une heure planifiee ?
    print("\ny moins chaque delta planifie (une valeur quasi constante = reconstructible) :")
    for c in ["d_sched", "d_eobt", "d_iobt", "d_lobt", "d_aobt3"]:
        if c not in ext.columns:
            continue
        r = (ext["TAXITIME_SEC_mvt"] - ext[c]).dropna()
        if len(r) < 3:
            continue
        print("  y - %-9s n=%4d  med %10.0f  ecart-type %10.0f  "
              "ecart interquartile %10.0f" % (c, len(r), r.median(), r.std(),
                                              r.quantile(0.75) - r.quantile(0.25)))

    cols = ["AIRPORT", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
            "d_sched", "d_eobt", "d_aobt3", "STAND_mvt", "RUNWAY_mvt",
            "AIRCRAFT_OPERATOR_flt", "ADES_mvt"]
    cols = [c for c in cols if c in ext.columns]
    print("\npremieres lignes :")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ext[cols].head(args.rows).to_string())

    # ces lignes sont-elles separables du reste de leur voie ?
    print("\ncomparaison avec les lignes normales de la meme voie :")
    lane_all = lane_mask(dep)
    normal = dep[lane_all & (y <= args.thr)]
    for c in ["d_sched", "d_eobt", "hour", "dow"]:
        if c not in dep.columns:
            continue
        a = ext[c].dropna()
        b = normal[c].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        print("  %-9s aberrantes med %9.0f   normales med %9.0f"
              % (c, a.median(), b.median()))
    print("\nlecture : si les valeurs se repetent et que y moins un delta planifie est")
    print("quasi constant, le defaut est deterministe et se recalcule. Sinon, ces lignes")
    print("ne sont pas apprenables sur 2025 et il faut se contenter d'une esperance.")


def cmd_run(args):
    try:
        import lightgbm as lgb
    except ImportError:
        sys.exit("pip install lightgbm")

    metar = load_metar(args.data) if args.with_metar else None
    if args.with_metar and metar is None:
        print("attention : pas de METAR en cache, lance d'abord la sous-commande metar")

    print("lecture du training")
    tr = load_training(args.data)
    print("construction des features training (%d lignes contexte)" % len(tr))
    tr = build(tr, metar)
    tr = tr[tr["PHASE_mvt"].eq("DEP") & tr["TAXITIME_SEC_mvt"].notna()].copy()
    tr["TAXITIME_SEC_mvt"] = tr["TAXITIME_SEC_mvt"].astype(float)
    print("departs exploitables : %d" % len(tr))

    print("lecture du ranking")
    rk = load_ranking(args.data)
    print("construction des features ranking (%d lignes contexte)" % len(rk))
    rk = build(rk, metar)
    rk = rk[rk["PHASE_mvt"].eq("DEP")].copy()
    print("departs a predire : %d" % len(rk))

    if args.drop:
        dropped = [c for c in args.drop if c in tr.columns or c in rk.columns]
        tr = tr.drop(columns=[c for c in dropped if c in tr.columns])
        rk = rk.drop(columns=[c for c in dropped if c in rk.columns])
        print("features retirees : %s" % ", ".join(dropped) if dropped else
              "aucune des features demandees n'existe")

    # controle de derive : une feature dont l'echelle change entre 2025 et 2026
    # fait perdre du score sans jamais apparaitre en validation
    num_cols = [c for c in rk.columns
                if c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])
                and c not in ("TAXITIME_SEC_mvt", "t", "t_block", "MVT_ID_mvt",
                              "FLIGHT_ID_mvt", "month", "doy")]
    drift = []
    for c in num_cols:
        a = pd.to_numeric(tr[c], errors="coerce")
        b = pd.to_numeric(rk[c], errors="coerce")
        sa = a.std()
        if not np.isfinite(sa) or sa == 0:
            continue
        smd = abs(a.mean() - b.mean()) / sa
        dna = abs(a.isna().mean() - b.isna().mean())
        drift.append((c, smd, dna))
    drift.sort(key=lambda z: -max(z[1], z[2] * 3))
    print("\nderive 2025 vs 2026 (ecart de moyenne en ecarts-types, ecart de taux de vide) :")
    for c, smd, dna in drift[:12]:
        flag = "  <-- a verifier" if smd > 0.25 or dna > 0.05 else ""
        print("  %-24s smd %6.3f   delta_na %6.3f%s" % (c, smd, dna, flag))

    month = tr["MVT_TIME_UTC_mvt"].dt.month
    fit_mask = ~month.isin([1, 7])
    val_jan = month.eq(1)
    val_jul = month.eq(7)

    prior = tr.loc[fit_mask, "TAXITIME_SEC_mvt"].mean()
    key_sets = [
        (["AIRPORT", "STAND_mvt", "RUNWAY_mvt"], "sr"),
        (["AIRPORT", "STAND_mvt"], "st"),
        (["AIRPORT", "RUNWAY_mvt"], "rw"),
        (["AIRPORT", "RUNWAY_mvt", "hour"], "rwh"),
        (["AIRPORT", "AIRCRAFT_OPERATOR_flt"], "op"),
    ]

    def encode(frame, source, prior_mean):
        for keys, prefix in key_sets:
            stats = fit_target_stats(source, keys, prior_mean)
            frame = apply_target_stats(frame, stats, keys, prefix, prior_mean)
        return frame

    # phase 1 : validation. Encodage calcule hors janvier et juillet.
    src = tr.loc[fit_mask]
    tr_enc = encode(tr.copy(), src, prior)
    X, maps, feats = prepare_matrix(tr_enc)
    y = tr_enc["TAXITIME_SEC_mvt"].to_numpy(dtype=float)

    params = dict(
        objective="regression", metric="rmse", learning_rate=args.lr,
        num_leaves=args.leaves, min_data_in_leaf=args.min_leaf,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=5.0, max_cat_threshold=64, cat_smooth=20, verbose=-1,
        num_threads=args.threads,
    )

    print("\nentrainement de validation")
    dtrain = lgb.Dataset(X[fit_mask.to_numpy()], y[fit_mask.to_numpy()])
    dj = lgb.Dataset(X[val_jan.to_numpy()], y[val_jan.to_numpy()], reference=dtrain)
    djl = lgb.Dataset(X[val_jul.to_numpy()], y[val_jul.to_numpy()], reference=dtrain)
    evals = {}
    model = lgb.train(
        params, dtrain, num_boost_round=args.rounds,
        valid_sets=[dj, djl], valid_names=["janvier", "juillet"],
        callbacks=[lgb.early_stopping(args.patience, verbose=False),
                   lgb.log_evaluation(100), lgb.record_evaluation(evals)],
    )
    best = model.best_iteration or args.rounds
    rj = evals["janvier"]["rmse"][best - 1]
    rl = evals["juillet"]["rmse"][best - 1]
    print("\nRMSE janvier 2025 : %.2f s" % rj)
    print("RMSE juillet 2025 : %.2f s" % rl)
    print("moyenne quadratique des deux mois : %.2f s"
          % np.sqrt((rj ** 2 + rl ** 2) / 2))
    print("meilleure iteration : %d" % best)

    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\n20 features les plus contributrices (gain) :")
    for name, val in imp.head(20).items():
        print("  %-28s %12.0f" % (name, val))

    # decomposition de l'erreur et choix de l'ecretage
    val_mask = (val_jan | val_jul).to_numpy()
    pv = model.predict(X[val_mask], num_iteration=best)
    yv = y[val_mask]
    apv = tr_enc.loc[val_mask, "AIRPORT"].to_numpy()
    print("\nRMSE par aeroport (validation) :")
    print("  %-6s %9s %10s %10s %8s %9s"
          % ("", "n", "RMSE", "RMSE propre", "n>seuil", "part MSE"))
    clean = yv <= EXTREME_THR
    rows = []
    for a in sorted(pd.unique(apv)):
        m = apv == a
        sq = (pv[m] - yv[m]) ** 2
        c = clean[m]
        r_all = float(np.sqrt(np.mean(sq)))
        r_cl = float(np.sqrt(np.mean(sq[c]))) if c.any() else float("nan")
        share = 100 * float(sq[~c].sum()) / float(sq.sum()) if sq.sum() else 0.0
        rows.append((a, int(m.sum()), r_all, r_cl, int((~c).sum()), share))
    for a, n, r_all, r_cl, n_ext, share in sorted(rows, key=lambda z: -z[3]):
        print("  %-6s n=%7d  %9.1f s %9.1f s %7d %7.1f %%"
              % (a, n, r_all, r_cl, n_ext, share))
    sq_all = (pv - yv) ** 2
    print("  ensemble : RMSE %.1f s, RMSE hors labels > %d s : %.1f s"
          % (np.sqrt(sq_all.mean()), EXTREME_THR, np.sqrt(sq_all[clean].mean())))

    # concentration de l'erreur sur la voie propre
    sq_c = np.sort(sq_all[clean])[::-1]
    tot = sq_c.sum()
    print("\nconcentration de l'erreur sur la voie propre :")
    for frac in (0.0001, 0.001, 0.01, 0.1):
        k = max(int(len(sq_c) * frac), 1)
        print("  les %6.2f %% pires lignes (%6d) portent %5.1f %% du MSE propre"
              % (100 * frac, k, 100 * sq_c[:k].sum() / tot))
    err = np.abs(pv[clean] - yv[clean])
    print("  erreur absolue : mediane %5.0f s, p90 %5.0f s, p99 %6.0f s, max %7.0f s"
          % (np.median(err), np.percentile(err, 90), np.percentile(err, 99), err.max()))

    # decoupe par voie : avec ou sans heure d'off-block NM
    lane_v = lane_mask(tr_enc.loc[val_mask]).to_numpy()
    print("\ncontribution au MSE par voie (validation) :")
    total_sq = float(np.sum((pv - yv) ** 2))
    for name, m in [("avec off-block NM", ~lane_v), ("sans off-block NM", lane_v)]:
        if m.sum() == 0:
            continue
        sq = float(np.sum((pv[m] - yv[m]) ** 2))
        print("  %-20s n=%7d (%5.2f %% des lignes)  RMSE %8.1f s  "
              "part du MSE %5.1f %%"
              % (name, int(m.sum()), 100 * m.mean() if m.size else 0,
                 np.sqrt(sq / m.sum()), 100 * sq / total_sq))
    n_ext = int((yv > EXTREME_THR).sum())
    st_v = match_state(tr_enc.loc[val_mask]).to_numpy()
    print("  detail par etat d'appariement NM :")
    for name in ["apparie avec AOBT", "apparie sans AOBT", "non apparie NM"]:
        m = st_v == name
        if m.sum() == 0:
            continue
        sq = float(np.sum((pv[m] - yv[m]) ** 2))
        print("    %-20s n=%7d (%5.2f %%)  RMSE %8.1f s  part du MSE %5.1f %%"
              % (name, int(m.sum()), 100 * m.mean(),
                 np.sqrt(sq / m.sum()), 100 * sq / total_sq))
    print("  labels > %d s : %d lignes, %.1f %% du MSE total"
          % (EXTREME_THR, n_ext,
             100 * float(np.sum((pv[yv > EXTREME_THR] - yv[yv > EXTREME_THR]) ** 2)) / total_sq
             if n_ext else 0.0))

    # estimateur de melange sur la voie sans off-block NM
    lane_fit = lane_mask(tr_enc.loc[fit_mask]).to_numpy()
    mix = fit_mixture(tr_enc.loc[fit_mask], y[fit_mask.to_numpy()], lane_fit)
    if mix:
        gp, ghi, glo, gn = mix["__global__"]
        print("\nmelange estime sur la voie sans off-block NM (%d lignes d'apprentissage) :" % gn)
        print("  proportion globale de labels par defaut : %.3f" % gp)
        print("  moyenne composante par defaut : %.0f s, composante normale : %.0f s"
              % (ghi if np.isfinite(ghi) else float("nan"),
                 glo if np.isfinite(glo) else float("nan")))
        for a in sorted(k for k in mix if k != "__global__"):
            p, m_hi, m_lo, n = mix[a]
            if n >= 20:
                print("    %-6s n=%5d  p=%.3f  defaut %8.0f s  normal %6.0f s"
                      % (a, n, p, m_hi, m_lo))
        pv_mix, changed = apply_mixture(pv, tr_enc.loc[val_mask], lane_v, mix)
        r_mix = float(np.sqrt(np.mean((pv_mix - yv) ** 2)))
        r_raw = float(np.sqrt(np.mean((pv - yv) ** 2)))
        print("  RMSE validation sans melange : %.2f s" % r_raw)
        print("  RMSE validation avec melange : %.2f s  (%d lignes modifiees)"
              % (r_mix, changed))
        use_mixture = r_mix < r_raw
        print("  -> melange %s" % ("retenu" if use_mixture else "ecarte"))
    else:
        use_mixture = False

    # regle des labels reconstruits : evaluee sur les predictions reelles du modele
    print("\nregle y = MVT_TIME - SCHED_TIME appliquee aux predictions (validation) :")
    ds_all = tr_enc["d_sched"].to_numpy(dtype=float)
    y_all = tr_enc["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    # la signature est BLOCK_TIME proche de SCHED_TIME, a n'importe quelle
    # amplitude : exiger y > 20000 s sous-estimait fortement la precision
    is_def_all = np.abs(y_all - ds_all) < 120
    fit_np = fit_mask.to_numpy()
    rmse0 = float(np.sqrt(np.mean((pv - yv) ** 2)))
    print("  RMSE de reference : %.2f s" % rmse0)
    print("  %-18s %7s %6s %7s %10s %10s"
          % ("variante", "seuil", "p", "cand.", "dur", "doux"))
    best_rule = (rmse0, None)
    for vname, kw in RULE_VARIANTS:
        for thr in args.rule_grid:
            c_fit = rule_candidates(tr_enc, thr, **kw) & fit_np
            c_val = rule_candidates(tr_enc.loc[val_mask], thr, **kw)
            if c_fit.sum() < 3 or c_val.sum() == 0:
                continue
            p = float((c_fit & is_def_all).sum()) / float(c_fit.sum())
            dsv = ds_all[val_mask]
            r = {}
            for mode in ("dur", "doux"):
                pp = pv.copy()
                repl = dsv[c_val] if mode == "dur" \
                    else p * dsv[c_val] + (1 - p) * pp[c_val]
                pp[c_val] = repl
                r[mode] = float(np.sqrt(np.mean((pp - yv) ** 2)))
                if r[mode] < best_rule[0]:
                    best_rule = (r[mode], (vname, kw, thr, p, mode))
            print("  %-18s %7d %6.3f %7d %10.2f %10.2f"
                  % (vname, thr, p, int(c_val.sum()), r["dur"], r["doux"]))
    if best_rule[1] is None:
        print("  aucune combinaison n'ameliore le RMSE, regle non appliquee")
        rule_cfg = None
    else:
        vname, kw, thr, p, mode = best_rule[1]
        print("  retenu : %s, seuil %d, mode %s -> %.2f s (gain %.2f s)"
              % (vname, thr, mode, best_rule[0], rmse0 - best_rule[0]))
        rule_cfg = (kw, thr, mode)

    print("\nbalayage de l'ecretage haut (validation) :")
    base_rmse = float(np.sqrt(np.mean((pv - yv) ** 2)))
    print("  sans ecretage           RMSE %.2f s" % base_rmse)
    best_hi, best_rmse = None, base_rmse
    for hi in [1800, 2400, 3000, 3600, 4200, 4800, 6000, 7200]:
        r = float(np.sqrt(np.mean((np.clip(pv, 0, hi) - yv) ** 2)))
        flag = ""
        if r < best_rmse:
            best_rmse, best_hi, flag = r, hi, "  <-- meilleur"
        print("  ecretage a %5d s      RMSE %.2f s%s" % (hi, r, flag))
    if best_hi is None:
        print("  aucun ecretage ne fait mieux, les predictions extremes sont justifiees")
    else:
        print("  gain de %.2f s en ecretant a %d s" % (base_rmse - best_rmse, best_hi))

    clip_hi = args.clip_hi if args.clip_hi > 0 else (best_hi or 0)

    # phase 2 : reentrainement sur tout 2025, encodage recalcule sur tout 2025
    print("\nreentrainement sur l'annee complete")
    prior_full = tr["TAXITIME_SEC_mvt"].mean()
    tr_full = encode(tr.copy(), tr, prior_full)
    Xf, maps_full, feats_full = prepare_matrix(tr_full)
    yf = tr_full["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    rounds_final = int(best * 1.1) + 1
    final = lgb.train(params, lgb.Dataset(Xf, yf), num_boost_round=rounds_final)

    rk_enc = encode(rk.copy(), tr, prior_full)
    for c in feats_full:
        if c not in rk_enc.columns:
            rk_enc[c] = np.nan
    Xr, _, _ = prepare_matrix(rk_enc[feats_full], cat_maps=maps_full)
    pred = final.predict(Xr[feats_full])
    pred = np.clip(pred, 0, None)
    if clip_hi > 0:
        n_clipped = int((pred > clip_hi).sum())
        pred = np.clip(pred, 0, clip_hi)
        print("ecretage applique a %d s (%d predictions concernees)" % (clip_hi, n_clipped))
    if use_mixture:
        mix_full = fit_mixture(tr_full, yf, lane_mask(tr_full).to_numpy())
        lane_r = lane_mask(rk_enc).to_numpy()
        pred, changed = apply_mixture(pred, rk_enc, lane_r, mix_full)
        print("melange applique sur %d lignes sans off-block NM" % changed)
    print("predictions : moyenne %.0f s, mediane %.0f s, min %.0f, max %.0f"
          % (pred.mean(), np.median(pred), pred.min(), pred.max()))

    if rule_cfg is not None:
        kw, thr, mode = rule_cfg
        ds_full = tr_full["d_sched"].to_numpy(dtype=float)
        y_full = tr_full["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
        is_def_full = np.abs(y_full - ds_full) < 120
        c_full = rule_candidates(tr_full, thr, **kw)
        p_full = (float(is_def_full[c_full].sum()) / float(max(c_full.sum(), 1))
                  if c_full.sum() else 0.0)
        c_r = rule_candidates(rk_enc, thr, **kw)
        ds_r = rk_enc["d_sched"].to_numpy(dtype=float)
        ok = c_r & np.isfinite(ds_r)
        if ok.sum():
            pred[ok] = (ds_r[ok] if mode == "dur"
                        else p_full * ds_r[ok] + (1 - p_full) * pred[ok])
            print("regle appliquee (%s, seuil %d, p=%.3f) sur %d lignes"
                  % (mode, thr, p_full, int(ok.sum())))
    sub_path = os.path.join(args.data, "submitting.parquet")
    sub = pd.read_parquet(sub_path)
    mapping = pd.Series(pred, index=rk_enc["MVT_ID_mvt"].to_numpy())
    mapping = mapping[~mapping.index.duplicated()]
    sub["TAXITIME_SEC_mvt"] = sub["MVT_ID_mvt"].map(mapping).astype(float)
    missing = int(sub["TAXITIME_SEC_mvt"].isna().sum())
    if missing:
        print("attention : %d identifiants sans prediction, remplis par la moyenne" % missing)
        sub["TAXITIME_SEC_mvt"] = sub["TAXITIME_SEC_mvt"].fillna(float(np.mean(pred)))
    sub.to_parquet(args.out, index=False)
    print("\nsoumission ecrite : %s (%d lignes)" % (args.out, len(sub)))
    print("nom attendu dans le bucket : motivated-ostrich_v<N>.parquet")


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="PRC Data Challenge 2026 - taxi-out")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="diagnostics fuite et baselines")
    d.add_argument("--data", default="./data")
    d.set_defaults(func=cmd_diagnose)

    m = sub.add_parser("metar", help="telecharge les METAR")
    m.add_argument("--data", default="./data")
    m.add_argument("--force", action="store_true")
    m.set_defaults(func=cmd_metar)

    o = sub.add_parser("outliers", help="analyse un aeroport a forte erreur")
    o.add_argument("--data", default="./data")
    o.add_argument("--airport", default="LIRF")
    o.set_defaults(func=cmd_outliers)

    f = sub.add_parser("defaults", help="inspecte les labels aberrants")
    f.add_argument("--data", default="./data")
    f.add_argument("--thr", type=float, default=20000)
    f.add_argument("--rows", type=int, default=40)
    f.set_defaults(func=cmd_defaults)

    w = sub.add_parser("dayshift", help="famille des labels decales d'un jour")
    w.add_argument("--data", default="./data")
    w.add_argument("--thr", type=float, default=20000)
    w.set_defaults(func=cmd_dayshift)

    u = sub.add_parser("rule", help="precision et rappel de la regle y = decollage - horaire")
    u.add_argument("--data", default="./data")
    u.add_argument("--tol", type=float, default=120)
    u.add_argument("--thr-extreme", type=float, default=20000)
    u.add_argument("--grid", type=float, nargs="+",
                   default=[3600, 7200, 10800, 14400, 20000, 30000, 40000])
    u.add_argument("--sep-thr", type=float, default=10800,
                   help="seuil utilise pour comparer vrais et faux candidats")
    u.set_defaults(func=cmd_rule)

    r = sub.add_parser("run", help="features, entrainement, soumission")
    r.add_argument("--data", default="./data")
    r.add_argument("--out", default="submission.parquet")
    r.add_argument("--with-metar", action="store_true")
    r.add_argument("--rounds", type=int, default=6000)
    r.add_argument("--patience", type=int, default=200)
    r.add_argument("--lr", type=float, default=0.03)
    r.add_argument("--leaves", type=int, default=255)
    r.add_argument("--min-leaf", type=int, default=100)
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--drop", nargs="*", default=[],
                   help="features a retirer avant l'entrainement")
    r.add_argument("--rule-grid", type=float, nargs="+",
                   default=[1800, 2700, 3600, 5400, 7200, 10800, 14400, 20000, 30000])
    r.add_argument("--clip-hi", type=float, default=0,
                   help="ecretage haut en secondes ; 0 = choisi automatiquement sur la validation")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
