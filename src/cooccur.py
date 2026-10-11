#!/usr/bin/env python3
"""
Co-occurrence des bugs "+1 jour" entre arrivees et departs.

Hypothese : le decalage d'un jour sur le block time est un defaut du systeme
de l'aeroport, actif sur certaines nuits. Les arrivees du ranking 2026 sont
labellisees, donc si les bugs d'arrivee et de depart tombent sur les memes
nuits en 2025, les arrivees 2026 designent les nuits a risque.

Le script ne fait AUCUNE soumission et n'utilise aucun retour du leaderboard.

  1. 2025 : definit les bugs de depart (label aberrant non couvert par la
     regle y = MVT - SCHED) et les bugs d'arrivee (taxi-in aberrant).
  2. 2025 : mesure P(bug depart | bug arrivee proche) contre la base,
     globalement, par aeroport, sur les candidats "decollage le lendemain",
     avec un placebo (arrivees decalees de +/- 7 jours).
  3. 2026 : repere les arrivees buggees du ranking et ecrit la liste des
     departs candidats dans results/cooccur_candidates_2026.csv.
  4. Esperance et dispersion du gain RMSE si on applique la correction.

Usage :
  python src/cooccur.py --data data
  python src/cooccur.py --data data --thr 20000 --windows 1 3 6 12 24
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prc_taxiout import load_training, load_ranking, add_calendar_and_plan, to_epoch  # noqa: E402

DAY = 86400.0
TZ = {
    "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}


# ----------------------------------------------------------------- definitions

def local_dates(df, col, shift_h=0):
    """Date locale (jour operationnel si shift_h > 0) par aeroport."""
    out = pd.Series(pd.NaT, index=df.index, dtype="object")
    for ap, tz in TZ.items():
        m = df["AIRPORT"].eq(ap)
        if m.any():
            loc = df.loc[m, col].dt.tz_convert(tz) - pd.Timedelta(hours=shift_h)
            out[m] = loc.dt.date
    return out


def flag_arrival_bugs(arr, thr):
    y = arr["TAXITIME_SEC_mvt"].astype(float)
    arr = arr.assign(
        arr_day=((y - DAY).abs() < 7200) | ((y + DAY).abs() < 7200),
        arr_ext=(y > thr) | (y < -thr),
    )
    return arr


def flag_departure_bugs(dep, thr):
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    by_rule = (y - dep["d_sched"]).abs() < 120
    dep = dep.assign(dep_bug=(~by_rule) & ((y > thr) | (y < -thr)))
    return dep


def add_candidate_flags(dep):
    """Candidats structurels : decollage a une date locale posterieure a l'horaire."""
    d_mvt = local_dates(dep, "MVT_TIME_UTC_mvt")
    d_sch = local_dates(dep, "SCHED_TIME_UTC_mvt")
    nextday = pd.Series(False, index=dep.index)
    ok = d_mvt.notna() & d_sch.notna()
    nextday[ok] = [a > b for a, b in zip(d_mvt[ok], d_sch[ok])]
    rule_cand = (dep["FLIGHT_ID_mvt"].isna() & dep["AIRCRAFT_TYPE_mvt"].isna()
                 & (dep["d_sched"] > 10800))
    return dep.assign(next_day=nextday, rule_cand=rule_cand,
                      unmatched=dep["FLIGHT_ID_mvt"].isna())


def nearest_distance(dep, arr_bug, shift_s=0.0):
    """Distance (s) de chaque depart a l'arrivee buggee la plus proche, meme aeroport."""
    dist = np.full(len(dep), np.inf)
    t_dep = dep["t"].to_numpy(dtype=float)
    ap_dep = dep["AIRPORT"].to_numpy()
    for ap in np.unique(ap_dep):
        tb = np.sort(arr_bug.loc[arr_bug["AIRPORT"].eq(ap), "t"].to_numpy(dtype=float) + shift_s)
        if len(tb) == 0:
            continue
        idx = np.where(ap_dep == ap)[0]
        t = t_dep[idx]
        j = np.searchsorted(tb, t)
        left = np.abs(t - tb[np.clip(j - 1, 0, len(tb) - 1)])
        right = np.abs(tb[np.clip(j, 0, len(tb) - 1)] - t)
        dist[idx] = np.minimum(left, right)
    return dist


def same_opday(dep, arr_bug):
    key_d = dep["AIRPORT"].astype(str) + "|" + local_dates(dep, "MVT_TIME_UTC_mvt", 4).astype(str)
    key_a = set(arr_bug["AIRPORT"].astype(str) + "|"
                + local_dates(arr_bug, "MVT_TIME_UTC_mvt", 4).astype(str))
    return key_d.isin(key_a).to_numpy()


# ----------------------------------------------------------------- statistiques

def fisher_p(a, b, c, d):
    try:
        from scipy.stats import fisher_exact
        return fisher_exact([[a, b], [c, d]], alternative="greater")[1]
    except ImportError:
        return np.nan


def lift_table(dep, flags, label):
    y = dep["dep_bug"].to_numpy()
    rows = []
    n_bug = int(y.sum())
    for name, f in flags.items():
        a = int((f & y).sum()); b = int((f & ~y).sum())
        c = int((~f & y).sum()); d = int((~f & ~y).sum())
        r_in = a / max(a + b, 1); r_out = c / max(c + d, 1)
        rows.append({"signal": name, "flagged": a + b, "bugs_in": a,
                     "recall": a / max(n_bug, 1), "rate_in": r_in, "rate_out": r_out,
                     "lift": r_in / r_out if r_out > 0 else np.inf,
                     "fisher_p": fisher_p(a, b, c, d)})
    print("\n%s (bugs de depart : %d sur %d departs)" % (label, n_bug, len(dep)))
    if rows:
        with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
            print(pd.DataFrame(rows).to_string(index=False))


def build_flags(dep, arr_bug, windows, shift_s=0.0):
    dist = nearest_distance(dep, arr_bug, shift_s)
    flags = {"<=%dh" % w: dist <= w * 3600 for w in windows}
    if shift_s == 0.0:
        flags["meme jour op."] = same_opday(dep, arr_bug)
    return flags, dist


# ----------------------------------------------------------------- esperance

def expected_gain(k, q, n_rank, score, S=DAY):
    """Gain RMSE si on ajoute q*S a k departs ayant chacun une proba q de bug."""
    mse = score ** 2
    mean_red = k * q * q * S * S / n_rank
    none_loss = k * q * q * S * S / n_rank          # cout si aucun bug parmi les k
    one_gain = (S * S * (2 * q - q * q) - (k - 1) * q * q * S * S) / n_rank
    p_none = (1 - q) ** k
    print("  k=%d candidats, q=%.3f : RMSE attendu %.2f s (actuel %.2f)"
          % (k, q, np.sqrt(max(mse - mean_red, 0)), score))
    print("    si aucun bug parmi eux (proba %.2f) : %.2f s" % (p_none, np.sqrt(mse + none_loss)))
    print("    si exactement un bug : %.2f s" % np.sqrt(max(mse - one_gain, 0)))


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--thr", type=float, default=20000)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 3, 6, 12, 24])
    ap.add_argument("--score", type=float, default=319.21, help="score public actuel")
    ap.add_argument("--out", default="results/cooccur_candidates_2026.csv")
    args = ap.parse_args()

    print("lecture du training 2025")
    tr = load_training(args.data)
    arr = tr[tr["PHASE_mvt"].eq("ARR") & tr["TAXITIME_SEC_mvt"].notna()].copy()
    dep = tr[tr["PHASE_mvt"].eq("DEP") & tr["TAXITIME_SEC_mvt"].notna()].copy()
    del tr
    arr = flag_arrival_bugs(arr, args.thr)
    dep = add_candidate_flags(flag_departure_bugs(add_calendar_and_plan(dep), args.thr))

    print("\n=== 2025 : inventaire ===")
    inv = pd.DataFrame({
        "arr_day": arr.groupby("AIRPORT")["arr_day"].sum(),
        "arr_ext": arr.groupby("AIRPORT")["arr_ext"].sum(),
        "dep_bug": dep.groupby("AIRPORT")["dep_bug"].sum(),
        "n_dep": dep.groupby("AIRPORT").size(),
    }).fillna(0).astype(int)
    print(inv.to_string())

    for sig in ("arr_day", "arr_ext"):
        arr_bug = arr[arr[sig]]
        if arr_bug.empty:
            print("\naucune arrivee %s, signal ignore" % sig)
            continue
        print("\n=== 2025 : signal %s (%d arrivees) ===" % (sig, len(arr_bug)))
        flags, dist = build_flags(dep, arr_bug, args.windows)
        lift_table(dep, flags, "tous departs")
        m = dep["next_day"].to_numpy()
        lift_table(dep[m], {k: v[m] for k, v in flags.items()}, "candidats decollage le lendemain")
        m2 = m & dep["unmatched"].to_numpy()
        lift_table(dep[m2], {k: v[m2] for k, v in flags.items()},
                   "candidats lendemain + non apparies NM")
        for a in inv.index[inv["dep_bug"] > 0]:
            ma = dep["AIRPORT"].eq(a).to_numpy()
            lift_table(dep[ma], {k: v[ma] for k, v in flags.items()}, "aeroport %s seul" % a)

        print("\n  placebo (arrivees decalees de +/- 7 jours) :")
        for s in (-7, 7):
            pf, _ = build_flags(dep, arr_bug, args.windows, shift_s=s * DAY)
            lift_table(dep, pf, "placebo %+d j" % s)

        print("\n  chaque bug de depart et l'arrivee buggee la plus proche :")
        b = dep[dep["dep_bug"]].assign(dist_h=dist[dep["dep_bug"].to_numpy()] / 3600)
        cols = ["AIRPORT", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
                "d_sched", "next_day", "unmatched", "dist_h"]
        with pd.option_context("display.width", 220):
            print(b[cols].sort_values("MVT_TIME_UTC_mvt").to_string(index=False))

    print("\n=== 2026 : arrivees buggees du ranking ===")
    rk = load_ranking(args.data)
    rarr = flag_arrival_bugs(rk[rk["PHASE_mvt"].eq("ARR") & rk["TAXITIME_SEC_mvt"].notna()].copy(),
                             args.thr)
    rdep = add_candidate_flags(add_calendar_and_plan(rk[rk["PHASE_mvt"].eq("DEP")].copy()))
    n_rank = len(rdep)
    for sig in ("arr_day", "arr_ext"):
        rb = rarr[rarr[sig]]
        print("\n%s : %d arrivees" % (sig, len(rb)))
        if rb.empty:
            continue
        print(rb.groupby(["AIRPORT", local_dates(rb, "MVT_TIME_UTC_mvt", 4)]).size()
              .rename("n").to_string())
        dist = nearest_distance(rdep, rb)
        rdep["dist_h_" + sig] = dist / 3600
    rdep["same_opday_day"] = same_opday(rdep, rarr[rarr["arr_day"]]) if rarr["arr_day"].any() else False

    keep = [c for c in rdep.columns if c.startswith("dist_h_")]
    if keep:
        near = np.column_stack([rdep[c].to_numpy() for c in keep]).min(axis=1)
        cand = rdep[(near <= max(args.windows)) & rdep["next_day"] & ~rdep["rule_cand"]].copy()
        cand["dist_h_min"] = near[cand.index.map(rdep.index.get_loc)]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cols = ["MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "d_sched",
                "unmatched", "same_opday_day", "dist_h_min"] + keep
        cand[cols].sort_values("dist_h_min").to_csv(args.out, index=False)
        print("\n%d departs candidats 2026 (lendemain, hors regle, <= %d h d'une arrivee buggee)"
              % (len(cand), max(args.windows)))
        print("ecrits dans %s" % args.out)
        print(cand["AIRPORT"].value_counts().to_string())

    print("\n=== Ordre de grandeur (N ranking = %d departs) ===" % n_rank)
    print("q = taux de bug mesure en 2025 dans le groupe signale ; a relire dans les tables.")
    for k, q in [(2, 0.5), (5, 0.3), (10, 0.2), (20, 0.1), (50, 0.07)]:
        expected_gain(k, q, n_rank, args.score)


if __name__ == "__main__":
    main()
