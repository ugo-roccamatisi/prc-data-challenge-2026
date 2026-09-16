#!/usr/bin/env python3
"""
Applique la regle des labels reconstruits a une soumission existante.

Certains mouvements n'ont jamais ete apparies aux donnees NM. Pour ceux-la,
l'aeroport a rempli BLOCK_TIME_UTC_mvt avec l'heure programmee, si bien que
la cible vaut exactement MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt. Les deux
colonnes sont disponibles dans le fichier de ranking, donc ces lignes se
recalculent au lieu de se predire.

Le script ne touche a rien d'autre : il lit une soumission produite par
n'importe quel pipeline et en reecrit une copie corrigee.

Exemples :
  python apply_rule.py --ranking ./data/ranking.parquet --sub submission.parquet \\
      --out submission_dur.parquet --mode dur
  python apply_rule.py --ranking ./data/ranking.parquet --sub submission.parquet \\
      --out submission_doux.parquet --mode doux --p 0.731
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
    ap = argparse.ArgumentParser(description="regle des labels reconstruits")
    ap.add_argument("--ranking", default="./data/ranking.parquet")
    ap.add_argument("--sub", required=True, help="soumission a corriger")
    ap.add_argument("--out", help="fichier de sortie (omis avec --dry-run)")
    ap.add_argument("--mode", choices=["dur", "doux"], default="dur")
    ap.add_argument("--p", type=float, default=0.731,
                    help="proportion de vrais defauts, utilisee en mode doux")
    ap.add_argument("--thr", type=float, default=10800,
                    help="seuil sur MVT_TIME - SCHED_TIME, en secondes")
    ap.add_argument("--airports", nargs="*", default=["LIRF"],
                    help="aeroports concernes ; liste vide pour ne pas filtrer")
    ap.add_argument("--no-type-filter", action="store_true",
                    help="ne pas exiger AIRCRAFT_TYPE_mvt vide")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        sys.exit("--out est requis sauf en --dry-run")

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

    print("candidats : %d sur %d departs" % (len(cand), len(dep)))
    if len(cand):
        print("  d_sched : min %.0f s, mediane %.0f s, max %.0f s"
              % (cand["d_sched"].min(), cand["d_sched"].median(),
                 cand["d_sched"].max()))
        print("  par aeroport : %s" % dict(cand["ADEP_mvt"].value_counts()))

    sub = pd.read_parquet(args.sub)
    if "MVT_ID_mvt" not in sub.columns or "TAXITIME_SEC_mvt" not in sub.columns:
        sys.exit("la soumission doit contenir MVT_ID_mvt et TAXITIME_SEC_mvt")
    before = sub["TAXITIME_SEC_mvt"].astype(float).copy()

    new_vals = pd.Series(cand["d_sched"].to_numpy(dtype=float),
                         index=cand["MVT_ID_mvt"].to_numpy())
    new_vals = new_vals[~new_vals.index.duplicated()]
    target = sub["MVT_ID_mvt"].map(new_vals)
    hit = target.notna()
    print("  dont presents dans la soumission : %d" % int(hit.sum()))
    if int(hit.sum()) == 0:
        print("aucune ligne a corriger, rien a faire")
        return

    if args.mode == "dur":
        updated = target[hit].astype(float)
    else:
        updated = args.p * target[hit].astype(float) + (1 - args.p) * before[hit]

    delta = (updated - before[hit]).abs()
    print("\nmode %s%s" % (args.mode, "" if args.mode == "dur" else " (p=%.3f)" % args.p))
    print("  avant  : mediane %9.0f s, max %10.0f s"
          % (before[hit].median(), before[hit].max()))
    print("  apres  : mediane %9.0f s, max %10.0f s"
          % (updated.median(), updated.max()))
    print("  ecart absolu : mediane %9.0f s, max %10.0f s"
          % (delta.median(), delta.max()))

    if args.dry_run:
        print("\n--dry-run : aucun fichier ecrit")
        return

    out = sub.copy()
    out.loc[hit, "TAXITIME_SEC_mvt"] = updated.to_numpy()
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].astype(float).clip(lower=0)
    if out["TAXITIME_SEC_mvt"].isna().any():
        sys.exit("des valeurs manquantes subsistent, sortie annulee")
    if len(out) != len(sub):
        sys.exit("le nombre de lignes a change, sortie annulee")
    out.to_parquet(args.out, index=False)
    print("\necrit : %s (%d lignes, %d modifiees)" % (args.out, len(out), int(hit.sum())))


if __name__ == "__main__":
    main()
