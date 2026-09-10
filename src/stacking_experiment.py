"""
PRC Data Challenge 2026 - Taxi-out v17 : correcteur de residu hors-fold
========================================================================

Principe (proposition d'un coequipier) :

    prediction finale = prediction du melange + lambda * correction predite

Le correcteur recoit les features disponibles au ranking PLUS les sorties
du premier etage (p_c, mu_normal, prediction du melange, ecart s - n) et
apprend directement  residu = duree reelle - prediction du melange.
Contrairement au modele unique (v6, refute), il part de la structure du
melange et l'ajuste ; contrairement au recalibrage (v16), il peut corriger
aussi les erreurs systematiques du regresseur normal.

Protocole :
  1. Predictions HORS-FOLD sur les 10 mois d'entrainement : 5 plis de deux
     mois, premier etage reentraine sur les 8 autres mois (une seed, rounds
     figes), target encodings et temps non entrave recalcules par pli.
  2. Correcteur entraine sur les residus hors-fold des plis 1-4, avec
     lambda et le nombre d'iterations choisis sur le pli 5 (novembre-
     decembre), lambda = 0 inclus. Loss Huber, residu ecrete a +/- CLIP :
     les deux lignes de CDG a 84 000 s et les "+1 jour" ne doivent pas
     dicter le correcteur (lecon de la branche Rome).
  3. Une configuration figee, evaluee une seule fois sur janvier + juillet
     2025 (premier etage entraine sur les 10 mois), par mois et par
     aeroport.
  4. Ranking : premier etage sur 12 mois, correcteur applique tel quel.
     Deux soumissions : reference (lambda = 0) et corrigee.

Garde-fou : si le gain sur janvier + juillet n'est pas positif dans les
deux mois, la soumission corrigee est quand meme ecrite (le leaderboard
tranche) mais le script le signale.

Usage : python src/stacking_experiment.py   (pipeline.py dans src/)
"""

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v16", SCRIPT_DIR / "pipeline.py")
v16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v16)

DATA_DIR, OUT_DIR = v16.DATA_DIR, v16.OUT_DIR
TARGET, CORRUPT = v16.TARGET, v16.CORRUPT
TE_FEATURES, UNIMP_FEATURES = v16.TE_FEATURES, v16.UNIMP_FEATURES
TAXI_MIN_S, TAXI_MAX_S = v16.TAXI_MIN_S, v16.TAXI_MAX_S
rmse = v16.rmse

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

TEAM, START_V = "motivated-ostrich", 18


# ---- Premier etage (version compacte, une seed, rounds figes) -------------

def fit_stage1(train_df: pd.DataFrame) -> dict:
    """Classifieur + regresseurs A/B, target encodings et temps non entrave,
    tout ajuste sur train_df. Rounds figes (pas de validation interne)."""
    tr = v16.add_te_out_of_fold(train_df, CORRUPT)
    maps, prior = v16.fit_te_maps(tr, tr[CORRUPT].values)
    tr = v16.add_unimpeded_out_of_fold(tr)
    umaps = v16.fit_unimpeded_maps(tr)
    extra = TE_FEATURES + UNIMP_FEATURES

    catsC = v16.get_categories(tr)
    clf = v16.train_lgb(dict(v16.LGB_CLF, seed=SEED), v16.prepare_matrix(tr, extra=extra),
                        tr[CORRUPT].values, num_rounds=ROUNDS_CLF)

    clean = tr[tr[CORRUPT] == 0]
    ha = clean["delta_mvt_aobt3_s"].notna()
    tA = clean[ha].copy()
    tA["residual"] = tA[TARGET] - tA["delta_mvt_aobt3_s"]
    cA = v16.get_categories(tA)
    mA = v16.train_lgb(dict(v16.LGB_REG, seed=SEED), v16.prepare_matrix(tA, extra=extra),
                       tA["residual"].values, num_rounds=ROUNDS_A)
    cB = v16.get_categories(clean)
    med = float(clean[TARGET].median())
    yB = (clean[TARGET] - clean["unimpeded_p50"].fillna(med)).values
    mB = v16.train_lgb(dict(v16.LGB_REG, seed=SEED), v16.prepare_matrix(clean, extra=extra),
                       yB, num_rounds=ROUNDS_B)
    return dict(maps=maps, prior=prior, umaps=umaps, catsC=catsC, clf=clf,
                cA=cA, mA=mA, cB=cB, mB=mB, med=med)


def predict_stage1(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Applique le premier etage a df (avec encodages de models). Retourne df
    enrichi des sorties STAGE1_OUT."""
    d = v16.apply_te(df, models["maps"], models["prior"])
    d = v16.apply_unimpeded(d, models["umaps"])
    extra = TE_FEATURES + UNIMP_FEATURES

    p_c = models["clf"].predict(v16.prepare_matrix(d, models["catsC"], extra=extra))
    mu = np.full(len(d), np.nan)
    ha = d["delta_mvt_aobt3_s"].notna().values
    if ha.any():
        mu[ha] = (d.loc[ha, "delta_mvt_aobt3_s"].values
                  + models["mA"].predict(v16.prepare_matrix(d[ha], models["cA"], extra=extra)))
    if (~ha).any():
        base = d.loc[~ha, "unimpeded_p50"].fillna(models["med"]).values
        mu[~ha] = base + models["mB"].predict(
            v16.prepare_matrix(d[~ha], models["cB"], extra=extra))
    mu_raw = mu.copy()
    mu = np.clip(mu, TAXI_MIN_S, TAXI_MAX_S)

    sched = d["delta_mvt_sched_s"].values
    ok = np.isfinite(sched)
    pg = np.where(ok, p_c ** GAMMA, 0.0)
    mix = np.maximum(pg * np.where(ok, sched, 0.0) + (1 - pg) * mu, TAXI_MIN_S)
    gap = np.where(ok, sched - mu, 0.0)

    d["p_c"], d["mu_normal"], d["mu_raw"], d["mix"] = p_c, mu, mu_raw, mix
    d["gap"] = gap
    d["log_gap"] = np.sign(gap) * np.log1p(np.abs(gap))
    return d


def corrector_matrix(d: pd.DataFrame):
    return v16.prepare_matrix(d, None, extra=TE_FEATURES + UNIMP_FEATURES + STAGE1_OUT)


def align(X: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = X.copy()
    for c, cs in cats.items():
        if c in X.columns:
            s = X[c].astype("object")
            X[c] = pd.Categorical(s.where(s.isin(set(cs))), categories=cs)
    return X


# ---- Pipeline ----------------------------------------------------------------

def main():
    dep = v16.load_or_build(v16.CACHE_PATH)
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
    models_tr = fit_stage1(train)
    v = predict_stage1(models_tr, val)
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
    models_full = fit_stage1(dep)
    rank = pd.read_parquet(rank_path)
    rdep = v16.build_all(rank)
    r = predict_stage1(models_full, rdep)
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
        name = f"{TEAM}_v{START_V + j}.parquet"
        s.to_parquet(OUT_DIR / name, index=False)
        print(f"  {name:38s} {tag:12s} mediane {np.median(pv):5.0f} s  "
              f"moyenne {pv.mean():5.0f} s  max {pv.max():7.0f} s")
        print(f"    .\\mc.exe cp submissions\\{name} dc26/prc-2026-{TEAM}/")


if __name__ == "__main__":
    main()
