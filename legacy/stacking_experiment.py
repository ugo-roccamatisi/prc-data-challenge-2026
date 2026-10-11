"""
PRC Data Challenge 2026 - Taxi-out v17: out-of-fold residual corrector
========================================================================

Principle (suggested by a teammate):

    final prediction = mixture prediction + lambda * predicted correction

The corrector receives the features available at ranking time PLUS the outputs
of the first stage (p_c, mu_normal, mixture prediction, gap s - n) and
directly learns  residual = true duration - mixture prediction.
Unlike the single model (v6, refuted), it starts from the structure of the
mixture and adjusts it; unlike the recalibration (v16), it can also correct
the systematic errors of the normal regressor.

Protocol:
  1. OUT-OF-FOLD predictions on the 10 training months: 5 folds of two
     months, first stage retrained on the other 8 months (one seed, frozen
     rounds), target encodings and unimpeded time recomputed per fold.
  2. Corrector trained on the out-of-fold residuals of folds 1-4, with
     lambda and the number of iterations chosen on fold 5 (November-
     December), lambda = 0 included. Huber loss, residual clipped at +/- CLIP:
     the two CDG rows at 84,000 s and the "+1 day" rows must not
     drive the corrector (lesson from the Rome branch).
  3. A frozen configuration, evaluated only once on January + July
     2025 (first stage trained on the 10 months), per month and per
     airport.
  4. Ranking: first stage on 12 months, corrector applied as is.
     Two submissions: reference (lambda = 0) and corrected.

Safeguard: if the gain on January + July is not positive in both
months, the corrected submission is still written (the leaderboard
decides) but the script reports it.

Usage: python src/stacking_experiment.py   (pipeline.py in src/)
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

TEAM, START_V = "motivated-ostrich", 18


# ---- First stage (compact version, one seed, frozen rounds) --------------

def fit_stage1(train_df: pd.DataFrame) -> dict:
    """Classifier + regressors A/B, target encodings and unimpeded time,
    all fitted on train_df. Frozen rounds (no internal validation)."""
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
    """Applies the first stage to df (with the encodings in models). Returns df
    enriched with the STAGE1_OUT outputs."""
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
    models_tr = fit_stage1(train)
    v = predict_stage1(models_tr, val)
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
    models_full = fit_stage1(dep)
    rank = pd.read_parquet(rank_path)
    rdep = v16.build_all(rank)
    r = predict_stage1(models_full, rdep)
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
        name = f"{TEAM}_v{START_V + j}.parquet"
        s.to_parquet(OUT_DIR / name, index=False)
        print(f"  {name:38s} {tag:12s} median {np.median(pv):5.0f} s  "
              f"mean {pv.mean():5.0f} s  max {pv.max():7.0f} s")
        print(f"    .\\mc.exe cp submissions\\{name} dc26/prc-2026-{TEAM}/")


if __name__ == "__main__":
    main()
