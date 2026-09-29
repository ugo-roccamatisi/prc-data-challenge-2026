#!/usr/bin/env python3
"""PRC: reproducible experiments, internal selection then a frozen control.
See README.md. No submission is ever uploaded.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prc_v16', ROOT / 'pipeline.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
TARGET, ID, CORRUPT = v.TARGET, 'MVT_ID_mvt', v.CORRUPT
CONTROL = [1, 7]
SELECT = [11, 12]
BASE = 'mix075'
COMPONENTS = ['normal_a', 'normal_b', 'classifier', 'direct_l2', 'direct_huber']
CLOCK_COLS = ['SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt']


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def code_hash():
    return {p.name: sha(p) for p in [Path(__file__), ROOT / 'pipeline.py']}


def score(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0 or y.shape != p.shape or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError('RMSE: empty, non-finite or mismatched arrays.')
    return float(np.sqrt(np.mean((y - p) ** 2)))


def check_ids(df, name):
    if ID not in df or df[ID].isna().any() or df[ID].duplicated().any():
        raise ValueError(f'{name}: identifiers absent, missing or duplicated.')


def feature_names(df):
    return [c for c in v.CAT_FEATURES + v.NUM_FEATURES + v.TE_FEATURES + v.UNIMP_FEATURES
            + [c for c in df if c.startswith('clock_')] if c in df]


def add_clocks(d):
    """Clock times, date changes, disagreements between observable sources."""
    d = d.copy()
    mvt = d['MVT_TIME_UTC_mvt']
    for col in CLOCK_COLS:
        if col not in d:
            d[col] = pd.NaT
        t = pd.to_datetime(d[col], utc=True, errors='coerce')
        tag = col.split('_')[0].lower()
        sec = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
        d[f'clock_{tag}_missing'] = t.isna().astype(float)
        d[f'clock_{tag}_sin'] = np.sin(2 * np.pi * sec / 86400)
        d[f'clock_{tag}_cos'] = np.cos(2 * np.pi * sec / 86400)
        d[f'clock_{tag}_seconds'] = t.dt.second.astype(float)
        d[f'clock_{tag}_day_gap'] = (mvt.dt.floor('D') - t.dt.floor('D')).dt.total_seconds() / 86400
    for a, b in [('AOBT_3_flt', 'LOBT_flt'), ('AOBT_3_flt', 'IOBT_flt'),
                 ('AOBT_3_flt', 'EOBT_1_flt'), ('LOBT_flt', 'IOBT_flt')]:
        delta = (d[a] - d[b]).dt.total_seconds()
        key = f'clock_{a.split("_")[0].lower()}_{b.split("_")[0].lower()}'
        d[key] = delta
        d[key + '_wrapped'] = (delta + 43200) % 86400 - 43200
    return d


def build(raw):
    """DEP labels are removed BEFORE any contextual feature is built."""
    raw = raw.copy().reset_index(drop=True)
    required = [ID, 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'ADEP_mvt', 'ADES_mvt',
                'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'SCHED_TIME_UTC_mvt']
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    for c in v.TIME_COLS:
        if c not in raw:
            raw[c] = pd.NaT
    if TARGET not in raw:
        raw[TARGET] = np.nan
    raw = v.parse_times(raw)
    is_dep = raw['PHASE_mvt'].eq('DEP')
    labels = raw.loc[is_dep, [ID, TARGET, 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                             'MVT_TIME_UTC_mvt']].copy()
    check_ids(labels, 'Departures')
    gap = (labels['BLOCK_TIME_UTC_mvt'] - labels['SCHED_TIME_UTC_mvt']).dt.total_seconds()
    labels[CORRUPT] = np.where(gap.notna(), (gap.abs() <= v.CORRUPT_TOL_S).astype(float), np.nan)
    labels['observed_block'] = labels['BLOCK_TIME_UTC_mvt']
    reconstructed = (labels['MVT_TIME_UTC_mvt'] - labels['BLOCK_TIME_UTC_mvt']).dt.total_seconds()
    labels['target_identity_error'] = labels[TARGET] - reconstructed
    raw.loc[is_dep, TARGET] = np.nan
    raw.loc[is_dep, 'BLOCK_TIME_UTC_mvt'] = pd.NaT
    d = add_clocks(v.build_all(raw))
    d = d.drop(columns=[TARGET, CORRUPT], errors='ignore').merge(
        labels[[ID, TARGET, CORRUPT, 'observed_block', 'target_identity_error']],
        on=ID, how='left', validate='one_to_one')
    # The diagnostic columns above never appear in feature_names.
    return d


def input_signature(files):
    return [{ 'path': str(p.resolve()), 'bytes': p.stat().st_size,
              'mtime_ns': p.stat().st_mtime_ns} for p in files]


def prepare(args):
    files = sorted(args.data.glob('training_*.parquet'))
    if not files:
        raise FileNotFoundError(f'No training_*.parquet in {args.data}')
    args.cache.mkdir(parents=True, exist_ok=True)
    meta = {'inputs': input_signature(files), 'code': code_hash()}
    path = args.cache / 'features.parquet'
    mp = args.cache / 'manifest.json'
    if path.exists() and mp.exists() and not args.rebuild:
        old = json.loads(mp.read_text(encoding='utf-8'))
        if old['source'] == meta and old['feature_sha256'] == sha(path):
            log('Cache checked, already ready.'); return
        raise RuntimeError('Cache differs: rerun prepare with --rebuild.')
    log('Reading and rebuilding the features, without using the v15 cache.')
    raw = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    d = build(raw)
    finite = np.isfinite(pd.to_numeric(d[TARGET], errors='coerce'))
    excluded = int((~finite).sum())
    d = d.loc[finite].copy()
    if d.empty or d['month'].isna().any():
        raise ValueError('No usable target or missing movement date.')
    years = sorted(d['MVT_TIME_UTC_mvt'].dt.year.unique().tolist())
    if years != [2025]:
        raise ValueError(f'This protocol expects 2025 only, got {years}. Adapt the splits explicitly.')
    # No finite target is clipped or dropped, even if <30 s or negative.
    d.to_parquet(path, index=False)
    diagnostics = {'rows': len(d), 'nonfinite_targets_excluded': excluded,
                   'negative_targets_kept': int((d[TARGET] < 0).sum()),
                   'targets_below_30_kept': int((d[TARGET] < 30).sum()),
                   'targets_above_7200': int((d[TARGET] > 7200).sum()),
                   'unknown_corrupt_labels': int(d[CORRUPT].isna().sum()),
                   'identity_error_over_10s': int((d.target_identity_error.abs() > 10).sum()),
                   'identity_error_available': int(d.target_identity_error.notna().sum()),
                   'by_month': {str(k): int(n) for k, n in d.groupby('month').size().items()}}
    dump(args.cache / 'data_audit.json', diagnostics)
    dump(mp, {'source': meta, 'feature_sha256': sha(path)})
    log(json.dumps(diagnostics, ensure_ascii=False))


def fit_maps(d):
    known = d[CORRUPT].notna()
    if not known.any():
        raise ValueError('No known corruption label in the training set.')
    te, prior = v.fit_te_maps(d.loc[known], d.loc[known, CORRUPT].to_numpy())
    um = v.fit_unimpeded_maps(d)
    clean = d.loc[d[CORRUPT].eq(0), TARGET]
    med = float(clean.median()) if len(clean) else float(d[TARGET].median())
    return te, prior, um, med


def apply_maps(d, maps):
    te, prior, um, _ = maps
    return v.apply_unimpeded(v.apply_te(d, te, prior), um)


def encoded_training(d):
    # The labels of each month are all excluded from its own encodings.
    # The groups are the same for every training run; the validation set is never read.
    groups = {1: 0, 7: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 8: 3, 9: 4, 10: 4, 11: 5, 12: 5}
    fold = d['month'].map(groups)
    pieces = []
    for f in sorted(fold.unique()):
        inside = fold.eq(f)
        if (~inside).sum() == 0:
            raise ValueError('Encoding: several month groups are required.')
        pieces.append(apply_maps(d.loc[inside], fit_maps(d.loc[~inside])))
    return pd.concat(pieces).loc[d.index]


def matrices(tr, va):
    cols = feature_names(tr)
    if set(cols) - set(va.columns):
        raise ValueError('Missing features at prediction time.')
    a, b = tr[cols].copy(), va[cols].copy()
    for c in cols:
        if c in v.CAT_FEATURES:
            values = tr[c].astype('string')
            categories = sorted(values.dropna().unique().tolist())
            a[c] = pd.Categorical(values, categories=categories)
            b[c] = pd.Categorical(va[c].astype('string'), categories=categories)
        else:
            a[c] = pd.to_numeric(a[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            b[c] = pd.to_numeric(b[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return a, b


def baseline(d, median):
    # The AOBT reference is only used as an offset when it is physically plausible.
    a = d['delta_mvt_aobt3_s'].to_numpy(float)
    u = d['unimpeded_p50'].fillna(median).to_numpy(float)
    return np.where(np.isfinite(a) & (a >= 0) & (a <= 7200), a, u)


def mixture(p, normal, sched, gamma):
    valid = np.isfinite(sched)
    weight = np.where(valid, np.clip(p, 0, 1) ** gamma, 0)
    return np.maximum(normal + weight * (np.where(valid, sched, 0) - normal), 30)


def model_params(kind, seed, args):
    p = dict(objective='regression', metric='None', learning_rate=0.04,
             num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
             bagging_fraction=0.9, bagging_freq=1, lambda_l2=5.0,
             verbosity=-1, num_threads=args.threads, seed=seed,
             bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed,
             deterministic=True, force_col_wise=True)
    if kind.startswith('normal'):
        p.update(num_leaves=127, lambda_l2=2.0)
    if kind == 'classifier':
        p.update(objective='binary', learning_rate=0.02, num_leaves=127)
    if kind == 'direct_huber':
        p.update(objective='huber', alpha=600.0)
    return p


def learn(kind, X, y, V, ytrue, transform, seed, args, folder, rounds=None):
    path = folder / f'{kind}_{seed}.txt'
    info = path.with_suffix('.json')
    if path.exists() and info.exists():
        m = lgb.Booster(model_file=str(path))
        n = json.loads(info.read_text())['rounds']
        log(f'Resuming {kind} seed {seed}, {n} trees')
    else:
        if len(X) == 0 or not np.isfinite(y).all():
            raise ValueError(f'{kind}: empty training set or non-finite labels.')
        p = model_params(kind, seed, args)
        ds = lgb.Dataset(X, label=y)
        if rounds is None:
            if len(V) == 0:
                raise ValueError(f'{kind}: empty validation set.')
            # Dataset labels consistent with its objective; feval uses the true duration.
            validation_label = np.zeros(len(V))
            dv = lgb.Dataset(V, label=validation_label, reference=ds)
            def feval(pred, data):
                return 'final_rmse_seconds', score(ytrue, transform(pred)), False
            m = lgb.train(p, ds, num_boost_round=args.max_rounds,
                          valid_sets=[dv], feval=feval,
                          callbacks=[lgb.early_stopping(args.patience, verbose=False),
                                     lgb.log_evaluation(300)])
            n = m.best_iteration or m.current_iteration()
        else:
            n = int(rounds)
            m = lgb.train(p, ds, num_boost_round=n)
        tmp = path.with_suffix('.tmp')
        m.save_model(str(tmp), num_iteration=n)
        tmp.replace(path)
        dump(info, {'rounds': n})
        log(f'{kind} seed {seed} done, {n} trees')
    return m, n


def fit_predict(tr, va, args, folder, frozen=None):
    """No label of va is read when frozen is provided (control/ranking)."""
    folder.mkdir(parents=True, exist_ok=True)
    maps = fit_maps(tr)
    log(f'Encodings by month blocks: {len(tr):,} train, {len(va):,} predictions')
    t = encoded_training(tr)
    d = apply_maps(va, maps)
    X, V = matrices(t, d)
    clean = t[CORRUPT].eq(0).to_numpy()
    has = t.delta_mvt_aobt3_s.notna().to_numpy()
    vhas = d.delta_mvt_aobt3_s.notna().to_numpy()
    med = maps[3]
    ub = t.unimpeded_p50.fillna(med).to_numpy(float)
    uv = d.unimpeded_p50.fillna(med).to_numpy(float)
    bt, bv = baseline(t, med), baseline(d, med)
    y = t[TARGET].to_numpy(float)
    # Only the selection step may access the labels of va.
    vy = d[TARGET].to_numpy(float) if frozen is None else np.zeros(len(d))
    vc = d[CORRUPT].eq(0).to_numpy() if frozen is None else np.ones(len(d), bool)
    normals = []; direct = {'direct_l2': [], 'direct_huber': []}
    rounds_out = {kind: [] for kind in COMPONENTS}
    importance = []
    for si, seed in enumerate(args.seeds):
        normal = np.zeros(len(d))
        for kind in ['normal_a', 'normal_b']:
            mask = clean & has if kind == 'normal_a' else clean
            vm = vc & vhas if kind == 'normal_a' else vc & ~vhas
            if frozen is None and not vm.any():
                vm = vc  # B can be selected on all clean rows if no AOBT is missing.
                if kind == 'normal_a':
                    raise ValueError('No clean row with AOBT in the selection set.')
            offset = t.delta_mvt_aobt3_s.to_numpy(float) if kind == 'normal_a' else ub
            voffset = d.delta_mvt_aobt3_s.to_numpy(float) if kind == 'normal_a' else uv
            transform = lambda p, off=voffset[vm]: np.clip(off + p, 30, 7200)
            fixed = frozen[kind][si] if frozen is not None else None
            m, n = learn(kind, X.loc[mask], (y-offset)[mask], V.loc[vm], vy[vm],
                         transform, seed, args, folder, fixed)
            rounds_out[kind].append(n)
            use = vhas if kind == 'normal_a' else ~vhas
            if use.any():
                normal[use] = voffset[use] + m.predict(V.loc[use], num_threads=args.threads)
            importance.extend({'component': kind, 'seed': seed, 'feature': f, 'gain': float(g)}
                              for f, g in zip(X.columns, m.feature_importance('gain')))
        normals.append(np.clip(normal, 30, 7200))
    # The classifier is selected on the mixture, with the normal regime fixed above.
    normal = np.mean(normals, axis=0)
    ps = []
    for si, seed in enumerate(args.seeds):
        known = t[CORRUPT].notna().to_numpy()
        if t.loc[known, CORRUPT].nunique() < 2:
            raise ValueError('Classifier: a single class in the training set.')
        transform = lambda p: mixture(p, normal, d.delta_mvt_sched_s.to_numpy(float), .75)
        m, n = learn('classifier', X.loc[known], t.loc[known, CORRUPT].to_numpy(float),
                     V, vy, transform, seed, args, folder,
                     frozen['classifier'][si] if frozen is not None else None)
        rounds_out['classifier'].append(n)
        ps.append(m.predict(V, num_threads=args.threads))
        importance.extend({'component': 'classifier', 'seed': seed, 'feature': f, 'gain': float(g)}
                          for f, g in zip(X.columns, m.feature_importance('gain')))
        for kind in direct:
            # No label clipping; Huber is a robustness ablation.
            m, n = learn(kind, X, y-bt, V, vy, lambda p: np.maximum(bv+p, 0),
                         seed, args, folder, frozen[kind][si] if frozen is not None else None)
            rounds_out[kind].append(n)
            direct[kind].append(bv + m.predict(V, num_threads=args.threads))
            importance.extend({'component': kind, 'seed': seed, 'feature': f, 'gain': float(g)}
                              for f, g in zip(X.columns, m.feature_importance('gain')))
    p = np.mean(ps, axis=0)
    out = d[[ID, 'ADEP_mvt', 'month', TARGET, CORRUPT, 'delta_mvt_aobt3_s',
             'delta_mvt_sched_s', 'no_flight_id', 'target_identity_error']].copy()
    out['normal'] = normal
    out['p_corrupt'] = p
    out[BASE] = mixture(p, normal, d.delta_mvt_sched_s.to_numpy(float), .75)
    out['mix1'] = mixture(p, normal, d.delta_mvt_sched_s.to_numpy(float), 1.)
    for k, preds in direct.items():
        out[k] = np.maximum(np.mean(preds, axis=0), 0)
    for c in [BASE, 'mix1'] + list(direct):
        if not np.isfinite(out[c]).all():
            raise ValueError(f'Non-finite predictions: {c}')
    pd.DataFrame(importance).to_csv(folder / 'importance.csv', index=False)
    dump(folder / 'rounds.json', rounds_out)
    return out, rounds_out


def select_model(pred):
    """Small selection budget: 13 predefined configurations, no stacking."""
    y = pred[TARGET].to_numpy(float)
    base = pred[BASE].to_numpy(float)
    months = sorted(pred.month.unique())
    rb = score(y, base)
    best = {'other': BASE, 'weight': 0.0}
    best_score = rb
    rows = []
    configs = [(BASE, 0.)] + [(k, w) for k in ['mix1', 'direct_l2', 'direct_huber']
                             for w in [.25, .5, .75, 1.]]
    for other, w in configs:
        p = (1-w)*base + w*pred[other].to_numpy(float)
        r = score(y, p)
        gains = [score(y[pred.month.eq(m)], base[pred.month.eq(m)]) -
                 score(y[pred.month.eq(m)], p[pred.month.eq(m)]) for m in months]
        eligible = (other == BASE) or (rb-r >= 1.0 and min(gains) >= 0.)
        rows.append({'other': other, 'weight': w, 'rmse': r, 'gain': rb-r,
                     'eligible': eligible,
                     **{f'gain_month_{int(m)}': g for m, g in zip(months, gains)}})
        if eligible and r < best_score:
            best, best_score = {'other': other, 'weight': w}, r
    return best, pd.DataFrame(rows)


def apply_choice(pred, choice):
    w = choice['weight']
    return (1-w)*pred[BASE].to_numpy(float) + w*pred[choice['other']].to_numpy(float)


def diagnostics(pred, folder, name):
    y = pred[TARGET].to_numpy(float)
    groups = [('all', np.ones(len(pred), bool))]
    groups += [(f'month={m}', pred.month.eq(m).to_numpy()) for m in sorted(pred.month.unique())]
    groups += [(f'airport={a}', pred.ADEP_mvt.astype(str).eq(a).to_numpy())
               for a in sorted(pred.ADEP_mvt.astype(str).unique())]
    groups += [('AOBT_present', pred.delta_mvt_aobt3_s.notna().to_numpy()),
               ('AOBT_missing', pred.delta_mvt_aobt3_s.isna().to_numpy()),
               ('block_equals_schedule', pred[CORRUPT].eq(1).to_numpy()),
               ('other_known_block', pred[CORRUPT].eq(0).to_numpy()),
               ('unknown_block_or_schedule', pred[CORRUPT].isna().to_numpy()),
               ('target_over_7200', y > 7200), ('target_below_30', y < 30)]
    rows = []
    for c in [BASE, 'mix1', 'direct_l2', 'direct_huber', 'selected']:
        e = (y-pred[c].to_numpy(float))**2
        total = float(e.sum())
        for group, mask in groups:
            if mask.any():
                rows.append({'model': c, 'group': group, 'n': int(mask.sum()),
                             'rmse': float(np.sqrt(e[mask].mean())),
                             'share_sse': float(e[mask].sum()/total) if total else 0.})
    pd.DataFrame(rows).to_csv(folder / f'{name}_metrics.csv', index=False)
    worst = pred.copy()
    worst['squared_error_selected'] = (y-pred.selected.to_numpy(float))**2
    worst.sort_values('squared_error_selected', ascending=False).head(200).to_csv(
        folder / f'{name}_worst200.csv', index=False)
    pred.to_parquet(folder / f'{name}_predictions.parquet', index=False)


def load_features(cache):
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['source']['code'] != code_hash():
        raise RuntimeError('Code changed since prepare: rebuild the cache.')
    if manifest['feature_sha256'] != sha(cache / 'features.parquet'):
        raise RuntimeError('Cache changed: rebuild with prepare --rebuild.')
    d = pd.read_parquet(cache / 'features.parquet')
    check_ids(d, 'Cache')
    return d, manifest


def settings(args, manifest):
    return {'feature_sha256': manifest['feature_sha256'], 'code': code_hash(),
            'seeds': args.seeds, 'max_rounds': args.max_rounds, 'patience': args.patience,
            'threads': args.threads, 'lightgbm': lgb.__version__, 'pandas': pd.__version__,
            'numpy': np.__version__, 'python': platform.python_version(),
            'selection_months': SELECT, 'control_months': CONTROL}


def run(args):
    d, manifest = load_features(args.cache)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = settings(args, manifest)
    mp = args.out / 'run_config.json'
    if mp.exists():
        if json.loads(mp.read_text(encoding='utf-8')) != cfg:
            raise RuntimeError('This folder contains another experiment. Choose another --out.')
    else:
        if any(args.out.iterdir()):
            raise RuntimeError('The --out folder must be empty for a new experiment.')
        dump(mp, cfg)
    tuning = d[~d.month.isin(SELECT+CONTROL)].copy()
    validation = d[d.month.isin(SELECT)].copy()
    control = d[d.month.isin(CONTROL)].copy()
    if set(validation.month.unique()) != set(SELECT) or set(control.month.unique()) != set(CONTROL):
        raise ValueError('Selection or control months missing.')
    locked = args.out / 'locked_selection.json'
    if not locked.exists():
        pred, rounds = fit_predict(tuning, validation, args, args.out / 'selection_models')
        choice, table = select_model(pred)
        table.to_csv(args.out / 'selection_grid.csv', index=False)
        pred['selected'] = apply_choice(pred, choice)
        diagnostics(pred, args.out, 'selection')
        dump(locked, {'choice': choice, 'rounds': rounds, 'config': cfg})
        log(f'Configuration frozen BEFORE the control: {choice}')
    frozen = json.loads(locked.read_text(encoding='utf-8'))
    if frozen['config'] != cfg:
        raise RuntimeError('Incompatible frozen configuration.')
    if not (args.out / 'summary.json').exists():
        train_control = d[~d.month.isin(CONTROL)].copy()
        # The labels are also masked in the object passed to the control prediction.
        hidden = control.copy()
        hidden[[TARGET, CORRUPT]] = np.nan
        pred, _ = fit_predict(train_control, hidden, args, args.out / 'control_models', frozen['rounds'])
        pred[TARGET] = control[TARGET].to_numpy()
        pred[CORRUPT] = control[CORRUPT].to_numpy()
        pred['selected'] = apply_choice(pred, frozen['choice'])
        diagnostics(pred, args.out, 'control')
        result = {'choice': frozen['choice'],
                  'control_rmse_reference': score(pred[TARGET], pred[BASE]),
                  'control_rmse_selected': score(pred[TARGET], pred.selected),
                  'control_gain_by_month': {str(m): score(pred.loc[pred.month.eq(m), TARGET], pred.loc[pred.month.eq(m), BASE])-
                     score(pred.loc[pred.month.eq(m), TARGET], pred.loc[pred.month.eq(m), 'selected']) for m in CONTROL},
                  'note': 'Control not used for selection in this script; these months were explored earlier in the project.'}
        dump(args.out / 'summary.json', result)
    log((args.out / 'summary.json').read_text(encoding='utf-8'))
    log('Done. The configuration stays frozen, even if the control degrades.')


def submission_frame(template, pred, values):
    check_ids(template, 'Submission template')
    check_ids(pred, 'Ranking predictions')
    if not np.isfinite(values).all():
        raise ValueError('Submission: non-finite predictions.')
    pp = pred[[ID]].copy(); pp[TARGET] = values
    s = template[[ID]].merge(pp, on=ID, how='left', validate='one_to_one', sort=False)
    if len(s) != len(template) or s[TARGET].isna().any() or not s[ID].equals(template[ID].reset_index(drop=True)):
        raise ValueError('Submission: wrong coverage or order of the identifiers.')
    return s


def submit(args):
    d, manifest = load_features(args.cache)
    locked = json.loads((args.out / 'locked_selection.json').read_text(encoding='utf-8'))
    cfg = locked['config']
    for k in ['seeds', 'max_rounds', 'patience', 'threads']:
        setattr(args, k, cfg[k])
    if settings(args, manifest) != cfg:
        raise RuntimeError('Code, data or environment differ from the selection.')
    if not (args.out / 'summary.json').exists():
        raise RuntimeError('Run `run` and complete the control before submit.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.team):
        raise ValueError('Invalid team name.')
    dest = args.out / 'submissions'
    dest.mkdir(exist_ok=True)
    paths = [dest / f'{args.team}_v{args.start_version+i}.parquet' for i in range(2)]
    if any(p.exists() for p in paths):
        raise FileExistsError('Submission numbers already exist: choose a different --start-version.')
    raw = pd.read_parquet(args.data / 'ranking.parquet')
    rp = args.out / 'ranking_input.json'
    signature = input_signature([args.data / 'ranking.parquet', args.data / 'submitting.parquet'])
    if rp.exists() and json.loads(rp.read_text()) != signature:
        raise RuntimeError('Ranking file changed: use a new experiment folder.')
    dump(rp, signature)
    rank = build(raw)
    rank[[TARGET, CORRUPT]] = np.nan
    pred, _ = fit_predict(d, rank, args, args.out / 'full_models', locked['rounds'])
    pred['selected'] = apply_choice(pred, locked['choice'])
    template = pd.read_parquet(args.data / 'submitting.parquet').reset_index(drop=True)
    outputs = [submission_frame(template, pred, pred[c].to_numpy()) for c in [BASE, 'selected']]
    for p, frame in zip(paths, outputs):
        frame.to_parquet(p, index=False)
        log(f'Submission written: {p} ({len(frame):,} rows)')
    pred.drop(columns=[TARGET, CORRUPT]).to_parquet(args.out / 'ranking_predictions.parquet', index=False)
    dump(args.out / 'submission_manifest.json', {'reference': str(paths[0]), 'selected': str(paths[1]),
                                               'choice': locked['choice'], 'automatic_upload': False})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['prepare', 'run', 'submit']:
        q = sub.add_parser(name)
        q.add_argument('--cache', type=Path, default=Path('prc_cache'))
        if name in ['prepare', 'submit']:
            q.add_argument('--data', type=Path, default=Path('data'))
        if name == 'prepare':
            q.add_argument('--rebuild', action='store_true')
        else:
            q.add_argument('--out', type=Path, default=Path('prc_run'))
        if name == 'run':
            q.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123])
            q.add_argument('--threads', type=int, default=max(1, min(8, os.cpu_count() or 1)))
            q.add_argument('--max-rounds', type=int, default=2500)
            q.add_argument('--patience', type=int, default=150)
        if name == 'submit':
            q.add_argument('--team', default='motivated-ostrich')
            q.add_argument('--start-version', type=int, default=30)
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    if a.command == 'run' and (a.threads < 1 or a.max_rounds < 1 or a.patience < 1
                              or len(set(a.seeds)) != len(a.seeds)):
        raise SystemExit('Invalid parameters: threads/rounds/patience must be positive, seeds unique.')
    {'prepare': prepare, 'run': run, 'submit': submit}[a.command](a)
