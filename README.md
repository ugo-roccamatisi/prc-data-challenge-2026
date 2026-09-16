# PRC Data Challenge 2026: team motivated-ostrich

Prediction of taxi-out time (seconds) for departures at ten major European
airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM),
January and July 2026, RMSE metric.

Best public score: **319.21 s** (submission `motivated-ostrich_v26`, pipeline
`src/taxiout_v19.py` plus the reconstruction rule of `src/apply_rule.py`).

This repository contains the full pipeline, the experiment log (two campaigns,
30 tested hypotheses with verdicts) and the material for the accompanying
paper.

## Method in one paragraph

The provided `TAXITIME` is not always a taxi time. About 4.8 % of departures
have their off-block time silently replaced by the scheduled time (so the
"taxi" equals the departure delay plus the real taxi), and a handful of rows
at LIRF carry a date-rollover bug (taxi around 86,400 s plus the real taxi).
The model is therefore a **regime mixture**:

```
prediction = p^0.75 * (MVT_TIME - SCHED_TIME) + (1 - p^0.75) * mu_normal
```

where `p` is a LightGBM classifier estimating the probability that the
off-block was substituted, and `mu_normal` predicts the **residual over the
Network Manager off-block** (`TAXITIME - (MVT_TIME - AOBT_3)`) for the 98.5 %
of flights where `AOBT_3` is available, with a fallback regressor otherwise.
Around that core, the final configuration adds four things measured on the
leaderboard: a deterministic reconstruction rule for one family of corrupted
labels, non-causal sequence features, one model per airport with a global
corruption classifier, and a CatBoost/LightGBM ensemble.

Five findings we consider more interesting than the score:

1. **The error floor is a data floor, not a model floor.** Twenty-six rows out
   of 344,227 carry 58 % of the validation MSE. Once they are excluded, the
   clean-lane RMSE is 260 s and the median absolute error is 100 s.
2. **One corrupted family is exactly reconstructible from provided columns.**
   For movements never matched to NM data, the airport filled `BLOCK_TIME`
   with the scheduled time, so `TAXITIME = MVT_TIME - SCHED_TIME` holds to
   within a few seconds. Both columns are visible in the ranking file. The
   rule is worth 7.95 s on the leaderboard and requires no probing of any
   kind.
3. **Causal congestion features are screened off by the observed AOBT_3, but
   posterior sequence context is not.** The literature's top predictor (queue
   size at pushback) brings nothing here, which we confirmed twice. Features
   built from the takeoffs that *follow* the target flight are worth 3.6 s.
   They have no operational meaning (a controller does not know future
   takeoffs) and only make sense because the task is retrospective
   reconstruction rather than forecasting.
4. **Architecture beats features.** Over a second campaign, per-airport
   regressors with a global classifier plus a CatBoost ensemble brought 5.1 s
   on the leaderboard, while four separate families of engineered features
   brought nothing.
5. **Seed noise is 2.9 s on this task.** Three seeds of an identical
   configuration gave 366.43, 366.17 and 363.52 s. Any single-seed comparison
   below roughly 3 s is unmeasured, which invalidates a large share of the
   comparisons routinely reported in this kind of competition, including
   several of our own first-campaign ones.

The complete version-by-version log is in
[`results/versions.md`](results/versions.md).

## Repository layout

```
├── README.md
├── requirements.txt
├── reproduce.py                 # single entry point for the first-campaign pipeline
├── src/
│   ├── taxiout_v19.py           # production pipeline (second campaign), best score
│   ├── apply_rule.py            # post-processing: label reconstruction rule
│   ├── prc_taxiout.py           # diagnostics (corruption anatomy, rule validation, ablations)
│   ├── prc_experiment.py        # first-campaign production pipeline (329.41 s)
│   ├── pipeline.py              # v16 feature library and earlier end-to-end pipeline
│   └── stacking_experiment.py   # out-of-fold residual corrector (kept: +0.7 s, documented)
├── notebooks/
│   ├── 01_eda_corruption.ipynb  # figures: off-block bias, substitution rate, MSE anatomy
│   └── 02_ablations.ipynb       # figures: version history, local-vs-public calibration
├── results/
│   └── versions.md              # experiment log, both campaigns
├── paper/
│   └── outline.md
├── LICENSE
└── data/                        # not versioned; see below
```

## Data

All inputs come from the challenge buckets (OpenSky Network / EUROCONTROL).
With the MinIO client configured as per the challenge instructions:

```
mc cp --recursive dc26/prc-2026-datasets/ data/
```

Expected in `data/`: 12 monthly `training_*.parquet` files (2025),
`ranking.parquet`, `submitting.parquet`.

**External data**: none in the final model. Two sources were evaluated and
rejected. The OPDI flight list and flight events (EUROCONTROL, ADS-B-derived)
as a potential off-block source: ground coverage at these airports is
insufficient (0.2 % match at LTFM, 1.6 % at LEMD; `first_seen` trails the true
off-block by a median 700 to 1,300 s). Hourly METAR observations for the ten
airports (Iowa State ASOS archive): temperature and de-icing proxies act as
season surrogates and degraded July in our holdout. The METAR download code
is kept in `src/prc_taxiout.py` (`metar` subcommand) so the negative result is
reproducible.

## Reproducing

```
pip install -r requirements.txt
```

Production pipeline (best submission):

```
python src/taxiout_v19.py --following --rounds-scale 2 --clf-lean \
    --per-airport --clf-global --catboost --cat-weight 0.75 --seeds 3
python src/apply_rule.py --ranking data/ranking.parquet \
    --sub submissions/motivated-ostrich_v19.parquet \
    --out submissions/motivated-ostrich_v26.parquet --mode dur
```

Useful options: `--fast` trains the first stage once instead of seven times
(no out-of-fold folds, no corrector, no submission) and prints the per-airport
and per-month breakdown, the clean-lane RMSE and the error concentration;
`--submit-only` skips validation and goes straight to the ranking;
`--model-cache` reuses trained models across runs of an identical
configuration; `--sample 0.3` subsamples the training set for quick screening.

Diagnostics, each a few minutes:

```
python src/prc_taxiout.py diagnose --data data     # leak-candidate columns, baselines
python src/prc_taxiout.py defaults --data data     # anatomy of the aberrant labels
python src/prc_taxiout.py rule     --data data     # precision/recall of the reconstruction rule
python src/prc_taxiout.py dayshift --data data     # the non-reconstructible family
python src/prc_taxiout.py outliers --data data --airport LIRF
```

First run: 20 to 40 min of feature computation, cached in `data/`. Full
production run with three seeds, per-airport models and CatBoost: several
hours on 12 cores.

The first-campaign pipeline is still runnable:

```
python reproduce.py
python src/prc_experiment.py prepare
python src/prc_experiment.py run
python src/prc_experiment.py submit
```

Determinism: seeds are fixed and listed in the scripts. Results are
reproducible up to the threading nondeterminism of LightGBM and CatBoost.
Note the measured seed-to-seed spread of 2.9 s: reported differences smaller
than that should be read as noise.

## Honest limitations

- Twenty-six rows carry 58 % of the validation MSE. Eighteen of them belong
  to a date-rollover family that is explicable after the fact but not
  identifiable from any provided column: the candidate filter finds two rows
  in the 2026 ranking set against the three or four expected. Any score in
  this challenge carries a few-seconds lottery on such rows.
- The substitution classifier plateaus at AUC 0.88 and 11 % recall at
  p > 0.5 despite sixteen tested signatures. The substitution events appear
  to be system-level and unobservable from the flight's own attributes.
- LIRF remains at 557 s of clean-lane RMSE against 174 to 267 s elsewhere,
  and structured estimators do no better there than a constant, which
  suggests the target itself is noisy well below the aberrance threshold.
- The gap to the top of the leaderboard (around 75 s) is not explained by
  anything we could measure in the provided data. Published repositories from
  better-ranked teams suggest a much larger feature count and much larger
  models rather than a different insight.

## A note on leaderboard probing

The public leaderboard scores the full ranking set at four decimals, which
makes it possible to reveal individual labels by changing one row at a time
across submissions, as discussed publicly in the challenge forum. We did not
do this. The reconstruction rule described above was derived entirely from
the 2025 training data and uses only columns present in the ranking file; its
derivation is reproducible from `src/prc_taxiout.py rule` without any
submission.

## Author

Ugo Roccamatisi, dual-degree MSc student, Centrale Lille and Cranfield
University (Aerospace Vehicle Design).

## License

MIT for the code. Challenge data remain subject to the challenge's own terms.
