# PRC Data Challenge 2026 — Team motivated-ostrich

Prediction of taxi-out time (seconds) for departures at ten major European
airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM),
January and July 2026, RMSE metric.

Final public score: **329.41 s** (submission `motivated-ostrich_v31`, pipeline `src/prc_experiment.py`).

This repository contains the full pipeline, the experiment log (19 tested
hypotheses with verdicts), and the material for the accompanying paper.

## Method in one paragraph

The provided `TAXITIME` is not always a taxi time. About 4.8 % of departures
have their off-block time silently replaced by the scheduled time (so the
"taxi" equals the departure delay plus the real taxi), and a handful of rows
at LIRF carry a date-rollover bug (taxi ≈ 86 400 s + real taxi). Our final
model is therefore a **regime mixture**:

```
prediction = p^0.75 * (MVT_TIME − SCHED_TIME) + (1 − p^0.75) * μ_normal
```

where `p` is a LightGBM classifier estimating the probability that the
off-block was substituted, and `μ_normal` is a LightGBM regressor predicting
the **residual over the Network Manager off-block** (`TAXITIME − (MVT_TIME −
AOBT_3)`) for the 98.5 % of flights where `AOBT_3` is available, with a
fallback regressor otherwise. Both are 5-seed ensembles. Features include NM
time deltas, airport/stand/runway categoricals, out-of-fold target encodings
of the substitution rate (airport × operator, airport × flight number,
airport × hour), arrival-context features computed from the fully-observed
arrivals of the ranking period (contemporaneous taxi-in levels, share of
corrupted arrival in-blocks — a reporting-outage signal), stand-linking
features (turnaround bound from the previous arrival at the same stand),
PRU-style unimpeded times per stand–runway, and off-block-anchored queue
features from the literature.

Three findings we consider more interesting than the score:

1. **The error floor is a data floor, not a model floor.** Half of the MSE is
   the irreducible discrepancy between the airport's off-block and the NM
   off-block (per-airport hourly bias from −118 s to +296 s, σ 250–500 s);
   another quarter sits in *two* CDG rows (84 240 s and 58 206 s) with no
   identifiable signature.
2. **Classic congestion features are useless when the NM off-block is
   observed.** The literature's top predictor (queue size at pushback, Idris
   2001) brings nothing here: knowing the actual NM taxi duration screens it
   off.
3. **Data bugs are not statistically stationary at the two-month scale.** An
   expectation-based correction for the LIRF rollover bug, validated with a
   month-transfer protocol on 2025 (+19.8 s), *lost* 2.45 s on the 2026
   leaderboard: the bug occurs ~12 times a year, so a two-month window is a
   Poisson draw. We removed the branch and kept the analysis.

The complete version-by-version log (local and public scores, all 19
hypotheses and their verdicts) is in [`results/versions.md`](results/versions.md).

## Repository layout

```
├── README.md
├── requirements.txt
├── reproduce.py               # single entry point
├── src/
│   ├── prc_experiment.py      # production pipeline (prepare / run / submit CLI) — final score
│   ├── pipeline.py            # v16 feature library and earlier end-to-end pipeline
│   └── stacking_experiment.py # out-of-fold residual corrector (kept: +0.7 s, documented)
├── notebooks/
│   ├── 01_eda_corruption.ipynb   # figures: off-block bias, substitution rate, MSE anatomy
│   └── 02_ablations.ipynb        # figures: version history, local-vs-public calibration
├── results/
│   └── versions.md            # experiment log v2 → v17
├── paper/
│   └── outline.md
├── LICENSE
└── data/                      # not versioned; see below
```

`pipeline.py` is deliberately kept as one file: it is the exact script that
produced the scored submissions, and every design decision is documented
in-line with the measurement that motivated it.

## Data

All inputs come from the challenge buckets (OpenSky Network / EUROCONTROL).
With the MinIO client configured as per the challenge instructions:

```
mc cp --recursive dc26/prc-2026-datasets/ data/
```

Expected in `data/`: 12 monthly `training_*.parquet` files (2025),
`ranking.parquet`, `submitting.parquet`.

**External data**: none in the final model. We evaluated the OPDI flight
list and flight events (EUROCONTROL, ADS-B-derived, openly documented at
opdi.aero) as a potential off-block source and rejected it: ground coverage
at these airports is insufficient (0.2 % match at LTFM, 1.6 % at LEMD;
`first_seen` trails the true off-block by a median 700–1 300 s). The
evaluation script is available on request; the negative result is in the
paper.

## Reproducing

```
pip install -r requirements.txt
python reproduce.py                     # earlier end-to-end pipeline (v16)
# Production pipeline (final submission):
python src/prc_experiment.py prepare    # builds features (labels masked first)
python src/prc_experiment.py run        # selection on Nov-Dec, frozen control on Jan+Jul
python src/prc_experiment.py submit     # writes the submission files
```

First run: ~20–40 min (feature computation is cached in `data/`), then
~10–15 min per run. The notebooks in `notebooks/` regenerate the paper's
figures from those caches (run `reproduce.py` once first). Outputs land in `submissions/` following the
`<team>_v<k>.parquet` naming convention. The script prints the temporal
validation (train: Feb–Jun + Aug–Dec 2025; validation: Jan + Jul 2025,
mimicking the ranking's seasonality), per-airport RMSE, and feature
importances at every stage.

Determinism: seeds are fixed (5-seed ensemble, listed in the script);
results are reproducible up to LightGBM's threading nondeterminism (±0.5 s
in our experience).

## Honest limitations

- The two heaviest error contributors (CDG rows with day-scale recorded
  taxi times and no signature) are unpredictable with the provided data; any
  score in this challenge carries a few-seconds lottery on such rows.
- The substitution classifier plateaus at AUC 0.88 / recall 11 % at p > 0.5
  despite 16 tested signatures; we conclude the substitution events are
  mostly system-level and unobservable from the flight's own attributes.
- The gap to the top of the leaderboard (~80 s) is not explained by anything
  we could measure in the provided data.

## Team

Ugo Roccamatisi — dual-degree MSc student, Centrale Lille & Cranfield
University (Aerospace Vehicle Design).

## License

MIT for the code. Challenge data remain subject to the challenge's own
terms.
