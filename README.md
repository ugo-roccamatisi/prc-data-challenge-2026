# PRC Data Challenge 2026: team motivated-ostrich

Taxi-out time prediction (seconds, RMSE) for departures at ten European
airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM).

**Final submission:** `motivated-ostrich_final.parquet`, produced by
[`src/taxiout_general.py`](src/taxiout_general.py) on `final_ranking.parquet`
(January, February, June and July 2026, 670,790 departures).

The final model is a single, self-contained pipeline built for the blind
final phase. **No setting of this model was chosen with leaderboard
feedback.** Every design choice rests either on a structural property of the
data, verified on all of 2025, or on out-of-time validation with a selection
fold and a separate control fold. The earlier, partly leaderboard-tuned
pipelines are kept in [`legacy/`](legacy/) and documented in
[`results/versions.md`](results/versions.md).

## At a glance

| | |
|---|---|
| Model | Two-regime mixture: classifier for substituted off-blocks, regressors for the normal taxi |
| Regressors | 0.75 CatBoost + 0.25 per-airport XGBoost, weights chosen on the selection fold |
| Key structural fact | `\|BLOCK - LOBT\| <= 3606 s` holds for 100 % of the 2,062,251 matched 2025 departures |
| Control fold (Jan + Jul 2025) | **348.3 s** RMSE, 262.8 s on the clean lane (labels <= 20,000 s) |
| Selection fold (Apr + Oct 2025) | 216.1 s RMSE, 210.1 s on the clean lane |
| External data | None |
| Runtime | About 4 h 45 on a Kaggle T4 GPU, 17 min without CatBoost |

## 1. What the label really is

The target is `TAXITIME = MVT_TIME - BLOCK_TIME`, where `BLOCK_TIME` is the
off-block time reported by the airport. Three facts about it shape the whole
model.

**Some off-blocks are not off-blocks.** On 4.8 % of 2025 departures,
`|BLOCK - SCHED| <= 10 s`: the airport filled the off-block with the scheduled
time, so the label equals the departure delay plus the real taxi. These rows
carry 29 % of the squared error of a naive model.

**Every matched block lies within one hour of the last filed off-block.**
For every departure matched to the Network Manager flight list,
`|BLOCK - LOBT| <= 3606 s`, without a single exception in 2,062,251 rows (the
other NM clocks satisfy it on 99.997 % for IOBT, 99.97 % for EOBT_1 and
99.91 % for AOBT_3). It is the window used to match airport movements to NM
flights. Two consequences follow, and both can only reduce the error:

1. The true taxi lies in `[MVT - LOBT - 3606, MVT - LOBT + 3606]`. Every
   prediction is projected on that interval.
2. A substituted off-block is **impossible** when `|SCHED - LOBT| > 3606 s`.
   In the January + July 2025 holdout, 13,869 departures are in that case and
   none of them is substituted. On those rows the substitution probability is
   set to zero.

Applied after the fact to the predictions of an earlier pipeline (v15), these
two rules alone took the holdout RMSE from 345.0 to 336.7 s.

**A few labels are not explainable from any column.** Twenty-six holdout rows
have labels above 20,000 s, most of them a date-rollover bug at LIRF (the
block carries the right time of day and the wrong date). We searched for a
signature twice, row by row ([`src/prc_taxiout.py`](src/prc_taxiout.py)
`dayshift`) and through co-occurrence with buggy arrivals on the same night
([`src/cooccur.py`](src/cooccur.py)): neither finds one (section 5).

## 2. Model

```
p      = P(BLOCK = SCHED)                 XGBoost classifier, 0 when infeasible
mu     = MVT - AOBT_3 + residual          regressor A, when 0 <= MVT - AOBT_3 <= 7200 s
       = taxi                             regressor B, otherwise
pred   = p^g * (MVT - SCHED) + (1 - p^g) * mu
pred   = clip(pred, MVT - LOBT - 3606, MVT - LOBT + 3606)
```

- **Classifier.** Trained only on rows where a substitution is feasible.
- **Regressor A** predicts the residual over the NM off-block (`AOBT_3`), used
  only when it gives a plausible taxi (0 to 7,200 s). **Regressor B** predicts
  the taxi directly for the other rows. Both are trained on non-substituted
  labels between 30 and 7,200 s.
- **Expectation, not argmax.** Under squared loss the best prediction is the
  expectation over the two regimes. The exponent `g = 0.9` is chosen on the
  selection fold.
- **Unmatched movements.** For rows with no NM record and no aircraft type
  whose `MVT - SCHED` exceeds 3 h, the label equals `MVT - SCHED` on 67 % of
  the 244 such rows of 2025. The mixture weight is that rate, estimated per
  airport and shrunk towards the global rate (no airport is hard-coded).
- **Regressor families.** Three families are trained for A and B: global
  XGBoost, per-airport XGBoost for A, and global CatBoost. Their weights
  (grid in steps of 0.25) and the exponent are chosen jointly on the selection
  fold, with an eligibility rule: a blend must beat XGBoost alone by at least
  0.5 s, otherwise XGBoost alone is kept. The selected blend is
  0.75 CatBoost + 0.25 per-airport XGBoost.
- **Final model.** Trained on the twelve months of 2025 with 1.1 times the
  number of trees found by early stopping, three seeds averaged.

### Features

Departure labels (`BLOCK_TIME`, `TAXITIME`) are masked before any contextual
feature is built, so training features are computed under exactly the
conditions of the ranking file.

| Group | Examples |
|---|---|
| Plan and clocks | `MVT - SCHED`, `MVT - AOBT_3`, `MVT - LOBT`, `MVT - EOBT_1`, `AOBT_3 - LOBT`, NM delay, feasibility flag, day gap |
| Traffic around the take-off | departures and arrivals within 15 min, same-runway departures 10 min before and 5 min after, gaps to the previous and next take-off |
| Off-block anchored queue | aircraft taxiing at my off-block, take-offs between my off-block and my take-off (airport, runway), recent NM taxi times |
| Airport climate | hourly and daily mean NM taxi, NM delay, arrival taxi-in, arrival block substitution rate, daily departures |
| Stand | turnaround, previous arrival substituted, previous arrival taxi-in, same aircraft type |
| Categories | airport, runway, stand, aircraft type, destination, operator, market segment, wake category |
| Reference | median and 20th percentile of the NM taxi per stand and runway, computed on the training fold |

## 3. Validation protocol

| Fold | Training months (2025) | Validation months | Role |
|---|---|---|---|
| Selection | all but Jan, Apr, Jul, Oct | Apr + Oct | number of trees (early stopping), exponent, family weights |
| Control | all but Jan, Jul | Jan + Jul | scored once with the settings of the selection fold |

The control fold mimics the seasonality of the ranking set and is never used
to choose anything. Three seeds of one fixed configuration spanned 2.9 s in an
earlier campaign, so differences below that are read as noise.

## 4. Results

### Control fold (January + July 2025, 344,419 departures)

| Configuration | RMSE | Clean lane |
|---|---|---|
| Earlier best pipeline v26 (partly leaderboard-tuned) | 356.6 s | 259.2 s |
| XGBoost only | 355.0 s | 271.7 s |
| Per-airport XGBoost only | 354.2 s | 270.6 s |
| CatBoost only | 349.0 s | 263.7 s |
| **Selected blend (0.75 CatBoost + 0.25 per-airport)** | **348.3 s** | **262.8 s** |

The blend was selected on April + October, where it gained 5.9 s over
XGBoost alone. On the control fold it gains 6.7 s (8.9 s on the clean lane),
with the gain spread over all ten airports.

### Ablations of the structural components (selected blend)

| Removed component | Selection fold | Control fold |
|---|---|---|
| None (full model) | 216.1 s | 348.3 s |
| Unmatched-movement rule | 222.4 s | 392.6 s |
| Rule and LOBT projection | 223.7 s | 397.9 s |

The rule is worth 44 s on the control fold because July 2025 carries many
unmatched LIRF rows; it is the component with the largest variance across
months and should be read as such.

### Where the remaining error is

Decomposition of the control-fold squared error by regime (reduced CPU run
on 60 % of the training rows, 359.9 s; the proportions match the full run).

| Regime | Rows | Share of MSE | Reducible? |
|---|---|---|---|
| Aberrant labels (> 20,000 s) | 26 | 40 % | No signature found in any column |
| Normal departures | 323,650 | 40 % | Only with new information |
| Unmatched to NM | 5,227 | 15 % | Partly: a perfect classifier would remove 10 points |
| Substituted off-block | 15,324 | 5 % | Classifier plateaus at AUC 0.88 |

Twenty-six rows weigh as much as 323,650 normal flights. On the normal lane
the model regresses towards the mean (short taxis over-predicted by about
100 s, taxis of 30 to 60 min under-predicted by about 400 s), the signature
of measurement noise: the gap between the airport off-block and the NM
off-block is not observable from the provided columns. The only information
that could reduce it is a physical observation of the push-back, such as
surface ADS-B tracks, which we did not use.

## 5. Negative results worth knowing

| Idea | Test | Outcome |
|---|---|---|
| Recalibrate airport-month means on EUROCONTROL's monthly taxi-out series | Out of time on 2025 | Model means are already within 20 s of the truth (LTFM July: 31 s); the series differs from our labels by up to 166 s in winter (de-icing definition). Shifting towards it costs 0.7 to 3.1 s. Rejected. |
| Flag date-rollover bugs through buggy arrivals the same night | 2025, windows 1 to 24 h, placebo at +/- 7 days | No co-occurrence (nearest buggy arrival 27 h away or more, placebo identical). The bug is per flight, not per night. |
| Rome branch: expected value of the rollover bug from its 2025 rate | Holdout and leaderboard (earlier campaign) | +19.8 s on 2025, -2.45 s on 2026: about 12 events a year, a two-month window is a Poisson draw. Removed. |
| Off-block queue features (literature's top predictor) | Holdout, twice | Screened off once the NM off-block is known |
| OPDI ADS-B events as off-block source | Coverage and timing | 0.2 % match at LTFM, `first_seen` 700 to 1,300 s after the off-block |
| METAR temperature and de-icing proxies | Holdout | Act as season surrogates, degrade July |

Earlier campaigns tested about thirty hypotheses; the full log is in
[`results/versions.md`](results/versions.md).

## 6. Reproducing

### On Kaggle (recommended)

1. Upload the challenge files as a private dataset: the twelve
   `training_*.parquet` files, `final_ranking.parquet`,
   `final_submitting.parquet`, `ranking.parquet`, `submitting.parquet`.
2. Import [`notebooks/03_kaggle_taxiout_general.ipynb`](notebooks/03_kaggle_taxiout_general.ipynb),
   add the dataset, select a GPU accelerator.
3. *Save Version > Save & Run All (Commit)*. Outputs in `/kaggle/working`:
   `motivated-ostrich_final.parquet`, `settings.json` (trees, weights,
   exponent, fold scores), `oof_sel.parquet`, `oof_ctl.parquet`.

The notebook is generated from `src/taxiout_general.py` and runs the same code.

### Locally

```
pip install -r requirements.txt
python src/taxiout_general.py --data data --seeds 3 --target final
```

Useful options:

| Option | Effect |
|---|---|
| `--target public` | predicts `ranking.parquet` with `submitting.parquet` as template |
| `--components xgb` | XGBoost only (17 min end to end on a Kaggle T4) |
| `--skip-cv --rounds-json settings.json` | reuses the settings of a previous run |
| `--skip-final` | validation only |
| `--sample 0.3`, `--eta 0.1` | quick screening |

The ranking file and the template are always chosen as a pair, and the script
stops if more than 0.1 % of template ids are missing from the ranking file.
Seeds are fixed; GPU training is reproducible up to floating-point
nondeterminism.

Diagnostics (a few minutes each):

```
python src/prc_taxiout.py defaults --data data    # anatomy of the aberrant labels
python src/prc_taxiout.py rule     --data data    # precision of the unmatched-movement rule
python src/prc_taxiout.py dayshift --data data    # the non-reconstructible family
python src/cooccur.py --data data                 # arrival/departure co-occurrence test
```

## Repository layout

```
├── src/
│   ├── taxiout_general.py         # final pipeline (features, models, validation, submission)
│   ├── prc_taxiout.py             # diagnostics on the labels
│   └── cooccur.py                 # co-occurrence test of the date-rollover bug
├── notebooks/
│   ├── 01_eda_corruption.ipynb    # earlier figures: off-block bias, substitution rate
│   ├── 02_ablations.ipynb         # earlier figures: version history
│   └── 03_kaggle_taxiout_general.ipynb   # final pipeline as a Kaggle notebook
├── results/versions.md            # experiment log, all campaigns
├── legacy/                        # earlier pipelines (v16 to v26), kept for traceability
├── paper/                         # report and figures of the second campaign
├── requirements.txt
└── LICENSE
```

The report in `paper/report/` describes the second-campaign pipeline (v26),
not the final model; this README is the reference for the final submission.

## Limitations

- CatBoost reached its iteration cap (6,000 trees for regressor A, 4,000 for
  B) without triggering early stopping: it was still improving. A larger cap
  or learning rate is available through `--cat-max` and `--cat-lr` but was not
  run before the deadline.
- February and June are in the final set but not in a validation fold; they
  exist in the 2025 training data and sit between the two folds seasonally.
- The unmatched-movement rule depends on the frequency of an airport-side
  defect, mostly at LIRF. Its gain varies widely between months.
- The clean-lane gap to the earlier v26 pipeline (3.6 s) is most likely the
  posterior sequence features (take-offs after the target flight), which we
  left out because they have no operational meaning.

## A note on leaderboard probing

The public leaderboard scores the full ranking set at four decimals, which
makes it possible to reveal individual labels by changing one row at a time
across submissions. We never did this. No setting of the final model was
chosen on the leaderboard, and the unmatched-movement rule is estimated on
2025 only.

## Author

Ugo Roccamatisi, dual-degree MSc student, Centrale Lille and Cranfield
University (Aerospace Vehicle Design).

## License

MIT for the code. Challenge data remain subject to the challenge's own terms.
