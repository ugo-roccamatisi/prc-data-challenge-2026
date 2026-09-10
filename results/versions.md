# PRC Data Challenge 2026: experiment log

Team `motivated-ostrich`. Taxi-out prediction (seconds) for departures at ten
European airports, January and July 2026, RMSE metric.

Local validation: train on 10 months of 2025, validate on January and July
2025 (344,227 departures), mimicking the ranking's seasonality. Local RMSE
values are only comparable to each other from v4 onwards (see note in 1).

## 1. Version chronology

| Version | Tested idea | Local RMSE | Public | Verdict |
|---|---|---|---|---|
| v2 | Direct LightGBM, base features + NM deltas + airport congestion | 254.0* | | Working baseline |
| v3 | Residual over AOBT_3 + cleaning of aberrant matches | 274.1* then 252.9* | | Cleaning hurts (removed rows are in the ground truth). Without cleaning, 1 s gain |
| v4 | Discovery of "block = sched" rows; classifier + probabilistic mixture; long-taxi rows reinstated | 379.8 | | **29 % gain** (533.1 without mixture). First honest measurement, long taxis included |
| v5 | Out-of-fold target encoding, scale_pos_weight | (broken) | | Early stopping at iteration 7: weighting incompatible with the stopping metric |
| v5.1 | Same without weighting, AUC stopping, per-airport exponents | 375.4 | 347.80 (v1) | Classifier fixed but recall unchanged (11 %). 4 s gain |
| v6 | Comparison: single model vs mixture vs body/tail | 472.8 / 379.1 / 379.4 | | **Single model refuted** (+100 s). Body/tail split neutral |
| v7 | Joint sweep (gamma_c, gamma_t, kappa) with month-transfer control | 375.9 but rejected | | The 3.5 s gain does not survive transfer (−18 s from one month to the other). Back to ARCH 0 |
| v7.1 | Fix: frozen rounds at retraining (bug: early stopping impossible with train = val) | 379.4 | 341.33 (v7) | Clean pipeline |
| v8 | gamma_c calibration on the leaderboard (0.5 / 0.6 / 0.75 / 1.0) | 389.8 / 379.4 / 380.1 / 391.0 | 358.85 / 347.80 / **347.59** / 355.16 | Public curve parallel to the local one, constant ~32 s offset: **local validation is a reliable proxy** |
| v9 | Arrival context (taxi-in around takeoff) + stand linking | 371.1 | | +8.3 s, spread across all airports. CDG and Rome unchanged |
| v10 | Reporting-outage signal (corrupted arrivals in the window) | 368.2 | | +2.9 s, Rome −33 s, other airports slightly degraded. Real but diluted signal (0.3 % of flights) |
| v11 | Rome branch (unmatched, next-day takeoff): three-regime expectation | 348.4 | 340.40 (v11, v13 pipeline) | **+19.8 s local, −2.45 s public**. See 3.2 |
| v12 | 5-seed ensembling + per-runway queue features | 346.4 | | +2.0 s, entirely from ensembling |
| v13 | Flight-number target encoding + stand-linked arrival corruption | 345.2 | **337.95** (v10, no Rome) | +1.2 s local. Best public score to date |
| v14 | PRU-style unimpeded time per stand-runway, regressor B on the excess | 345.9 | | Neutral. `unimpeded_p50` 8th feature, but the geometry was already extracted from `stand_runway` |
| v15 | Off-block-anchored queue features (N, Q, recent taxi; Idris/Ravizza) | 345.0 | 338.02 (v15, with Rome) | Neutral. No `q_*` feature in the top 15: congestion adds nothing once the NM-side duration is known |

\* Validation on filtered data (taxi > 7,200 s removed), not comparable with later versions.

Best production version: **v15 pipeline without the Rome branch** (submission
v14, expected around 335 s public).

## 2. Diagnostics that steered the work

| Diagnostic | Result | Consequence |
|---|---|---|
| Naive `MVT − AOBT_3` | RMSE 380.8 s; per-airport median bias from −118 s (Rome) to +296 s (Istanbul) | Systematic definition gap between airport and NM off-blocks: target = residual over AOBT_3 |
| "Block = sched" rows (`\|BLOCK − SCHED\| ≤ 10 s`) | 4.80 % of departures, 29 % of squared error, median taxi 1,009 s (normal: 910 s) | The taxi-out is not a taxi-out on these rows: `TAXITIME = MVT − SCHED` exactly (median gap 3 s). Probabilistic mixture (v4) |
| Error concentration | 10 rows = 29.6 % of MSE; 100 rows = 44 %; 1,000 rows = 58 % | The RMSE is decided by a handful of records |
| MSE decomposition by regime (v11) | Normal 55 %, tail > 1 h 32 % (25 % in two CDG rows), Rome 8 %, block = sched 5.6 % | The floor is the normal regime's noise; the two CDG rows (84,240 and 58,206 s, no signature) are unpredictable |
| "+1 day" bug | 14 rows in 2025, 12 at Rome; block stamped with the scheduled date and the actual next-day time | A single such row is worth 21,500 of MSE. Rome branch (v11) |
| Rome subset (unmatched, next-day takeoff) | 168 flights/year; delay > 12 h: 11 "+1 day", 16 "block = sched", 0 normal | Three-regime expectation by delay bucket |
| Local vs public (v8) | Constant 31-36 s offset across four gamma values | Local gains transfer, except those relying on bug frequencies (v11) |

## 3. Hypotheses tested to improve corruption detection

The classifier stayed at AUC 0.88 and 11 % recall at p > 0.5 across all
versions. Tested signatures:

| Signature | Result | Kept |
|---|---|---|
| Operator, stand, hour (target encoding) | Used (top features) but no recall gain | Yes (v5.1) |
| Arrival context (taxi-in, congestion) | Helps the normal regime, not detection | Yes (v9) |
| Reporting outage (corrupted arrivals within the hour) | Correlation 0.34; 4 % → 23-30 % departure corruption when > 30 % of arrivals are corrupted, but 0.3 % of flights concerned | Yes (v10) |
| `MVT_ID` out of temporal sequence | Identical distributions corrupted / normal | No |
| Previous arrival at the same stand corrupted | 14.6 % vs 4.5 %; previous AND next: 38.5 % (3,000 flights/year) | Yes (v13) |
| Flight number (target encoding) | H1 → H2 persistence 0.815; 37 numbers > 80 % (precision 94 %) but numbers > 30 % carry only 10 % of corrupted rows | Yes (v13) |
| Date rollover (takeoff after local midnight) | 7.1 % of "+1 day" among unmatched LIRF rollovers; full signature = Rome + unmatched + next-day takeoff | Yes (v11), then removed (3.2) |
| Format fingerprints (round seconds, microseconds, field lengths, flight rules) | All ratios ~1 | No |
| Missing airport fields (`FLIGHT_mvt`, `STAND`, `RUNWAY`) | < 0.1 % of rows | No |
| Same-day duplicate flight numbers | Inconclusive | No |
| Turnaround ≈ taxi (block copied from the in-block) | 1 row out of 141 | No |
| OPDI (EUROCONTROL ADS-B): `first_seen`, `entry-taxiway` | Coverage 16 % (0.2 % at Istanbul, 1.6 % at Madrid); `first_seen` trails the off-block by 700-1,300 s; naive RMSE 1,330 s vs 370 for AOBT_3; weight 0.01 in combination | No |
| Airport/NM gap on arrivals as a contemporaneous system measure | `ARVT_3` is a landing time, not an in-block: untestable | No |

Conclusion: the corruption is mostly diffuse and unpredictable with the
provided data. The remaining 90 % of corrupted rows sit in flight numbers,
operators and stands with 2-10 % rates, without a signature.

### 3.1 What did not work on the normal regime

| Idea | Result |
|---|---|
| Single L2 model on all rows (v6) | 472.8 s vs 379.1: the mixture structure is necessary |
| Huber loss for the no-AOBT_3 regressor (v5) | Degrades (3,141 vs 3,047) |
| Body/tail decomposition with tail expectation (v6, v7) | Neutral, then a phantom gain that did not transfer |
| Per-runway queue at takeoff (v12) | No feature in the top 15 |
| PRU unimpeded time (v14) | Residual 226.1 → 225.4 s |
| Off-block queue, N / Q / recent taxi (v15) | Residual 225.4 → 224.6 s, no feature in the top 15 |
| Weather (METAR) | Not tested: three independent papers (Haneda, Zurich, JFK) find a null effect outside extreme events |

### 3.2 The Rome branch: a local gain that does not transfer

On the Rome subset (unmatched, next-day takeoff), the branch predicts the
expectation `r_+1d · (86,400 + mu) + r_bs · delta_sched + r_normal · mu` with
rates estimated on 2025 by delay bucket.

- Local: +19.8 s (v11), gain present in January (+14.9) and July (+23.8).
- Public: v10 without Rome 337.95, v11 with Rome 340.40, i.e. **−2.45 s**.
- The subset exists in 2026 (52 flights vs 46 in validation) but its
  composition changed: fewer bugs, more normal flights; the 2025-calibrated
  expectation overshoots. Arrival corruption rates are stable in 2026
  (ratios 0.88-1.19), so this is not a regime change: with ~12 events a
  year, a two-month window is a Poisson draw.

Lesson: modelling a data bug from its past frequency assumes the producing
system is statistically stationary at the evaluation scale. Rare bugs make
such corrections positive in expectation but undecidable over two months.
The month-transfer control on 2025 was not enough to detect it; only the
leaderboard was.

## 4. Takeaways for the paper

1. **The floor is a data floor, not a model floor.** The 226 s residual on
   the normal regime is dominated by the definition gap between airport and
   NM off-blocks (per-airport sigma 250-500 s). The literature reaches
   75-90 s RMSE on cleanly measured physical taxiing (Ravizza 2014, Kato
   2022); the difference is measurement noise, not physics.
2. **When the NM-side duration is observed, classic congestion features are
   screened off.** The literature's top predictor (queue at pushback, Idris
   2001) brings nothing here.
3. **A handful of records decides the RMSE.** Two rows carry 25 % of the
   validation MSE. The public ranking is noisy by construction.
4. **Block-time corruption is a diffuse system-level event**, weakly
   associated with handler, stand and aircraft, persistent by flight number
   for a fringe only, and unpredictable beyond AUC 0.88 with sixteen tested
   signatures.
5. **Bugs are not stationary at the two-month scale.** The Rome correction,
   validated by month transfer within 2025, loses on 2026.
6. **Local validation is faithful when it measures physical properties**
   (parallel gamma curve, constant offset) **and misleading when it measures
   information-system properties** (Rome branch).

## 5. Public scores (leaderboard)

| Submission | Content | Score |
|---|---|---|
| v1 | v5.1 pipeline, gamma = 0.6 | 347.80 |
| v2 | gamma = 0.5 | 358.85 |
| v3 | gamma = 0.75 | 347.59 |
| v4 | gamma = 1.0 | 355.16 |
| v7 | v11 pipeline, with Rome | 341.33 |
| v10 | v13 pipeline, no Rome | **337.95** |
| v11 | v13 pipeline, with Rome | 340.40 |
| v15 | v15 pipeline, with Rome | 338.02 |
| v14 | v15 pipeline, no Rome | to submit (expected ~335) |

Leaderboard head: 246.4 (1st), 260.9 (2nd). The ~80 s gap is not explained
by anything measurable in the provided data (see 4).
