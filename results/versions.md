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
| v16 | Mixture weight recalibration `w = f(p, log(s − n))` | transfer FAILS (366 → 409 s Jan→Jul) | not submitted | The earlier quick test excluded the Rome subset; applied to all rows, the calibrator blows up on extreme-gap flights. Third illustration of the tail-learning trap, caught by the internal transfer check |
| v17 | Out-of-fold residual corrector (5 folds, Huber, clipped residuals, lambda swept in-train) | +0.7 | 336.42 (v18 ref) / 335.78 (v19) | Converges in 11 trees: the first stage already extracts everything. The +0.7 s transfers exactly to public |
| v30 | Re-engineered pipeline (see 3.3) | selected on Nov-Dec | **329.41** | Label masking before feature building, per-second early stopping on Nov-Dec, plausibility-gated AOBT baseline, clock/rollover features, hard [30, 7200] clip |

\* Validation on filtered data (taxi > 7,200 s removed), not comparable with later versions.

Best production version of the first campaign: **v30**
(`src/prc_experiment.py`, submission `motivated-ostrich_v31`, public
327.70 s with the reconstruction rule of section 6, 329.41 s without). The
second campaign (section 6) supersedes it at 319.21 s.

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
| Mixture weight recalibration (v16) | Apparent +7.6 s gain on the Rome-free population; −42 s once tails are included. Rejected by the internal transfer check |
| Out-of-fold residual corrector (v17) | +0.7 s local and public. 11 trees at early stopping: residuals are not a function of the features |

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

### 3.3 The v30 re-engineering: +6.4 s public from protocol, not modelling

`src/prc_experiment.py` keeps the exact v16
architecture (two-regime mixture, p^0.75, A/B regressors, out-of-fold
encodings) and changes the engineering around it:

- **Departure labels are masked before any contextual feature is built**
  (`TAXITIME` and `BLOCK_TIME` set to NaN on departures), so training
  features are computed under exactly the ranking's conditions.
- **Early stopping measures the final RMSE in seconds** (custom feval
  applying the offset and clipping), per component, on a dedicated
  **November-December selection split**; January + July become a frozen,
  untouched control. The blend choice (13 predefined configurations, with
  an eligibility rule: ≥ 1 s gain and no month degraded) is locked before
  the control is ever scored.
- **Plausibility-gated baseline**: the AOBT_3 offset is only used when
  `0 ≤ MVT − AOBT_3 ≤ 7200 s`; otherwise the unimpeded p50 takes over.
  Rows with day-scale AOBT deltas no longer poison the residual regressor.
- **Clock features**: per-source time-of-day encodings, cross-source
  deltas (AOBT/LOBT/IOBT/EOBT, with wrapped variants) and **day-gap
  features** that capture date rollovers as a feature rather than a
  hand-made branch.
- **Hard [30, 7200] s clip** on the normal branch.

Public: 335.78 (v19) → **329.41** (v30). Several changes were introduced
together, so the 6.4 s are not individually attributable; the plausibility
gate and the day-gap features are the most likely main contributors, as
they target exactly the tail rows identified in the MSE decomposition.

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
| v14 | v15 pipeline, no Rome, 5 seeds | 336.13 |
| v18 | v17 reference (1 seed) | 336.42 |
| v19 | v17 corrector (lambda 0.75) | 335.78 |
| v6 | v10 pipeline, no Rome | 338.06 |
| v30/v31 | re-engineered pipeline (3.3) | **329.41** |

Consistency checks these scores provide: the corrector's +0.7 s local gain
transfers exactly (v19 − v18 = 0.64 s); 5-seed ensembling is worth only
0.3 s publicly (v14 vs v18); the Rome branch costs 1.9 s on the v15
pipeline too (v14 vs v15), confirming v10/v11.

Leaderboard head at the time of writing (16 Sep 2026): 241.75 (1st),
244.83 (2nd), 264.20 (10th). See section 6 for the second campaign and
section 7 for the updated reading of the remaining gap.

## 6. Second campaign (14 to 16 September 2026)

The first campaign optimised a two-regime mixture through feature
engineering. The second merged `pipeline.py` (v16) and
`stacking_experiment.py` (v17) into a single file, `src/taxiout_v19.py`, and
attacked the problem from the architecture side instead. Submission ids below
are the actual `motivated-ostrich_vNN.parquet` files, which is the only
unambiguous key; the first campaign's internal version names (v2 to v30) are
unrelated to them.

### 6.1 Public scores

| Submission | Configuration | Public |
|---|---|---|
| v31 | first-campaign v30 plus the reconstruction rule | 327.70 |
| v19 | merged pipeline, no rule, no new features | 338.37 |
| v21 | v19 plus reconstruction rule (hard mode) | 330.42 |
| v22 | v21 plus posterior sequence features | 326.80 |
| v23 | v22 plus double capacity, lean classifier, 3 seeds | 325.15 |
| v25 | v23 plus per-airport models, global classifier, CatBoost | 320.10 |
| v26 | v25 with CatBoost weight 0.75 instead of 0.5 | **319.21** |

Total: 19.16 s over three days, of which 12.4 s from architecture and
7.95 s from the reconstruction rule.

### 6.2 The reconstruction rule

For movements never matched to the NM flight list (`FLIGHT_ID_mvt` null, all
`_flt` columns null, `AIRCRAFT_TYPE_mvt` also null), the airport filled
`BLOCK_TIME_UTC_mvt` with `SCHED_TIME_UTC_mvt`. The label therefore satisfies
`TAXITIME = MVT_TIME - SCHED_TIME` to within a few seconds. Both columns are
present in the ranking file, so the label is recomputable rather than
predictable.

Derivation, entirely on 2025 (`src/prc_taxiout.py defaults` then `rule`):

- 76 labels above 20,000 s in 2025; 58 of them satisfy the identity within
  120 s, 72 are at LIRF.
- On those rows `y - d_sched` has median 1 s and interquartile range 6 s.
- A naive threshold on `d_sched` alone is useless: at 20,000 s it yields 245
  candidates for 57 true positives, and applying it costs more than it gains.
- Adding the two missing-field conditions raises precision from 0.23 to 0.85
  at the same threshold. Evaluated against the model's own predictions on a
  January plus July 2025 holdout, hard substitution at a 10,800 s threshold
  restricted to LIRF reduced the total RMSE from 472.74 to 371.23 s.
- The 2026 ranking file contains 74 candidates, all at LIRF, against 24 in
  the equivalent 2025 window: the defect transfers.
- Measured on the leaderboard: 338.37 to 330.42 s, i.e. 7.95 s. Applied to
  the first-campaign pipeline it is worth 1.71 s (329.41 to 327.70), which is
  consistent: that pipeline's mixture already handled part of those rows.

The remaining 18 aberrant labels belong to the date-rollover family and are
*not* reconstructible: `y - 86,400` matches a planned delta on 1 row out of
18, and `d_sched` on those rows ranges from 939 to 93,535 s. The block time
carries the correct time of day and the wrong date, which explains the values
after the fact but identifies nothing beforehand
(`src/prc_taxiout.py dayshift`).

### 6.3 Posterior sequence features (non-causal)

The task is retrospective reconstruction, and the ranking file contains every
movement of January and July 2026, including departures that took off after
the target flight. Forty-one features were built from that: gaps to the next
1, 2, 3, 5, 10 and 20 takeoffs on the same runway, forward density windows,
full burst treatment (position in the burst, burst length, takeoffs remaining
in it), throughput normalised by the runway's monthly median, and NM clocks of
the following flights, in particular the number of the next ten departures
that had *already* pushed back when the target took off, which is the queue
behind it.

Holdout: clean-lane RMSE 274.83 to 271.54 s, global 370.66 to 368.89 s, gain
spread over eight airports out of ten. Public: 330.42 to 326.80 s, i.e.
3.62 s.

This refines finding 2 of section 4 rather than contradicting it. Congestion
measured *before* the flight is screened off by the observed AOBT_3, which
v12 and v15 established and which the demand-queue features of section 6.5
reconfirmed. Congestion measured *after* it is not. These features have no
operational meaning and only make sense in a reconstruction setting.

### 6.4 Architecture

| Change | Holdout (global / clean) | Verdict |
|---|---|---|
| Reference (posterior features, 1 seed) | 368.89 / 271.54 | |
| Capacity x2 (1,160 trees for regressor A) | 366.43 / 269.80 | Kept, 2.5 s |
| Capacity x4 | 368.29 / 269.88 | Past the optimum |
| Capacity x8 | 375.69 / 271.14 | Clearly past it |
| Three-seed average, equal weights | 364.40 / 268.53 | Kept, 2.0 s |
| Per-airport models (10 dedicated plus fallback) | 373.21 / 264.51 | Clean lane improves, aberrant rows degrade |
| Per-airport plus global classifier | 364.12 / 267.84 | Kept: the local classifier overfits Rome's fifty corrupted labels |
| CatBoost/LightGBM ensemble | 360.07 / 264.11 | Kept, 3.9 s |
| All three together | 358.82 / 260.60 | Best of the campaign |
| CatBoost weight sweep (free, no retraining) | 0.75 optimal: 356.60 / 259.19 | Kept, 0.7 s |

Two readings worth recording. Setting the ensemble weight to 0 leaves 362.89 s
and to 1 leaves 357.38 s: on this problem CatBoost alone beats LightGBM
alone, it is not merely complementary. And the per-airport split only pays
once the corruption classifier is kept global, because a classifier dedicated
to LIRF sees its fifty corrupted labels in a far higher proportion than the
global one, which drowns them in 1.7 million rows.

### 6.5 Refuted in the second campaign

| Idea | Holdout effect | Verdict |
|---|---|---|
| Demand queue anchored on SCHED and EOBT (6 features) | -1.6 s | Refuted. Reconfirms v15: the AOBT-anchored queue already covers it |
| Stand-neighbour, airport-level and arrival context (34 features) | -3.0 s global | Refuted |
| Degraded-day detection (arrival taxi-in over its hourly norm) | 0.0 s | Refuted. The signal does not exist in taxi-in: a de-icing aircraft delays itself, not the arrivals |
| Removing `month` from the features | +5.2 s holdout | Not adopted. Another team reported the same change at +43 s on cross-validation and -7.6 s on the leaderboard; our effect is within seed noise |
| Upper clipping of predictions | Degrades at every threshold from 1,800 to 7,200 s | Refuted: the extreme labels are real |
| Constant per-airport mixture on the unmatched lane | -14 s | Refuted: the model already carries row-level information that a group mean destroys |
| Residual corrector (first campaign v17) at double capacity | lambda drops to 0 | Superseded: capacity and seed averaging absorb what it used to recover |

### 6.6 Measurement noise

Three seeds of a single identical configuration produced 366.43, 366.17 and
363.52 s, a spread of 2.9 s. This was measured late, and it retrospectively
disqualifies several single-seed comparisons of both campaigns, including
four of the refutations above whose magnitude sits inside that band. The
changes that survive the noise floor are the reconstruction rule (7.95 s
public), the posterior features (3.62 s public) and the architecture block
(5.05 s public), all three confirmed on the leaderboard rather than on the
holdout.

## 7. Where the remaining gap is

With the final configuration, the validation MSE splits as follows: 26 rows
out of 344,227 carry 58 % of it, and the clean lane carries the rest at
260 s of RMSE, with a median absolute error of 100 s, a 90th percentile of
306 s and a 99th percentile of 825 s. Per airport, clean lane: LIRF 557,
EGLL 267, LFPG 254, LTFM 247, LEBL 218, LSZH 206, EHAM 189, EDDF 183,
EDDM 179, LEMD 174.

Two consequences. The aberrant rows are in everyone's ranking set, so the
leader at 241.75 s subtracts them too: their clean-lane RMSE has to be
around 200 s against our 260. The gap is therefore in the ordinary taxi, not
in the artifacts. And LFPG, which looked like a problem airport at 577 s,
falls to 254 s once two rows are removed, whereas LIRF stays at 557 s with
structured estimators doing no better there than a constant.
