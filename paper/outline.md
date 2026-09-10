# JOAS paper plan (article format, pre-print like the 2025 teams)

**Working title:** What Is a Taxi-Out Time? Data Floors, Silent
Substitutions and the Limits of Prediction in the PRC Data Challenge 2026

**Angle:** not a winning-method report, but a study of what is predictable
and what is not in a real operational dataset. The paper's value lies in
cleanly measured negative results and artifact characterisation — things
practitioners (PRU, airports, airlines) meet daily and rarely publish.

Target length: 10-14 pages, JOAS template (use the 2025 pre-prints as a
model for tone and structure).

---

## 1. Introduction (1 page)

- Taxi-out as an efficiency metric (fuel, CO2, A-CDM); literature reaches
  75-90 s RMSE on physical taxiing (Ravizza 2014, Kato 2022).
- The challenge: predict TAXITIME for Jan/Jul 2026, 10 airports, RMSE.
- Thesis of the paper (the three findings of the README).
- Outline.

## 2. Data (1.5 pages)

- The two sources (airport movements, NM flight plans) and the join.
- **Figure 1**: hourly median bias BLOCK − AOBT_3 per airport (v3 table:
  −118 s Rome → +296 s Istanbul, with sigmas). Message: two off-block
  definitions coexist.
- What is masked in the ranking and what is not (AOBT_3 visible: major
  consequence, section 5).

## 3. Anatomy of the target: when TAXITIME is not a taxi (2.5 pages)

- 3.1 Schedule substitution: definition (|BLOCK − SCHED| ≤ 10 s), 4.80 %,
  TAXITIME = MVT − SCHED within 3 s. **Figure 2**: rate per airport (Rome
  18 %). Organisers' confirmation (kept as-is).
- 3.2 The LIRF date-rollover bug: mechanism (scheduled date + actual time),
  12-14 cases/year, **Table 1**: the 13 cases of 2025.
- 3.3 The unclassifiable: 141 taxis > 2 h that are neither; the two CDG
  rows. **Figure 3**: validation MSE decomposition by regime
  (55 / 32 / 8 / 5.6 %) and concentration curve (10 rows = 30 %).

## 4. Method (2 pages)

- The regime mixture, the conditional-expectation argument, the residual
  over AOBT_3.
- The protocol: Jan+Jul validation, month-transfer control, frozen rounds,
  ensembling. Leaderboard calibration: **Figure 4** (local vs public gamma
  curve, constant 32 s offset) — local validation is a reliable proxy for
  physical properties.
- Table 2: ablation (from versions.md, local/public columns).

## 5. What does not work, and why that is interesting (3 pages)

- 5.1 Substitution detection plateaus: 16 signatures, AUC 0.88, recall
  11 %. **Table 3**: signatures and verdicts (condensed from versions.md
  §3). Flight-number persistence (0.815) but mostly diffuse corruption.
- 5.2 Congestion features are screened off by the observed AOBT_3:
  N/Q/recent taxi (Idris, Ravizza) without effect. Discussion: the
  literature predicts before pushback; the challenge "predicts" after
  takeoff.
- 5.3 Open ADS-B provides no off-block: OPDI, ground coverage, first_seen
  at +700/1,300 s. **Table 4**: matching and RMSE per airport.
- 5.4 Bug non-stationarity: the Rome branch, +19.8 s local (positive month
  transfer!), −2.45 s public. 2026 arrival corruption rates ≈ 2025
  (**Table 5**): not a regime change but Poisson sampling over 12
  events/year. General lesson for operational data correction.

## 6. Discussion (1.5 pages)

- The floor: definition noise ≈ 226 s residual; comparison to the 75-90 s
  literature on cleanly measured taxiing.
- Implications for PRU and airports: the quality of the off-block field
  (and a substitution flag) would be worth more than any model.
- Implications for challenges: RMSE + rare tails = lottery over a few rows;
  suggestion (trimmed RMSE or per-regime scoring).
- Limitations: two test months, no access to airport systems.

## 7. Conclusion (0.5 page)

## Reproducibility

- Repository link, challenge data only, fixed seeds.

---

## Material already available

| Item | Source |
|---|---|
| Fig. 1 (per-airport bias) | notebook 01 |
| Fig. 2 (substitution rate) | notebook 01 |
| Tab. 1 (+1 day cases) | v11 diagnostic |
| Fig. 3 (MSE decomposition) | notebook 01 (val_preds_v15.parquet) |
| Fig. 4 (local/public gamma) | notebook 02 |
| Tab. 2 (ablation) | results/versions.md §1 and §5 |
| Tab. 3 (signatures) | results/versions.md §3 |
| Tab. 4 (OPDI) | opdi_diag.py output |
| Tab. 5 (2026 stability) | ranking-arrivals diagnostic |

## Proposed schedule (3 weeks)

- W1: figures and tables (all in caches and outputs), sections 2-3.
- W2: sections 4-5, public repo finalised and linked.
- W3: intro, discussion, proofreading (one pass by Fakhre Ali if possible),
  pre-print deposit and final challenge submission.
