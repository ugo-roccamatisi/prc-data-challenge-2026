# Plan du papier JOAS (format article, pré-print comme les équipes 2025)

**Titre de travail :** What Is a Taxi-Out Time? Data Floors, Silent
Substitutions and the Limits of Prediction in the PRC Data Challenge 2026

**Angle :** pas un rapport de méthode gagnante, mais une étude de ce qui est
prédictible et de ce qui ne l'est pas dans un jeu de données opérationnel
réel. La valeur du papier est dans les résultats négatifs mesurés proprement
et dans la caractérisation des artefacts, choses que les praticiens (PRU,
aéroports, compagnies) rencontrent tous les jours et publient rarement.

Longueur cible : 10-14 pages, gabarit JOAS (voir les pré-prints 2025 comme
modèle de ton et de structure).

---

## 1. Introduction (1 page)

- Le taxi-out comme métrique d'efficacité (fuel, CO2, A-CDM) ; littérature
  atteint 75-90 s de RMSE sur le roulage physique (Ravizza 2014, Kato 2022).
- Le challenge : prédire TAXITIME sur jan/juil 2026, 10 aéroports, RMSE.
- Thèse du papier (les trois findings du README).
- Annonce du plan.

## 2. Data (1,5 page)

- Les deux sources (mouvements aéroport, plans de vol NM) et la jointure.
- **Figure 1** : biais médian horaire BLOCK − AOBT_3 par aéroport (le tableau
  de la v3 : −118 s Rome → +296 s Istanbul, avec écarts-types). Message :
  deux définitions d'off-block coexistent.
- Ce qui est masqué dans le ranking et ce qui ne l'est pas (AOBT_3 visible :
  conséquence majeure, section 5).

## 3. Anatomie de la cible : quand TAXITIME n'est pas un taxi (2,5 pages)

- 3.1 Substitution par l'horaire : définition (|BLOCK − SCHED| ≤ 10 s),
  4,80 %, TAXITIME = MVT − SCHED à 3 s près. **Figure 2** : taux par
  aéroport (Rome 18 %). Confirmation des organisateurs (kept as-is).
- 3.2 Le bug de changement de date à LIRF : mécanisme (date programmée +
  heure réelle), 12-14 cas/an, **Table 1** : les 13 cas de 2025.
- 3.3 Les inclassables : 141 taxi > 2 h ni l'un ni l'autre ; les deux lignes
  de CDG. **Figure 3** : décomposition du MSE de validation par régime
  (55 / 32 / 8 / 5,6 %) et courbe de concentration (10 lignes = 30 %).

## 4. Méthode (2 pages)

- Le mélange par régimes, l'argument espérance conditionnelle, le résidu
  sur AOBT_3.
- Le protocole : validation jan+juil, contrôle de transfert entre mois,
  rounds figés, ensembling. Calibration au leaderboard : **Figure 4** (la
  courbe gamma locale vs publique, décalage constant 32 s) — la validation
  locale est un proxy fiable pour les propriétés physiques.
- Tableau 2 : ablation (reprendre versions.md, colonnes local/public).

## 5. Ce qui ne marche pas, et pourquoi c'est intéressant (3 pages)

- 5.1 La détection de la substitution plafonne : 16 signatures, AUC 0,88,
  rappel 11 %. **Table 3** : signatures et verdicts (condensé de
  versions.md §3). Persistance par numéro de vol (0,815) mais corruption
  majoritairement diffuse.
- 5.2 Les features de congestion sont éteintes par l'observation d'AOBT_3 :
  N/Q/taxi récent (Idris, Ravizza) sans effet — screening off. Discussion :
  la littérature prédit avant le push-back ; le challenge « prédit » après
  le décollage.
- 5.3 L'ADS-B ouvert ne fournit pas d'off-block : OPDI, couverture sol,
  first_seen à +700/1300 s. **Table 4** : appariement et RMSE par aéroport.
- 5.4 La non-stationnarité des bugs : la branche Rome, +19,8 s local
  (transfert entre mois positif !), −2,45 s public. Taux de corruption des
  arrivées 2026 ≈ 2025 (**Table 5**) : ce n'est pas un changement de
  régime, c'est du Poisson sur 12 événements/an. Leçon générale pour la
  correction de données opérationnelles.

## 6. Discussion (1,5 page)

- Le plancher : bruit de définition ≈ 226 s de résidu ; comparaison aux
  75-90 s de la littérature sur du roulage physique mesuré proprement.
- Implications pour la PRU et les aéroports : la qualité du champ off-block
  (et un flag de substitution) vaudrait plus que tout modèle.
- Implications pour les challenges : métrique RMSE + queues rares = loterie
  sur quelques lignes ; suggestion (RMSE tronqué ou score par régime).
- Limites : deux mois de test, pas d'accès aux systèmes des aéroports.

## 7. Conclusion (0,5 page)

## Reproductibilité

- Lien du dépôt, données du challenge uniquement, seeds fixées.

---

## Matériel déjà disponible

| Élément | Source |
|---|---|
| Fig. 1 (biais par aéroport) | diagnostic v3 (notebook) |
| Fig. 2 (taux substitution) | diagnostic v4 |
| Tab. 1 (cas +1 jour) | diagnostic v11 |
| Fig. 3 (décomposition MSE) | val_preds_v15.parquet |
| Fig. 4 (gamma local/public) | scores v1-v4 + validation v8 |
| Tab. 2 (ablation) | results/versions.md §1 et §5 |
| Tab. 3 (signatures) | results/versions.md §3 |
| Tab. 4 (OPDI) | sortie opdi_diag.py |
| Tab. 5 (stabilité 2026) | diagnostic piste 2 (arrivées du ranking) |

## Calendrier proposé (3 semaines)

- S1 : figures et tables (tout est dans les caches et les sorties), sections
  2-3.
- S2 : sections 4-5, repo public finalisé et lié.
- S3 : intro, discussion, relectures (une par Fakhre Ali si possible),
  dépôt du pré-print et soumission finale du challenge.
