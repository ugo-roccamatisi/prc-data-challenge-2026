# PRC Data Challenge 2026 : journal des tentatives et résultats

Équipe `motivated-ostrich`. Prédiction du taxi-out (secondes) pour les départs de dix aéroports européens, janvier et juillet 2026, métrique RMSE.

Validation locale : entraînement sur 10 mois de 2025, validation sur janvier et juillet 2025 (344 227 départs), pour mimer la saisonnalité du ranking. Les RMSE locaux ne sont comparables entre eux qu'à partir de la v4 (voir note en 1.3).

## 1. Chronologie des versions

| Version | Idée testée | RMSE local | Public | Verdict |
|---|---|---|---|---|
| v2 | LightGBM direct, features de base + deltas NM + congestion aéroport | 254,0* | | Baseline fonctionnelle |
| v3 | Résidu sur AOBT_3 + nettoyage des appariements aberrants | 274,1* puis 252,9* | | Le nettoyage dégrade (les lignes retirées sont dans la vérité terrain). Sans nettoyage, gain de 1 s |
| v4 | Découverte des lignes "block = sched" ; classifieur + mélange probabiliste ; réintégration des taxi longs | 379,8 | | **Gain de 29 %** (533,1 sans mélange). Première mesure honnête, taxi longs inclus |
| v5 | Target encoding hors-fold, scale_pos_weight | (cassé) | | Early stopping à l'itération 7 : pondération incompatible avec la métrique d'arrêt |
| v5.1 | Idem sans pondération, arrêt sur AUC, exposants par aéroport | 375,4 | 347,80 (v1) | Classifieur réparé mais rappel inchangé (11 %). Gain de 4 s |
| v6 | Comparatif : modèle unique vs mélange vs corps/queue | 472,8 / 379,1 / 379,4 | | **Modèle unique réfuté** (+100 s). Décomposition corps/queue neutre |
| v7 | Balayage joint (γ_c, γ_t, κ) avec contrôle de transfert janvier/juillet | 375,9 mais rejeté | | Le gain de 3,5 s ne survit pas au transfert (−18 s d'un mois à l'autre). Retour ARCH 0 |
| v7.1 | Correctif : rounds figés au réentraînement (bug : early stopping impossible sur train = val) | 379,4 | 341,33 (v7) | Pipeline propre |
| v8 | Calibration de γ_c au leaderboard (0,5 / 0,6 / 0,75 / 1,0) | 389,8 / 379,4 / 380,1 / 391,0 | 358,85 / 347,80 / **347,59** / 355,16 | Courbe publique parallèle à la locale, décalage constant ~32 s : **la validation locale est un proxy fiable** |
| v9 | Contexte arrivées (taxi-in autour du décollage) + enchaînement par poste | 371,1 | | +8,3 s, réparti sur tous les aéroports. CDG et Rome inchangés |
| v10 | Signal de panne de reporting (corruption des arrivées dans la fenêtre) | 368,2 | | +2,9 s, Rome −33 s, autres aéroports légèrement dégradés. Signal réel mais rare (0,3 % des vols) |
| v11 | Branche Rome (non apparié, décollage le lendemain) : espérance à trois régimes | 348,4 | 340,40 (v11, pipeline v13) | **+19,8 s local, −2,45 s public**. Voir 3.2 |
| v12 | Ensembling 5 seeds + file d'attente par piste | 346,4 | | +2,0 s, entièrement dû à l'ensembling |
| v13 | Target encoding par numéro de vol + corruption des arrivées liées au poste | 345,2 | **337,95** (v10, sans Rome) | +1,2 s local. Meilleur public à ce jour |
| v14 | Temps non entravé (PRU) par poste-piste, régresseur B sur l'excédent | 345,9 | | Neutre. `unimpeded_p50` 8e feature, mais la géométrie était déjà extraite de `stand_runway` |
| v15 | File d'attente ancrée sur l'off-block (N, Q, taxi récent, Idris/Ravizza) | 345,0 | 338,02 (v15, avec Rome) | Neutre. Aucune feature `q_*` dans le top 15 : la congestion n'apporte rien quand la durée réelle côté NM est connue |

\* Validation sur données filtrées (taxi > 7 200 s retirés), non comparable aux versions suivantes.

Meilleure version de production : **pipeline v15 sans branche Rome** (soumission v14, attendue autour de 335 s public).

## 2. Diagnostics qui ont orienté le travail

| Diagnostic | Résultat | Conséquence |
|---|---|---|
| Naïf `MVT − AOBT_3` | RMSE 380,8 s ; biais médian par aéroport de −118 s (Rome) à +296 s (Istanbul) | Écart de définition systématique entre off-block aéroport et NM : cible = résidu sur AOBT_3 |
| Lignes "block = sched" (`\|BLOCK − SCHED\| ≤ 10 s`) | 4,80 % des départs, 29 % de l'erreur quadratique, taxi médian 1 009 s (normal : 910 s) | Le taxi-out n'est pas un taxi-out sur ces lignes : `TAXITIME = MVT − SCHED` exactement (écart médian 3 s). Mélange probabiliste (v4) |
| Concentration de l'erreur | 10 lignes = 29,6 % du MSE ; 100 lignes = 44 % ; 1 000 lignes = 58 % | Le RMSE est décidé par une poignée d'enregistrements |
| Décomposition du MSE par régime (v11) | Normal 55 %, queue > 1 h 32 % (dont 25 % dans deux lignes de CDG), Rome 8 %, block = sched 5,6 % | Le plancher est le bruit du régime normal ; les deux lignes de CDG (84 240 et 58 206 s, sans signature) sont imprédictibles |
| Bug "+1 jour" | 14 lignes en 2025, 12 à Rome, block estampillé à la date programmée avec l'heure réelle du lendemain | Une seule de ces lignes vaut 21 500 de MSE. Branche Rome (v11) |
| Sous-ensemble Rome (non apparié, décollage le lendemain) | 168 vols/an ; retard > 12 h : 11 "+1 jour", 16 "block = sched", 0 normal | Espérance à trois régimes par tranche de retard |
| Local vs public (v8) | Décalage constant de 31 à 36 s sur quatre valeurs de γ | Les gains locaux se transposent, sauf ceux qui reposent sur la fréquence des bugs (v11) |

## 3. Hypothèses testées pour améliorer la détection de la corruption

Le classifieur est resté à AUC 0,88 et rappel 11 % à p > 0,5 sur toutes les versions. Signatures testées :

| Signature | Résultat | Retenue |
|---|---|---|
| Opérateur, poste, heure (target encoding) | Utilisées (top features) mais aucun gain de rappel | Oui (v5.1) |
| Contexte arrivées (taxi-in, congestion) | Aide le régime normal, pas la détection | Oui (v9) |
| Panne de reporting (arrivées corrompues dans l'heure) | Corrélation 0,34 ; 4 % → 23-30 % de corruption quand > 30 % des arrivées sont corrompues, mais 0,3 % des vols concernés | Oui (v10) |
| `MVT_ID` hors séquence temporelle | Distributions identiques corrompues / normales | Non |
| Arrivée précédente au même poste corrompue | 14,6 % contre 4,5 % ; précédente et suivante : 38,5 % (3 000 vols/an) | Oui (v13) |
| Numéro de vol (target encoding) | Persistance H1 → H2 de 0,815 ; 37 numéros à > 80 % (précision 94 %) mais les numéros à > 30 % ne portent que 10 % des corrompues | Oui (v13) |
| Rollover de date (décollage après minuit local) | 7,1 % de "+1 jour" parmi les rollover LIRF non appariés ; signature complète = Rome + non apparié + décollage le lendemain | Oui (v11), puis retirée (3.2) |
| Empreintes de format (secondes rondes, microsecondes, longueur des champs, règles de vol) | Tous les ratios ~1 | Non |
| Champs aéroport manquants (`FLIGHT_mvt`, `STAND`, `RUNWAY`) | < 0,1 % des lignes | Non |
| Doublons de numéro de vol le même jour | Non concluant | Non |
| Temps de rotation ≈ taxi (block copié de l'in-block) | 1 ligne sur 141 | Non |
| OPDI (ADS-B EUROCONTROL) : `first_seen`, `entry-taxiway` | Couverture 16 % (0,2 % à Istanbul, 1,6 % à Madrid) ; `first_seen` arrive 700 à 1 300 s après l'off-block ; RMSE naïf 1 330 s contre 370 pour AOBT_3 ; poids 0,01 en combinaison | Non |
| Écart aéroport/NM côté arrivées comme mesure contemporaine du système | `ARVT_3` est un atterrissage, pas un in-block : non testable | Non |

Conclusion : la corruption est majoritairement diffuse et non prédictible avec les données fournies. Les 90 % de lignes corrompues restantes sont dans des numéros de vol, opérateurs et postes à 2-10 % de taux, sans signature.

### 3.1 Ce qui n'a pas marché sur le régime normal

| Idée | Résultat |
|---|---|
| Modèle unique L2 sur toutes les lignes (v6) | 472,8 s contre 379,1 : la structure du mélange est nécessaire |
| Loss Huber pour le régresseur sans AOBT_3 (v5) | Dégrade (3 141 contre 3 047) |
| Décomposition corps / queue avec espérance de queue (v6, v7) | Neutre, puis gain fantôme non transféré |
| File d'attente par piste au décollage (v12) | Aucune feature dans le top 15 |
| Temps non entravé PRU (v14) | Résidu 226,1 → 225,4 s |
| File d'attente à l'off-block, N / Q / taxi récent (v15) | Résidu 225,4 → 224,6 s, aucune feature dans le top 15 |
| Météo (METAR) | Non testée : trois papiers indépendants (Haneda, Zurich, JFK) concluent à un effet nul hors événements extrêmes |

### 3.2 La branche Rome : un gain local qui ne se transpose pas

La branche prédit, sur le sous-ensemble Rome non apparié décollant le lendemain, l'espérance `r_+1j · (86 400 + μ) + r_bs · delta_sched + r_normal · μ` avec des taux estimés sur 2025 par tranche de retard.

- Local : +19,8 s (v11), gain présent sur janvier (+14,9) et juillet (+23,8).
- Public : v10 sans Rome 337,95, v11 avec Rome 340,40, soit **−2,45 s**.
- Le sous-ensemble existe en 2026 (52 vols contre 46 en validation) mais sa composition a changé : moins de bugs, plus de vols normaux ; l'espérance calibrée sur 2025 surestime.

Leçon : modéliser une corruption à partir de sa fréquence passée suppose que le système qui la produit est stationnaire. Ici il ne l'est pas. Le contrôle de transfert entre mois de 2025 ne suffit pas à le détecter ; seul le leaderboard l'a révélé.

## 4. Enseignements pour le papier

1. **Le plancher est un plancher de données, pas de modèle.** Le résidu de 226 s sur le régime normal est dominé par l'écart de définition entre l'off-block aéroport et l'off-block NM (écart-type par aéroport de 250 à 500 s). La littérature atteint 75 à 90 s de RMSE sur le roulage physique (Ravizza 2014, Kato 2022) ; la différence est le bruit de mesure, pas la physique.
2. **Quand la durée côté NM est observée, les features de congestion classiques sont sans effet.** La feature numéro un de la littérature (file au push-back, Idris 2001) n'apporte rien ici.
3. **Une poignée d'enregistrements décide du RMSE.** Deux lignes portent 25 % du MSE de validation. Le classement public est bruité par construction.
4. **La corruption des block times est un événement système diffus**, faiblement associé au handler, au poste et à l'avion, persistante par numéro de vol pour une frange seulement, et non prédictible au-delà d'une AUC de 0,88 avec seize signatures testées.
5. **Les bugs ne sont pas stationnaires.** La correction de Rome, validée par transfert entre mois de 2025, perd sur 2026.
6. **La validation locale est fidèle quand elle mesure des propriétés physiques** (courbe de γ parallèle, décalage constant), **et trompeuse quand elle mesure des propriétés du système d'information** (branche Rome).

## 5. Scores publics (leaderboard)

| Soumission | Contenu | Score |
|---|---|---|
| v1 | pipeline v5.1, γ = 0,6 | 347,80 |
| v2 | γ = 0,5 | 358,85 |
| v3 | γ = 0,75 | 347,59 |
| v4 | γ = 1,0 | 355,16 |
| v7 | pipeline v11, avec Rome | 341,33 |
| v10 | pipeline v13, sans Rome | **337,95** |
| v11 | pipeline v13, avec Rome | 340,40 |
| v15 | pipeline v15, avec Rome | 338,02 |
| v14 | pipeline v15, sans Rome | à soumettre (attendu ~335) |

Tête du classement : 246,4 (1er), 260,9 (2e). Écart d'environ 80 s, non expliqué par les données fournies (voir 4).
