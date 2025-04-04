# Rapport sur le Fonctionnement de la Section Prédiction

## 1. Vue d'ensemble

La section prédiction est un module sophistiqué qui permet de prédire la probabilité de chute de météorites dans une zone donnée. Elle combine plusieurs composants interactifs et utilise des modèles statistiques avancés pour fournir des prédictions précises.

## 2. Interface Utilisateur

### 2.1 Paramètres Avancés (Sidebar)

La sidebar contient les paramètres avancés qui permettent d'affiner les prédictions :

- **Horizon de prévision** (1-50 ans)
  - Permet de définir la période future pour laquelle on souhaite obtenir des prédictions
  - Impact la fiabilité des prédictions (plus l'horizon est long, moins la prédiction est fiable)

- **Rayon d'analyse** (0.5°-10°)
  - Définit la zone circulaire autour du point sélectionné pour l'analyse
  - Influence la quantité de données historiques prises en compte

- **Type de détection**
  - Found (météorites trouvées)
  - Fell (chutes observées)
  - Affecte les probabilités et la fiabilité des prédictions

- **Facteur environnemental** (0-1)
  - Ajuste l'influence des facteurs environnementaux
  - Impact l'estimation des probabilités

- **Poids des données historiques** (0-1)
  - Détermine l'importance accordée aux données anciennes
  - Équilibre entre données historiques et tendances récentes

- **Complexité du modèle** (1-10)
  - Ajuste la sophistication des algorithmes de prédiction
  - Compromis entre précision et risque de surajustement

### 2.2 Carte Interactive

- Permet de sélectionner un point pour l'analyse
- Affiche les météorites existantes avec une faible opacité
- Visualise le rayon d'analyse sélectionné

## 3. Fonctionnalités de Prédiction

### 3.1 Prédiction de Base

Le système calcule :
- Probabilité d'impact
- Masse estimée
- Classification probable
- Indice de fiabilité

### 3.2 Analyses Temporelles

Fournit des prédictions sur différentes périodes :
- Court terme (1/3 de l'horizon)
- Moyen terme (2/3 de l'horizon)
- Long terme (horizon complet)

Pour chaque catégorie de masse :
- Petite (10g - 1000g)
- Moyenne (1kg - 100kg)
- Grande (100kg - 10t)

### 3.3 Analyses Spatiales

Génère des cartes de probabilité basées sur :
- Densité locale de météorites
- Facteurs environnementaux
- Patterns historiques

## 4. Algorithmes et Calculs

### 4.1 Calcul des Probabilités

Les probabilités sont calculées en tenant compte de :
1. Données historiques locales
2. Densité régionale vs globale
3. Type de chute (Found/Fell)
4. Facteurs temporels
5. Facteurs environnementaux

### 4.2 Estimation des Masses

Utilise :
- Moyenne géométrique des masses locales
- Ajustement selon les tendances temporelles
- Facteurs environnementaux

### 4.3 Indice de Fiabilité

Calculé en fonction de :
- Proximité des données historiques
- Densité des données dans la zone
- Horizon temporel
- Qualité des données disponibles

## 5. Visualisations

### 5.1 Graphiques Principaux

- Carte de prédiction interactive
- Graphique d'évolution temporelle
- Distribution des masses
- Comparaison régionale

### 5.2 Analyses Supplémentaires

- Évolution des classifications
- Carte de densité
- Facteurs d'influence
- Tendances saisonnières

## 6. Mise à Jour en Temps Réel

Le système offre une option de mise à jour en temps réel qui permet :
- Actualisation automatique des prédictions
- Mise à jour dynamique des visualisations
- Réponse immédiate aux changements de paramètres

## 7. Limitations et Considérations

- La fiabilité diminue avec l'augmentation de l'horizon temporel
- Les prédictions sont plus précises dans les zones avec plus de données historiques
- Les facteurs environnementaux peuvent avoir un impact significatif
- La complexité du modèle doit être ajustée selon la qualité des données disponibles 