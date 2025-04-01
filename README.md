# Dashboard Analyse de Météorites

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Dash](https://img.shields.io/badge/Dash-2.11.1-blue)
![Plotly](https://img.shields.io/badge/Plotly-5.14.1-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0.0-blue)
![Status](https://img.shields.io/badge/Status-En%20développement-orange)

Application web interactive pour l'analyse et la visualisation de données sur les météorites terrestres. Ce dashboard permet d'explorer les impacts de météorites enregistrés dans le monde, avec des outils de filtrage, d'analyse statistique et de prédiction.

## Fonctionnalités

- **Carte mondiale interactive** : Visualisation de tous les impacts de météorites avec filtrage dynamique
- **Multiples modes de visualisation** :
  - Carte de points colorés par classe, masse, type de chute ou année
  - Carte de chaleur de densité d'impacts
  - Différents styles de cartes (sombre, standard, satellite, rues)
- **Analyses statistiques avancées** :
  - Distribution des masses (histogramme)
  - Évolution temporelle des découvertes par décennie
  - Distribution des classes de météorites (graphique circulaire)
  - Matrice de corrélation entre variables numériques
- **Outils de prédiction** :
  - Prédiction de masse potentielle par coordonnées géographiques
  - Analyse de zone pour comparer avec les météorites historiques
- **Interface utilisateur moderne** :
  - Design inspiré d'Apple avec une expérience utilisateur fluide
  - Filtres interactifs et mise à jour dynamique des graphiques
  - Affichage responsive adapté à différentes tailles d'écran

## Structure du Projet

```
dashboard/
├── data/                   # Données (CSV des météorites)
├── src/                    # Code source
│   ├── assets/            # Ressources statiques (CSS, images)
│   ├── components/        # Composants de l'interface
│   ├── models/           # Modèles de données et ML
│   ├── utils/            # Utilitaires
│   ├── app.py           # Point d'entrée de l'application
│   └── __init__.py      # Fichier d'initialisation du package
├── venv/                  # Environnement virtuel Python
├── run.py                # Script de lancement à la racine du projet
├── requirements.txt      # Dépendances Python
└── README.md            # Documentation principale
```

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/yourusername/dashboard-meteorites.git
   cd dashboard-meteorites
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   
   # Sur Windows
   venv\Scripts\activate
   
   # Sur macOS/Linux
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer l'application**
   ```bash
   python run.py
   ```

   Ou alternativement:
   ```bash
   python src/app.py
   ```

5. **Accéder au dashboard**
   
   Ouvrez votre navigateur à l'adresse: [http://127.0.0.1:8051/](http://127.0.0.1:8051/)

## Utilisation

### Filtrage des Données

- Utilisez les curseurs et menus déroulants dans la barre latérale pour filtrer les données par:
  - Plage de masse (échelle logarithmique)
  - Classe de météorite
  - Type de chute (observée ou trouvée)
  - Période de découverte

### Navigation sur la Carte

- Faites défiler pour zoomer, cliquez et faites glisser pour naviguer
- Basculez entre la vue des points et la carte de chaleur
- Changez le style de carte et le mode de coloration des points

### Prédiction

1. Cliquez sur un point de la carte dans la section "Prédiction de Météorites"
2. Appuyez sur "Prédire" pour obtenir une estimation de la masse
3. Ou appuyez sur "Analyser la zone" pour obtenir des statistiques sur les météorites connues dans cette région

## Données

Le jeu de données utilisé provient de NASA's Open Data Portal et contient des informations sur plus de 45,000 météorites tombées sur Terre.

Chaque entrée comprend:
- Nom et ID
- Classe de météorite
- Masse (en grammes)
- Année de découverte/chute
- Coordonnées de localisation
- Type de découverte (chute observée ou trouvaille)

## Développement

### Prérequis Techniques

- Python 3.8 ou supérieur
- Connaissance de Dash/Plotly
- Compréhension de base des données géospatiales
- Notions de machine learning pour la partie prédiction

### Extension du Projet

Pour ajouter de nouvelles fonctionnalités:
1. Modifiez `src/components/view.py` pour l'interface utilisateur
2. Implémentez la logique correspondante dans `src/components/controller.py`
3. Si nécessaire, ajoutez des modèles dans `src/models/`
4. Mettez à jour la configuration dans `src/utils/config.py`

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## Contact

Pour toute question ou suggestion concernant ce projet, veuillez contacter:
- email@exemple.com

---

© 2023 Dashboard Météorites | Développé avec ❤️ et Python
