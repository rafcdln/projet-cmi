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
├── assets/                 # Ressources statiques globales
├── data/                   # Données (CSV des météorites)
├── docs/                   # Documentation détaillée
├── models/                 # Modèles ML entraînés et sauvegardés
├── src/                    # Code source
│   ├── assets/            # Ressources statiques (CSS, images)
│   ├── callbacks/         # Callbacks Dash pour chaque page
│   ├── components/        # Composants réutilisables de l'interface
│   │   ├── controller.py  # Logique de contrôle principale
│   │   └── view.py        # Éléments d'interface réutilisables
│   ├── models/           # Modèles de données et ML
│   │   ├── ml_model.py    # Modèle de machine learning
│   │   └── model.py       # Modèle de données
│   ├── pages/            # Pages du dashboard multi-pages
│   │   ├── home.py        # Page d'accueil
│   │   ├── map.py         # Page de carte mondiale
│   │   ├── analysis.py    # Page d'analyse de données
│   │   └── predictions.py # Page de prédictions
│   ├── utils/            # Utilitaires
│   │   └── config.py      # Configuration globale
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

   Ouvrez votre navigateur à l'adresse: [http://127.0.0.1:8054/](http://127.0.0.1:8054/)

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
1. Créez ou modifiez une page dans `src/pages/`
2. Ajoutez les callbacks correspondants dans `src/callbacks/`
3. Pour les composants réutilisables, modifiez `src/components/view.py`
4. Pour la logique commune, modifiez `src/components/controller.py`
5. Si nécessaire, ajoutez ou modifiez les modèles dans `src/models/`
6. Mettez à jour la configuration dans `src/utils/config.py`
7. Les modèles ML entraînés sont sauvegardés dans le dossier `models/` à la racine du projet

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## Contact

Pour toute question ou suggestion concernant ce projet, veuillez contacter:
- email@exemple.com

## Documentation

- [Rapport détaillé sur le fonctionnement de la section prédiction](docs/section_prediction.md)

---

© 2025 Dashboard Météorites | Développé avec ❤️ et Python
