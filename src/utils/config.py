"""
Configuration centralisée pour l'application d'analyse de météorites.
"""
import os

# Chemins de fichiers
# S'assurer que le chemin pointe vers le dossier data à la racine du projet
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Meteorite_Landings.csv")

# Configuration de la carte
DEFAULT_MAP_STYLE = "standard"  # Style standard
DEFAULT_COLOR_MODE = "class"

# Configuration des filtres
MASS_RANGE = [0, 6]  # Exposant log10
YEAR_RANGE = [1800, 2020]
DEFAULT_CLASSES = "all"
DEFAULT_FALLS = ["Found", "Fell"]

# Configuration du serveur
PORT = 8080
DEBUG_MODE = True

# Paramètres d'analyse de zone
ZONE_RADIUS_DEGREES = 2.5

# Styles d'affichage
GRAPH_HEIGHT = 350
MAP_HEIGHT = 650
PREDICTION_MAP_HEIGHT = 450

# Palette de couleurs
COLOR_SCHEMES = {
    "primary": "#0071e3",
    "success": "#34c759",
    "warning": "#ff9500",
    "danger": "#ff3b30",
    "info": "#5ac8fa",
    "light": "#f5f5f7",
    "dark": "#1d1d1f",
}

# Configuration des colorscales pour les différents modes
COLOR_SCALES = {
    "mass": "Viridis",
    "year": "Plasma",
    "fall": {
        "Found": "#5ac8fa",
        "Fell": "#ff9500"
    },
    "class": "Pastel"
}

# Configuration du débogage
STOP_ON_ERROR = False  # Si True, l'application s'arrêtera en cas d'erreur au lieu de continuer
VERBOSE_ERRORS = True  # Si True, affiche des informations détaillées sur les erreurs