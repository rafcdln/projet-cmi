"""
Fonctions utilitaires pour l'application d'analyse de météorites.
"""
import numpy as np
import pandas as pd
from ..utils.config import ZONE_RADIUS_DEGREES


def log_transform(value, min_val=0.1):
    """Applique une transformation log pour afficher des valeurs sur une échelle logarithmique."""
    if value <= 0:
        value = min_val
    return np.log10(value)


def filter_by_coordinates(df, lat, lon, radius_degrees=ZONE_RADIUS_DEGREES):
    """Filtre un DataFrame par coordonnées géographiques dans un rayon spécifié en degrés."""
    mask = ((df['reclat'] - lat).abs() <= radius_degrees) & ((df['reclong'] - lon).abs() <= radius_degrees)
    return df[mask]


def create_circle_points(center_lat, center_lon, radius_degrees, num_points=100):
    """Crée un cercle de points autour d'un centre, utilisé pour visualiser une zone sur la carte."""
    angles = np.linspace(0, 2*np.pi, num_points)
    circle_lats = center_lat + radius_degrees * np.sin(angles)
    circle_lons = center_lon + radius_degrees * np.cos(angles)
    return pd.DataFrame({'lat': circle_lats, 'lon': circle_lons})


def format_number(value, decimal_places=2):
    """Formate un nombre avec des séparateurs de milliers et un nombre fixe de décimales."""
    if isinstance(value, (int, float)):
        if value == int(value):
            return f"{int(value):,}"
        else:
            return f"{value:,.{decimal_places}f}"
    return value 