import pandas as pd
from datetime import datetime
import numpy as np
import traceback

class MeteoriteData:
    def __init__(self, data_path, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print("Chargement des données depuis:", data_path)
        self.data = pd.read_csv(data_path)
        self._clean_data()
        self._prepare_data()
        if self.verbose:
            print(f"Données chargées avec succès: {len(self.data)} météorites")

    def _clean_data(self):
        """Nettoyer les données de base"""
        # Convertir des colonnes en numérique
        self.data['mass (g)'] = pd.to_numeric(self.data['mass (g)'], errors='coerce')
        self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')

        # Supprimer les points avec latitude = 0 ET longitude = 0 (erreurs de données)
        invalid_coords = (self.data['reclat'] == 0) & (self.data['reclong'] == 0)
        if invalid_coords.any():
            if self.verbose:
                print(f"Suppression de {invalid_coords.sum()} météorites avec coordonnées (0,0) invalides")
            self.data = self.data[~invalid_coords]

        # Supprimer les lignes avec des valeurs manquantes dans les colonnes essentielles
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['reclat', 'reclong', 'mass (g)'])
        dropped_count = initial_count - len(self.data)
        if dropped_count > 0 and self.verbose:
            print(f"Suppression de {dropped_count} météorites avec des valeurs manquantes")

        # Correction des masses nulles ou négatives
        mask_invalid_mass = self.data['mass (g)'] <= 0
        if mask_invalid_mass.any():
            if self.verbose:
                print(f"Correction de {mask_invalid_mass.sum()} masses invalides (≤0)")
            min_valid_mass = self.data[self.data['mass (g)'] > 0]['mass (g)'].min()
            self.data.loc[mask_invalid_mass, 'mass (g)'] = min_valid_mass
            if self.verbose:
                print(f"Masse minimale après correction: {min_valid_mass}g")

    def _prepare_data(self):
        try:
            # Éliminer les lignes avec des coordonnées manquantes (essentielles pour la carte)
            self.data = self.data.dropna(subset=['reclat', 'reclong'])

            # Filtrer les années aberrantes
            current_year = datetime.now().year
            self.data = self.data[(self.data['year'].isna()) | self.data['year'].between(1000, current_year, inclusive='both')]

            # Nettoyage des masses et conversion en numérique
            self.data['mass (g)'] = pd.to_numeric(self.data['mass (g)'], errors='coerce')

            # Remplacer les valeurs NaN de masse par la moyenne (pour éviter les pertes de données)
            median_mass = self.data['mass (g)'].median()
            self.data['mass (g)'].fillna(median_mass, inplace=True)

            # CORRECTION IMPORTANTE: S'assurer que les valeurs de masse sont > 0 pour permettre le log
            # Remplacer toutes les valeurs ≤ 0 par 1 gramme comme valeur minimum
            self.data['mass (g)'] = self.data['mass (g)'].apply(lambda x: max(float(x), 1.0))

            # NOUVELLE COLONNE: Créer une colonne 'adjusted_mass' qui garantit des valeurs positives
            self.data['adjusted_mass'] = self.data['mass (g)']

            # Maintenant on peut calculer le log en toute sécurité
            self.data['log_mass'] = np.log10(self.data['adjusted_mass'])

            # Vérification des valeurs problématiques (pour le débogage)
            problematic_mass = self.data[self.data['mass (g)'] <= 0]
            if not problematic_mass.empty and self.verbose:
                print(f"ATTENTION: {len(problematic_mass)} valeurs de masse problématiques ont été corrigées")

            # Vérification du résultat
            min_mass = self.data['mass (g)'].min()
            if self.verbose:
                print(f"Masse minimale après correction: {min_mass}g")

            # Convertir les années en int pour éviter les problèmes avec les filtres
            # après avoir traité les NaN
            self.data['year'] = self.data['year'].fillna(0).astype(int)

            # Création des intervalles temporels (décennies)
            self.data['decade'] = (self.data['year'] // 10) * 10

            # Nettoyer les valeurs de fall (Fell/Found)
            self.data['fall'] = self.data['fall'].fillna('Unknown')

            # Nettoyer les valeurs de classe
            self.data['recclass'] = self.data['recclass'].fillna('Unknown')

        except Exception as e:
            if self.verbose:
                print(f"ERREUR lors de la préparation des données: {str(e)}")
                print(traceback.format_exc())

    def get_filtered_data(self, mass_range=None, classification=None, fall_type=None, decade_range=None):
        try:
            filtered = self.data.copy()

            # Filtre par plage de masse (en exposant log10)
            if mass_range:
                try:
                    min_mass = 10 ** mass_range[0]
                    max_mass = 10 ** mass_range[1]
                    filtered = filtered[filtered['mass (g)'].between(min_mass, max_mass)]
                except (TypeError, ValueError) as e:
                    if self.verbose:
                        print(f"Erreur dans le filtre de masse: {e}")
                    # En cas d'erreur, ne pas appliquer ce filtre

            # Filtre par classe de météorite
            if classification:
                try:
                    # Gestion du cas 'all' ou d'une chaîne unique
                    if classification == 'all':
                        pass  # Ne pas filtrer si 'all' est sélectionné
                    elif isinstance(classification, str):
                        # Convertir en liste si c'est une chaîne simple
                        filtered = filtered[filtered['recclass'] == classification]
                    else:
                        # Si c'est déjà une liste, utiliser isin
                        filtered = filtered[filtered['recclass'].isin(classification)]
                except Exception as e:
                    if self.verbose:
                        print(f"Erreur dans le filtre de classification: {e}")

            # Filtre par type de chute
            if fall_type:
                try:
                    # Même logique pour fall_type
                    if isinstance(fall_type, str):
                        filtered = filtered[filtered['fall'] == fall_type]
                    else:
                        filtered = filtered[filtered['fall'].isin(fall_type)]
                except Exception as e:
                    if self.verbose:
                        print(f"Erreur dans le filtre de type de chute: {e}")

            # Filtre par plage de décennies
            if decade_range:
                try:
                    min_decade = int(decade_range[0])
                    max_decade = int(decade_range[1])
                    filtered = filtered[filtered['decade'].between(min_decade, max_decade)]
                except (TypeError, ValueError) as e:
                    if self.verbose:
                        print(f"Erreur dans le filtre de décennie: {e}")

            return filtered

        except Exception as e:
            if self.verbose:
                print(f"ERREUR lors du filtrage des données: {str(e)}")
                print(traceback.format_exc())
            return pd.DataFrame()  # Retourner un dataframe vide en cas d'erreur

    def get_meteorites_in_radius(self, lat, lon, radius_degrees):
        """
        Récupère les météorites dans un rayon donné autour d'un point

        Args:
            lat (float): Latitude du point central
            lon (float): Longitude du point central
            radius_degrees (float): Rayon en degrés

        Returns:
            DataFrame: Les météorites dans le rayon spécifié
        """
        try:
            # Calculer la distance approximative en degrés (simplifié)
            df = self.data.copy()

            # Formule de distance euclidienne simplifiée (suffisante pour de petites distances)
            df['distance'] = np.sqrt((df['reclat'] - lat)**2 + (df['reclong'] - lon)**2)

            # Filtrer par rayon
            nearby = df[df['distance'] <= radius_degrees]

            return nearby

        except Exception as e:
            if self.verbose:
                print(f"ERREUR lors de la récupération des météorites dans le rayon: {str(e)}")
                print(traceback.format_exc())
            return pd.DataFrame()  # Retourner un dataframe vide en cas d'erreur