import pandas as pd
from datetime import datetime
import numpy as np
import traceback

class MeteoriteData:
    def __init__(self, data_path):
        try:
            self.df = pd.read_csv(data_path)
            self._clean_data()
            print(f"Données chargées avec succès: {len(self.df)} météorites")
        except Exception as e:
            print(f"ERREUR lors du chargement des données: {str(e)}")
            print(traceback.format_exc())
            # Initialiser un dataframe vide en cas d'erreur
            self.df = pd.DataFrame()
        
    def _clean_data(self):
        try:
            # Nettoyage numérique des années
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
            
            # Éliminer les lignes avec des coordonnées manquantes (essentielles pour la carte)
            self.df = self.df.dropna(subset=['reclat', 'reclong'])
            
            # Filtrer les années aberrantes
            current_year = datetime.now().year
            self.df = self.df[(self.df['year'].isna()) | self.df['year'].between(1000, current_year, inclusive='both')]
            
            # Nettoyage des masses et conversion en numérique
            self.df['mass (g)'] = pd.to_numeric(self.df['mass (g)'], errors='coerce')
            
            # Remplacer les valeurs NaN de masse par la moyenne (pour éviter les pertes de données)
            median_mass = self.df['mass (g)'].median()
            self.df['mass (g)'].fillna(median_mass, inplace=True)
            
            # CORRECTION IMPORTANTE: S'assurer que les valeurs de masse sont > 0 pour permettre le log
            # Remplacer toutes les valeurs ≤ 0 par 1 gramme comme valeur minimum
            self.df['mass (g)'] = self.df['mass (g)'].apply(lambda x: max(float(x), 1.0))
            
            # NOUVELLE COLONNE: Créer une colonne 'adjusted_mass' qui garantit des valeurs positives
            self.df['adjusted_mass'] = self.df['mass (g)']
            
            # Maintenant on peut calculer le log en toute sécurité
            self.df['log_mass'] = np.log10(self.df['adjusted_mass'])
            
            # Vérification des valeurs problématiques (pour le débogage)
            problematic_mass = self.df[self.df['mass (g)'] <= 0]
            if not problematic_mass.empty:
                print(f"ATTENTION: {len(problematic_mass)} valeurs de masse problématiques ont été corrigées")
            
            # Vérification du résultat
            min_mass = self.df['mass (g)'].min()
            print(f"Masse minimale après correction: {min_mass}g")
            
            # Convertir les années en int pour éviter les problèmes avec les filtres
            # après avoir traité les NaN
            self.df['year'] = self.df['year'].fillna(0).astype(int)
            
            # Création des intervalles temporels (décennies)
            self.df['decade'] = (self.df['year'] // 10) * 10
            
            # Nettoyer les valeurs de fall (Fell/Found)
            self.df['fall'] = self.df['fall'].fillna('Unknown')
            
            # Nettoyer les valeurs de classe
            self.df['recclass'] = self.df['recclass'].fillna('Unknown')
            
        except Exception as e:
            print(f"ERREUR lors du nettoyage des données: {str(e)}")
            print(traceback.format_exc())
        
    def get_filtered_data(self, mass_range=None, classification=None, fall_type=None, decade_range=None):
        try:
            filtered = self.df.copy()
            
            # Filtre par plage de masse (en exposant log10)
            if mass_range:
                try:
                    min_mass = 10 ** mass_range[0]
                    max_mass = 10 ** mass_range[1]
                    filtered = filtered[filtered['mass (g)'].between(min_mass, max_mass)]
                except (TypeError, ValueError) as e:
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
                    print(f"Erreur dans le filtre de type de chute: {e}")
            
            # Filtre par plage de décennies
            if decade_range:
                try:
                    min_decade = int(decade_range[0])
                    max_decade = int(decade_range[1])
                    filtered = filtered[filtered['decade'].between(min_decade, max_decade)]
                except (TypeError, ValueError) as e:
                    print(f"Erreur dans le filtre de décennie: {e}")
            
            return filtered
            
        except Exception as e:
            print(f"ERREUR lors du filtrage des données: {str(e)}")
            print(traceback.format_exc())
            return pd.DataFrame()  # Retourner un dataframe vide en cas d'erreur