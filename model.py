import pandas as pd
from datetime import datetime

class MeteoriteData:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self._clean_data()
        
    def _clean_data(self):
        # Nettoyage numérique des années
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        self.df = self.df.dropna(subset=['reclat', 'reclong', 'year'])
        self.df = self.df[self.df['year'].between(1000, datetime.now().year)]
        
        # Nettoyage des masses et filtrage supérieur à 1900g
        self.df['mass (g)'] = pd.to_numeric(self.df['mass (g)'], errors='coerce')
        self.df = self.df[self.df['mass (g)'] > 1900]  # Seuil de 1900 grammes
        
        # Création des intervalles temporels
        self.df['decade'] = (self.df['year'] // 10) * 10
        
    def get_filtered_data(self, mass_range=None, classification=None, fall_type=None, decade_range=None):
        filtered = self.df.copy()
        if mass_range:
            filtered = filtered[filtered['mass (g)'].between(*mass_range)]
        if classification:
            filtered = filtered[filtered['recclass'].isin(classification)]
        if fall_type:
            filtered = filtered[filtered['fall'].isin(fall_type)]
        if decade_range:
            filtered = filtered[filtered['decade'].between(*decade_range)]
        return filtered