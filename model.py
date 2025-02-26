import pandas as pd
from datetime import datetime

class MeteoriteData:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self._clean_data()
        
    def _clean_data(self):
        # Nettoyage des données
        self.df = self.df.dropna(subset=['reclat', 'reclong'])
        self.df['year'] = pd.to_datetime(self.df['year'], errors='coerce').dt.year
        self.df = self.df[self.df['year'] <= datetime.now().year]
        self.df['mass (g)'] = pd.to_numeric(self.df['mass (g)'], errors='coerce')
        self.df['decade'] = (self.df['year'] // 10) * 10
        self.df = self.df[self.df['mass (g)'] > 0]
        
    def get_filtered_data(self, mass_range, classification, fall_type, decade_range):
        filtered = self.df.copy()
        filtered = filtered[(filtered['mass (g)'] >= mass_range[0]) & 
                          (filtered['mass (g)'] <= mass_range[1])]
        
        if classification:
            filtered = filtered[filtered['recclass'].isin(classification)]
            
        if fall_type:
            filtered = filtered[filtered['fall'].isin(fall_type)]
            
        filtered = filtered[(filtered['decade'] >= decade_range[0]) & 
                          (filtered['decade'] <= decade_range[1])]
        
        return filtered