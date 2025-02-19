import pandas as pd

class MeteoriteData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load_data()

    def _load_data(self):
        """Charge les données depuis le fichier CSV."""
        df = pd.read_csv(self.filepath)
        # Nettoyage des données
        df = df.dropna(subset=['reclat', 'reclong', 'year', 'mass (g)'])
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['mass (g)'] = pd.to_numeric(df['mass (g)'], errors='coerce')
        df = df[df['year'] >= 1900]  # Filtrer les années récentes
        return df

    def get_data(self):
        """Retourne le DataFrame nettoyé."""
        return self.data