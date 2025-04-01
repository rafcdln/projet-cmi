import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

class MeteoriteML:
    def __init__(self, data):
        self.data = data
        self.le = LabelEncoder()
        self.mass_scaler = StandardScaler()  # Scaler spécifique pour la prédiction de masse
        self.class_scaler = StandardScaler()  # Scaler spécifique pour la classification
        self.mass_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.class_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.location_clusters = KMeans(n_clusters=5, random_state=42)
        self._prepare_data()
        
    def _prepare_data(self):
        # Nettoyage et préparation des données
        self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')
        self.data['mass (g)'] = pd.to_numeric(self.data['mass (g)'], errors='coerce')
        self.data = self.data.dropna(subset=['reclat', 'reclong', 'year', 'mass (g)'])
        
        # Garantir que 'fall' contient bien "Fell" et "Found"
        self.known_falls = sorted(self.data['fall'].unique().tolist())
        
        # Encodage des variables catégorielles - s'assurer de réaliser un fit complet
        self.fall_le = LabelEncoder().fit(['Fell', 'Found'])  # Encodeur spécifique pour fall avec valeurs fixes
        self.data['fall_encoded'] = self.fall_le.transform(self.data['fall'])
        
        self.recclass_le = LabelEncoder().fit(self.data['recclass'])
        self.data['recclass_encoded'] = self.recclass_le.transform(self.data['recclass'])
        
    def train_mass_predictor(self):
        # Préparation des features pour la prédiction de la masse
        features = ['reclat', 'reclong', 'year', 'fall_encoded']
        X = self.data[features]
        y = np.log1p(self.data['mass (g)'])  # Log transform pour normaliser
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_scaled = self.mass_scaler.fit_transform(X_train)  # Utiliser mass_scaler
        X_test_scaled = self.mass_scaler.transform(X_test)
        
        # Entraînement du modèle
        self.mass_predictor.fit(X_train_scaled, y_train)
        y_pred = self.mass_predictor.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        return np.sqrt(mse)
    
    def predict_mass(self, lat, lon, year, fall):
        # Prédiction de la masse pour de nouvelles données
        try:
            # Vérification des entrées
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                raise ValueError("Les coordonnées doivent être numériques")
            
            if not isinstance(year, (int, float)):
                raise ValueError("L'année doit être numérique")
                
            if not isinstance(fall, str) or fall not in ['Fell', 'Found']:
                raise ValueError("Le type de chute doit être 'Fell' ou 'Found'")
                
            # Utilisation de l'encodeur spécifique pour 'fall' qui connaît 'Fell' et 'Found'
            fall_encoded = self.fall_le.transform([fall])[0]
            
            # Création et normalisation des features
            features = np.array([[lat, lon, year, fall_encoded]])
            
            # Vérifier qu'il existe des données pour la normalisation
            if not hasattr(self.mass_scaler, 'mean_') or not hasattr(self.mass_predictor, 'feature_importances_'):
                # Si le modèle n'est pas encore entraîné, le faire maintenant
                self.train_mass_predictor()
                
            # Normalisation des features avec le mass_scaler
            features_scaled = self.mass_scaler.transform(features)
            
            # Prédiction
            mass_pred = self.mass_predictor.predict(features_scaled)
            
            # Inversion de la transformation logarithmique et limitation des valeurs extrêmes
            mass = np.expm1(mass_pred[0])
            
            # Limiter les valeurs extrêmes
            if mass < 0.1:
                mass = 0.1  # Minimum de 0.1g
            elif mass > 10000000:
                mass = 10000000  # Maximum de 10 tonnes
                
            return mass
            
        except Exception as e:
            import traceback
            print(f"Erreur dans la prédiction de masse: {str(e)}")
            print(traceback.format_exc())
            # Retourner une valeur par défaut en cas d'erreur
            return 10.0  # Valeur par défaut raisonnable
    
    def train_class_predictor(self):
        # Préparation des features pour la classification
        features = ['reclat', 'reclong', 'mass (g)', 'year', 'fall_encoded']
        X = self.data[features]
        y = self.data['recclass']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_scaled = self.class_scaler.fit_transform(X_train)  # Utiliser class_scaler
        X_test_scaled = self.class_scaler.transform(X_test)
        
        # Entraînement du classificateur
        self.class_predictor.fit(X_train_scaled, y_train)
        y_pred = self.class_predictor.predict(X_test_scaled)
        return accuracy_score(y_test, y_pred)
    
    def cluster_locations(self):
        # Clustering des localisations
        locations = self.data[['reclat', 'reclong']].copy()
        self.location_clusters.fit(locations)
        return self.location_clusters.labels_
    
    def get_feature_importance(self):
        # Analyse de l'importance des features
        features = ['reclat', 'reclong', 'year', 'fall_encoded']
        importance = self.mass_predictor.feature_importances_
        return dict(zip(features, importance))
    
    def save_models(self, path_prefix='models/'):
        # Sauvegarde des modèles entraînés
        joblib.dump(self.mass_predictor, f'{path_prefix}mass_predictor.joblib')
        joblib.dump(self.class_predictor, f'{path_prefix}class_predictor.joblib')
        joblib.dump(self.mass_scaler, f'{path_prefix}mass_scaler.joblib')
        joblib.dump(self.class_scaler, f'{path_prefix}class_scaler.joblib')
        joblib.dump(self.le, f'{path_prefix}label_encoder.joblib') 