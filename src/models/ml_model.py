import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

class MeteoriteML:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.mass_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.class_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.location_clusters = KMeans(n_clusters=5, random_state=42)
        self._prepare_data()
        
    def _prepare_data(self):
        # Nettoyage et préparation des données
        self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')
        self.data['mass (g)'] = pd.to_numeric(self.data['mass (g)'], errors='coerce')
        self.data = self.data.dropna(subset=['reclat', 'reclong', 'year', 'mass (g)'])
        
        # Encodage des variables catégorielles
        self.data['fall_encoded'] = self.le.fit_transform(self.data['fall'])
        self.data['recclass_encoded'] = self.le.fit_transform(self.data['recclass'])
        
    def train_mass_predictor(self):
        # Préparation des features pour la prédiction de la masse
        features = ['reclat', 'reclong', 'year', 'fall_encoded']
        X = self.data[features]
        y = np.log1p(self.data['mass (g)'])  # Log transform pour normaliser
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entraînement du modèle
        self.mass_predictor.fit(X_train_scaled, y_train)
        y_pred = self.mass_predictor.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        return np.sqrt(mse)
    
    def predict_mass(self, lat, lon, year, fall):
        # Prédiction de la masse pour de nouvelles données
        fall_encoded = self.le.transform([fall])[0]
        features = np.array([[lat, lon, year, fall_encoded]])
        features_scaled = self.scaler.transform(features)
        mass_pred = self.mass_predictor.predict(features_scaled)
        return np.expm1(mass_pred[0])  # Inverse log transform
    
    def train_class_predictor(self):
        # Préparation des features pour la classification
        features = ['reclat', 'reclong', 'mass (g)', 'year', 'fall_encoded']
        X = self.data[features]
        y = self.data['recclass']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
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
        joblib.dump(self.scaler, f'{path_prefix}scaler.joblib')
        joblib.dump(self.le, f'{path_prefix}label_encoder.joblib') 