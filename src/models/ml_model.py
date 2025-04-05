import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os
import pickle
import time

class MeteoriteML:
    def __init__(self, data, models_dir='models'):
        self.data = data
        self.models_dir = models_dir
        self.le = LabelEncoder()
        self.mass_scaler = StandardScaler()  # Scaler spécifique pour la prédiction de masse
        self.class_scaler = StandardScaler()  # Scaler spécifique pour la classification
        self.mass_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.class_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.location_clusters = KMeans(n_clusters=5, random_state=42)

        # Créer le répertoire des modèles s'il n'existe pas
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        self._prepare_data()

        # Essayer de charger les modèles existants
        self.load_models()

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

    def train_mass_predictor(self, force_train=False):
        """Entraîne le modèle de prédiction de masse si nécessaire"""
        # Vérifier si le modèle est déjà entraîné
        if not force_train and hasattr(self.mass_predictor, 'feature_importances_'):
            print("Le modèle de prédiction de masse est déjà entraîné.")
            return 0.0  # Retourne une valeur par défaut pour l'erreur

        print("Entraînement du modèle de prédiction de masse...")
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
        rmse = np.sqrt(mse)

        # Sauvegarder le modèle entraîné
        self.save_models()

        print(f"Modèle de prédiction de masse entraîné avec RMSE: {rmse:.4f}")
        return rmse

    def predict_mass(self, lat, lon, year, fall):
        """Prédiction de la masse pour de nouvelles données"""
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

            # Vérifier que le modèle est entraîné
            if not hasattr(self.mass_scaler, 'mean_') or not hasattr(self.mass_predictor, 'feature_importances_'):
                print("Le modèle de prédiction de masse n'est pas encore entraîné. Entraînement en cours...")
                self.train_mass_predictor(force_train=True)

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

    def train_class_predictor(self, force_train=False):
        """Entraîne le modèle de classification si nécessaire"""
        # Vérifier si le modèle est déjà entraîné
        if not force_train and hasattr(self.class_predictor, 'feature_importances_'):
            print("Le modèle de classification est déjà entraîné.")
            return 0.0  # Retourne une valeur par défaut pour l'accuracy

        print("Entraînement du modèle de classification...")
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
        accuracy = accuracy_score(y_test, y_pred)

        # Sauvegarder le modèle entraîné (si pas déjà fait par train_mass_predictor)
        self.save_models()

        print(f"Modèle de classification entraîné avec une précision de: {accuracy:.4f}")
        return accuracy

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

    def save_models(self):
        """Sauvegarde tous les modèles entraînés dans le répertoire des modèles"""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            models_data = {
                'mass_predictor': self.mass_predictor,
                'class_predictor': self.class_predictor,
                'mass_scaler': self.mass_scaler,
                'class_scaler': self.class_scaler,
                'fall_le': self.fall_le,
                'timestamp': timestamp
            }

            # Sauvegarde avec pickle (plus rapide et plus compact que joblib pour plusieurs objets)
            with open(os.path.join(self.models_dir, 'meteorite_models.pkl'), 'wb') as f:
                pickle.dump(models_data, f)

            # Sauvegarde individuelle avec joblib (alternative)
            joblib.dump(self.mass_predictor, os.path.join(self.models_dir, 'mass_predictor.joblib'))
            joblib.dump(self.class_predictor, os.path.join(self.models_dir, 'class_predictor.joblib'))
            joblib.dump(self.mass_scaler, os.path.join(self.models_dir, 'mass_scaler.joblib'))
            joblib.dump(self.class_scaler, os.path.join(self.models_dir, 'class_scaler.joblib'))

            print(f"Modèles sauvegardés avec succès dans {self.models_dir}")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des modèles: {str(e)}")
            return False

    def load_models(self):
        """Charge les modèles précédemment entraînés s'ils existent"""
        try:
            # Vérifier si le fichier pickle existe
            pickle_path = os.path.join(self.models_dir, 'meteorite_models.pkl')
            if os.path.exists(pickle_path):
                print("Chargement des modèles depuis le fichier pickle...")
                with open(pickle_path, 'rb') as f:
                    models_data = pickle.load(f)

                self.mass_predictor = models_data['mass_predictor']
                self.class_predictor = models_data['class_predictor']
                self.mass_scaler = models_data['mass_scaler']
                self.class_scaler = models_data['class_scaler']
                self.fall_le = models_data['fall_le']

                print(f"Modèles chargés avec succès (timestamp: {models_data.get('timestamp', 'inconnu')})")
                return True

            # Alternative: vérifier les fichiers joblib individuels
            elif os.path.exists(os.path.join(self.models_dir, 'mass_predictor.joblib')):
                print("Chargement des modèles depuis les fichiers joblib individuels...")
                self.mass_predictor = joblib.load(os.path.join(self.models_dir, 'mass_predictor.joblib'))
                self.class_predictor = joblib.load(os.path.join(self.models_dir, 'class_predictor.joblib'))
                self.mass_scaler = joblib.load(os.path.join(self.models_dir, 'mass_scaler.joblib'))
                self.class_scaler = joblib.load(os.path.join(self.models_dir, 'class_scaler.joblib'))
                print("Modèles chargés avec succès depuis les fichiers joblib")
                return True
            else:
                print("Aucun modèle pré-entraîné trouvé. Les modèles seront entraînés.")
                return False
        except Exception as e:
            print(f"Erreur lors du chargement des modèles: {str(e)}")
            print("Les modèles seront entraînés à nouveau.")
            return False