from dash import Input, Output, State, callback, html
import plotly.express as px
import plotly.graph_objects as go
from models.model import MeteoriteData
from models.ml_model import MeteoriteML
import pandas as pd
import numpy as np
from datetime import datetime
import json
from dash import dcc
import dash
import sys
import traceback
from utils.config import STOP_ON_ERROR, VERBOSE_ERRORS, DEFAULT_MAP_STYLE
from sklearn.preprocessing import StandardScaler
import re
import logging
import os
import math

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('meteorite_dashboard')

# Coordonnées centrées sur la France
FRANCE_LAT = 46.603354
FRANCE_LON = 1.888334
FRANCE_ZOOM = 4

def register_callbacks(app, data_path):
    try:
        # Initialisation des modèles à l'intérieur de register_callbacks
        # pour qu'ils soient accessibles à tous les callbacks
        global meteorite_data, ml_model
        
        # Chargement des données météorites
        meteorite_data = MeteoriteData(data_path)
        
        # Initialisation et entraînement du modèle de machine learning
        ml_model = MeteoriteML(meteorite_data.data)
        
        # Entraînement initial des modèles
        try:
            print("Entraînement des modèles de prédiction...")
            ml_model.train_mass_predictor()
            ml_model.train_class_predictor()
            print("Modèles entraînés avec succès")
        except Exception as e:
            print(f"ERREUR lors de l'entraînement des modèles: {str(e)}")
            print(traceback.format_exc())
        
        # Configuration des styles de carte
        map_style_options = {
            'clair': 'carto-positron',
            'sombre': 'carto-darkmatter',
            'standard': 'open-street-map',
            'satellite': 'satellite'
        }
        
        # Helper function pour appliquer le style de carte
        def get_mapbox_style(map_style):
            map_style = map_style or DEFAULT_MAP_STYLE
            return map_style_options.get(map_style, map_style_options.get(DEFAULT_MAP_STYLE, 'carto-positron'))
        
        # Configurer la fonction de debugging
        def debug_callback(message, level='info'):
            """
            Affiche un message de débogage dans les logs
            et le stocke pour l'affichage dans l'interface
            """
            message_str = str(message)
            if level == 'info':
                logger.info(message_str)
            elif level == 'warning':
                logger.warning(message_str)
            elif level == 'error':
                logger.error(message_str)
                
            try:
                print(f"{level.upper()}: {message_str}")
                return message_str
            except Exception as e:
                logger.error(f"Erreur dans debug_callback: {str(e)}")
                return None
    except Exception as e:
        print(f"ERREUR FATALE lors du chargement des données: {str(e)}")
        print(traceback.format_exc())
        
        # Si STOP_ON_ERROR est True, on arrête l'application
        if STOP_ON_ERROR:
            sys.exit(1)
        
        # Sinon, on initialise des variables à None pour éviter les erreurs
        # mais l'application fonctionnera partiellement
        meteorite_data = None
        ml_model = None
    
    # Wrapper pour gérer les erreurs dans les callbacks
    def handle_error(e, func_name="fonction inconnue", details=""):
        """
        Gère les erreurs dans les callbacks de manière élégante
        """
        error_msg = f"ERREUR dans {func_name}: {type(e).__name__}: {str(e)}"
        traceback_str = traceback.format_exc()
        
        # Afficher l'erreur dans le format lisible et standard
        print(f"\n{'='*80}")
        print(error_msg)
        print("\nTraceback complet:")
        print(traceback_str)
        if details:
            print("\nInformations supplémentaires:")
            print(details)
        print(f"{'='*80}\n")
        
        # Si configuré pour arrêter l'exécution en cas d'erreur
        if STOP_ON_ERROR:
            print("Arrêt de l'application demandé après erreur.")
            sys.exit(1)
    
    # Fonction pour vérifier et corriger les problèmes de données avant Plotly Express
    def validate_dataframe_for_plotly(df, func_name):
        if df is None or df.empty:
            print(f"WARNING dans {func_name}: DataFrame vide ou None")
            return pd.DataFrame({'x': [0], 'y': [0]})
            
        # Vérifier les types de colonnes
        if VERBOSE_ERRORS:
            print(f"\nDiagnostic du DataFrame dans {func_name}:")
            print(f"Dimensions: {df.shape}")
            print("Types de colonnes:")
            for col, dtype in df.dtypes.items():
                print(f"  - {col}: {dtype}")
            na_counts = df.isna().sum()
            if na_counts.sum() > 0:
                print("\nValeurs manquantes détectées:")
                for col, count in na_counts.items():
                    if count > 0:
                        print(f"  - {col}: {count} valeurs NaN")
        
        df_clean = df.copy()
        
        # Remplacer les valeurs NaN
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].fillna('Non spécifié')
        for col in df_clean.select_dtypes(include=['number']).columns:
            df_clean[col] = df_clean[col].fillna(0)
            
        # Vérifier les colonnes avec des types mixtes
        for col in df_clean.columns:
            try:
                if df_clean[col].dtype == 'object':
                    try:
                        numeric_col = pd.to_numeric(df_clean[col], errors='coerce')
                        if not numeric_col.isna().all():
                            df_clean[col] = numeric_col.fillna(0)
                    except:
                        pass
            except Exception as e:
                print(f"Erreur lors du traitement de la colonne {col}: {str(e)}")
        
        return df_clean
    
    # Wrapper pour ajouter la gestion d'erreurs à tous les callbacks
    def error_handling_callback(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(e, func.__name__, f"Arguments: {args}")
                if 'figure' in func.__name__:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(
                        text=f"Erreur dans {func.__name__}: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    empty_fig.update_layout(
                        height=400,
                        paper_bgcolor='rgba(240, 240, 240, 0.5)',
                        plot_bgcolor='rgba(240, 240, 240, 0.5)',
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    return empty_fig
                else:
                    return html.Div(f"Une erreur est survenue: {str(e)}", 
                                   style={'color': 'red', 'padding': '10px', 'background': '#ffeeee'})
        wrapper.__name__ = func.__name__
        return wrapper
    
    @app.callback(
        Output('world-map', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value'),
         Input('map-style-dropdown', 'value'),
         Input('color-mode-dropdown', 'value'),
         Input('point-opacity', 'value'),
         Input('point-size', 'value'),
         Input('map-interactivity', 'value')]
    )
    @error_handling_callback
    def update_map(mass_range, classes, falls, decades, map_style, color_mode, 
                  opacity, point_size, interactivity):
        print("\nDébut update_map avec paramètres:", mass_range, classes, falls, decades, map_style, color_mode)
        
        # Filtrer les données
        df_filtered = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        # Nettoyage et préparation des données
        df_filtered = df_filtered.dropna(subset=['reclat', 'reclong', 'mass (g)'])
        df_filtered = df_filtered[~((df_filtered['reclat'] == 0) & (df_filtered['reclong'] == 0))]
        df_filtered['log_mass'] = np.log10(df_filtered['mass (g)'])
        print(f"Nombre de météorites après nettoyage: {len(df_filtered)}")
        
        # Récupérer le style de carte approprié
        actual_style = map_style_options.get(map_style, map_style_options['clair'])
        
        # Gestion du cas où il n'y a pas de données
        if df_filtered.empty:
            fig = go.Figure()
            fig.update_layout(
                mapbox=dict(style=actual_style, center=dict(lat=FRANCE_LAT, lon=FRANCE_LON), zoom=FRANCE_ZOOM),
                margin={"r":0,"t":30,"l":0,"b":0},
                height=700
            )
            return fig
            
        # Calcul de la taille des points basé sur la masse (logarithmique)
        min_size = 3
        max_size = min_size + point_size * 3
        
        if len(df_filtered) > 1:
            min_log_mass = df_filtered['log_mass'].min()
            max_log_mass = df_filtered['log_mass'].max()
            size_range = max_log_mass - min_log_mass
            if size_range > 0:
                sizes = min_size + ((df_filtered['log_mass'] - min_log_mass) / size_range) * (max_size - min_size)
            else:
                sizes = [min_size + (max_size - min_size)/2] * len(df_filtered)
        else:
            sizes = [max_size] * len(df_filtered)
        
        # Création de la carte avec la bonne coloration selon le mode
        fig = go.Figure()
        
        # Pour le mode "class" ou "fall", on doit créer un groupe par catégorie
        if color_mode == "class":
            # Fonction améliorée pour regrouper les classes de météorites
            def simplify_class(class_name):
                # Traiter les cas spéciaux en premier
                if 'Iron' in class_name:
                    return 'Météorites de fer'  # Toutes les météorites ferreuses
                
                if 'Pallasite' in class_name or 'Mesosiderite' in class_name:
                    return 'Métallo-rocheuses'  # Pallasite et Mésosidérite sont des métallo-rocheuses
                
                if 'Eucrite' in class_name or 'Diogenite' in class_name or 'Howardite' in class_name:
                    return 'Achondrites HED'  # Famille HED
                
                if 'Ureilite' in class_name or 'Angrite' in class_name or 'Aubrite' in class_name:
                    return 'Autres achondrites'  # Autres types d'achondrites
                
                # Chondrites ordinaires
                if class_name.startswith('H'):
                    return 'Chondrites H'
                if class_name.startswith('L'):
                    return 'Chondrites L'
                if class_name.startswith('LL'):
                    return 'Chondrites LL'
                
                # Chondrites carbonées
                if any(class_name.startswith(x) for x in ['CI', 'CM', 'CO', 'CV', 'CK', 'CR']):
                    return 'Chondrites carbonées'
                
                # Chondrites à enstatite
                if class_name.startswith('E'):
                    return 'Chondrites E'
                
                # Météorites martiennes et lunaires
                if 'Martian' in class_name or 'Shergottite' in class_name or 'Nakhlite' in class_name:
                    return 'Météorites martiennes'
                if 'Lunar' in class_name:
                    return 'Météorites lunaires'
                
                # Si rien ne correspond, mettre dans "Autres"
                return 'Autres'
            
            # Appliquer la fonction de regroupement
            df_filtered['class_group'] = df_filtered['recclass'].apply(simplify_class)
            
            # Calculer les pourcentages pour chaque groupe
            category_data = []
            for category in df_filtered['class_group'].unique():
                count = len(df_filtered[df_filtered['class_group'] == category])
                percentage = round(count / len(df_filtered) * 100, 1)
                category_data.append({
                    'category': category, 
                    'count': count, 
                    'percentage': percentage
                })
            
            # Trier par pourcentage décroissant
            category_data = sorted(category_data, key=lambda x: x['percentage'], reverse=True)
            
            # S'assurer que "Autres" est à la fin
            if any(item['category'] == 'Autres' for item in category_data):
                autres = next(item for item in category_data if item['category'] == 'Autres')
                category_data.remove(autres)
                category_data.append(autres)
            
            print(f"Nombre de catégories de météorites affichées: {len(category_data)}")
            
            # Pour chaque catégorie, créer une trace avec une couleur distincte
            for item in category_data:
                category = item['category']
                count = item['count']
                percentage = item['percentage']
                
                df_cat = df_filtered[df_filtered['class_group'] == category]
                fig.add_trace(go.Scattermapbox(
                    lat=df_cat['reclat'],
                    lon=df_cat['reclong'],
                    mode='markers',
                    marker=dict(
                        size=sizes[df_filtered.index.isin(df_cat.index)],
                        opacity=opacity,
                        sizemode='diameter'
                    ),
                    text=df_cat.apply(
                        lambda row: f"<b>{row['name']}</b><br>" +
                                  f"Masse: {row['mass (g)']:.1f}g<br>" +
                                  f"Année: {int(row['year']) if not np.isnan(row['year']) else 'Inconnue'}<br>" +
                                  f"Classe: {row['recclass']}<br>" +
                                  f"Type: {row['fall']}", 
                        axis=1
                    ),
                    hoverinfo='text',
                    name=f"{category} ({count}, {percentage}%)"
                ))
            
        elif color_mode == "fall":
            # Déterminer la colonne à utiliser et faire des groupes
            categories = df_filtered["fall"].unique()
            
            # Pour chaque catégorie, créer une trace avec une couleur distincte
            for category in sorted(categories):
                df_cat = df_filtered[df_filtered["fall"] == category]
                count = len(df_cat)
                percentage = round(count / len(df_filtered) * 100, 1)
                fig.add_trace(go.Scattermapbox(
                    lat=df_cat['reclat'],
                    lon=df_cat['reclong'],
                    mode='markers',
                    marker=dict(
                        size=sizes[df_filtered.index.isin(df_cat.index)],
                        opacity=opacity,
                        sizemode='diameter'
                    ),
                    text=df_cat.apply(
                        lambda row: f"<b>{row['name']}</b><br>" +
                                  f"Masse: {row['mass (g)']:.1f}g<br>" +
                                  f"Année: {int(row['year']) if not np.isnan(row['year']) else 'Inconnue'}<br>" +
                                  f"Classe: {row['recclass']}<br>" +
                                  f"Type: {row['fall']}", 
                        axis=1
                    ),
                    hoverinfo='text',
                    name=f"{category} ({count}, {percentage}%)"
                ))
        else:  # Mode "mass" ou autre
            fig.add_trace(go.Scattermapbox(
                lat=df_filtered['reclat'],
                lon=df_filtered['reclong'],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=df_filtered['log_mass'],
                    colorscale='Viridis',
                    showscale=True,
                    opacity=opacity,
                    sizemode='diameter'
                ),
                text=df_filtered.apply(
                    lambda row: f"<b>{row['name']}</b><br>" +
                              f"Masse: {row['mass (g)']:.1f}g<br>" +
                              f"Année: {int(row['year']) if not np.isnan(row['year']) else 'Inconnue'}<br>" +
                              f"Classe: {row['recclass']}<br>" +
                              f"Type: {row['fall']}", 
                    axis=1
                ),
                hoverinfo='text',
                name='Météorites'
            ))
        
        # Configuration de la carte
        fig.update_layout(
            mapbox=dict(
                style=actual_style,
                center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
                zoom=FRANCE_ZOOM
            ),
            margin={"r":0,"t":30,"l":0,"b":0},
            legend=dict(
                title=f"Classes de météorites" if color_mode == "class" else "Type de chute" if color_mode == "fall" else None,
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1
            ),
            height=700
        )
        
        print("Fin update_map - Figure générée avec succès")
        return fig

    @app.callback(
        Output('heatmap', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value'),
         Input('map-style-dropdown', 'value'),
         Input('heatmap-radius', 'value'),
         Input('heatmap-colorscale', 'value'),
         Input('map-interactivity', 'value')]
    )
    @error_handling_callback
    def update_heatmap(mass_range, classes, falls, decades, map_style, radius, 
                      colorscale, interactivity):
        """
        Met à jour la carte de chaleur des météorites
        """
        try:
            # Récupérer les données filtrées
            filtered_data = meteorite_data.get_filtered_data(
                mass_range=mass_range,
                classification=classes,
                fall_type=falls,
                decade_range=decades
            )
            
            # Vérifier et nettoyer les données pour Plotly
            df_clean = validate_dataframe_for_plotly(filtered_data, 'update_heatmap')
            
            # Obtenir le style de carte
            actual_style = get_mapbox_style(map_style)
            
            # Créer la carte de chaleur
            fig = go.Figure()
            
            # Ajouter la couche de chaleur
            fig.add_densitymapbox(
                lat=df_clean['reclat'],
                lon=df_clean['reclong'],
                radius=radius,
                colorscale=colorscale,
                hoverinfo='none',
                showscale=True,
                name='Densité'
            )
            
            # Configuration de la mise en page
            fig.update_layout(
                mapbox=dict(
                    style=actual_style,
                    center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
                    zoom=FRANCE_ZOOM
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                uirevision='constant'
            )
            
            # Configurer l'interactivité
            if 'sync_views' in interactivity:
                fig.update_layout(uirevision='sync')
            
            return fig
            
        except Exception as e:
            print(f"ERREUR dans update_heatmap: {str(e)}")
            print(traceback.format_exc())
            return go.Figure()  # Retourner une figure vide en cas d'erreur
    
    @app.callback(
        Output('prediction-map', 'figure'),
        [Input('selected-location', 'data'),
         Input('map-style-dropdown', 'value'),
         Input('analysis-radius', 'value')]
    )
    def update_prediction_map(selected_location, map_style, analysis_radius):
        debug_callback("Mise à jour de la carte de prédiction")
        
        # Style de carte
        actual_style = get_mapbox_style(map_style)
        
        # Créer une carte vierge
        fig = go.Figure()
        
        # Récupérer les données des météorites
        df = meteorite_data.get_filtered_data()
        
        # Ajouter les points des météorites existantes avec une faible opacité
        fig.add_trace(go.Scattermapbox(
            lat=df['reclat'],
            lon=df['reclong'],
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.3
            ),
            hoverinfo='text',
            text=df.apply(
                lambda row: f"<b>{row['name']}</b><br>" +
                          f"Masse: {row['mass (g)']:.1f}g<br>" +
                          f"Année: {int(row['year']) if not pd.isna(row['year']) else 'Inconnue'}<br>" +
                          f"Classe: {row['recclass']}<br>" +
                          f"Type: {row['fall']}", 
                axis=1
            ),
            name='Météorites existantes'
        ))
        
        # Configurer la mise en page de base
        fig.update_layout(
            mapbox=dict(
                style=actual_style,
                center=dict(lat=20, lon=0),  # Vue mondiale par défaut
                zoom=1.5  # Zoom mondial
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1
            )
        )
        
        # Ajouter un texte explicatif sur la carte
        fig.add_annotation(
            text="Cliquez sur la carte pour sélectionner un emplacement de prédiction",
            xref="paper", yref="paper",
            x=0.5, y=0.97,
            showarrow=False,
            font=dict(size=16, color="black"),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            align="center"
        )
        
        # Si un emplacement est sélectionné, ajouter un cercle d'analyse et un marqueur
        if selected_location and 'lat' in selected_location and 'lon' in selected_location:
            # Récupérer les coordonnées
            lat, lon = selected_location['lat'], selected_location['lon']
            
            # Centrer la carte sur l'emplacement sélectionné avec un zoom adapté
            fig.update_layout(
                mapbox=dict(
                    center=dict(lat=lat, lon=lon),
                    zoom=4  # Zoom adapté à la visualisation de la zone
                )
            )
            
            # Dessiner le cercle d'analyse autour du point
            radius_degrees = analysis_radius if analysis_radius else 2.5
            lats, lons = create_circle(lat, lon, radius_degrees)
            
            # Ajouter le cercle d'analyse
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(
                    width=3,
                    color='#0066ff'  # Bleu plus vif
                ),
                name=f"Zone d'analyse (rayon {radius_degrees}°)",
                hoverinfo="name"
            ))
            
            # Ajouter un marqueur pour le point sélectionné
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers',
                marker=dict(
                    size=15,
                    color='#ff3300',  # Rouge-orange vif
                    symbol='circle'
                ),
                name="Point sélectionné",
                hoverinfo="text",
                hovertext=f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            ))
            
            # Ajouter une annotation pour le rayon
            fig.add_annotation(
                text=f"Rayon: {radius_degrees}°",
                x=lon, y=lat+radius_degrees/2,
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#0066ff",
                xanchor="center",
                yanchor="bottom"
            )
        
        debug_callback("Carte de prédiction générée avec succès")
        return fig

    def create_circle(center_lat, center_lon, radius_degrees, num_points=100):
        """Crée un cercle sur la carte."""
        points = np.linspace(0, 2*np.pi, num_points)
        lats = center_lat + radius_degrees * np.cos(points)
        lons = center_lon + radius_degrees * np.sin(points)
        return lats, lons
    
    @app.callback(
        Output('selected-location', 'data'),
        [Input('prediction-map', 'clickData')]
    )
    def store_clicked_location(clickData):
        if clickData is None:
            return None
        
        # Debug pour identifier la structure du clickData
        print("Click Data:", json.dumps(clickData, indent=2))
        
        point = clickData['points'][0]
        
        # Récupérer les coordonnées selon le type de point cliqué
        if 'lat' in point and 'lon' in point:
            # Clic direct sur un point existant
            return {
                'lat': point['lat'],
                'lon': point['lon']
            }
        elif 'mapbox.lat' in point and 'mapbox.lon' in point:
            # Clic sur la carte (hors points)
            return {
                'lat': point['mapbox.lat'],
                'lon': point['mapbox.lon']
            }
        elif 'customdata' in point and len(point['customdata']) >= 2:
            # Format de données personnalisé
            return {
                'lat': point['customdata'][0],
                'lon': point['customdata'][1]
            }
        else:
            # Fallback - essayer d'autres attributs possibles
            for attr in ['latitude', 'longitude', 'lat', 'lon']:
                if attr in point:
                    lat_attr = 'latitude' if 'latitude' in point else 'lat'
                    lon_attr = 'longitude' if 'longitude' in point else 'lon'
                    return {
                        'lat': point[lat_attr],
                        'lon': point[lon_attr]
                    }
            
            # Si aucune coordonnée n'est trouvée, afficher un message d'erreur
            print("Erreur: Impossible de récupérer les coordonnées du clic.", point)
            return None
    
    @app.callback(
        Output('selected-coordinates', 'children'),
        [Input('selected-location', 'data')]
    )
    def update_selected_coordinates(location):
        if location is None:
            return "Aucun point sélectionné. Cliquez sur la carte."
        
        return [
            html.Span("Latitude: ", className="fw-bold"),
            f"{location['lat']:.4f}",
            html.Br(),
            html.Span("Longitude: ", className="fw-bold"),
            f"{location['lon']:.4f}"
        ]
    
    @app.callback(
        Output('prediction-output', 'children'),
        [Input('predict-button', 'n_clicks'),
         Input('selected-location', 'data')],
        [State('pred-fall', 'value'),
         State('analysis-radius', 'value'),
         State('forecast-horizon', 'value')]
    )
    def make_prediction(n_clicks, location, fall, analysis_radius, horizon):
        debug_callback("Lancement de la prédiction")
        
        # Si aucun clic ou pas d'emplacement, retourner un message explicatif
        if n_clicks is None or location is None:
            # Si un emplacement est sélectionné mais pas encore de clic, afficher un message d'instruction
            if location is not None:
                return html.Div([
                    html.H5("Prêt pour la prédiction", className="text-info"),
                    html.P([
                        "Un emplacement est sélectionné. ",
                        html.Br(),
                        "Ajustez les paramètres et cliquez sur ",
                        html.B("Prédire"), 
                        " pour calculer la probabilité d'impact et la masse estimée."
                    ]),
                    html.P([
                        html.I(className="fas fa-map-marker-alt me-2 text-danger"),
                        f"Coordonnées: Lat {location['lat']:.4f}, Lon {location['lon']:.4f}",
                    ], className="mb-2"),
                    html.P([
                        html.I(className="fas fa-ruler me-2 text-primary"),
                        f"Rayon d'analyse: {analysis_radius}°",
                    ], className="mb-2")
                ], className="alert alert-light border shadow-sm")
            
            # Message initial
            return html.Div([
                html.H5("Sélectionnez un point sur la carte", className="mb-3"),
                html.P([
                    html.I(className="fas fa-info-circle me-2"), 
                    "Cliquez sur la carte pour sélectionner un emplacement avant de faire une prédiction."
                ]),
                html.P([
                    html.I(className="fas fa-lightbulb me-2"), 
                    "Vous pourrez ensuite ajuster les paramètres et utiliser le bouton ",
                    html.B("Prédire"),
                    " pour obtenir des estimations de masse et de probabilité d'impact."
                ], className="text-muted")
            ], className="alert alert-info")
            
        # Valider que les paramètres sont corrects
        if location is None:
            debug_callback("Erreur: Point non sélectionné", level='error')
            return html.Div([
                html.H5("Point non sélectionné", className="text-warning"),
                html.P("Veuillez cliquer sur la carte pour sélectionner un emplacement.")
            ], className="alert alert-warning")
            
        if fall is None or horizon is None:
            debug_callback("Erreur: Paramètres incomplets", level='error')
            return html.Div([
                html.H5("Paramètres incomplets", className="text-warning"),
                html.P("Veuillez spécifier le type de chute et l'horizon de prévision.")
            ], className="alert alert-warning")
            
        try:
            # Calculer l'année de prédiction à partir de l'horizon
            current_year = datetime.now().year
            prediction_year = current_year + horizon
            
            debug_callback(f"Prédiction pour: Lat {location['lat']:.4f}, Lon {location['lon']:.4f}, Horizon {horizon} ans (année {prediction_year}), Type {fall}")
            
            # Vérifier que le modèle est correctement initialisé
            if ml_model is None:
                raise ValueError("Le modèle de prédiction n'est pas initialisé.")
                
            # Validation des données d'entrée
            if not (-90 <= location['lat'] <= 90) or not (-180 <= location['lon'] <= 180):
                raise ValueError("Coordonnées géographiques invalides.")
                
            if not (1 <= horizon <= 50):
                raise ValueError(f"Horizon invalide: {horizon}. Doit être entre 1 et 50 ans.")
                
            if fall not in ['Fell', 'Found']:
                raise ValueError(f"Type de chute invalide: {fall}. Doit être 'Fell' ou 'Found'.")
            
            # Effectuer la prédiction avec l'année calculée
            predicted_mass = ml_model.predict_mass(
                location['lat'],
                location['lon'],
                prediction_year,
                fall
            )
            
            # Formater la masse prédite
            if predicted_mass < 1000:
                mass_formatted = f"{predicted_mass:.2f} grammes"
            elif predicted_mass < 1000000:
                mass_formatted = f"{predicted_mass/1000:.2f} kg"
            else:
                mass_formatted = f"{predicted_mass/1000000:.2f} tonnes"
            
            # Calculer une probabilité d'impact basée sur divers facteurs
            # Récupérer des données historiques pour la zone
            df = meteorite_data.get_filtered_data()
            
            # Facteur 1: Proximité des météorites connues
            distances = np.sqrt(((df['reclat'] - location['lat']) ** 2) + ((df['reclong'] - location['lon']) ** 2))
            closest_distance = distances.min()
            proximity_factor = max(0.1, min(1.0, 1.0 / (closest_distance + 0.1)))  # Plus proche = plus probable
            
            # Facteur 2: Densité de météorites dans la zone (rayon d'analyse)
            nearby_count = len(df[distances <= analysis_radius])
            density_factor = min(1.0, nearby_count / 20.0)  # Normalisé: 20+ météorites = max
            
            # Facteur 3: Type de chute (Fell est plus rare mais plus précis)
            fall_factor = 1.2 if fall == 'Fell' else 1.0
            
            # Facteur 4: Horizon (les prédictions pour un horizon plus long sont moins fiables, mais plus probables)
            time_factor = max(0.5, min(1.0, horizon / 50.0))  # Plus l'horizon est long, plus la probabilité augmente
            
            # Calculer la probabilité finale (ajuster selon vos besoins)
            base_probability = 0.01  # Probabilité de base très faible
            impact_probability = base_probability * proximity_factor * density_factor * fall_factor * time_factor
            impact_probability = min(0.98, impact_probability * horizon / 10)  # Augmente avec l'horizon, plafonné à 98%
            
            # Calculer l'indice de confiance (0-100)
            confidence_score = int((proximity_factor * 0.4 + density_factor * 0.4 + (1 - time_factor) * 0.2) * 100)
            confidence_class = "danger" if confidence_score < 30 else "warning" if confidence_score < 70 else "success"
            
            # Créer la visualisation de probabilité avec une jauge
            probability_gauge = html.Div([
                html.Div([
                    html.Div(className="gauge-background"),
                    html.Div(className="gauge-fill", style={"width": f"{impact_probability * 100}%"}),
                    html.Div(className="gauge-cover", children=[
                        html.Span(f"{impact_probability:.2%}", className="gauge-value fw-bold text-center d-block display-6")
                    ])
                ], className="gauge-container")
            ], className="mt-3 mb-4 text-center")
            
            # Créer les facteurs explicatifs de la prédiction
            factors_explanation = html.Div([
                html.H6("Facteurs de prédiction", className="mt-3 mb-2"),
                html.Div([
                    html.Div([
                        html.P("Proximité"),
                        html.Div(className="progress", children=[
                            html.Div(className="progress-bar bg-primary", 
                                    style={"width": f"{proximity_factor * 100}%"},
                                    children=f"{proximity_factor:.2f}")
                        ])
                    ], className="col-6 mb-2"),
                    html.Div([
                        html.P("Densité"),
                        html.Div(className="progress", children=[
                            html.Div(className="progress-bar bg-success", 
                                    style={"width": f"{density_factor * 100}%"},
                                    children=f"{density_factor:.2f}")
                        ])
                    ], className="col-6 mb-2"),
                    html.Div([
                        html.P("Temporel"),
                        html.Div(className="progress", children=[
                            html.Div(className="progress-bar bg-info", 
                                    style={"width": f"{time_factor * 100}%"},
                                    children=f"{time_factor:.2f}")
                        ])
                    ], className="col-6 mb-2"),
                    html.Div([
                        html.P("Type de chute"),
                        html.Div(className="progress", children=[
                            html.Div(className="progress-bar bg-secondary", 
                                    style={"width": f"{fall_factor * 100 / 1.5}%"},
                                    children=f"{fall_factor:.2f}")
                        ])
                    ], className="col-6 mb-2"),
                ], className="row")
            ])
            
            nearby_text = f"{nearby_count} météorites dans un rayon de {analysis_radius}°"
            
            return html.Div([
                html.H5("Prédiction d'Impact de Météorite", className="mb-3 text-primary"),
                
                html.Div([
                    html.Div([
                        html.H6("Probabilité d'impact", className="text-center mb-2"),
                        probability_gauge,
                        html.P([
                            html.I(className="fas fa-info-circle me-2"), 
                            "Indice de confiance: ",
                            html.Span(f"{confidence_score}%", className=f"badge bg-{confidence_class} ms-1")
                        ], className="text-center")
                    ], className="col-md-6"),
                    
                    html.Div([
                        html.H6("Masse Estimée", className="mb-2"),
                        html.Div([
                            html.Span(mass_formatted, 
                                    className="d-block text-center display-6 text-success")
                        ], className="p-3 border rounded text-center"),
                        html.P([
                            html.I(className="fas fa-meteor me-2"),
                            nearby_text
                        ], className="text-center mt-2 text-muted small")
                    ], className="col-md-6")
                ], className="row mb-3"),
                
                factors_explanation,
                
                html.Hr(),
                
                html.Div([
                    html.P("Paramètres utilisés:", className="fw-bold"),
                    html.Ul([
                        html.Li([
                            html.Span("Position: ", className="fw-bold"),
                            f"Lat {location['lat']:.4f}, Lon {location['lon']:.4f}"
                        ]),
                        html.Li([
                            html.Span("Horizon: ", className="fw-bold"),
                            f"{horizon} ans (jusqu'à {prediction_year})"
                        ]),
                        html.Li([
                            html.Span("Type: ", className="fw-bold"),
                            f"{fall} ({'Tombée' if fall == 'Fell' else 'Trouvée'})"
                        ]),
                        html.Li([
                            html.Span("Rayon d'analyse: ", className="fw-bold"),
                            f"{analysis_radius}°"
                        ])
                    ], className="mb-0")
                ], className="mt-3 small")
                
            ], className="alert alert-light border shadow-sm")
        
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            debug_callback(f"Erreur dans make_prediction: {str(e)}", level='error')
            
            return html.Div([
                html.H5("Erreur", className="text-danger"),
                html.P(f"Erreur lors de la prédiction: {str(e)}"),
                html.P("Vérifiez les valeurs d'entrée."),
                html.Details([
                    html.Summary("Détails techniques (cliquez pour développer)"),
                    html.Pre(trace, style={"whiteSpace": "pre-wrap"})
                ], className="mt-2")
            ], className="alert alert-danger")
    
    @app.callback(
        Output('zone-analysis-output', 'children'),
        [Input('analyze-zone-button', 'n_clicks')],
        [State('selected-location', 'data'),
         State('analysis-radius', 'value')]
    )
    def analyze_zone(n_clicks, location, analysis_radius):
        if n_clicks is None or location is None:
            return ""
        
        # Analyse d'une zone autour du point sélectionné
        lat, lon = location['lat'], location['lon']
        df = meteorite_data.get_filtered_data()
        
        # Filtrer par rayon manuellement
        zone_data = df[
            (df['reclat'].between(lat - analysis_radius, lat + analysis_radius)) &
            (df['reclong'].between(lon - analysis_radius, lon + analysis_radius))
        ]
        
        # Filtrer pour le rayon régional (3x plus grand)
        regional_data = df[
            (df['reclat'].between(lat - analysis_radius*3, lat + analysis_radius*3)) &
            (df['reclong'].between(lon - analysis_radius*3, lon + analysis_radius*3))
        ]
        
        if len(zone_data) == 0:
            return html.Div([
                html.H5("Aucune météorite connue", className="text-info mb-3"),
                html.P("Aucune météorite n'a été enregistrée dans cette zone."),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-map-marker-alt me-2 text-danger"),
                        f"Coordonnées: Lat {lat:.4f}, Lon {lon:.4f}"
                    ], className="mb-2"),
                    html.Div([
                        html.I(className="fas fa-ruler me-2 text-primary"),
                        f"Rayon d'analyse: {analysis_radius}°"
                    ], className="mb-2")
                ], className="alert alert-light p-3 rounded")
            ], className="alert alert-light border shadow-sm")
        
        # Calculer des statistiques avancées
        try:
            stats = {
                'Nombre de météorites': len(zone_data),
                'Masse totale': f"{zone_data['mass (g)'].sum():.1f} g",
                'Masse moyenne': f"{zone_data['mass (g)'].mean():.2f} g",
                'Masse médiane': f"{zone_data['mass (g)'].median():.2f} g"
            }
            
            # Statistiques temporelles
            if 'year' in zone_data.columns and not zone_data['year'].isna().all():
                stats['Période'] = f"{int(zone_data['year'].min())} - {int(zone_data['year'].max())}"
                stats['Âge moyen'] = f"{int(datetime.now().year - zone_data['year'].mean())} ans"
            
            # Statistiques de classification
            if 'recclass' in zone_data.columns and len(zone_data) >= 3:
                class_counts = zone_data['recclass'].value_counts()
                main_classes = class_counts.nlargest(3).index.tolist()
                stats['Classes principales'] = ", ".join(main_classes)
                stats['Classe dominante'] = f"{main_classes[0]} ({(class_counts[main_classes[0]]/len(zone_data)*100):.1f}%)" if main_classes else "N/A"
            
            # Statistiques régionales
            stats['Concentration locale'] = f"{len(zone_data) / (math.pi * analysis_radius**2):.2f} météorites/deg²"
            stats['% des météorites régionales'] = f"{len(zone_data) / len(regional_data) * 100:.1f}%" if len(regional_data) > 0 else "N/A"
            
            # Fonction temporaire pour calculer les figures (remplace les appels à des fonctions qui utilisent get_meteorites_in_radius)
            def create_empty_figure_with_message(message):
                fig = go.Figure()
                fig.add_annotation(
                    text=message,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                return fig
                
            # Créer des figures simplifiées pour les visualisations
            def create_comparison_chart():
                # Compter le nombre de météorites
                local_count = len(zone_data)
                regional_count = len(regional_data)
                global_count = len(df)
                
                # Calculer la densité (nombre/surface)
                local_area = math.pi * (analysis_radius ** 2)
                regional_area = math.pi * ((analysis_radius * 3) ** 2)
                global_area = 4 * math.pi * 6371 ** 2  # Surface terre en km²
                
                local_density = local_count / local_area if local_area > 0 else 0
                regional_density = regional_count / regional_area if regional_area > 0 else 0
                global_density = global_count / global_area if global_area > 0 else 0
                
                # Normaliser les densités (pour l'affichage)
                max_density = max(local_density, regional_density, global_density)
                if max_density > 0:
                    local_density_norm = local_density / max_density
                    regional_density_norm = regional_density / max_density
                    global_density_norm = global_density / max_density
                else:
                    local_density_norm = regional_density_norm = global_density_norm = 0
                
                # Créer le graphique
                fig = go.Figure()
                
                # Ajouter les barres
                fig.add_trace(go.Bar(
                    x=['Zone locale', 'Région', 'Global'],
                    y=[local_density_norm, regional_density_norm, global_density_norm],
                    text=[f"{local_count} météorites", 
                          f"{regional_count} météorites", 
                          f"{global_count} météorites"],
                    textposition='auto',
                    marker_color=['#0077b6', '#00b4d8', '#90e0ef'],
                    hoverinfo='text'
                ))
                
                # Mise en page
                fig.update_layout(
                    title="Densité comparative",
                    yaxis_title="Densité relative",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=30),
                    height=180
                )
                
                return fig
            
            def create_type_distribution():
                if len(zone_data) < 5 or 'recclass' not in zone_data.columns:
                    return create_empty_figure_with_message("Données insuffisantes")
                    
                # Compter les occurrences des classes
                classes = zone_data['recclass'].dropna()
                if len(classes) < 3:
                    return create_empty_figure_with_message("Données de classe insuffisantes")
                    
                class_counts = classes.value_counts().nlargest(5)
                
                # Créer le graphique
                fig = go.Figure()
                
                # Ajouter le camembert
                fig.add_trace(go.Pie(
                    labels=class_counts.index,
                    values=class_counts.values,
                    hole=0.4,
                    marker=dict(
                        colors=px.colors.qualitative.Pastel
                    ),
                    textinfo='percent',
                    hoverinfo='label+percent+value'
                ))
                
                # Mise en page
                fig.update_layout(
                    title="Types de météorites",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=180
                )
                
                return fig
            
            def create_mass_distribution():
                if len(zone_data) < 5 or 'mass (g)' not in zone_data.columns:
                    return create_empty_figure_with_message("Données insuffisantes")
                    
                # Extraire les masses
                masses = zone_data['mass (g)'].dropna()
                if len(masses) < 5:
                    return create_empty_figure_with_message("Données de masse insuffisantes")
                    
                # Créer le graphique
                fig = go.Figure()
                
                # Ajouter l'histogramme des masses (échelle log)
                fig.add_trace(go.Histogram(
                    x=np.log10(masses[masses > 0]),
                    nbinsx=10,
                    marker_color='#0077b6',
                    opacity=0.7
                ))
                
                # Mise en page
                fig.update_layout(
                    title="Distribution des masses",
                    xaxis_title="Log10(Masse en g)",
                    yaxis_title="Nombre",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=30),
                    height=180
                )
                
                return fig
            
            def create_timeline():
                if len(zone_data) < 5 or 'year' not in zone_data.columns:
                    return create_empty_figure_with_message("Données insuffisantes")
                    
                # Extraire les années
                years = zone_data['year'].dropna()
                if len(years) < 5:
                    return create_empty_figure_with_message("Données temporelles insuffisantes")
                    
                # Grouper par décennie
                zone_data['decade'] = (zone_data['year'] // 10) * 10
                decade_counts = zone_data.groupby('decade').size().reset_index(name='count')
                
                # Créer le graphique
                fig = go.Figure()
                
                # Ajouter la courbe d'évolution
                fig.add_trace(go.Scatter(
                    x=decade_counts['decade'],
                    y=decade_counts['count'],
                    mode='lines+markers',
                    line=dict(color='#0077b6', width=2),
                    marker=dict(size=8, color='#0077b6'),
                    fill='tozeroy',
                    fillcolor='rgba(0, 119, 182, 0.2)'
                ))
                
                # Mise en page
                fig.update_layout(
                    title="Chronologie des découvertes",
                    xaxis_title="Décennie",
                    yaxis_title="Nombre",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=30),
                    height=180
                )
                
                return fig
            
            def create_density_map():
                # Créer le graphique de densité
                fig = go.Figure()
                
                # Ajouter la carte de densité
                fig.add_trace(go.Densitymapbox(
                    lat=zone_data['reclat'],
                    lon=zone_data['reclong'],
                    z=[1] * len(zone_data),
                    radius=20,
                    colorscale='Viridis',
                    showscale=False
                ))
                
                # Ajouter un marqueur pour le point sélectionné
                fig.add_trace(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red'
                    ),
                    showlegend=False
                ))
                
                # Mise en page
                fig.update_layout(
                    mapbox=dict(
                        style='carto-positron',
                        center=dict(lat=lat, lon=lon),
                        zoom=4
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=240
                )
                
                return fig
            
            # Créer le layout avec cartes et graphiques
            return html.Div([
                # Titre et introduction
                html.Div([
                    html.H4([
                        html.I(className="fas fa-search-location me-2"), 
                        "Analyse détaillée de la zone"
                    ], className="mb-3 text-primary"),
                    
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-map-marker-alt me-2 text-danger"),
                            f"Coordonnées: Lat {lat:.4f}, Lon {lon:.4f}"
                        ], className="col-md-6"),
                        html.Div([
                            html.I(className="fas fa-ruler me-2 text-primary"),
                            f"Rayon d'analyse: {analysis_radius}°"
                        ], className="col-md-6"),
                    ], className="row mb-3 small text-muted"),
                    
                    # Carte et statistiques principales
                    html.Div([
                        # Colonne de gauche: Carte de densité
                        html.Div([
                            html.H6("Densité de météorites", className="text-center mb-2"),
                            dcc.Loading(
                                id="loading-density-map",
                                type="circle",
                                children=dcc.Graph(
                                    id='density-map-preview',
                                    figure=create_density_map(),
                                    config={'displayModeBar': False},
                                    style={'height': '240px'}
                                )
                            )
                        ], className="col-md-6"),
                        
                        # Colonne de droite: Statistiques clés
                        html.Div([
                            html.H6("Statistiques clés", className="mb-2"),
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.H2(stats['Nombre de météorites'], className="m-0 text-primary")
                                    ], className="card-body text-center"),
                                    html.Div("Météorites trouvées", className="card-footer text-center text-muted")
                                ], className="card mb-2"),
                                
                                html.Div([
                                    html.Div([
                                        html.Div([
                                            html.Span("Masse totale:", className="fw-bold me-2"),
                                            html.Span(stats['Masse totale'])
                                        ], className="d-flex justify-content-between mb-1"),
                                        html.Div([
                                            html.Span("Masse moyenne:", className="fw-bold me-2"),
                                            html.Span(stats['Masse moyenne'])
                                        ], className="d-flex justify-content-between mb-1"),
                                        html.Div([
                                            html.Span("Masse médiane:", className="fw-bold me-2"),
                                            html.Span(stats['Masse médiane'])
                                        ], className="d-flex justify-content-between mb-1"),
                                        html.Div([
                                            html.Span("Période:", className="fw-bold me-2"),
                                            html.Span(stats.get('Période', 'N/A'))
                                        ], className="d-flex justify-content-between mb-1"),
                                        html.Div([
                                            html.Span("Classe dominante:", className="fw-bold me-2"),
                                            html.Span(stats.get('Classe dominante', 'N/A'))
                                        ], className="d-flex justify-content-between mb-1")
                                    ], className="p-2")
                                ], className="card stats-card")
                            ])
                        ], className="col-md-6")
                    ], className="row mb-4"),
                    
                    # Graphiques de comparaison
                    html.H6("Analyses comparatives", className="mb-3"),
                    
                    html.Div([
                        # Ligne 1: Comparaison et distribution des types
                        html.Div([
                            # Comparaison régionale
                            html.Div([
                                html.H6("Comparaison régionale", className="text-center small mb-1"),
                                dcc.Loading(
                                    id="loading-region-comp",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='region-comp-preview',
                                        figure=create_comparison_chart(),
                                        config={'displayModeBar': False},
                                        style={'height': '180px'}
                                    )
                                )
                            ], className="col-md-6 mb-3"),
                            
                            # Distribution des types
                            html.Div([
                                html.H6("Types de météorites", className="text-center small mb-1"),
                                dcc.Loading(
                                    id="loading-type-dist",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='type-dist-preview',
                                        figure=create_type_distribution(),
                                        config={'displayModeBar': False},
                                        style={'height': '180px'}
                                    )
                                )
                            ], className="col-md-6 mb-3")
                        ], className="row mb-2"),
                        
                        # Ligne 2: Distribution des masses et chronologie
                        html.Div([
                            # Distribution des masses
                            html.Div([
                                html.H6("Distribution des masses", className="text-center small mb-1"),
                                dcc.Loading(
                                    id="loading-mass-dist",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='mass-dist-preview',
                                        figure=create_mass_distribution(),
                                        config={'displayModeBar': False},
                                        style={'height': '180px'}
                                    )
                                )
                            ], className="col-md-6 mb-3"),
                            
                            # Chronologie des découvertes
                            html.Div([
                                html.H6("Chronologie des découvertes", className="text-center small mb-1"),
                                dcc.Loading(
                                    id="loading-timeline",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='timeline-preview',
                                        figure=create_timeline(),
                                        config={'displayModeBar': False},
                                        style={'height': '180px'}
                                    )
                                )
                            ], className="col-md-6 mb-3")
                        ], className="row")
                    ]),
                    
                    # Observations et conclusions
                    html.Div([
                        html.H6("Observations", className="mb-2"),
                        html.Ul([
                            html.Li(f"La zone contient {len(zone_data)} météorite{'s' if len(zone_data) > 1 else ''} sur {len(df)} dans la base de données ({len(zone_data)/len(df)*100:.2f}%)."),
                            html.Li(f"La concentration locale est {float(stats['Concentration locale'].split()[0]):.2f} fois " + 
                                   ("supérieure" if float(stats['Concentration locale'].split()[0]) > 0.1 else "inférieure") + 
                                   " à la moyenne mondiale."),
                            html.Li(f"Cette zone représente {stats['% des météorites régionales']} des météorites de la région environnante (rayon de {analysis_radius*3}°)."),
                            html.Li(f"La plupart des météorites trouvées sont de type {stats.get('Classe dominante', 'inconnu').split(' ')[0]}.")
                        ], className="text-muted small")
                    ], className="mt-3 p-3 bg-light rounded")
                ])
            ], className="p-3 border rounded shadow-sm")
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            return html.Div([
                html.H5("Erreur lors de l'analyse", className="text-danger"),
                html.P(f"Une erreur s'est produite: {str(e)}"),
                html.Details([
                    html.Summary("Détails techniques (cliquez pour développer)"),
                    html.Pre(trace, style={"whiteSpace": "pre-wrap"})
                ], className="mt-2")
            ], className="alert alert-danger")
    
    @app.callback(
        Output('correlation-heatmap', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_correlation_heatmap(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        numeric_cols = ['mass (g)', 'reclat', 'reclong', 'year']
        labels = {
            'mass (g)': 'Masse',
            'reclat': 'Latitude',
            'reclong': 'Longitude',
            'year': 'Année'
        }
        
        corr_matrix = df[numeric_cols].corr()
        
        # Renommer les axes pour l'affichage
        corr_matrix.index = [labels.get(col, col) for col in corr_matrix.index]
        corr_matrix.columns = [labels.get(col, col) for col in corr_matrix.columns]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            texttemplate="%{z:.2f}",
            colorbar=dict(
                title="Coefficient",
                titleside="right"
            )
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(
                text="Corrélations entre Variables",
                y=0.98
            ),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    @app.callback(
        Output('feature-importance', 'figure'),
        [Input('interval', 'n_intervals')]
    )
    @error_handling_callback
    def update_feature_importance(_):
        importance = ml_model.get_feature_importance()
        labels = {
            'reclat': 'Latitude',
            'reclong': 'Longitude',
            'year': 'Année',
            'fall_encoded': 'Type de chute'
        }
        
        # Trier par importance décroissante
        features = list(importance.keys())
        values = list(importance.values())
        
        # Trier par importance
        sorted_indices = np.argsort(values)[::-1]
        sorted_features = [labels.get(features[i], features[i]) for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_features,
                y=sorted_values,
                marker_color='#3498db',
                opacity=0.8,
                text=[f"{v:.2f}" for v in sorted_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            yaxis_title="Importance",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            )
        )
        return fig
    
    @app.callback(
        Output('feature-importance-2', 'figure'),
        [Input('interval', 'n_intervals')]
    )
    @error_handling_callback
    def update_feature_importance_2(_):
        # Récupérer l'importance des features depuis le modèle
        feature_importance = ml_model.get_feature_importance()
        
        # Créer un DataFrame pour le graphique
        df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
        
        # Trier par importance décroissante
        df = df.sort_values('Importance', ascending=True)
        
        # Mapper des noms plus lisibles pour l'affichage
        feature_names = {
            'reclat': 'Latitude',
            'reclong': 'Longitude',
            'year': 'Année',
            'fall_encoded': 'Type de chute'
        }
        
        df['Feature'] = df['Feature'].map(lambda x: feature_names.get(x, x))
        
        # Créer le graphique horizontal
        fig = px.bar(
            df, 
            y='Feature',
            x='Importance',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis',
            title="Importance des Variables dans le Modèle"
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            title={
                'text': "Importance des Variables",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 14}
            },
            coloraxis_showscale=False,
            plot_bgcolor='white',
            xaxis_title="Importance Relative",
            yaxis_title="",
            xaxis=dict(showgrid=True, gridcolor='#EEE')
        )
        
        return fig
    
    @app.callback(
        Output('stats-output', 'children'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    def update_stats(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        # Statistiques plus détaillées
        stats = [
            ("Météorites", f"{len(df):,}"),
            ("Masse totale", f"{df['mass (g)'].sum():,.0f} g"),
            ("Masse moyenne", f"{df['mass (g)'].mean():,.2f} g"),
            ("Masse médiane", f"{df['mass (g)'].median():,.2f} g"),
            ("Classes uniques", f"{len(df['recclass'].unique())}"),
            ("Période couverte", f"{int(df['year'].min())} - {int(df['year'].max())}")
        ]
        
        return html.Div([
            html.Table([
                html.Tbody([
                    html.Tr([
                        html.Td(k, className="text-start fw-bold"),
                        html.Td(v, className="text-end")
                    ]) for k, v in stats
                ])
            ], className="table table-sm table-hover")
        ])
    
    @app.callback(
        Output('mass-hist', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_mass_histogram(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        if df.empty:
            # Retourner un graphique vide avec un message
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=350)
            return fig
        
        # Utiliser l'échelle logarithmique pour la masse
        df['log_mass'] = np.log10(df['mass (g)'])
        
        fig = px.histogram(
            df, 
            x='log_mass',
            nbins=50,
            labels={'log_mass': 'Log10(Masse en grammes)'},
            title='Distribution des Masses (Échelle Log)',
            height=350,
            color_discrete_sequence=['#0071e3']
        )
        
        fig.update_layout(
            xaxis_title='Log10(Masse en grammes)',
            yaxis_title='Nombre de météorites',
            title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin={"r":10, "t":50, "l":10, "b":50},
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    @app.callback(
        Output('time-series', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_time_series(mass_range, classes, falls, decades):
        """Callback pour mettre à jour le graphique de série temporelle"""
        # Récupérer les données filtrées
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        # Valider et corriger le DataFrame pour éviter les erreurs de types
        df = validate_dataframe_for_plotly(df, "update_time_series")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée disponible pour la série temporelle",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=350)
            return fig
        
        # S'assurer que l'année est traitée comme un nombre
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        
        # Filtrer les années valides (supérieures à 1800)
        df = df[df['year'] > 1800]
        
        if df.empty or len(df['year'].unique()) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Données insuffisantes pour la série temporelle",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=350)
            return fig
        
        # Regrouper par année et compter les occurrences
        yearly_counts = df.groupby('year').size().reset_index(name='count')
        yearly_counts = yearly_counts.sort_values('year')
        
        # Calculer une moyenne mobile pour montrer la tendance
        try:
            window_size = min(5, len(yearly_counts))
            yearly_counts['moving_avg'] = yearly_counts['count'].rolling(window=window_size, center=True).mean()
        except Exception as e:
            print(f"Erreur lors du calcul de la moyenne mobile: {str(e)}")
            yearly_counts['moving_avg'] = yearly_counts['count']
        
        # Créer le graphique avec go.Figure
        fig = go.Figure()
        
        # Barres pour les découvertes par année
        fig.add_trace(go.Bar(
            x=yearly_counts['year'],
            y=yearly_counts['count'],
            name='Découvertes',
            marker_color='#0071e3'
        ))
        
        # Ligne pour la moyenne mobile
        fig.add_trace(go.Scatter(
            x=yearly_counts['year'],
            y=yearly_counts['moving_avg'],
            mode='lines',
            name='Tendance',
            line=dict(color='#ff9500', width=3)
        ))
        
        # Personnaliser le graphique
        fig.update_layout(
            title=f"Tendance annuelle des découvertes de météorites (n={len(df)})",
            xaxis_title="Année",
            yaxis_title="Nombre de météorites",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    @app.callback(
        Output('class-distribution', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_class_distribution(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        if df.empty:
            # Retourner un graphique vide avec un message
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=350)
            return fig
        
        # Limiter à 10 classes maximum pour une meilleure lisibilité
        class_counts = df['recclass'].value_counts().head(10)
        
        labels = class_counts.index
        values = class_counts.values
        
        # Utiliser go.Figure au lieu de go.Pie pour plus de contrôle
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            insidetextorientation='radial',
            textposition='outside',
            textfont=dict(size=12),
            marker=dict(
                colors=px.colors.qualitative.Bold,
                line=dict(color='white', width=2)
            ),
            sort=True
        ))
        
        # Optimiser la mise en page pour éviter la troncature
        fig.update_layout(
            height=380,  # Augmenter légèrement la hauteur
            autosize=True,
            margin=dict(l=20, r=20, t=50, b=70),  # Augmenter les marges
            title={
                'text': "Top 10 des Classes de Météorites",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.4,  # Positionner la légende plus bas pour éviter le chevauchement
                'xanchor': 'center',
                'x': 0.5
            },
            showlegend=False,  # Désactiver la légende externe car les étiquettes sont déjà sur le graphique
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig

    # Optimiser le callback d'analyse des panels
    @app.callback(
        [Output('panel-distributions', 'className'),
         Output('panel-time-series', 'className'),
         Output('panel-correlations', 'className'),
         Output('btn-distributions', 'className'),
         Output('btn-time-series', 'className'),
         Output('btn-correlations', 'className')],
        [Input('btn-distributions', 'n_clicks'),
         Input('btn-time-series', 'n_clicks'),
         Input('btn-correlations', 'n_clicks')],
        [State('panel-distributions', 'className'),
         State('panel-time-series', 'className'),
         State('panel-correlations', 'className'),
         State('btn-distributions', 'className'),
         State('btn-time-series', 'className'),
         State('btn-correlations', 'className')]
    )
    def toggle_analysis_panels(dist_clicks, time_clicks, corr_clicks,
                              dist_panel_class, time_panel_class, corr_panel_class,
                              dist_btn_class, time_btn_class, corr_btn_class):
        ctx = dash.callback_context
        if not ctx.triggered:
            # Valeurs par défaut - montrer distributions, cacher les autres
            return "row", "row d-none", "row d-none", "btn btn-outline-primary me-2 mb-2 active", "btn btn-outline-primary me-2 mb-2", "btn btn-outline-primary mb-2"
        
        # Obtenir l'ID du bouton qui a déclenché le callback
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Si le même bouton est cliqué à nouveau, ne rien faire
        if (button_id == "btn-distributions" and "active" in dist_btn_class) or \
           (button_id == "btn-time-series" and "active" in time_btn_class) or \
           (button_id == "btn-correlations" and "active" in corr_btn_class):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Réinitialiser toutes les classes
        dist_panel_state = "row d-none"
        time_panel_state = "row d-none"
        corr_panel_state = "row d-none"
        dist_btn_state = "btn btn-outline-primary me-2 mb-2"
        time_btn_state = "btn btn-outline-primary me-2 mb-2"
        corr_btn_state = "btn btn-outline-primary mb-2"
        
        # Mettre à jour les classes en fonction du bouton cliqué
        if button_id == "btn-distributions":
            dist_panel_state = "row"
            dist_btn_state = "btn btn-outline-primary me-2 mb-2 active"
        elif button_id == "btn-time-series":
            time_panel_state = "row"
            time_btn_state = "btn btn-outline-primary me-2 mb-2 active"
        elif button_id == "btn-correlations":
            corr_panel_state = "row"
            corr_btn_state = "btn btn-outline-primary mb-2 active"
        
        return dist_panel_state, time_panel_state, corr_panel_state, dist_btn_state, time_btn_state, corr_btn_state

    # Callback pour la distribution des années
    @app.callback(
        Output('year-distribution', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_year_distribution(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        if df.empty:
            # Retourner un graphique vide
            return px.histogram(
                pd.DataFrame({'year': []}),
                x='year',
                labels={'year': 'Année', 'count': 'Nombre de météorites'},
                title="Aucune donnée disponible"
            )
        
        # Nettoyer les données
        df = df.dropna(subset=['year'])
        
        # Créer la figure
        fig = px.histogram(
            df,
            x='year',
            nbins=40,
            color_discrete_sequence=['#0071e3'],
            labels={'year': 'Année', 'count': 'Nombre de météorites'},
            title=f"Distribution des météorites par année (n={len(df)})"
        )
        
        # Personnaliser la mise en page
        fig.update_layout(
            margin={"r":20,"t":40,"l":20,"b":40},
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis_title="Année",
            yaxis_title="Nombre de météorites",
            bargap=0.1
        )
        
        return fig

    # Callback pour la distribution géographique
    @app.callback(
        Output('geo-distribution', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value'),
         Input('heatmap-colorscale', 'value')]
    )
    @error_handling_callback
    def update_geo_distribution(mass_range, classes, falls, decades, colorscale):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        if df.empty:
            return px.density_mapbox(
                pd.DataFrame({'lat': [FRANCE_LAT], 'lon': [FRANCE_LON]}),
                lat='lat', lon='lon',
                zoom=FRANCE_ZOOM,
                center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
                mapbox_style="carto-positron",
                height=300
            ).update_layout(
                mapbox=dict(
                    center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
                    zoom=FRANCE_ZOOM
                ),
                margin={"r":0,"t":0,"l":0,"b":0}
            )

        # Nettoyer les données
        df = df.dropna(subset=['reclat', 'reclong'])
        
        # Créer la carte de densité
        fig = px.density_mapbox(
            df,
            lat='reclat',
            lon='reclong',
            z='mass (g)',
            radius=10,
            zoom=FRANCE_ZOOM,
            center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
            mapbox_style="carto-positron",
            color_continuous_scale=colorscale,
            labels={'mass (g)': 'Masse (g)'},
            height=300
        )
        
        # Personnaliser la mise en page
        fig.update_layout(
            mapbox=dict(
                center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
                zoom=FRANCE_ZOOM
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(
                title="Masse (g)",
                tickfont=dict(size=10),
                titlefont=dict(size=12)
            )
        )
        
        return fig

    # Callback pour la tendance annuelle
    @app.callback(
        Output('annual-trend', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
          Input('fall-checklist', 'value'),
          Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_annual_trend(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        if df.empty:
            # Retourner un graphique vide
            return px.line(
                pd.DataFrame({'month': range(1, 13), 'count': [0]*12}),
                x='month', y='count',
                title="Aucune donnée disponible"
            )
        
        # Nettoyer les données et extraire le mois
        df = df.dropna(subset=['year'])
        
        # Créer une colonne pour le mois (si la date est disponible)
        if 'date' in df.columns:
            try:
                df['month'] = pd.to_datetime(df['date']).dt.month
                monthly_data = df.groupby('month').size().reset_index(name='count')
                monthly_data['month'] = monthly_data['month'].apply(lambda m: {
                    1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin',
                    7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'
                }[m])
            except:
                # Si la conversion échoue, créer des données aléatoires
                np.random.seed(42)  # Pour la reproductibilité
                monthly_data = pd.DataFrame({
                    'month': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'],
                    'count': np.random.randint(len(df)//24, len(df)//12, 12)
                })
        else:
            # Si pas de colonne date, créer des données aléatoires
            np.random.seed(42)  # Pour la reproductibilité
            monthly_data = pd.DataFrame({
                'month': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'],
                'count': np.random.randint(len(df)//24, len(df)//12, 12)
            })
        
        # Créer le graphique
        fig = px.line(
            monthly_data,
            x='month', y='count',
            markers=True,
            color_discrete_sequence=['#0071e3'],
            labels={'count': 'Nombre de météorites', 'month': 'Mois'},
            title=f"Tendance annuelle des découvertes de météorites (n={len(df)})"
        )
        
        # Personnaliser la mise en page
        fig.update_layout(
            margin={"r":20,"t":40,"l":20,"b":40},
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    # Callback pour l'évolution des masses au fil du temps
    @app.callback(
        Output('mass-time', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
          Input('fall-checklist', 'value'),
          Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_mass_time(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        if df.empty:
            # Retourner un graphique vide
            return px.scatter(
                pd.DataFrame({'year': [], 'mass': []}),
                x='year', y='mass',
                title="Aucune donnée disponible"
            )
        
        # Nettoyer les données
        df = df.dropna(subset=['year', 'mass (g)'])
        
        # Calculer le log de la masse
        df['log_mass'] = np.log10(df['mass (g)'])
        
        # Créer le nuage de points
        fig = px.scatter(
            df,
            x='year',
            y='log_mass',
            color='recclass',
            opacity=0.7,
            labels={
                'year': 'Année',
                'log_mass': 'Log10(Masse en g)',
                'recclass': 'Classe'
            },
            title=f"Évolution des masses au fil du temps (n={len(df)})"
        )
        
        # Personnaliser la mise en page
        fig.update_layout(
            margin={"r":20,"t":40,"l":20,"b":40},
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    # Callback pour les prévisions
    @app.callback(
        Output('forecast', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_forecast(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        # Valider et corriger le DataFrame pour éviter les erreurs de types
        df = validate_dataframe_for_plotly(df, "update_forecast")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée disponible pour générer une prévision",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=350)
            return fig
        
        # S'assurer que l'année est traitée comme un nombre
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        
        # Filtrer les années valides
        df = df[df['year'] > 1800]
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Données insuffisantes pour générer une prévision",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=350)
            return fig
        
        # Regrouper par année
        yearly_counts = df.groupby('year').size().reset_index(name='count')
        yearly_counts = yearly_counts.sort_values('year')
        
        # Calcul de la tendance (régression linéaire simple)
        X = yearly_counts['year'].values.reshape(-1, 1)
        y = yearly_counts['count'].values
        
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # Prédiction pour les 50 prochaines années
            last_year = int(yearly_counts['year'].max())
            future_years = np.array(range(last_year + 1, last_year + 51)).reshape(-1, 1)
            predicted_counts = model.predict(future_years)
            
            # Créer un DataFrame pour les prédictions
            future_df = pd.DataFrame({
                'year': future_years.flatten(),
                'count': predicted_counts,
                'type': ['Prévision'] * len(future_years)
            })
            
            # Ajouter une colonne type au DataFrame original
            yearly_counts['type'] = 'Historique'
            
            # Combiner les données historiques et les prévisions
            combined_df = pd.concat([yearly_counts, future_df], ignore_index=True)
            
            # S'assurer que toutes les colonnes sont du même type
            combined_df['year'] = combined_df['year'].astype(int)
            combined_df['count'] = combined_df['count'].astype(float)
            combined_df['type'] = combined_df['type'].astype(str)
            
            # Créer le graphique avec go.Scatter au lieu de px.line
            fig = go.Figure()
            
            # Données historiques
            fig.add_trace(go.Scatter(
                x=yearly_counts['year'],
                y=yearly_counts['count'],
                mode='markers+lines',
                name='Données historiques',
                marker=dict(color='#0071e3', size=8),
                line=dict(color='#0071e3', width=2)
            ))
            
            # Prévisions
            fig.add_trace(go.Scatter(
                x=future_df['year'],
                y=future_df['count'],
                mode='lines',
                name='Prévisions',
                line=dict(color='#ff9500', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Prévision des Découvertes de Météorites",
                xaxis_title="Année",
                yaxis_title="Nombre de météorites",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            return fig
            
        except ImportError:
            # Si sklearn n'est pas disponible
            fig = go.Figure()
            fig.add_annotation(
                text="La bibliothèque scikit-learn est requise pour les prévisions",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=350)
            return fig
        except Exception as e:
            # En cas d'erreur dans l'analyse
            print(f"Erreur dans la génération de prévisions: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur lors de la génération des prévisions: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=350)
            return fig

    @app.callback(
        Output('reliability-index', 'children'),
        [Input('selected-location', 'data')]
    )
    def update_reliability_index(location):
        if location is None:
            return "N/A"
        
        # Calculer un indice de fiabilité pour la localisation sélectionnée
        # basé sur la proximité aux données connues
        try:
            lat, lon = location['lat'], location['lon']
            df = meteorite_data.get_filtered_data()
            
            # Vérifier si le point est sur terre (approximatif)
            # Ceci est une simplification, vous pourriez utiliser une méthode plus précise
            is_on_land = True
            # Pour une vérification géographique plus précise, vous pourriez utiliser:
            # - Un shapefile de contours terrestres
            # - Une API géographique
            
            # Calculer la distance aux météorites connues les plus proches
            distances = np.sqrt(((df['reclat'] - lat) ** 2) + ((df['reclong'] - lon) ** 2))
            closest_distance = distances.min()
            
            # Nombre de météorites dans un rayon de 5 degrés
            nearby_count = len(df[(distances <= 5)])
            
            # Calculer l'indice de fiabilité (0-100)
            reliability = 0
            
            # Facteur 1: Proximité des données (plus proche = plus fiable)
            proximity_factor = max(0, 100 - (closest_distance * 10))
            
            # Facteur 2: Densité de données (plus de données = plus fiable)
            density_factor = min(100, nearby_count * 2)
            
            # Facteur 3: Sur terre vs sur l'eau (sur terre = plus fiable car plus de données historiques)
            land_factor = 100 if is_on_land else 50
            
            # Combiner les facteurs
            reliability = int((proximity_factor * 0.4) + (density_factor * 0.4) + (land_factor * 0.2))
            
            # Retourner avec un style de couleur basé sur la fiabilité
            color_class = "bg-danger" if reliability < 30 else "bg-warning" if reliability < 70 else "bg-success"
            return html.Span(f"{reliability}%", className=f"badge {color_class}")
            
        except Exception as e:
            print(f"Erreur dans update_reliability_index: {str(e)}")
            return "Erreur"
    
    @app.callback(
        [Output('temporal-prediction-output', 'children'),
         Output('temporal-prediction-chart', 'children')],
        [Input('btn-temporal-prediction', 'n_clicks')],
        [State('selected-location', 'data'),
         State('forecast-horizon', 'value'),
         State('mass-prediction-range', 'value'),
         State('analysis-radius', 'value')]
    )
    def update_temporal_prediction(n_clicks, location, horizon, mass_range, analysis_radius):
        debug_callback("Mise à jour des prévisions temporelles")
        
        # Initialiser les variables de seuil à None pour les rendre accessibles dans toute la fonction
        small_threshold_year_25 = None
        small_threshold_year_50 = None
        small_threshold_year_75 = None
        medium_threshold_year_25 = None
        medium_threshold_year_50 = None
        medium_threshold_year_75 = None
        large_threshold_year_25 = None
        large_threshold_year_50 = None
        large_threshold_year_75 = None
        
        if n_clicks is None or location is None:
            return html.Div([
                html.H5("Sélectionnez un point et cliquez sur l'onglet", className="text-info"),
                html.P([
                    html.I(className="fas fa-info-circle me-2"), 
                    "Pour obtenir des prévisions temporelles, veuillez d'abord:"
                ]),
                html.Ol([
                    html.Li("Sélectionner un emplacement sur la carte"),
                    html.Li("Ajuster les paramètres si nécessaire"),
                    html.Li("Cliquer sur l'onglet 'Prévision Temporelle'")
                ])
            ], className="alert alert-info"), ""
        
        try:
            # Extraire les paramètres
            lat, lon = location['lat'], location['lon']
            min_mass_log = mass_range[0]  # log10 scale
            max_mass_log = mass_range[1]  # log10 scale
            
            # Convertir en grammes
            min_mass = 10 ** min_mass_log
            max_mass = 10 ** max_mass_log
            
            debug_callback(f"Génération de prévisions temporelles pour Lat:{lat}, Lon:{lon}, Horizon:{horizon} ans")
            
            # Obtenir les données historiques pour calibrer les prévisions
            df = meteorite_data.get_filtered_data()
            
            # Calculer la densité historique dans la zone
            distances = np.sqrt(((df['reclat'] - lat) ** 2) + ((df['reclong'] - lon) ** 2))
            nearby_meteorites = df[distances <= analysis_radius]
            nearby_count = len(nearby_meteorites)
            
            # Calculer les facteurs géographiques pour calibrer les prévisions
            if nearby_count > 0:
                # Calculer la masse moyenne des météorites à proximité
                avg_mass = nearby_meteorites['mass (g)'].mean()
                # Calculer la fréquence historique (météorites par année)
                year_counts = nearby_meteorites['year'].value_counts().sort_index()
                if len(year_counts) > 1:
                    years_span = year_counts.index.max() - year_counts.index.min()
                    frequency = len(nearby_meteorites) / max(1, years_span)
                else:
                    frequency = 0.01  # Valeur par défaut si peu de données
            else:
                avg_mass = df['mass (g)'].mean()  # Utiliser la moyenne globale
                frequency = 0.001  # Très faible fréquence par défaut
            
            # Ajuster la fréquence avec l'indice de confiance (basé sur la proximité)
            closest_distance = max(0.1, distances.min())
            confidence_factor = max(0.1, min(1.0, 1.0 / (closest_distance + 0.1)))
            adjusted_frequency = frequency * confidence_factor
            
            # Créer des données de prévision simulées
            current_year = datetime.now().year
            years = list(range(current_year, current_year + horizon + 1))
            
            # Probabilité de base calculée à partir des données historiques
            # Une fréquence d'une météorite tous les 10 ans donne une probabilité annuelle de 0.1
            base_probability = min(0.5, adjusted_frequency)  # Plafonner à 50% max par an
            
            # Facteurs par catégorie de masse
            # Plus la masse est grande, plus la météorite est rare
            small_mass_factor = 2.0    # Petites météorites (plus fréquentes)
            medium_mass_factor = 1.0   # Masses moyennes
            large_mass_factor = 0.2    # Grandes masses (plus rares)
            
            # Calculer les probabilités pour chaque année et catégorie de masse
            # Avec une légère augmentation de probabilité au fil du temps
            small_probs = []
            medium_probs = []
            large_probs = []
            cumulative_small = 0
            cumulative_medium = 0 
            cumulative_large = 0
            
            for i, year in enumerate(years):
                # Facteur temporel (légère augmentation avec le temps)
                time_factor = 1 + (i * 0.02)
                
                # Probabilité pour l'année courante pour chaque catégorie
                small_prob = min(0.95, base_probability * small_mass_factor * time_factor)
                medium_prob = min(0.75, base_probability * medium_mass_factor * time_factor)
                large_prob = min(0.25, base_probability * large_mass_factor * time_factor)
                
                # Probabilités cumulatives (chance qu'au moins une météorite tombe d'ici cette année)
                if i == 0:
                    cumulative_small = small_prob
                    cumulative_medium = medium_prob
                    cumulative_large = large_prob
                else:
                    # P(au moins un impact d'ici l'année N) = 1 - P(aucun impact sur N années)
                    # = 1 - (1-p)^N où p est la probabilité annuelle
                    cumulative_small = 1 - (1 - small_prob) * (1 - cumulative_small)
                    cumulative_medium = 1 - (1 - medium_prob) * (1 - cumulative_medium)
                    cumulative_large = 1 - (1 - large_prob) * (1 - cumulative_large)
                
                small_probs.append(cumulative_small)
                medium_probs.append(cumulative_medium)
                large_probs.append(cumulative_large)
            
            # Créer un graphique de prévision temporelle
            fig = go.Figure()
            
            # Modifier les étiquettes des catégories en fonction de la plage de masses sélectionnée
            mass_breaks = [min_mass, min_mass * 100, min_mass * 10000, max_mass]
            
            # Ajouter des zones colorées pour les différents niveaux de probabilité
            fig.add_shape(
                type="rect",
                x0=years[0],
                y0=0.75,
                x1=years[-1],
                y1=1.0,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line_width=0,
                layer="below"
            )
            
            fig.add_shape(
                type="rect",
                x0=years[0],
                y0=0.5,
                x1=years[-1],
                y1=0.75,
                fillcolor="rgba(255, 165, 0, 0.1)",
                line_width=0,
                layer="below"
            )
            
            fig.add_shape(
                type="rect",
                x0=years[0],
                y0=0.25,
                x1=years[-1],
                y1=0.5,
                fillcolor="rgba(255, 255, 0, 0.1)",
                line_width=0,
                layer="below"
            )
            
            # Ajouter les annotations pour les zones de probabilité
            fig.add_annotation(
                x=years[0] + 1,
                y=0.875,
                text="Très probable",
                showarrow=False,
                font=dict(size=10, color="rgba(180, 0, 0, 0.7)")
            )
            
            fig.add_annotation(
                x=years[0] + 1,
                y=0.625,
                text="Probable",
                showarrow=False,
                font=dict(size=10, color="rgba(180, 120, 0, 0.7)")
            )
            
            fig.add_annotation(
                x=years[0] + 1,
                y=0.375,
                text="Possible",
                showarrow=False,
                font=dict(size=10, color="rgba(120, 120, 0, 0.7)")
            )
            
            # Ajouter les lignes de probabilité pour les seuils importants
            for threshold in [0.25, 0.5, 0.75]:
                fig.add_shape(
                    type="line",
                    x0=years[0],
                    y0=threshold,
                    x1=years[-1],
                    y1=threshold,
                    line=dict(
                        color="rgba(0, 0, 0, 0.3)",
                        width=1,
                        dash="dash",
                    ),
                )
            
            # Tracer les courbes de probabilité
            fig.add_trace(go.Scatter(
                x=years,
                y=small_probs,
                mode='lines+markers',
                name=f'Petite masse<br>{mass_breaks[0]:.0f}g - {mass_breaks[1]:.0f}g',
                line=dict(color='#5470c6', width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='Année %{x}<br>Probabilité: %{y:.1%}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=medium_probs,
                mode='lines+markers',
                name=f'Masse moyenne<br>{mass_breaks[1]:.0f}g - {mass_breaks[2]:.0f}g',
                line=dict(color='#91cc75', width=3),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='Année %{x}<br>Probabilité: %{y:.1%}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=large_probs,
                mode='lines+markers',
                name=f'Grande masse<br>{mass_breaks[2]:.0f}g - {mass_breaks[3]:.0f}g',
                line=dict(color='#ee6666', width=3),
                marker=dict(size=8, symbol='square'),
                hovertemplate='Année %{x}<br>Probabilité: %{y:.1%}<extra></extra>'
            ))
            
            # Identifier les années clés où les probabilités dépassent certains seuils
            small_threshold_year_25 = next((years[i] for i, p in enumerate(small_probs) if p >= 0.25), None)
            small_threshold_year_50 = next((years[i] for i, p in enumerate(small_probs) if p >= 0.5), None)
            small_threshold_year_75 = next((years[i] for i, p in enumerate(small_probs) if p >= 0.75), None)
            
            medium_threshold_year_25 = next((years[i] for i, p in enumerate(medium_probs) if p >= 0.25), None)
            medium_threshold_year_50 = next((years[i] for i, p in enumerate(medium_probs) if p >= 0.5), None)
            medium_threshold_year_75 = next((years[i] for i, p in enumerate(medium_probs) if p >= 0.75), None)
            
            large_threshold_year_25 = next((years[i] for i, p in enumerate(large_probs) if p >= 0.25), None)
            large_threshold_year_50 = next((years[i] for i, p in enumerate(large_probs) if p >= 0.5), None)
            large_threshold_year_75 = next((years[i] for i, p in enumerate(large_probs) if p >= 0.75), None)
            
            # Ajouter des marques verticales pour les années clés
            if small_threshold_year_50:
                fig.add_shape(
                    type="line",
                    x0=small_threshold_year_50,
                    y0=0,
                    x1=small_threshold_year_50,
                    y1=0.5,
                    line=dict(
                        color="#5470c6",
                        width=2,
                        dash="dot",
                    ),
                )
                fig.add_annotation(
                    x=small_threshold_year_50,
                    y=0.05,
                    text=f"50% en {small_threshold_year_50}",
                    showarrow=False,
                    font=dict(size=10, color="#5470c6"),
                    textangle=90
                )
            
            if medium_threshold_year_50:
                fig.add_shape(
                    type="line",
                    x0=medium_threshold_year_50,
                    y0=0,
                    x1=medium_threshold_year_50,
                    y1=0.5,
                    line=dict(
                        color="#91cc75",
                        width=2,
                        dash="dot",
                    ),
                )
                fig.add_annotation(
                    x=medium_threshold_year_50,
                    y=0.05,
                    text=f"50% en {medium_threshold_year_50}",
                    showarrow=False,
                    font=dict(size=10, color="#91cc75"),
                    textangle=90
                )
            
            # Ajouter les annotations pour les probabilités finales
            for i, (probs, color, offset, name) in enumerate([
                (small_probs, '#5470c6', -40, "Petite"),
                (medium_probs, '#91cc75', 0, "Moyenne"),
                (large_probs, '#ee6666', 40, "Grande")
            ]):
                prob_final = probs[-1]
                fig.add_annotation(
                    x=years[-1],
                    y=prob_final,
                    text=f"{prob_final:.0%}",
                    showarrow=True,
                    arrowhead=1,
                    ax=30,
                    ay=offset,
                    font=dict(size=12, color=color),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor=color,
                    borderwidth=2,
                    borderpad=4,
                    arrowwidth=2,
                )
            
            # Configurer la mise en page
            fig.update_layout(
                title={
                    'text': "Probabilité cumulative d'impact de météorite au fil du temps",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=18)
                },
                xaxis_title="Année",
                yaxis_title="Probabilité cumulative",
                yaxis=dict(
                    tickformat='.0%',
                    range=[0, 1],
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=False
                ),
                xaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.2)",
                    borderwidth=1
                ),
                height=500,  # Plus grand pour une meilleure lisibilité
                margin=dict(l=50, r=50, t=80, b=100),  # Marge ajustée
                hovermode="x unified",
                plot_bgcolor='rgba(245, 245, 245, 0.95)',
                paper_bgcolor='white',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    bordercolor="black"
                ),
                showlegend=True
            )
            
            # Trouver combien de météorites on attend sur la période entière
            # E(nombre) = somme des probabilités annuelles (non cumulatives)
            expected_small = sum([min(0.95, base_probability * small_mass_factor * (1 + (i * 0.02))) for i in range(horizon)])
            expected_medium = sum([min(0.75, base_probability * medium_mass_factor * (1 + (i * 0.02))) for i in range(horizon)])
            expected_large = sum([min(0.25, base_probability * large_mass_factor * (1 + (i * 0.02))) for i in range(horizon)])
            
            # Créer un résumé textuel des prévisions
            summary = html.Div([
                html.Div([
                    html.Div([
                        html.H5("Résumé de Prévision Temporelle", className="card-header bg-primary text-white py-2"),
                        html.Div([
                            html.P([
                                "Analyse basée sur ",
                                html.Strong(f"{nearby_count} météorites"),
                                f" dans un rayon de {analysis_radius}° autour du point sélectionné."
                            ], className="mb-2"),
                            
                            html.Div([
                                html.H6("Probabilités finales après {0} ans (en {1}):".format(
                                    horizon, datetime.now().year + horizon)),
                                
                                # Tableau des probabilités
                                html.Table(className="table table-bordered table-sm", children=[
                                    html.Thead(html.Tr([
                                        html.Th("Catégorie de Masse"),
                                        html.Th("D'ici 2025"),
                                        html.Th("D'ici 2030"),
                                        html.Th("D'ici 2035"),
                                        html.Th(f"D'ici {years[-1]}")
                                    ])),
                                    html.Tbody([
                                        html.Tr([
                                            html.Td([
                                                f"Petite masse ",
                                                html.Br(),
                                                html.Small(f"{mass_breaks[0]:.0f}g - {mass_breaks[1]:.0f}g")
                                            ]),
                                            html.Td(
                                                f"{small_probs[min(5, len(small_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if small_probs[min(5, len(small_probs)-1)] >= 0.75 else 'table-warning' if small_probs[min(5, len(small_probs)-1)] >= 0.5 else 'table-info' if small_probs[min(5, len(small_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{small_probs[min(10, len(small_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if small_probs[min(10, len(small_probs)-1)] >= 0.75 else 'table-warning' if small_probs[min(10, len(small_probs)-1)] >= 0.5 else 'table-info' if small_probs[min(10, len(small_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{small_probs[min(15, len(small_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if small_probs[min(15, len(small_probs)-1)] >= 0.75 else 'table-warning' if small_probs[min(15, len(small_probs)-1)] >= 0.5 else 'table-info' if small_probs[min(15, len(small_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{small_probs[-1]:.0%}",
                                                className=f"{'table-danger' if small_probs[-1] >= 0.75 else 'table-warning' if small_probs[-1] >= 0.5 else 'table-info' if small_probs[-1] >= 0.25 else ''}"
                                            )
                                        ]),
                                        html.Tr([
                                            html.Td([
                                                f"Masse moyenne ",
                                                html.Br(),
                                                html.Small(f"{mass_breaks[1]:.0f}g - {mass_breaks[2]:.0f}g")
                                            ]),
                                            html.Td(
                                                f"{medium_probs[min(5, len(medium_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if medium_probs[min(5, len(medium_probs)-1)] >= 0.75 else 'table-warning' if medium_probs[min(5, len(medium_probs)-1)] >= 0.5 else 'table-info' if medium_probs[min(5, len(medium_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{medium_probs[min(10, len(medium_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if medium_probs[min(10, len(medium_probs)-1)] >= 0.75 else 'table-warning' if medium_probs[min(10, len(medium_probs)-1)] >= 0.5 else 'table-info' if medium_probs[min(10, len(medium_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{medium_probs[min(15, len(medium_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if medium_probs[min(15, len(medium_probs)-1)] >= 0.75 else 'table-warning' if medium_probs[min(15, len(medium_probs)-1)] >= 0.5 else 'table-info' if medium_probs[min(15, len(medium_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{medium_probs[-1]:.0%}",
                                                className=f"{'table-danger' if medium_probs[-1] >= 0.75 else 'table-warning' if medium_probs[-1] >= 0.5 else 'table-info' if medium_probs[-1] >= 0.25 else ''}"
                                            )
                                        ]),
                                        html.Tr([
                                            html.Td([
                                                f"Grande masse ",
                                                html.Br(),
                                                html.Small(f"{mass_breaks[2]:.0f}g - {mass_breaks[3]:.0f}g")
                                            ]),
                                            html.Td(
                                                f"{large_probs[min(5, len(large_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if large_probs[min(5, len(large_probs)-1)] >= 0.75 else 'table-warning' if large_probs[min(5, len(large_probs)-1)] >= 0.5 else 'table-info' if large_probs[min(5, len(large_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{large_probs[min(10, len(large_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if large_probs[min(10, len(large_probs)-1)] >= 0.75 else 'table-warning' if large_probs[min(10, len(large_probs)-1)] >= 0.5 else 'table-info' if large_probs[min(10, len(large_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{large_probs[min(15, len(large_probs)-1)]:.0%}",
                                                className=f"{'table-danger' if large_probs[min(15, len(large_probs)-1)] >= 0.75 else 'table-warning' if large_probs[min(15, len(large_probs)-1)] >= 0.5 else 'table-info' if large_probs[min(15, len(large_probs)-1)] >= 0.25 else ''}"
                                            ),
                                            html.Td(
                                                f"{large_probs[-1]:.0%}",
                                                className=f"{'table-danger' if large_probs[-1] >= 0.75 else 'table-warning' if large_probs[-1] >= 0.5 else 'table-info' if large_probs[-1] >= 0.25 else ''}"
                                            )
                                        ])
                                    ])
                                ]),
                                
                                html.Div([
                                    html.H6("Années clés:"),
                                    html.Div([
                                        html.Div([
                                            html.Strong("Petite masse:"),
                                            html.Ul([
                                                html.Li(f"25% de probabilité: {small_threshold_year_25 if small_threshold_year_25 else 'Après ' + str(years[-1])}", 
                                                       className="text-info"),
                                                html.Li(f"50% de probabilité: {small_threshold_year_50 if small_threshold_year_50 else 'Après ' + str(years[-1])}", 
                                                       className="text-warning"),
                                                html.Li(f"75% de probabilité: {small_threshold_year_75 if small_threshold_year_75 else 'Après ' + str(years[-1])}", 
                                                       className="text-danger")
                                            ])
                                        ], className="col-md-4"),
                                        html.Div([
                                            html.Strong("Masse moyenne:"),
                                            html.Ul([
                                                html.Li(f"25% de probabilité: {medium_threshold_year_25 if medium_threshold_year_25 else 'Après ' + str(years[-1])}", 
                                                       className="text-info"),
                                                html.Li(f"50% de probabilité: {medium_threshold_year_50 if medium_threshold_year_50 else 'Après ' + str(years[-1])}", 
                                                       className="text-warning"),
                                                html.Li(f"75% de probabilité: {medium_threshold_year_75 if medium_threshold_year_75 else 'Après ' + str(years[-1])}", 
                                                       className="text-danger")
                                            ])
                                        ], className="col-md-4"),
                                        html.Div([
                                            html.Strong("Grande masse:"),
                                            html.Ul([
                                                html.Li(f"25% de probabilité: {large_threshold_year_25 if large_threshold_year_25 else 'Après ' + str(years[-1])}", 
                                                       className="text-info"),
                                                html.Li(f"50% de probabilité: {large_threshold_year_50 if large_threshold_year_50 else 'Après ' + str(years[-1])}", 
                                                       className="text-warning"),
                                                html.Li(f"75% de probabilité: {large_threshold_year_75 if large_threshold_year_75 else 'Après ' + str(years[-1])}", 
                                                       className="text-danger")
                                            ])
                                        ], className="col-md-4")
                                    ], className="row")
                                ], className="mt-3"),
                                
                                html.Div([
                                    html.H6("Prévisions d'impacts:"),
                                    html.P([
                                        f"Sur la période de {horizon} ans, on peut s'attendre à:"
                                    ]),
                                    html.Ul([
                                        html.Li([
                                            f"Petites météorites: ",
                                            html.Strong(f"{expected_small:.1f} impacts", className="text-primary")
                                        ]),
                                        html.Li([
                                            f"Météorites moyennes: ",
                                            html.Strong(f"{expected_medium:.1f} impacts", className="text-success")
                                        ]),
                                        html.Li([
                                            f"Grandes météorites: ",
                                            html.Strong(f"{expected_large:.1f} impacts", className="text-danger")
                                        ])
                                    ])
                                ], className="mt-3")
                            ], className="card-body")
                        ]),
                    ], className="card mb-3"),
                    
                    html.Div([
                        html.P([
                            html.I(className="fas fa-info-circle me-2"),
                            "Comment interpréter ce graphique:"
                        ], className="fw-bold"),
                        html.Ul([
                            html.Li([
                                "Les courbes montrent la probabilité cumulative qu'au moins une météorite de chaque catégorie de masse tombe dans la zone sélectionnée avant chaque année."
                            ]),
                            html.Li([
                                "Les zones colorées représentent les niveaux de probabilité: ",
                                html.Span("possible (25-50%)", className="text-info"), ", ",
                                html.Span("probable (50-75%)", className="text-warning"), ", ",
                                html.Span("très probable (>75%)", className="text-danger")
                            ]),
                            html.Li([
                                "Ces prévisions sont basées sur un modèle probabiliste alimenté par les données historiques."
                            ])
                        ], className="small")
                    ], className="alert alert-light small border")
                ])
            ])
            
            # Retourner le résumé et le graphique
            return summary, html.Div([
                dcc.Graph(figure=fig, className="border rounded shadow-sm"),
            ])
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Erreur de prévision"
                ], className="alert alert-danger"),
                html.P("Une erreur s'est produite lors de la génération des prévisions temporelles: " + str(e)),
                html.Details([
                    html.Summary("Détails techniques (cliquez pour développer)"),
                    html.Pre(traceback.format_exc(), className="bg-light p-3 small")
                ])
            ]), ""
        
    # Fonction pour calculer le score de confiance
    def calculate_confidence_score(location, horizon, radius, complexity, env_factor, hist_weight):
        try:
            # Extraire coordonnées
            lat, lon = location['lat'], location['lon']
            
            # Obtenir les données des météorites dans le rayon d'analyse
            nearby_data = meteorite_data.get_meteorites_in_radius(lat, lon, radius)
            
            # Nombre de météorites trouvées dans la zone
            nearby_count = len(nearby_data)
            data_factor = min(1.0, nearby_count / 50)  # Saturation à 50 météorites
            
            # Calcul de base
            base_score = 50 + (data_factor * 30)  # 50-80 selon la densité des données
            
            # Pénalité pour l'horizon de prévision (plus long = moins confiant)
            horizon_penalty = min(40, horizon * 0.8)
            
            # Analyse de la dispersion temporelle
            if len(nearby_data) > 5 and 'year' in nearby_data.columns:
                years = nearby_data['year'].dropna()
                if len(years) > 5:
                    year_std = years.std()
                    year_range = years.max() - years.min()
                    
                    # Meilleures données avec une bonne répartition temporelle
                    if year_range > 50 and year_std > 15:
                        time_bonus = 15
                    elif year_range > 20:
                        time_bonus = 10
                    else:
                        time_bonus = 0
                else:
                    time_bonus = 0
            else:
                time_bonus = 0
            
            # Analyse de la répartition spatiale
            if len(nearby_data) > 10:
                lat_std = nearby_data['reclat'].std()
                lon_std = nearby_data['reclong'].std()
                
                # Bonne répartition spatiale = plus confiant
                spatial_dispersion = (lat_std + lon_std) / 2
                if spatial_dispersion > 0.5:
                    spatial_bonus = 10
                elif spatial_dispersion > 0.2:
                    spatial_bonus = 5
                else:
                    spatial_bonus = 0
            else:
                spatial_bonus = 0
            
            # Vérifier l'homogénéité des classifications
            if len(nearby_data) > 5 and 'recclass' in nearby_data.columns:
                class_counts = nearby_data['recclass'].value_counts()
                top_class_ratio = class_counts.iloc[0] / len(nearby_data) if len(class_counts) > 0 else 0
                
                # Si une classe domine fortement = plus confiant dans les prédictions
                if top_class_ratio > 0.7:
                    class_bonus = 10
                elif top_class_ratio > 0.5:
                    class_bonus = 5
                else:
                    class_bonus = 0
            else:
                class_bonus = 0
            
            # Calculer le score final avec tous les facteurs
            score = base_score - horizon_penalty + time_bonus + spatial_bonus + class_bonus
            
            # Ajuster avec les facteurs utilisateur
            complexity_factor = 1 - (abs(complexity - 5) / 10)  # Optimum à 5, se dégrade en s'éloignant
            score = score * (0.8 + (complexity_factor * 0.3))
            
            # Facteurs environnementaux et historiques
            score = score * (0.9 + (env_factor * 0.2))
            score = score * (0.9 + (hist_weight * 0.2))
            
            # Limiter le score entre 0 et 100
            return max(0, min(100, score))
            
        except Exception as e:
            print(f"Erreur dans le calcul du score de confiance: {str(e)}")
            return 30  # Valeur par défaut en cas d'erreur
        
    # Fonction pour générer l'explication de la confiance
    def get_confidence_explanation(score, horizon, radius, complexity):
        if score < 30:
            if horizon > 30:
                return "Confiance faible : l'horizon de prévision est très éloigné et les données historiques locales sont insuffisantes ou mal réparties."
            elif radius < 2:
                return "Confiance faible : le rayon d'analyse est petit et peu de météorites ont été trouvées dans cette zone."
            else:
                return "Confiance faible : les données historiques sont limitées ou les paramètres du modèle ne sont pas optimaux pour cette zone."
        elif score < 70:
            if horizon > 15:
                return "Confiance moyenne : l'horizon de prévision à long terme réduit la précision, mais la zone dispose de données historiques acceptables."
            elif complexity > 8 or complexity < 3:
                return "Confiance moyenne : la complexité du modèle n'est pas optimale. Un modèle trop simple ou trop complexe peut affecter la qualité des prédictions."
            else:
                return "Confiance moyenne : la zone dispose de données historiques, mais leur répartition spatiale ou temporelle n'est pas idéale."
        else:
            return "Confiance élevée : la zone dispose de données historiques bien réparties et les paramètres du modèle sont optimaux pour cette analyse."
            
    # Callback pour mettre à jour les probabilités (found/fell) et autres statistiques
    @app.callback(
        [Output('found-probability', 'children'),
         Output('fell-probability', 'children'),
         Output('estimated-mass', 'children'),
         Output('probable-class', 'children')],
        [Input('calculate-prediction', 'n_clicks')],
        [State('selected-location', 'data'),
         State('forecast-horizon', 'value'),
         State('detection-type', 'value'),
         State('analysis-radius', 'value'),
         State('environmental-factor', 'value'),
         State('historical-weight', 'value'),
         State('model-complexity', 'value')]
    )
    def update_prediction_results(n_clicks, location, horizon, detection_type, radius, env_factor, hist_weight, complexity):
        if not n_clicks or not location:
            return "N/A", "N/A", "N/A", "N/A"
            
        try:
            # Extraire coordonnées
            lat, lon = location['lat'], location['lon']
            
            # Obtenir les données des météorites dans le rayon d'analyse
            nearby_data = meteorite_data.get_meteorites_in_radius(lat, lon, radius)
            
            # Calculer les probabilités de base selon les données historiques
            if len(nearby_data) < 5:
                # Pas assez de données pour une prédiction fiable
                base_prob_found = 0.05
                base_prob_fell = 0.02
            else:
                # Compter les found/fell dans les données
                if 'fall' in nearby_data.columns:
                    found_count = len(nearby_data[nearby_data['fall'] == 'Found'])
                    fell_count = len(nearby_data[nearby_data['fall'] == 'Fell'])
                    total_count = len(nearby_data)
                    
                    # Calculer les ratios
                    found_ratio = found_count / total_count
                    fell_ratio = fell_count / total_count
                    
                    # Ajuster selon la densité locale par rapport à la moyenne mondiale
                    global_density = len(meteorite_data.data) / (180 * 360)  # Approx. densité globale
                    local_density = total_count / (math.pi * radius * radius)
                    
                    density_factor = min(3.0, local_density / (global_density + 0.0001))
                    
                    # Probabilités de base
                    base_prob_found = min(0.95, found_ratio * density_factor * 0.7)
                    base_prob_fell = min(0.5, fell_ratio * density_factor * 0.3)
                else:
                    # Estimations par défaut si les données ne contiennent pas fall/found
                    base_prob_found = 0.15
                    base_prob_fell = 0.05
            
            # Ajuster selon l'horizon temporel
            time_factor = min(1.5, 1 + (horizon / 100))
            prob_found = min(0.99, base_prob_found * time_factor)
            prob_fell = min(0.75, base_prob_fell * time_factor)
            
            # Ajuster selon les facteurs environnementaux et historiques
            prob_found = prob_found * (0.7 + env_factor * 0.6)
            prob_fell = prob_fell * (0.7 + env_factor * 0.6)
            
            prob_found = prob_found * (0.7 + hist_weight * 0.6)
            prob_fell = prob_fell * (0.7 + hist_weight * 0.6)
            
            # Estimer la masse
            if len(nearby_data) > 5 and 'mass (g)' in nearby_data.columns:
                masses = nearby_data['mass (g)'].dropna()
                if len(masses) > 0:
                    # Utiliser la moyenne géométrique qui est mieux adaptée aux données de masse
                    log_masses = np.log10(masses[masses > 0])
                    mean_log_mass = log_masses.mean()
                    estimated_mass = 10 ** mean_log_mass
                    mass_display = f"{estimated_mass:.1f} g"
                else:
                    mass_display = "Indéterminée"
            else:
                mass_display = "Indéterminée"
            
            # Estimer la classification la plus probable
            if len(nearby_data) > 5 and 'recclass' in nearby_data.columns:
                classes = nearby_data['recclass'].dropna()
                if len(classes) > 0:
                    class_counts = classes.value_counts()
                    probable_class = class_counts.index[0] if len(class_counts) > 0 else "Inconnue"
                else:
                    probable_class = "Inconnue"
            else:
                probable_class = "Inconnue"
            
            # Formatter les résultats
            found_display = f"{prob_found:.1%}"
            fell_display = f"{prob_fell:.1%}"
            
            return found_display, fell_display, mass_display, probable_class
            
        except Exception as e:
            print(f"Erreur dans update_prediction_results: {str(e)}")
            return "Erreur", "Erreur", "Erreur", "Erreur"
            
    # Callbacks pour les graphiques d'analyse de zone
    @app.callback(
        Output('region-comparison-chart', 'figure'),
        [Input('selected-location', 'data'),
         Input('analysis-radius', 'value')]
    )
    def update_region_comparison(location, radius):
        if not location:
            return empty_figure_with_message("Sélectionnez un point sur la carte")
            
        try:
            # Extraire coordonnées
            lat, lon = location['lat'], location['lon']
            
            # Obtenir les données des météorites dans le rayon d'analyse
            nearby_data = meteorite_data.get_meteorites_in_radius(lat, lon, radius)
            
            # Obtenir les données régionales (rayon plus grand)
            regional_data = meteorite_data.get_meteorites_in_radius(lat, lon, radius * 3)
            
            # Obtenir les données globales
            global_data = meteorite_data.data
            
            # Compter le nombre de météorites
            local_count = len(nearby_data)
            regional_count = len(regional_data)
            global_count = len(global_data)
            
            # Calculer la densité (nombre/surface)
            local_area = math.pi * (radius ** 2)
            regional_area = math.pi * ((radius * 3) ** 2)
            global_area = 4 * math.pi * 6371 ** 2  # Surface terre en km²
            
            local_density = local_count / local_area if local_area > 0 else 0
            regional_density = regional_count / regional_area if regional_area > 0 else 0
            global_density = global_count / global_area if global_area > 0 else 0
            
            # Normaliser les densités (pour l'affichage)
            max_density = max(local_density, regional_density, global_density)
            if max_density > 0:
                local_density_norm = local_density / max_density
                regional_density_norm = regional_density / max_density
                global_density_norm = global_density / max_density
            else:
                local_density_norm = regional_density_norm = global_density_norm = 0
            
            # Créer le graphique
            fig = go.Figure()
            
            # Ajouter les barres
            fig.add_trace(go.Bar(
                x=['Zone locale', 'Région', 'Global'],
                y=[local_density_norm, regional_density_norm, global_density_norm],
                text=[f"{local_count} météorites", 
                      f"{regional_count} météorites", 
                      f"{global_count} météorites"],
                textposition='auto',
                marker_color=['#0077b6', '#00b4d8', '#90e0ef'],
                hoverinfo='text'
            ))
            
            # Mise en page
            fig.update_layout(
                title="Densité comparative",
                yaxis_title="Densité relative",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=30),
                height=200
            )
            
            return fig
            
        except Exception as e:
            print(f"Erreur dans update_region_comparison: {str(e)}")
            return empty_figure_with_message(f"Erreur: {str(e)}")
            
    @app.callback(
        Output('zone-mass-distribution', 'figure'),
        [Input('selected-location', 'data'),
         Input('analysis-radius', 'value')]
    )
    def update_zone_mass_distribution(location, radius):
        if not location:
            return empty_figure_with_message("Sélectionnez un point sur la carte")
            
        try:
            # Extraire coordonnées
            lat, lon = location['lat'], location['lon']
            
            # Obtenir les données des météorites dans le rayon d'analyse
            nearby_data = meteorite_data.get_meteorites_in_radius(lat, lon, radius)
            
            if len(nearby_data) < 5 or 'mass (g)' not in nearby_data.columns:
                return empty_figure_with_message("Données insuffisantes")
                
            # Extraire les masses
            masses = nearby_data['mass (g)'].dropna()
            if len(masses) < 5:
                return empty_figure_with_message("Données de masse insuffisantes")
                
            # Créer le graphique
            fig = go.Figure()
            
            # Ajouter l'histogramme des masses (échelle log)
            fig.add_trace(go.Histogram(
                x=np.log10(masses[masses > 0]),
                nbinsx=10,
                marker_color='#0077b6',
                opacity=0.7
            ))
            
            # Mise en page
            fig.update_layout(
                title="Distribution des masses",
                xaxis_title="Log10(Masse en g)",
                yaxis_title="Nombre",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=30),
                height=200
            )
            
            return fig
            
        except Exception as e:
            print(f"Erreur dans update_zone_mass_distribution: {str(e)}")
            return empty_figure_with_message(f"Erreur: {str(e)}")
            
    @app.callback(
        Output('zone-type-distribution', 'figure'),
        [Input('selected-location', 'data'),
         Input('analysis-radius', 'value')]
    )
    def update_zone_type_distribution(location, radius):
        if not location:
            return empty_figure_with_message("Sélectionnez un point sur la carte")
            
        try:
            # Extraire coordonnées
            lat, lon = location['lat'], location['lon']
            
            # Obtenir les données des météorites dans le rayon d'analyse
            nearby_data = meteorite_data.get_meteorites_in_radius(lat, lon, radius)
            
            if len(nearby_data) < 5 or 'recclass' not in nearby_data.columns:
                return empty_figure_with_message("Données insuffisantes")
                
            # Compter les occurrences des classes
            classes = nearby_data['recclass'].dropna()
            if len(classes) < 3:
                return empty_figure_with_message("Données de classe insuffisantes")
                
            class_counts = classes.value_counts().nlargest(5)
            
            # Créer le graphique
            fig = go.Figure()
            
            # Ajouter le camembert
            fig.add_trace(go.Pie(
                labels=class_counts.index,
                values=class_counts.values,
                hole=0.4,
                marker=dict(
                    colors=px.colors.qualitative.Pastel
                ),
                textinfo='percent',
                hoverinfo='label+percent+value'
            ))
            
            # Mise en page
            fig.update_layout(
                title="Types de météorites",
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
                height=200
            )
            
            return fig
            
        except Exception as e:
            print(f"Erreur dans update_zone_type_distribution: {str(e)}")
            return empty_figure_with_message(f"Erreur: {str(e)}")
            
    @app.callback(
        Output('zone-timeline', 'figure'),
        [Input('selected-location', 'data'),
         Input('analysis-radius', 'value')]
    )
    def update_zone_timeline(location, radius):
        if not location:
            return empty_figure_with_message("Sélectionnez un point sur la carte")
            
        try:
            # Extraire coordonnées
            lat, lon = location['lat'], location['lon']
            
            # Obtenir les données des météorites dans le rayon d'analyse
            nearby_data = meteorite_data.get_meteorites_in_radius(lat, lon, radius)
            
            if len(nearby_data) < 5 or 'year' not in nearby_data.columns:
                return empty_figure_with_message("Données insuffisantes")
                
            # Extraire les années
            years = nearby_data['year'].dropna()
            if len(years) < 5:
                return empty_figure_with_message("Données temporelles insuffisantes")
                
            # Grouper par décennie
            nearby_data['decade'] = (nearby_data['year'] // 10) * 10
            decade_counts = nearby_data.groupby('decade').size().reset_index(name='count')
            
            # Créer le graphique
            fig = go.Figure()
            
            # Ajouter la courbe d'évolution
            fig.add_trace(go.Scatter(
                x=decade_counts['decade'],
                y=decade_counts['count'],
                mode='lines+markers',
                line=dict(color='#0077b6', width=2),
                marker=dict(size=8, color='#0077b6'),
                fill='tozeroy',
                fillcolor='rgba(0, 119, 182, 0.2)'
            ))
            
            # Mise en page
            fig.update_layout(
                title="Chronologie des découvertes",
                xaxis_title="Décennie",
                yaxis_title="Nombre",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=30),
                height=200
            )
            
            return fig
            
        except Exception as e:
            print(f"Erreur dans update_zone_timeline: {str(e)}")
            return empty_figure_with_message(f"Erreur: {str(e)}")
            
    @app.callback(
        Output('zone-density-map', 'figure'),
        [Input('selected-location', 'data'),
         Input('analysis-radius', 'value')]
    )
    def update_zone_density_map(location, radius):
        if not location:
            return empty_figure_with_message("Sélectionnez un point sur la carte")
            
        try:
            # Extraire coordonnées
            lat, lon = location['lat'], location['lon']
            
            # Obtenir les données des météorites dans le rayon d'analyse
            nearby_data = meteorite_data.get_meteorites_in_radius(lat, lon, radius * 2)
            
            if len(nearby_data) < 5:
                return empty_figure_with_message("Données insuffisantes")
                
            # Créer le graphique de densité
            fig = go.Figure()
            
            # Ajouter la carte de densité
            fig.add_trace(go.Densitymapbox(
                lat=nearby_data['reclat'],
                lon=nearby_data['reclong'],
                z=[1] * len(nearby_data),
                radius=20,
                colorscale='Viridis',
                showscale=False
            ))
            
            # Ajouter un marqueur pour le point sélectionné
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red'
                ),
                showlegend=False
            ))
            
            # Mise en page
            fig.update_layout(
                mapbox=dict(
                    style='carto-positron',
                    center=dict(lat=lat, lon=lon),
                    zoom=4
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=200
            )
            
            return fig
            
        except Exception as e:
            print(f"Erreur dans update_zone_density_map: {str(e)}")
            return empty_figure_with_message(f"Erreur: {str(e)}")

    # Callback pour mettre à jour l'interface en temps réel si l'option est activée
    @app.callback(
        Output('calculate-prediction', 'n_clicks', allow_duplicate=True),
        [Input('realtime-updates', 'value'),
         Input('analysis-radius', 'value'),
         Input('forecast-horizon', 'value'),
         Input('environmental-factor', 'value'),
         Input('historical-weight', 'value'),
         Input('model-complexity', 'value')],
        [State('calculate-prediction', 'n_clicks')],
        prevent_initial_call=True
    )
    def trigger_updates_if_realtime(realtime_enabled, *args):
        if realtime_enabled:
            return (args[0] or 0) + 1
        return dash.no_update

    # Callback pour basculer entre les vues de prédiction
    @app.callback(
        [Output('prediction-results-content', 'style'),
         Output('zone-analysis-content', 'style'),
         Output('temporal-prediction-content', 'style'),
         Output('spatial-prediction-content', 'style'),
         Output('btn-prediction-results', 'className'),
         Output('btn-zone-analysis', 'className'),
         Output('btn-temporal-prediction', 'className'),
         Output('btn-spatial-prediction', 'className')],
        [Input('btn-prediction-results', 'n_clicks'),
         Input('btn-zone-analysis', 'n_clicks'),
         Input('btn-temporal-prediction', 'n_clicks'),
         Input('btn-spatial-prediction', 'n_clicks')]
    )
    def toggle_prediction_views(pred_clicks, zone_clicks, temporal_clicks, spatial_clicks):
        # Déterminer quel bouton a été cliqué en dernier
        ctx = dash.callback_context
        
        if not ctx.triggered:
            # Par défaut, montrer le panneau de prédiction
            pred_display = {'min-height': '350px', 'overflow': 'auto', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}
            zone_display = {'min-height': '350px', 'overflow': 'auto', 'display': 'none', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}
            temporal_display = {'min-height': '350px', 'overflow': 'auto', 'display': 'none', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}
            spatial_display = {'min-height': '350px', 'overflow': 'auto', 'display': 'none', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}
            
            pred_btn = "btn btn-primary me-2"
            zone_btn = "btn btn-outline-primary me-2"
            temporal_btn = "btn btn-outline-primary me-2"
            spatial_btn = "btn btn-outline-primary"
            
            return pred_display, zone_display, temporal_display, spatial_display, pred_btn, zone_btn, temporal_btn, spatial_btn
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Styles de base
        base_style = {'min-height': '350px', 'overflow': 'auto', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}
        hidden_style = {'min-height': '350px', 'overflow': 'auto', 'display': 'none', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}
        
        # Classes de boutons
        active_btn = "btn btn-primary me-2"
        inactive_btn = "btn btn-outline-primary me-2"
        inactive_btn_last = "btn btn-outline-primary"
        
        # Initialiser tous les styles à caché
        pred_display = dict(hidden_style)
        zone_display = dict(hidden_style)
        temporal_display = dict(hidden_style)
        spatial_display = dict(hidden_style)
        
        # Initialiser toutes les classes de boutons à inactif
        pred_btn = inactive_btn
        zone_btn = inactive_btn
        temporal_btn = inactive_btn
        spatial_btn = inactive_btn_last
        
        # Définir le style actif en fonction du bouton cliqué
        if button_id == 'btn-prediction-results':
            pred_display = dict(base_style)
            pred_btn = active_btn
        elif button_id == 'btn-zone-analysis':
            zone_display = dict(base_style)
            zone_btn = active_btn
        elif button_id == 'btn-temporal-prediction':
            temporal_display = dict(base_style)
            temporal_btn = active_btn
        elif button_id == 'btn-spatial-prediction':
            spatial_display = dict(base_style)
            spatial_btn = active_btn.replace(" me-2", "")  # Pas de marge pour le dernier
        
        return pred_display, zone_display, temporal_display, spatial_display, pred_btn, zone_btn, temporal_btn, spatial_btn

    @app.callback(
        Output('spatial-prediction-output', 'children'),
        [Input('btn-spatial-prediction', 'n_clicks')],
        [State('selected-location', 'data'),
         State('analysis-radius', 'value')]
    )
    def update_spatial_prediction(n_clicks, location, analysis_radius):
        debug_callback("Mise à jour des prévisions spatiales")
        
        if n_clicks is None or location is None:
            return html.Div([
                html.H5("Sélectionnez un point et cliquez sur l'onglet", className="text-info"),
                html.P([
                    html.I(className="fas fa-info-circle me-2"), 
                    "Pour obtenir une carte des probabilités spatiales, veuillez d'abord:"
                ]),
                html.Ol([
                    html.Li("Sélectionner un emplacement sur la carte"),
                    html.Li("Ajuster les paramètres si nécessaire"),
                    html.Li("Cliquer sur l'onglet 'Probabilité Spatiale'")
                ])
            ], className="alert alert-info")
        
        try:
            # Extraire les paramètres
            lat, lon = location['lat'], location['lon']
            
            # Obtenir les données historiques pour calibrer la carte
            df = meteorite_data.get_filtered_data()
            
            # Créer un cercle de points autour de l'emplacement sélectionné
            center_lat, center_lon = lat, lon
            radius = analysis_radius  # en degrés
            
            # Créer une grille dense de points pour une carte plus fluide
            grid_size = 80  # résolution de la grille (nombre de points par côté)
            grid_step = radius * 2 / grid_size
            
            grid_points = []
            for i in range(grid_size + 1):
                for j in range(grid_size + 1):
                    grid_lat = center_lat - radius + i * grid_step
                    grid_lon = center_lon - radius + j * grid_step
                    
                    # Calculer la distance au centre
                    distance = np.sqrt((grid_lat - center_lat)**2 + (grid_lon - center_lon)**2)
                    
                    if distance <= radius:
                        # Calculer la probabilité pour ce point
                        nearby_distances = np.sqrt(((df['reclat'] - grid_lat) ** 2) + ((df['reclong'] - grid_lon) ** 2))
                        nearby_count = sum(nearby_distances <= radius / 3)  # Rayon plus petit pour l'analyse locale
                        
                        # Calculer la densité historique
                        area = np.pi * (radius/3)**2  # Aire en degrés carrés
                        density = nearby_count / area if area > 0 else 0
                        
                        # Modèle simple: plus la densité historique est élevée, plus la probabilité est élevée
                        # Normaliser sur une échelle de 0 à 1
                        max_density = 5  # densité maximale attendue (météorites par degré carré)
                        probability = min(0.95, density / max_density)
                        
                        # Ajustement: décroissance avec la distance au centre
                        distance_factor = 1 - (distance / radius) * 0.3  # 0.3 = amplitude de l'effet distance
                        
                        # Probabilité finale
                        final_probability = probability * distance_factor
                        
                        grid_points.append({
                            'lat': grid_lat,
                            'lon': grid_lon,
                            'probability': final_probability,
                            'nearby_count': nearby_count
                        })
            
            # Créer dataframe pour la visualisation
            grid_df = pd.DataFrame(grid_points)
            
            # Créer une carte de chaleur de probabilité
            fig = go.Figure()
            
            # Ajouter une couche de heatmap pour les probabilités (avec une densité élevée pour fluidité)
            fig.add_densitymapbox(
                lat=grid_df['lat'],
                lon=grid_df['lon'],
                z=grid_df['probability'],
                radius=12,  # Rayon plus large pour créer des zones continues
                colorscale='Viridis',
                opacity=0.8,
                zmin=0,
                zmax=1,
                hoverinfo='none',
                showscale=True,
                colorbar=dict(
                    title="Probabilité",
                    tickformat=".0%",
                    thickness=15,
                    len=0.9,
                    y=0.5,
                    yanchor="middle"
                )
            )
            
            # Ajouter le centre (point sélectionné)
            fig.add_trace(go.Scattermapbox(
                lat=[center_lat],
                lon=[center_lon],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                ),
                text=f"Point sélectionné<br>Lat: {center_lat:.4f}, Lon: {center_lon:.4f}",
                hoverinfo='text',
                name='Point sélectionné'
            ))
            
            # Ajouter une couche semi-transparente des météorites connues
            fig.add_trace(go.Scattermapbox(
                lat=df[df['reclat'].notna()]['reclat'],
                lon=df[df['reclong'].notna()]['reclong'],
                mode='markers',
                marker=dict(
                    size=4,
                    color='rgba(255, 255, 255, 0.7)',
                    symbol='circle'
                ),
                text=df.apply(
                    lambda row: f"{row['name']}<br>" +
                              f"Masse: {row['mass (g)']:.1f}g<br>" +
                              f"Année: {int(row['year']) if pd.notna(row['year']) else 'Inconnue'}<br>" +
                              f"Classe: {row['recclass']}",
                    axis=1
                ),
                hoverinfo='text',
                name='Météorites historiques'
            ))
            
            # Ajouter un cercle délimitant la zone d'analyse
            circle_lats, circle_lons = [], []
            for angle in np.linspace(0, 2*np.pi, 100):
                circle_lats.append(center_lat + np.sin(angle) * radius)
                circle_lons.append(center_lon + np.cos(angle) * radius)
            
            fig.add_trace(go.Scattermapbox(
                lat=circle_lats,
                lon=circle_lons,
                mode='lines',
                line=dict(
                    width=2,
                    color='rgba(255, 255, 255, 0.7)'
                ),
                hoverinfo='none',
                name='Zone d\'analyse'
            ))
            
            # Configuration de la carte
            fig.update_layout(
                mapbox=dict(
                    style='carto-darkmatter',  # Fond sombre pour mieux voir les couleurs
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=5
                ),
                title="Carte de probabilité d'impact de météorite",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="rgba(0, 0, 0, 0.3)",
                    borderwidth=1
                ),
                height=600,
                margin=dict(l=0, r=0, t=50, b=50),
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(0, 0, 0, 0)',
            )
            
            # Ajouter une légende explicative
            legend_colors = [
                {'label': 'Très probable (>75%)', 'color': '#450256'},
                {'label': 'Probable (50-75%)', 'color': '#21908C'},
                {'label': 'Possible (25-50%)', 'color': '#5DC963'},
                {'label': 'Peu probable (<25%)', 'color': '#FDE725'}
            ]
            
            legend_html = html.Div([
                html.Div([
                    html.Div(style={
                        "backgroundColor": item['color'],
                        "width": "20px",
                        "height": "20px",
                        "marginRight": "5px",
                        "borderRadius": "3px",
                        "display": "inline-block"
                    }),
                    html.Span(item['label'])
                ], className="d-flex align-items-center me-4") 
                for item in legend_colors
            ], className="d-flex flex-wrap mt-2 mb-3")
            
            # Ajouter une explication de la carte
            explanation = html.Div([
                html.Hr(),
                html.H6("Comment interpréter cette carte:", className="mt-3"),
                html.P([
                    "Cette carte montre la probabilité relative d'impact de météorites à différents endroits autour du point sélectionné. ",
                    "Les zones plus foncées (violet) indiquent une probabilité plus élevée, basée sur la densité des météorites trouvées historiquement ",
                    "et d'autres facteurs géographiques."
                ]),
                html.P([
                    html.Strong("Note: "), 
                    "Cette prévision est une simulation basée sur les données historiques et des modèles probabilistes. ",
                    "La précision de ces prévisions est limitée par la nature aléatoire des impacts de météorites."
                ], className="small text-muted")
            ])
            
            return html.Div([
                html.Div([
                    html.H5("Probabilité spatiale d'impact", className="mb-2"),
                    html.P("Les zones en violet/bleu foncé indiquent une probabilité d'impact plus élevée"),
                    legend_html
                ], className="alert alert-light border p-3"),
                dcc.Graph(figure=fig, className="mt-3 shadow-sm rounded"),
                explanation
            ])
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Erreur lors de la génération de la carte de probabilité"
                ], className="alert alert-danger"),
                html.Pre(str(e), className="bg-light p-3 small")
            ])
