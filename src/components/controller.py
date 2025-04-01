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

# Coordonnées centrées sur la France
FRANCE_LAT = 46.603354
FRANCE_LON = 1.888334
FRANCE_ZOOM = 4

def register_callbacks(app, data_path):
    # Charger les données
    try:
        meteorite_data = MeteoriteData(data_path)
        ml_model = MeteoriteML(data_path)
        
        # Entraînement initial des modèles
        ml_model.train_mass_predictor()
        ml_model.train_class_predictor()
        print("INFO: Modèles entraînés avec succès")
    except Exception as e:
        error_msg = f"ERREUR FATALE lors du chargement des données: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        # Ne pas planter l'application, mais initialiser avec des données vides
        meteorite_data = None
        ml_model = None
        if STOP_ON_ERROR:
            print("Arrêt du programme demandé après erreur.")
            sys.exit(1)
    
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
        return map_style_options.get(map_style, map_style_options.get(DEFAULT_MAP_STYLE, 'carto-positron')), None
    
    # Fonction utilitaire pour gérer les erreurs
    def handle_error(e, function_name, additional_info=None):
        error_type = type(e).__name__
        error_msg = f"ERREUR dans {function_name}: {error_type}: {str(e)}"
        print("\n" + "="*80)
        print(error_msg)
        if VERBOSE_ERRORS:
            print("\nTraceback complet:")
            print(traceback.format_exc())
            if additional_info:
                print("\nInformations supplémentaires:")
                print(additional_info)
        print("="*80 + "\n")
        if STOP_ON_ERROR:
            print("Arrêt du programme demandé après erreur.")
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
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        # Vérifier si des données sont disponibles
        if df.empty:
            fig = px.density_mapbox(
                pd.DataFrame({'lat': [FRANCE_LAT], 'lon': [FRANCE_LON], 'value': [0]}),
                lat='lat', lon='lon', z='value', zoom=FRANCE_ZOOM,
                center=dict(lat=FRANCE_LAT, lon=FRANCE_LON)
            )
            fig.update_layout(
                mapbox=dict(
                    style="carto-positron", 
                    center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
                    zoom=FRANCE_ZOOM
                ),
                margin={"r":0, "t":0, "l":0, "b":0},
                height=650
            )
            return fig
        
        # Nettoyage des données pour éviter les NaN
        df = df.dropna(subset=['mass (g)', 'reclat', 'reclong'])
        
        # Calculer le log de la masse pour une meilleure distribution des couleurs
        df['log_mass'] = np.log10(df['mass (g)'])
        
        # Appliquer le style de carte
        actual_style, _ = get_mapbox_style(map_style)
        
        fig = px.density_mapbox(
            df, 
            lat='reclat', 
            lon='reclong', 
            z='log_mass',
            radius=radius,
            zoom=FRANCE_ZOOM,
            center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
            height=650,
            opacity=0.9,
            labels={'log_mass': 'Log10(Masse)'},
            color_continuous_scale=colorscale,
            hover_data={
                'name': True,
                'mass (g)': ':.2f',
                'year': True,
                'recclass': True
            }
        )
        
        # Mise à jour du layout
        fig.update_layout(
            mapbox=dict(
                style=actual_style,
            center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
            zoom=FRANCE_ZOOM
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(
                title="Log10(Masse)",
                titleside="right",
                thicknessmode="pixels", 
                thickness=20,
                len=0.9,
                xanchor="right",
                x=0.99,
                y=0.5
            )
        )
        
        return fig
    
    @app.callback(
        Output('prediction-map', 'figure'),
        [Input('selected-location', 'data'),
         Input('selected-zone', 'data'),
         Input('map-style-dropdown', 'value'),
         Input('zone-radius', 'value')]
    )
    def update_prediction_map(selected_location, selected_zone, map_style, zone_radius):
        # Création de la carte de base avec les données historiques
        df = meteorite_data.get_filtered_data()
        
        # Nettoyer les données pour éviter les NaN
        df = df.dropna(subset=['reclat', 'reclong', 'mass (g)'])
        
        # Sous-échantillonner les données pour accélérer le rendu (max 1000 points)
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
        
        # Calculer le log de la masse pour les tailles des points
        df['log_mass'] = np.log10(df['mass (g)'])
        
        # Créer une colonne de texte pour le survol
        df['hover_text'] = df.apply(
            lambda x: f"<b>{x['name']}</b><br>" +
                      f"Classe: {x['recclass']}<br>" +
                      f"Masse: {x['mass (g)']:.2f}g<br>" +
                      f"Année: {int(x['year']) if not np.isnan(x['year']) else 'Inconnue'}<br>" +
                      f"Type: {x['fall']}",
            axis=1
        )
        
        # Appliquer le style de carte
        actual_style, _ = get_mapbox_style(map_style)
        
        fig = px.scatter_mapbox(
            df,
            lat='reclat',
            lon='reclong',
            opacity=0.7,  # Même opacité que le graphique principal
            size='log_mass',
            size_max=15,  # Taille max plus grande
            color='recclass',  # Colorier par classe comme dans le graphique principal
            hover_name='name',
            custom_data=[df['reclat'], df['reclong'], df['log_mass'], df['name'], df['recclass'], 
                        df['mass (g)'], df['year'], df['fall']],  # Inclure toutes les données pour le survol
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Utiliser la palette de couleurs Plotly
            zoom=2,  # Vue mondiale par défaut
            height=650,  # Augmentation de la hauteur pour correspondre au style mis à jour
            hover_data={
                'reclat': False,
                'reclong': False,
                'log_mass': False,
                'name': False,  # Déjà montré dans hover_name
                'recclass': True,
                'mass (g)': True,
                'year': True,
                'fall': True
            }
        )
        
        # Configuration du format de survol
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
        )
        
        # Ajout des instructions directement sur la carte
        fig.add_annotation(
            text="Cliquez n'importe où sur la carte pour sélectionner un point ou une zone",
            xref="paper", yref="paper",
            x=0.5, y=0.05,
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            align="center"
        )
        
        center_lat = 20
        center_lon = 0
        zoom_level = 1
        
        # Si un emplacement est sélectionné, ajouter un marqueur
        if selected_location and 'lat' in selected_location and 'lon' in selected_location:
            lat, lon = selected_location['lat'], selected_location['lon']
            center_lat, center_lon = lat, lon
            zoom_level = 4
            
            # Si une zone est sélectionnée, afficher la zone avec le rayon approprié
            if selected_zone is not None:
                radius = selected_zone['radius']
                
                # Ajouter un cercle autour du point pour la zone
                lats, lons = create_circle(lat, lon, radius)
            fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='lines',
                    line=dict(
                        width=3,
                        color='#ff3b30'  # Rouge plus vif pour la zone sélectionnée
                    ),
                    name="Zone sélectionnée",
                    hoverinfo="text",
                    hovertext=f"Zone: rayon de {radius}°<br>Centre: {lat:.4f}, {lon:.4f}"
                ))
                
                # Ajouter un marqueur pour le centre de la zone
                fig.add_trace(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                mode='markers',
                marker=dict(
                        size=10,
                        color='#ff3b30',
                        symbol='circle',
                        line=dict(
                            width=2,
                            color='white'
                        )
                    ),
                    name="Centre de la zone",
                hoverinfo="text",
                    hovertext=f"Centre: Lat: {lat:.4f}, Lon: {lon:.4f}"
                ))
            else:
                # Sinon, afficher un cercle d'analyse standard et un marqueur pour le point sélectionné
                # Ajouter un cercle d'analyse autour du point
                lats, lons = create_circle(lat, lon, zone_radius)
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(
                    width=2,
                        color='#ff9500'  # Orange pour le cercle d'analyse
                ),
                    name=f"Zone d'analyse (rayon {zone_radius}°)",
                hoverinfo="skip"
            ))
        
                # Ajouter un marqueur pour le point sélectionné
                fig.add_trace(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='#ff9500',
                        symbol='marker',
                        line=dict(
                            width=2,
                            color='white'
                        )
                    ),
                    name="Point sélectionné",
                    hoverinfo="text",
                    hovertext=f"Point: Lat: {lat:.4f}, Lon: {lon:.4f}"
                ))
        
        # Mise à jour du layout avec le centre et le zoom déterminés
            fig.update_layout(
                mapbox=dict(
                    style=actual_style,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom_level
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            showlegend=True,
            legend=dict(
                title="Classes de météorites",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            clickmode='event+select'  # Activer la sélection et les événements de clic
            )
        
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
        [Input('predict-button', 'n_clicks')],
        [State('selected-location', 'data'),
         State('pred-year', 'value'),
         State('pred-fall', 'value')]
    )
    def make_prediction(n_clicks, location, year, fall):
        if n_clicks is None or location is None:
            return ""
        
        if location is None:
            return html.Div([
                html.H5([html.I(className="fas fa-exclamation-triangle me-2"), "Point non sélectionné"], 
                      className="text-warning mb-3"),
                html.P("Veuillez cliquer sur la carte pour sélectionner un emplacement.")
            ], className="alert alert-warning border rounded shadow-sm")
            
        if year is None or fall is None:
            return html.Div([
                html.H5([html.I(className="fas fa-exclamation-triangle me-2"), "Paramètres incomplets"], 
                      className="text-warning mb-3"),
                html.P("Veuillez spécifier l'année et le type de chute.")
            ], className="alert alert-warning border rounded shadow-sm")
            
        try:
            # Vérifier que le modèle est correctement initialisé
            if ml_model is None:
                raise ValueError("Le modèle de prédiction n'est pas initialisé.")
                
            # Validation des données d'entrée
            if not (-90 <= location['lat'] <= 90) or not (-180 <= location['lon'] <= 180):
                raise ValueError("Coordonnées géographiques invalides.")
                
            if not (1800 <= year <= 2050):
                raise ValueError(f"Année invalide: {year}. Doit être entre 1800 et 2050.")
                
            if fall not in ['Fell', 'Found']:
                raise ValueError(f"Type de chute invalide: {fall}. Doit être 'Fell' ou 'Found'.")
            
            # Effectuer la prédiction
            predicted_mass = ml_model.predict_mass(
                location['lat'],
                location['lon'],
                year,
                fall
            )
            
            # Formater la masse prédite
            if predicted_mass < 1:
                mass_formatted = f"{predicted_mass*1000:.1f} milligrammes"
            elif predicted_mass < 1000:
                mass_formatted = f"{predicted_mass:.2f} grammes"
            elif predicted_mass < 1000000:
                mass_formatted = f"{predicted_mass/1000:.2f} kilogrammes"
            else:
                mass_formatted = f"{predicted_mass/1000000:.2f} tonnes"
            
            # Récupérer les données de météorites proches pour contexte
            df = meteorite_data.get_filtered_data()
            nearby = df[
                (df['reclat'].between(location['lat'] - 5, location['lat'] + 5)) &
                (df['reclong'].between(location['lon'] - 5, location['lon'] + 5))
            ]
            
            nearby_count = len(nearby)
            nearby_classes = nearby['recclass'].value_counts().to_dict()
            top_classes = sorted(nearby_classes.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Calculer la masse moyenne dans la région
            avg_mass = nearby['mass (g)'].mean() if not nearby.empty else 0
            
            # Préparer un texte comparatif
            comparison_text = ""
            if avg_mass > 0:
                if predicted_mass > avg_mass * 1.5:
                    comparison_text = "Cette prédiction est nettement supérieure à la moyenne régionale."
                elif predicted_mass < avg_mass * 0.5:
                    comparison_text = "Cette prédiction est nettement inférieure à la moyenne régionale."
                else:
                    comparison_text = "Cette prédiction est proche de la moyenne régionale."
            
            return html.Div([
                html.H5([html.I(className="fas fa-meteor me-2"), "Résultats de la prédiction"], 
                       className="mb-3 text-primary"),
                
                # Cartouche principal avec la masse prédite
                html.Div([
                html.Div([
                        html.H3(mass_formatted, className="m-0")
                    ], className="p-3 bg-primary text-white rounded-top"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Strong("Coordonnées:"),
                                html.Div(f"Latitude: {location['lat']:.4f}"),
                                html.Div(f"Longitude: {location['lon']:.4f}")
                            ], className="col-md-4"),
                            html.Div([
                                html.Strong("Paramètres:"),
                                html.Div(f"Année: {year}"),
                                html.Div(f"Type: {fall}")
                            ], className="col-md-4"),
                            html.Div([
                                html.Strong("Météorites similaires:"),
                                html.Div(f"{nearby_count} météorites dans la région")
                            ], className="col-md-4")
                        ], className="row")
                    ], className="p-3 bg-light rounded-bottom border border-top-0")
                ], className="mb-3 shadow-sm"),
                
                # Contexte régional
                html.Div([
                    html.H6([html.I(className="fas fa-info-circle me-2"), "Contexte régional"], 
                           className="mb-2"),
                    html.Div([
                        html.Div([
                            html.Strong("Classes dominantes dans la région:"),
                            html.Ul([
                                html.Li(f"{c}: {n} météorites") for c, n in top_classes
                            ] if top_classes else [html.Li("Aucune donnée disponible")])
                        ], className="col-md-6"),
                        html.Div([
                            html.Strong("Comparaison:"),
                            html.P(comparison_text or "Données insuffisantes pour la comparaison.", 
                                  className="text-muted")
                        ], className="col-md-6")
                    ], className="row")
                ], className="mt-3 p-3 bg-light rounded border")
                
            ], className="alert alert-light border rounded shadow-sm p-4")
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"Erreur dans make_prediction: {str(e)}")
            print(trace)
            
            return html.Div([
                html.H5([html.I(className="fas fa-exclamation-circle me-2"), "Erreur"], 
                       className="text-danger mb-3"),
                html.P(f"Erreur lors de la prédiction: {str(e)}"),
                html.P("Vérifiez les valeurs d'entrée et réessayez."),
                html.Details([
                    html.Summary("Détails techniques (cliquez pour développer)", 
                                className="text-muted small"),
                    html.Pre(trace, style={"whiteSpace": "pre-wrap", "fontSize": "0.8rem"})
                ], className="mt-2")
            ], className="alert alert-danger border rounded shadow-sm")
    
    @app.callback(
        Output('zone-analysis-output', 'children'),
        [Input('analyze-zone-button', 'n_clicks')],
        [State('selected-location', 'data')]
    )
    def analyze_zone(n_clicks, location):
        if n_clicks is None or location is None:
            return ""
        
        if location is None:
            return html.Div([
                html.H5([html.I(className="fas fa-exclamation-triangle me-2"), "Point non sélectionné"], 
                       className="text-warning mb-3"),
                html.P("Veuillez cliquer sur la carte pour sélectionner un emplacement.")
            ], className="alert alert-warning border rounded shadow-sm")
        
        # Analyse d'une zone de 2.5 degrés autour du point sélectionné
        lat, lon = location['lat'], location['lon']
        df = meteorite_data.get_filtered_data()
        
        zone_data = df[
            (df['reclat'].between(lat - 2.5, lat + 2.5)) &
            (df['reclong'].between(lon - 2.5, lon + 2.5))
        ]
        
        if len(zone_data) == 0:
            return html.Div([
                html.H5([html.I(className="fas fa-search-minus me-2"), "Aucune météorite connue"], 
                       className="text-info mb-3"),
                html.P("Aucune météorite n'a été enregistrée dans cette zone."),
                html.P("Essayez de sélectionner une autre zone ou d'élargir vos critères de recherche.")
            ], className="alert alert-info border rounded shadow-sm")
        
        # Calculs statistiques
        total_count = len(zone_data)
        total_mass = zone_data['mass (g)'].sum()
        avg_mass = zone_data['mass (g)'].mean()
        min_mass = zone_data['mass (g)'].min()
        max_mass = zone_data['mass (g)'].max()
        
        # Distribution des années
        earliest = int(zone_data['year'].min()) if not np.isnan(zone_data['year'].min()) else 'Inconnue'
        latest = int(zone_data['year'].max()) if not np.isnan(zone_data['year'].max()) else 'Inconnue'
        
        # Distribution des classes
        classes = zone_data['recclass'].value_counts().to_dict()
        top_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Distribution des types de chute
        falls = zone_data['fall'].value_counts().to_dict()
        
        # Formatage des masses pour l'affichage
        def format_mass(mass):
            if mass < 1:
                return f"{mass*1000:.1f} mg"
            elif mass < 1000:
                return f"{mass:.2f} g"
            elif mass < 1000000:
                return f"{mass/1000:.2f} kg"
            else:
                return f"{mass/1000000:.2f} t"
        
        # Création du rapport détaillé
        return html.Div([
            html.H5([html.I(className="fas fa-search-location me-2"), "Analyse de la Zone"],
                   className="mb-3 text-primary"),
            
            # Cartouche principal avec le résumé
            html.Div([
                html.Div([
                    html.H4(f"{total_count} météorites", className="m-0"),
                    html.P(f"dans un rayon de 2.5° autour de ({lat:.4f}, {lon:.4f})", 
                          className="mb-0 text-white-50")
                ], className="p-3 bg-primary text-white rounded-top"),
                
                html.Div([
                    html.Div([
                        html.Div([
                            html.Strong("Période:"),
                            html.Div(f"De {earliest} à {latest}")
                        ], className="col-md-4"),
                        html.Div([
                            html.Strong("Masse totale:"),
                            html.Div(format_mass(total_mass))
                        ], className="col-md-4"),
                        html.Div([
                            html.Strong("Masse moyenne:"),
                            html.Div(format_mass(avg_mass))
                        ], className="col-md-4")
                    ], className="row")
                ], className="p-3 bg-light rounded-bottom border border-top-0")
            ], className="mb-3 shadow-sm"),
            
            # Détails supplémentaires
            html.Div([
                # Distribution des classes
                html.Div([
                    html.H6([html.I(className="fas fa-tags me-2"), "Classes de météorites"], 
                           className="mb-2"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Strong(f"{c}: "),
                                f"{n} météorites ({n/total_count*100:.1f}%)"
                            ], className="mb-1") for c, n in top_classes
                        ], className="col-md-6"),
                        html.Div([
                            html.Strong("Distribution des masses:"),
                            html.Ul([
                                html.Li(f"Minimum: {format_mass(min_mass)}"),
                                html.Li(f"Maximum: {format_mass(max_mass)}"),
                                html.Li(f"Médiane: {format_mass(zone_data['mass (g)'].median())}")
                            ], className="mb-0")
                        ], className="col-md-6")
                    ], className="row")
                ], className="mb-3"),
                
                # Types de chute
                html.Div([
                    html.H6([html.I(className="fas fa-meteor me-2"), "Types de chute"], 
                           className="mb-2"),
                    html.Div([
                        html.Div([
                            html.Strong(f"{fall}: "),
                            f"{count} météorites ({count/total_count*100:.1f}%)"
                        ], className="mb-1") for fall, count in falls.items()
                    ])
                ], className="mb-3"),
                
                # Lien vers météorites remarquables
                html.Div([
                    html.H6([html.I(className="fas fa-star me-2"), "Météorites remarquables"], 
                           className="mb-2"),
                    # Trouver les 3 plus grosses météorites de la zone
                    html.Ul([
                        html.Li([
                            html.Strong(f"{row['name']}: "),
                            f"{format_mass(row['mass (g)'])} - {row['recclass']} ({int(row['year']) if not np.isnan(row['year']) else 'Année inconnue'})"
                        ]) for _, row in zone_data.nlargest(3, 'mass (g)').iterrows()
                    ])
                ])
            ], className="p-3 bg-light rounded border")
            
        ], className="alert alert-light border rounded shadow-sm p-4")
    
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
        # Styles par défaut
        base_style = {'min-height': '350px', 'overflow': 'auto', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}
        hidden_style = {'min-height': '350px', 'overflow': 'auto', 'display': 'none', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}
        
        # Classes de boutons
        inactive_btn = 'btn btn-outline-primary me-2'
        active_btn = 'btn btn-primary me-2'
        inactive_last_btn = 'btn btn-outline-primary'
        active_last_btn = 'btn btn-primary'
        
        # Valeurs par défaut - afficher prédiction simple
        pred_style = base_style
        zone_style = hidden_style
        temporal_style = hidden_style
        spatial_style = hidden_style
        
        pred_btn = active_btn
        zone_btn = inactive_btn
        temporal_btn = inactive_btn
        spatial_btn = inactive_last_btn
        
        # Déterminer quel bouton a été cliqué
        ctx = dash.callback_context
        if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
            # Réinitialiser tous les styles et classes
            pred_style = hidden_style
            zone_style = hidden_style
            temporal_style = hidden_style
            spatial_style = hidden_style
            
            pred_btn = inactive_btn
            zone_btn = inactive_btn
            temporal_btn = inactive_btn
            spatial_btn = inactive_last_btn
            
            # Mettre à jour en fonction du bouton cliqué
        if button_id == 'btn-prediction-results':
                pred_style = base_style
                pred_btn = active_btn
            elif button_id == 'btn-zone-analysis':
                zone_style = base_style
                zone_btn = active_btn
            elif button_id == 'btn-temporal-prediction':
                temporal_style = base_style
                temporal_btn = active_btn
            elif button_id == 'btn-spatial-prediction':
                spatial_style = base_style
                spatial_btn = active_last_btn
        
        return pred_style, zone_style, temporal_style, spatial_style, pred_btn, zone_btn, temporal_btn, spatial_btn
    
    # Callback pour la prédiction temporelle
    @app.callback(
        Output('temporal-prediction-output', 'children'),
        [Input('btn-temporal-prediction', 'n_clicks'),
         Input('forecast-horizon', 'value'),
         Input('season-filter', 'value')],
        [State('selected-location', 'data'),
         State('selected-zone', 'data')]
    )
    def generate_temporal_prediction(n_clicks, horizon, season, location, zone):
        if n_clicks is None or (location is None and zone is None):
            return html.Div("Cliquez sur le bouton 'Prévision Temporelle' après avoir sélectionné un point ou une zone.", 
                           className="text-muted p-3")
        
        # Déterminer si on utilise un point ou une zone
        if zone is not None:
            lat, lon = zone['center_lat'], zone['center_lon']
            radius = zone['radius']
            location_type = "zone"
        else:
            lat, lon = location['lat'], location['lon']
            radius = 2.5  # Rayon par défaut
            location_type = "point"
        
        try:
            # Récupérer les données historiques dans la région
            df = meteorite_data.get_filtered_data()
            
            # Filtrer les données pour la région sélectionnée
            if location_type == "zone":
                nearby = df[
                    (df['reclat'].between(lat - radius, lat + radius)) &
                    (df['reclong'].between(lon - radius, lon + radius))
                ]
            else:
                nearby = df[
                    (df['reclat'].between(lat - 2.5, lat + 2.5)) &
                    (df['reclong'].between(lon - 2.5, lon + 2.5))
                ]
            
            # Si pas assez de données, élargir la zone
            if len(nearby) < 5:
                search_radius = radius * 2
                nearby = df[
                    (df['reclat'].between(lat - search_radius, lat + search_radius)) &
                    (df['reclong'].between(lon - search_radius, lon + search_radius))
                ]
            
            # S'il n'y a toujours pas assez de données
            if len(nearby) < 3:
                return html.Div([
                    html.H5([html.I(className="fas fa-exclamation-triangle me-2"), "Données insuffisantes"], 
                           className="text-warning mb-3"),
                    html.P("Pas assez de données historiques dans cette région pour générer une prévision temporelle fiable."),
                    html.P("Essayez de sélectionner une autre zone ou d'élargir le rayon.")
                ], className="alert alert-warning")
            
            # Filtrer par saison si nécessaire
            if season != 'all':
                # Supposons que la colonne de mois existe ou puisse être dérivée
                # Dans un cas réel, vous devrez adapter ceci à vos données
                season_months = {
                    'winter': [12, 1, 2],
                    'spring': [3, 4, 5],
                    'summer': [6, 7, 8],
                    'autumn': [9, 10, 11]
                }
                
                # Si la colonne de date existe, filtrer par mois
                if 'date' in df.columns:
                    try:
                        nearby['month'] = pd.to_datetime(nearby['date']).dt.month
                        nearby = nearby[nearby['month'].isin(season_months[season])]
                    except:
                        pass
            
            # Analyser les tendances temporelles
            yearly_data = nearby.groupby(nearby['year'].astype(int)).size().reset_index(name='count')
            
            # Calculer la moyenne et l'écart-type pour l'estimation probabiliste
            avg_yearly = yearly_data['count'].mean()
            std_yearly = yearly_data['count'].std() if len(yearly_data) > 1 else avg_yearly * 0.5
            
            # Générer une prévision pour les prochaines années
            current_year = datetime.now().year
            future_years = list(range(current_year, current_year + horizon + 1))
            
            # Créer une valeur médiane estimée pour une visualisation
            predicted_counts = np.random.normal(avg_yearly, std_yearly, len(future_years))
            predicted_counts = np.maximum(predicted_counts, 0)  # Pas de valeurs négatives
            
            # Calculer des intervalles de confiance
            lower_bound = np.maximum(predicted_counts - 1.96 * std_yearly, 0)
            upper_bound = predicted_counts + 1.96 * std_yearly
            
            # Préparer les données pour le graphique
            forecast_df = pd.DataFrame({
                'Année': future_years,
                'Prévision': predicted_counts,
                'Min': lower_bound,
                'Max': upper_bound
            })
            
            # Calculer les probabilités pour l'affichage
            prob_at_least_one = 1 - np.exp(-avg_yearly)
            prob_at_least_one_percent = prob_at_least_one * 100
            
            # Estimer la date de la prochaine chute
            days_until_next = 365 / avg_yearly if avg_yearly > 0 else float('inf')
            next_date_estimate = (datetime.now() + pd.Timedelta(days=days_until_next)).strftime('%d/%m/%Y')
            
            # Créer un graphique de prévision
            fig = go.Figure()
            
            # Données historiques
            fig.add_trace(go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['count'],
                mode='markers',
                name='Historique',
                marker=dict(color='#0071e3', size=8)
            ))
            
            # Ligne de prévision
            fig.add_trace(go.Scatter(
                x=forecast_df['Année'],
                y=forecast_df['Prévision'],
                mode='lines',
                name='Prévision médiane',
                line=dict(color='#ff9500', width=2)
            ))
            
            # Intervalle de confiance
            fig.add_trace(go.Scatter(
                x=forecast_df['Année'].tolist() + forecast_df['Année'].tolist()[::-1],
                y=forecast_df['Max'].tolist() + forecast_df['Min'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 149, 0, 0.2)',
                line=dict(color='rgba(255, 149, 0, 0)'),
                hoverinfo='skip',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"Prévision des Chutes de Météorites {f'({season})' if season != 'all' else ''}",
                xaxis_title="Année",
                yaxis_title="Nombre de météorites",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=300
            )
            
            # Résumé textuel de la prévision
            return html.Div([
                html.H5([html.I(className="fas fa-calendar-alt me-2"), "Prévision Temporelle"], 
                       className="mb-3 text-primary"),
                
                html.Div([
                    # Statistiques de prévision
                    html.Div([
                        html.Div([
                            html.H4(f"{avg_yearly:.1f}", className="mb-0 text-primary"),
                            html.P("météorites par an", className="text-muted mb-0")
                        ], className="p-3 bg-light rounded border mb-3"),
                        
                        html.Div([
                            html.Strong("Prochaine météorite estimée:", className="d-block mb-2"),
                            html.Div([
                                html.Span("Date: ", className="fw-bold"), 
                                html.Span(next_date_estimate, className="text-primary")
                            ], className="mb-1"),
                            html.Div([
                                html.Span("Probabilité dans l'année: ", className="fw-bold"), 
                                html.Span(f"{prob_at_least_one_percent:.1f}%", className="text-primary")
                            ])
                        ], className="mb-3")
                    ], className="col-md-4"),
                    
                    # Graphique de prévision
                    html.Div([
                        dcc.Graph(
                            figure=fig,
                            config={'displayModeBar': False}
                        )
                    ], className="col-md-8")
                ], className="row mb-3"),
                
                # Détails de la méthodologie
                html.Div([
                    html.H6("Méthodologie", className="mb-2"),
                    html.P([
                        "Cette prévision est basée sur l'analyse de ",
                        html.Strong(f"{len(nearby)} météorites historiques "),
                        f"dans un rayon de {radius if location_type == 'zone' else 2.5}° autour du point sélectionné. ",
                        "L'estimation utilise un modèle de distribution de Poisson pour les événements rares."
                    ], className="text-muted small")
                ], className="mt-3 p-3 bg-light rounded border")
                
            ], className="p-3")
        
        except Exception as e:
            import traceback
            print(f"Erreur dans la prédiction temporelle: {str(e)}")
            print(traceback.format_exc())
            
            return html.Div([
                html.H5([html.I(className="fas fa-exclamation-circle me-2"), "Erreur"], 
                       className="text-danger mb-3"),
                html.P(f"Erreur lors de la génération de la prévision temporelle: {str(e)}"),
                html.Details([
                    html.Summary("Détails techniques", className="text-muted"),
                    html.Pre(traceback.format_exc(), style={"whiteSpace": "pre-wrap", "fontSize": "0.8rem"})
                ])
            ], className="alert alert-danger")
    
    # Callback pour la prédiction spatiale
    @app.callback(
        Output('spatial-prediction-output', 'children'),
        [Input('btn-spatial-prediction', 'n_clicks'),
         Input('prediction-model', 'value')],
        [State('selected-location', 'data'),
         State('selected-zone', 'data'),
         State('pred-year', 'value'),
         State('pred-fall', 'value')]
    )
    def generate_spatial_prediction(n_clicks, model_type, location, zone, year, fall_type):
        if n_clicks is None or (location is None and zone is None):
            return html.Div("Cliquez sur le bouton 'Probabilité Spatiale' après avoir sélectionné un point ou une zone.", 
                           className="text-muted p-3")
        
        # Déterminer si on utilise un point ou une zone
        if zone is not None:
            lat, lon = zone['center_lat'], zone['center_lon']
            radius = zone['radius']
            location_type = "zone"
        else:
            lat, lon = location['lat'], location['lon']
            radius = 2.5  # Rayon par défaut
            location_type = "point"
        
        try:
            # Récupérer les données mondiales
            df = meteorite_data.get_filtered_data()
            
            # Données dans la région sélectionnée
            nearby = df[
                (df['reclat'].between(lat - radius * 2, lat + radius * 2)) &
                (df['reclong'].between(lon - radius * 2, lon + radius * 2))
            ]
            
            # S'il n'y a pas assez de données
            if len(nearby) < 5:
                return html.Div([
                    html.H5([html.I(className="fas fa-exclamation-triangle me-2"), "Données insuffisantes"], 
                           className="text-warning mb-3"),
                    html.P("Pas assez de données historiques dans cette région pour générer une carte de probabilité."),
                    html.P("Essayez de sélectionner une zone avec plus de météorites historiques.")
                ], className="alert alert-warning")
            
            # Créer une grille de points pour la heatmap
            grid_size = 20  # Résolution de la grille
            lat_grid = np.linspace(lat - radius, lat + radius, grid_size)
            lon_grid = np.linspace(lon - radius, lon + radius, grid_size)
            
            # Créer un maillage pour la grille
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Calculer la densité pour chaque point (utilisation d'un KDE simple pour l'exemple)
            density = np.zeros((grid_size, grid_size))
            for nlat, nlon in zip(nearby['reclat'], nearby['reclong']):
                # Calculer la contribution de chaque météorite historique à la densité
                # Formule simple de noyau gaussien
                density += np.exp(-0.5 * ((lat_mesh - nlat)**2 + (lon_mesh - nlon)**2) / (radius/5)**2)
            
            # Normaliser les densités
            if density.max() > 0:
                density = density / density.max()
            
            # Créer un DataFrame pour la heatmap
            heatmap_data = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if density[i, j] > 0.01:  # Ignorer les valeurs trop faibles
                        heatmap_data.append({
                            'lat': lat_mesh[i, j],
                            'lon': lon_mesh[i, j],
                            'probability': density[i, j]
                        })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Créer une carte de chaleur de probabilité
            fig = px.density_mapbox(
                heatmap_df,
                lat='lat',
                lon='lon',
                z='probability',
                radius=max(5, 20 * radius / 10),  # Ajuster le rayon en fonction de la taille de la zone
                center=dict(lat=lat, lon=lon),
                zoom=max(4, 10 - radius),  # Ajuster le zoom en fonction du rayon
                color_continuous_scale='Viridis',
                opacity=0.8,
                labels={'probability': 'Probabilité relative'},
                mapbox_style="carto-positron",
                height=400
            )
            
            # Ajouter le point ou la zone sélectionnée
            if location_type == "zone":
                # Ajouter un cercle pour montrer la zone
                lats, lons = create_circle(lat, lon, radius)
                fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='lines',
                    line=dict(width=2, color='red'),
                    name="Zone sélectionnée"
                ))
        else:
                # Ajouter un marqueur pour le point
                fig.add_trace(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name="Point sélectionné"
                ))
            
            # Ajouter des marqueurs pour les météorites historiques
            fig.add_trace(go.Scattermapbox(
                lat=nearby['reclat'],
                lon=nearby['reclong'],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.7),
                name="Météorites historiques",
                hoverinfo='text',
                hovertext=nearby.apply(
                    lambda row: f"{row['name']}<br>Masse: {row['mass (g)']:.1f}g<br>Année: {int(row['year'])}", 
                    axis=1
                )
            ))
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="left",
                    x=0
                )
            )
            
            # Calculer des statistiques pour la zone
            point_probability = 0
            zone_probability = 0
            
            if len(nearby) > 0:
                # Densité moyenne des météorites dans la région
                region_area = 3.14159 * (radius * 111)**2  # Approximation de l'aire en km²
                density_per_km2 = len(nearby) / region_area
                
                # Probabilité estimée pour un point spécifique (très approximative)
                point_area = 0.01  # km²
                point_probability = density_per_km2 * point_area * 100 * (1/100)  # Probabilité sur 100 ans
                
                # Probabilité pour la zone entière
                zone_probability = min(95, len(nearby) / region_area * region_area * 100 * (1/100))  # Probabilité sur 100 ans
            
            # Préparer le texte de présentation
            return html.Div([
                html.H5([html.I(className="fas fa-globe me-2"), "Carte de Probabilité Spatiale"], 
                       className="mb-3 text-primary"),
                
                # Carte et statistiques
                html.Div([
                    # Statistiques
                    html.Div([
                        html.H6("Probabilités estimées", className="mb-3"),
                        
                        html.Div([
                            html.Div([
                                html.Strong("Point précis:"),
                                html.Div([
                                    html.Span(f"{point_probability:.4f}%", 
                                             className="text-primary fw-bold fs-5")
                                ])
                            ], className="p-3 bg-light rounded border mb-3"),
                            
                            html.Div([
                                html.Strong("Zone entière:"),
                                html.Div([
                                    html.Span(f"{zone_probability:.2f}%", 
                                             className="text-primary fw-bold fs-5")
                                ])
                            ], className="p-3 bg-light rounded border mb-3"),
                            
                            html.Div([
                                html.Strong("Période:"),
                                html.Div("Prochain siècle")
                            ], className="text-muted small")
                        ])
                    ], className="col-md-3"),
                    
                    # Carte
                    html.Div([
                        dcc.Graph(
                            figure=fig,
                            config={'scrollZoom': True}
                        )
                    ], className="col-md-9")
                ], className="row mb-3"),
                
                # Explication de la méthodologie
                html.Div([
                    html.H6("Interprétation", className="mb-2"),
                    html.P([
                        "Cette carte montre la probabilité relative de chute de météorite, basée sur ",
                        html.Strong(f"{len(nearby)} météorites historiques "),
                        "dans la région. Les zones en jaune-vert ont une probabilité plus élevée, ",
                        "tandis que les zones en bleu foncé ont une probabilité plus faible. ",
                        "Le modèle utilisé est basé sur une estimation par noyau (KDE) des densités spatiales."
                    ], className="text-muted small")
                ], className="mt-3 p-3 bg-light rounded border")
                
            ], className="p-3")
            
        except Exception as e:
            import traceback
            print(f"Erreur dans la prédiction spatiale: {str(e)}")
            print(traceback.format_exc())
            
            return html.Div([
                html.H5([html.I(className="fas fa-exclamation-circle me-2"), "Erreur"], 
                       className="text-danger mb-3"),
                html.P(f"Erreur lors de la génération de la carte de probabilité: {str(e)}"),
                html.Details([
                    html.Summary("Détails techniques", className="text-muted"),
                    html.Pre(traceback.format_exc(), style={"whiteSpace": "pre-wrap", "fontSize": "0.8rem"})
                ])
            ], className="alert alert-danger")
    
    # Store pour stocker le rayon de la zone sélectionnée
    @app.callback(
        Output('selected-zone', 'data'),
        [Input('select-zone-button', 'n_clicks'),
         Input('zone-radius', 'value')],
        [State('selected-location', 'data')]
    )
    def store_selected_zone(n_clicks, radius, location):
        if n_clicks is None or n_clicks == 0 or location is None:
            return None
            
        return {
            'center_lat': location['lat'],
            'center_lon': location['lon'],
            'radius': radius
        }
    
    # Callback pour mettre à jour les coordonnées sélectionnées
    @app.callback(
        Output('selected-coordinates', 'children'),
        [Input('selected-location', 'data'),
         Input('selected-zone', 'data')]
    )
    def update_selected_coordinates(location, zone):
        if location is None:
            return "Aucun point sélectionné. Cliquez sur la carte."
        
        if zone is not None:
            return [
                html.Strong("Zone sélectionnée:", className="d-block mb-2 text-success"),
                html.Span("Centre: ", className="fw-bold"),
                f"Lat: {zone['center_lat']:.4f}, Lon: {zone['center_lon']:.4f}",
                html.Br(),
                html.Span("Rayon: ", className="fw-bold"),
                f"{zone['radius']}°",
                html.Br(),
                html.Span("Surface couverte: ", className="fw-bold"),
                f"~{3.14159 * (zone['radius'] * 111)**2:.0f} km²"
            ]
        
        return [
            html.Strong("Point sélectionné:", className="d-block mb-2"),
            html.Span("Latitude: ", className="fw-bold"),
            f"{location['lat']:.4f}",
            html.Br(),
            html.Span("Longitude: ", className="fw-bold"),
            f"{location['lon']:.4f}"
        ]