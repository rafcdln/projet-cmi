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
         Input('map-style-dropdown', 'value')]
    )
    def update_prediction_map(selected_location, map_style):
        # Création de la carte de base avec les données historiques
        df = meteorite_data.get_filtered_data()
        
        # Nettoyer les données pour éviter les NaN
        df = df.dropna(subset=['reclat', 'reclong', 'mass (g)'])
        
        # Sous-échantillonner les données pour accélérer le rendu (max 1000 points)
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
        
        # Calculer le log de la masse pour les tailles des points
        df['log_mass'] = np.log10(df['mass (g)'])
        
        # Appliquer le style de carte
        actual_style, _ = get_mapbox_style(map_style)
        
        fig = px.scatter_mapbox(
            df,
            lat='reclat',
            lon='reclong',
            opacity=0.6,
            size='log_mass',
            size_max=10,
            color_discrete_sequence=['#0071e3'],
            zoom=FRANCE_ZOOM,
            center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
            height=450,
            hover_data=None
        )
        
        fig.update_traces(hoverinfo="skip", hovertemplate=None)
        
        # Ajout des instructions directement sur la carte
        fig.add_annotation(
            text="Cliquez n'importe où sur la carte pour sélectionner un point",
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
        
        # Si un emplacement est sélectionné, ajouter un marqueur
        if selected_location and 'lat' in selected_location and 'lon' in selected_location:
            fig.add_trace(go.Scattermapbox(
                lat=[selected_location['lat']],
                lon=[selected_location['lon']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='#ff9500',
                    symbol='circle'
                ),
                name="Emplacement sélectionné",
                hoverinfo="text",
                hovertext=f"Lat: {selected_location['lat']:.4f}, Lon: {selected_location['lon']:.4f}"
            ))
            
            # Ajouter un cercle autour du point
            lats, lons = create_circle(selected_location['lat'], selected_location['lon'], 2.5)
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(
                    width=2,
                    color='#ff9500'
                ),
                name="Zone d'analyse (rayon 2.5°)",
                hoverinfo="skip"
            ))
        
        # Mise à jour du layout
            fig.update_layout(
                mapbox=dict(
                    style=actual_style,
                center=dict(lat=FRANCE_LAT, lon=FRANCE_LON),
                zoom=FRANCE_ZOOM
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            showlegend=True
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
        return {
            'lat': point['lat'],
            'lon': point['lon']
        }
    
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
        
        try:
            predicted_mass = ml_model.predict_mass(
                location['lat'],
                location['lon'],
                year,
                fall
            )
            
            return html.Div([
                html.H5("Résultats de la prédiction:", className="mb-3 text-success"),
                html.Div([
                    html.Span("Masse prédite: ", className="fw-bold"),
                    html.Span(f"{predicted_mass:.2f} grammes", 
                             className="text-primary fs-5")
                ], className="mb-2"),
                html.Div([
                    html.Span("Localisation: ", className="fw-bold"),
                    html.Br(),
                    f"Latitude: {location['lat']:.4f}", html.Br(),
                    f"Longitude: {location['lon']:.4f}"
                ], className="text-muted small")
            ], className="alert alert-light border")
        except Exception as e:
            return html.Div([
                html.H5("Erreur", className="text-danger"),
                html.P(f"Erreur lors de la prédiction: {str(e)}"),
                html.P("Vérifiez les valeurs d'entrée.")
            ], className="alert alert-danger")
    
    @app.callback(
        Output('zone-analysis-output', 'children'),
        [Input('analyze-zone-button', 'n_clicks')],
        [State('selected-location', 'data')]
    )
    def analyze_zone(n_clicks, location):
        if n_clicks is None or location is None:
            return ""
        
        # Analyse d'une zone de 2.5 degrés autour du point sélectionné
        lat, lon = location['lat'], location['lon']
        df = meteorite_data.get_filtered_data()
        
        zone_data = df[
            (df['reclat'].between(lat - 2.5, lat + 2.5)) &
            (df['reclong'].between(lon - 2.5, lon + 2.5))
        ]
        
        if len(zone_data) == 0:
            return html.Div([
                html.H5("Aucune météorite connue", className="text-info mb-3"),
                html.P("Aucune météorite n'a été enregistrée dans cette zone."),
                html.P([
                    "Coordonnées analysées: ",
                    html.Br(),
                    f"Latitude: {lat:.4f} ± 2.5°",
                    html.Br(),
                    f"Longitude: {lon:.4f} ± 2.5°"
                ], className="text-muted small")
            ], className="alert alert-light border")
        
        # Calculer des statistiques avancées
        stats = {
            'Nombre de météorites': len(zone_data),
            'Masse totale': f"{zone_data['mass (g)'].sum():.1f} g",
            'Masse moyenne': f"{zone_data['mass (g)'].mean():.2f} g",
            'Masse médiane': f"{zone_data['mass (g)'].median():.2f} g",
            'Masse minimale': f"{zone_data['mass (g)'].min():.2f} g",
            'Masse maximale': f"{zone_data['mass (g)'].max():.2f} g",
            'Période': f"{int(zone_data['year'].min())} - {int(zone_data['year'].max())}",
            'Classes principales': ", ".join(zone_data['recclass'].value_counts().nlargest(3).index.tolist())
        }
        
        return html.Div([
            html.H5("Analyse de la Zone", className="text-info mb-3"),
            html.P(f"Analyse d'une zone de 2.5° autour de ({lat:.4f}, {lon:.4f})"),
            
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Caractéristique", className="text-start"),
                        html.Th("Valeur", className="text-end")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(k, className="text-start fw-bold"),
                        html.Td(v, className="text-end")
                    ]) for k, v in stats.items()
                ])
            ], className="table table-sm table-hover")
            
        ], className="alert alert-light border")
    
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

    @app.callback(
        Output('class-distribution-sorted', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_class_distribution_sorted(mass_range, classes, falls, decades):
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
        
        # Fonction pour grouper les classes de météorites
        def simplify_class(class_name):
            components = []
            # Chondrites ordinaires
            if class_name.startswith('H'):
                components.extend(['Silicates (Olivine/Pyroxène)', 'Alliage Fer-Nickel (20-25%)'])
            if class_name.startswith('L'):
                components.extend(['Silicates (Olivine/Pyroxène)', 'Alliage Fer-Nickel (15-20%)'])
            if class_name.startswith('LL'):
                components.extend(['Silicates (Olivine/Pyroxène)', 'Alliage Fer-Nickel (10-15%)'])
            
            # Chondrites carbonées
            if any(x in class_name for x in ['CI', 'CM', 'CO', 'CV', 'CK', 'CR']):
                components.extend(['Matière organique', 'Minéraux hydratés', 'Carbone'])
            
            # Chondrites à enstatite
            if class_name.startswith('E'):
                components.append('Enstatite')
            
            # Achondrites HED
            if 'Eucrite' in class_name or 'Diogenite' in class_name or 'Howardite' in class_name:
                components.extend(['Pyroxène', 'Plagioclase'])
            
            # Météorites de fer
            if 'Iron' in class_name:
                components.extend(['Fer (90-95%)', 'Nickel (5-10%)'])
            
            # Métallo-rocheuses
            if 'Pallasite' in class_name:
                components.extend(['Olivine (50%)', 'Alliage Fer-Nickel (50%)'])
            if 'Mesosiderite' in class_name:
                components.extend(['Silicates', 'Alliage Fer-Nickel'])
            
            # Météorites martiennes (SNC)
            if 'Martian' in class_name or 'Shergottite' in class_name or 'Nakhlite' in class_name:
                components.extend(['Pyroxène', 'Olivine', 'Plagioclase'])
            
            # Météorites lunaires
            if 'Lunar' in class_name:
                components.extend(['Anorthite', 'Pyroxène', 'Olivine'])
            
            if not components:
                components.append('Composition inconnue')
            return components
        
        # Extraire et compter tous les composants
        all_components = []
        for class_name in df['recclass']:
            all_components.extend(extract_components(class_name))
        
        # Compter les composants
        component_counts = pd.Series(all_components).value_counts()
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter les barres
        fig.add_trace(go.Bar(
            y=component_counts.index,
            x=component_counts.values,
            orientation='h',
            marker_color='#0071e3',
            text=component_counts.values,
            textposition='auto',
        ))
        
        # Mise en page
        fig.update_layout(
            title={
                'text': "Composants Principaux des Météorites",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Nombre de météorites",
            yaxis_title="Composant",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                automargin=True
            ),
            bargap=0.2
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
            return px.histogram(
                pd.DataFrame({'year': []}),
                x='year',
                labels={'year': 'Année', 'count': 'Nombre de météorites'},
                title="Aucune donnée disponible"
            )
        
        # Nettoyer les données
        df = df.dropna(subset=['year'])
        
        # Fonction pour grouper les classes de météorites
        def simplify_class(class_name):
            if 'Iron' in class_name:
                return 'Météorites de fer'
            if 'Pallasite' in class_name or 'Mesosiderite' in class_name:
                return 'Métallo-rocheuses'
            if 'Eucrite' in class_name or 'Diogenite' in class_name or 'Howardite' in class_name:
                return 'Achondrites HED'
            if 'Ureilite' in class_name or 'Angrite' in class_name or 'Aubrite' in class_name:
                return 'Autres achondrites'
            if class_name.startswith('H'):
                return 'Chondrites H'
            if class_name.startswith('L'):
                return 'Chondrites L'
            if class_name.startswith('LL'):
                return 'Chondrites LL'
            if any(class_name.startswith(x) for x in ['CI', 'CM', 'CO', 'CV', 'CK', 'CR']):
                return 'Chondrites carbonées'
            if class_name.startswith('E'):
                return 'Chondrites E'
            if 'Martian' in class_name or 'Shergottite' in class_name or 'Nakhlite' in class_name:
                return 'Météorites martiennes'
            if 'Lunar' in class_name:
                return 'Météorites lunaires'
            return 'Autres'
        
        # Appliquer le groupement
        df['class_group'] = df['recclass'].apply(simplify_class)
        
        # Créer l'histogramme avec coloration par type
        fig = px.histogram(
            df,
            x='year',
            color='class_group',
            nbins=40,
            labels={'year': 'Année', 'count': 'Nombre de météorites', 'class_group': 'Type'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Personnaliser la mise en page
        fig.update_layout(
            margin={"r":20,"t":40,"l":20,"b":40},
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis_title="Année",
            yaxis_title="Nombre de météorites",
            bargap=0.1,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    @app.callback(
        Output('raw-components', 'figure'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
          Input('fall-checklist', 'value'),
          Input('decade-slider', 'value')]
    )
    @error_handling_callback
    def update_raw_components(mass_range, classes, falls, decades):
        df = meteorite_data.get_filtered_data(
            mass_range=mass_range,
            classification=classes,
            fall_type=falls,
            decade_range=decades
        )
        
        if df.empty:
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
            
        # Fonction pour extraire les composants principaux
        def extract_components(class_name):
            components = []
            # Chondrites ordinaires
            if class_name.startswith('H'):
                components.extend(['Silicates (Olivine/Pyroxène)', 'Alliage Fer-Nickel (20-25%)'])
            if class_name.startswith('L'):
                components.extend(['Silicates (Olivine/Pyroxène)', 'Alliage Fer-Nickel (15-20%)'])
            if class_name.startswith('LL'):
                components.extend(['Silicates (Olivine/Pyroxène)', 'Alliage Fer-Nickel (10-15%)'])
            
            # Chondrites carbonées
            if any(x in class_name for x in ['CI', 'CM', 'CO', 'CV', 'CK', 'CR']):
                components.extend(['Matière organique', 'Minéraux hydratés', 'Carbone'])
            
            # Chondrites à enstatite
            if class_name.startswith('E'):
                components.append('Enstatite')
            
            # Achondrites HED
            if 'Eucrite' in class_name or 'Diogenite' in class_name or 'Howardite' in class_name:
                components.extend(['Pyroxène', 'Plagioclase'])
            
            # Météorites de fer
            if 'Iron' in class_name:
                components.extend(['Fer (90-95%)', 'Nickel (5-10%)'])
            
            # Métallo-rocheuses
            if 'Pallasite' in class_name:
                components.extend(['Olivine (50%)', 'Alliage Fer-Nickel (50%)'])
            if 'Mesosiderite' in class_name:
                components.extend(['Silicates', 'Alliage Fer-Nickel'])
            
            # Météorites martiennes (SNC)
            if 'Martian' in class_name or 'Shergottite' in class_name or 'Nakhlite' in class_name:
                components.extend(['Pyroxène', 'Olivine', 'Plagioclase'])
            
            # Météorites lunaires
            if 'Lunar' in class_name:
                components.extend(['Anorthite', 'Pyroxène', 'Olivine'])
            
            if not components:
                components.append('Composition inconnue')
            return components
        
        # Extraire et compter tous les composants
        all_components = []
        for class_name in df['recclass']:
            all_components.extend(extract_components(class_name))
        
        # Compter les composants
        component_counts = pd.Series(all_components).value_counts()
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter les barres
        fig.add_trace(go.Bar(
            y=component_counts.index,
            x=component_counts.values,
            orientation='h',
            marker_color='#0071e3',
            text=component_counts.values,
            textposition='auto',
        ))
        
        # Mise en page
        fig.update_layout(
            title={
                'text': "Composants Principaux des Météorites",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Nombre de météorites",
            yaxis_title="Composant",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                automargin=True
            ),
            bargap=0.2
        )
        
        return fig