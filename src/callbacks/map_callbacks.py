"""
Callbacks pour la page de carte mondiale des météorites
"""
from dash import Input, Output, State, callback, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from models.model import MeteoriteData
from utils.config import DATA_PATH, DEFAULT_MAP_STYLE, COLOR_SCHEMES
import json

# Charger une seule fois les données au démarrage du serveur
meteorite_data = MeteoriteData(DATA_PATH)

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

# Callback pour la carte mondiale
@callback(
    Output('world-map', 'figure'),
    [Input('mass-slider', 'value'),
     Input('class-dropdown', 'value'),
     Input('fall-checklist', 'value'),
     Input('decade-slider', 'value'),
     Input('color-mode-dropdown', 'value'),
     Input('map-style-dropdown', 'value'),
     Input('point-opacity', 'value'),
     Input('point-size', 'value')]
)
def update_world_map(mass_range, classes, falls, decades, color_mode, map_style, opacity, point_size):
    """
    Met à jour la carte du monde en fonction des filtres
    """
    # Filtrer les données
    filtered_data = filter_data(mass_range, classes, falls, decades)

    # Si aucune donnée après filtrage, retourner une carte vide
    if filtered_data.empty:
        empty_fig = create_empty_map(map_style)
        return empty_fig

    # Définir le hover_data en fonction du mode de couleur
    hover_data = {
        'name': True,
        'class': True,
        'mass (g)': ':.2f',
        'year': True,
        'fall': True,
        'lat': False,
        'lon': False
    }

    # Préparer les données de couleur selon le mode sélectionné
    if color_mode == 'mass':
        color_data = np.log10(filtered_data['mass (g)'])
        color_title = 'Log10 Masse (g)'
        colorscale = 'Viridis'
    elif color_mode == 'year':
        color_data = filtered_data['year']
        color_title = 'Année'
        colorscale = 'Plasma'
    elif color_mode == 'fall':
        color_data = filtered_data['fall']
        color_title = 'Type de chute'
        colorscale = [[0, COLOR_SCHEMES['primary']], [1, COLOR_SCHEMES['warning']]]
    else:  # color_mode == 'class' ou autre
        color_data = filtered_data['class']
        color_title = 'Classe'
        colorscale = px.colors.qualitative.Pastel

    # Créer la figure Plotly
    fig = px.scatter_mapbox(
        filtered_data,
        lat="lat",
        lon="lon",
        color=color_data,
        size=[point_size] * len(filtered_data),  # Taille fixe des points
        hover_name="name",
        hover_data=hover_data,
        color_continuous_scale=colorscale if color_mode in ['mass', 'year'] else None,
        category_orders={'fall': ['Fell', 'Found']} if color_mode == 'fall' else None,
        opacity=opacity,
        zoom=2,
        height=650,
        title=f"Distribution des météorites ({len(filtered_data)} points)"
    )

    # Configurer le style de la carte
    fig.update_layout(
        mapbox_style=get_mapbox_style(map_style),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        coloraxis_colorbar=dict(
            title=color_title
        ) if color_mode in ['mass', 'year'] else {}
    )

    return fig

# Callback pour la carte de densité
@callback(
    Output('heatmap', 'figure'),
    [Input('mass-slider', 'value'),
     Input('class-dropdown', 'value'),
     Input('fall-checklist', 'value'),
     Input('decade-slider', 'value'),
     Input('map-style-dropdown', 'value'),
     Input('heatmap-radius', 'value'),
     Input('heatmap-colorscale', 'value')]
)
def update_heatmap(mass_range, classes, falls, decades, map_style, radius, colorscale):
    """
    Met à jour la carte de densité de chaleur des météorites
    """
    # Filtrer les données
    filtered_data = filter_data(mass_range, classes, falls, decades)

    # Si aucune donnée après filtrage, retourner une carte vide
    if filtered_data.empty:
        empty_fig = create_empty_map(map_style)
        return empty_fig

    # Créer la carte de densité
    fig = go.Figure()

    # Ajouter la couche de densité
    fig.add_densitymapbox(
        lat=filtered_data['lat'],
        lon=filtered_data['lon'],
        radius=radius,
        colorscale=colorscale,
        zmax=30,
        below='',
        opacity=0.8,
    )

    # Configurer le style de la carte
    fig.update_layout(
        mapbox_style=get_mapbox_style(map_style),
        mapbox=dict(
            center=dict(lat=0, lon=0),
            zoom=1
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=650,
    )

    return fig

# Variable globale pour stocker les données filtrées
filtered_data_json = None

# Fonction utilitaire pour filtrer les données
def filter_data(mass_range, classes, falls, decades):
    """
    Filtre les données selon les critères sélectionnés
    """
    global filtered_data_json

    df = meteorite_data.data.copy()

    # Filtrer par masse (en log10)
    mass_min, mass_max = 10 ** mass_range[0], 10 ** mass_range[1]
    df = df[(df['mass (g)'] >= mass_min) & (df['mass (g)'] <= mass_max)]

    # Filtrer par classe si une sélection spécifique est faite
    if classes != 'all' and classes:
        if not isinstance(classes, list):
            classes = [classes]
        df = df[df['class'].isin(classes)]

    # Filtrer par type de chute
    if falls and len(falls) > 0:
        df = df[df['fall'].isin(falls)]

    # Filtrer par décennie
    min_decade, max_decade = decades
    df = df[(df['year'] >= min_decade) & (df['year'] <= max_decade)]

    # Stocker les données filtrées dans la variable globale
    filtered_data_json = df.to_json(date_format='iso', orient='split')

    return df

# Callback pour stocker les données filtrées et les rendre disponibles pour d'autres callbacks
@callback(
    Output('filtered-data', 'data'),
    [Input('mass-slider', 'value'),
     Input('class-dropdown', 'value'),
     Input('fall-checklist', 'value'),
     Input('decade-slider', 'value')]
)
def store_filtered_data(mass_range, classes, falls, decades):
    """
    Stocke les données filtrées dans le composant dcc.Store
    """
    # Filtrer les données
    df = filter_data(mass_range, classes, falls, decades)

    # Retourner les données filtrées au format JSON
    return df.to_json(date_format='iso', orient='split')

# Callback pour générer les statistiques
@callback(
    [Output('stats-count', 'children'),
     Output('stats-mass', 'children'),
     Output('stats-years', 'children'),
     Output('stats-classes', 'children'),
     Output('timeline-mini', 'figure'),
     Output('class-pie-mini', 'figure')],
    [Input('filtered-data', 'data')]
)
def update_statistics(json_data):
    """
    Met à jour les différentes statistiques basées sur les données filtrées
    """
    if not json_data:
        return "0", "0 g", "N/A", "N/A", {}, {}

    # Convertir les données JSON en DataFrame
    df = pd.read_json(json_data, orient='split')

    if df.empty:
        return "0", "0 g", "N/A", "N/A", {}, {}

    # Nombre de météorites
    count = len(df)

    # Masse totale
    total_mass = df['mass (g)'].sum()
    if total_mass >= 1e9:
        mass_str = f"{total_mass/1e9:.1f}T"
    elif total_mass >= 1e6:
        mass_str = f"{total_mass/1e6:.1f}M"
    elif total_mass >= 1e3:
        mass_str = f"{total_mass/1e3:.1f}k"
    else:
        mass_str = f"{total_mass:.1f}"

    # Période couverte
    min_year = df['year'].min()
    max_year = df['year'].max()
    years_str = f"{min_year}-{max_year}"

    # Classes principales
    top_classes = df['class'].value_counts().head(3).index.tolist()
    classes_str = ", ".join(top_classes)

    # Timeline mini figure
    timeline_fig = px.histogram(
        df,
        x='year',
        nbins=30,
        color_discrete_sequence=[COLOR_SCHEMES['primary']]
    )
    timeline_fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    # Class pie mini figure
    class_counts = df['class'].value_counts().head(5)
    class_fig = px.pie(
        names=class_counts.index,
        values=class_counts.values,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    class_fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    return str(count), mass_str, years_str, classes_str, timeline_fig, class_fig

# Fonction utilitaire pour créer une carte vide
def create_empty_map(map_style):
    """
    Crée une carte vide avec un message
    """
    fig = go.Figure()

    fig.update_layout(
        mapbox_style=get_mapbox_style(map_style),
        mapbox=dict(
            center=dict(lat=0, lon=0),
            zoom=1
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=650,
        annotations=[
            dict(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="Aucune météorite ne correspond aux critères de filtrage",
                showarrow=False,
                font=dict(size=16)
            )
        ]
    )

    return fig