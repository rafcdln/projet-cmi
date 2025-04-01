from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from utils.config import DEFAULT_MAP_STYLE

def create_layout():
    return html.Div([
        # Section Entête avec style Apple
        html.Div([
            html.Div([
                html.H1('Analyse des Météorites', 
                    className='mb-0', 
                    style={
                        'fontWeight': '600',
                        'fontSize': '48px',
                        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                        'letterSpacing': '-0.5px',
                        'color': '#1d1d1f'
                    }
                ),
                html.P('Exploration interactive des données de météorites terrestres', 
                    className='text-muted', 
                    style={
                        'fontWeight': '400',
                        'fontSize': '21px',
                        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                        'color': '#86868b'
                    }
                ),
            ], className='ps-3 py-4')
        ], className='mb-4 bg-white shadow-sm rounded-3'),
        
        # Corps principal
        html.Div([
            # Colonne gauche - Filtres
            html.Div([
                html.Div([
                    html.H5('Filtres', 
                        className='card-header py-3', 
                        style={
                            'fontWeight': '500',
                            'fontSize': '24px',
                            'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                            'color': '#1d1d1f',
                            'borderBottom': '1px solid #f5f5f7'
                        }
                    ),
                    html.Div([
                        # Filtre de masse
                        html.Div([
                            html.Label([
                                'Plage de masse (g)',
                                html.I(className='fas fa-info-circle ms-2', id='mass-slider-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.RangeSlider(
                                id='mass-slider',
                                min=0,
                                max=6,
                                step=0.1,
                                marks={i: f'10^{i}' for i in range(7)},
                                value=[0, 6],
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], className='mb-4'),
                        
                        # Mode de couleur
                        html.Div([
                            html.Label([
                                'Mode de couleur',
                                html.I(className='fas fa-info-circle ms-2', id='color-mode-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.Dropdown(
                                id='color-mode-dropdown',
                                options=[
                                    {'label': 'Classe', 'value': 'class'},
                                    {'label': 'Masse', 'value': 'mass'},
                                    {'label': 'Type de chute', 'value': 'fall'},
                                    {'label': 'Année', 'value': 'year'}
                                ],
                                value='class',
                                clearable=False,
                                searchable=False,
                                className='modern-dropdown'
                            ),
                        ], className='mb-4'),
                        
                        # Sélection de classe
                        html.Div([
                            html.Label([
                                'Classe de météorite',
                                html.I(className='fas fa-info-circle ms-2', id='class-dropdown-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.Dropdown(
                                id='class-dropdown',
                                options=[
                                    {'label': 'Toutes les classes', 'value': 'all'},
                                    {'label': 'H5', 'value': 'H5'},
                                    {'label': 'L6', 'value': 'L6'},
                                    {'label': 'LL6', 'value': 'LL6'},
                                    {'label': 'H4', 'value': 'H4'},
                                    {'label': 'L5', 'value': 'L5'},
                                    {'label': 'LL5', 'value': 'LL5'},
                                    {'label': 'H6', 'value': 'H6'},
                                    {'label': 'L4', 'value': 'L4'},
                                    {'label': 'LL4', 'value': 'LL4'}
                                ],
                                value='all',
                                multi=True,
                                searchable=True,
                                className='modern-dropdown'
                            ),
                        ], className='mb-4'),
                        
                        # Filtre de chute
                        html.Div([
                            html.Label([
                                'Type de chute',
                                html.I(className='fas fa-info-circle ms-2', id='fall-checklist-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.Checklist(
                                id='fall-checklist',
                                options=[
                                    {'label': ' Trouvé', 'value': 'Found'},
                                    {'label': ' Observé', 'value': 'Fell'}
                                ],
                                value=['Found', 'Fell'],
                                labelStyle={'display': 'block', 'marginBottom': '5px'},
                                className='ps-2'
                            ),
                        ], className='mb-4'),
                        
                        # Filtre de décennie
                        html.Div([
                            html.Label([
                                'Période de découverte',
                                html.I(className='fas fa-info-circle ms-2', id='decade-slider-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.RangeSlider(
                                id='decade-slider',
                                min=1800,
                                max=2020,
                                step=10,
                                marks={i: str(i) for i in range(1800, 2021, 40)},
                                value=[1800, 2020],
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], className='mb-4'),
                        
                        # Style de carte
                        html.Div([
                            html.Label([
                                'Style de carte',
                                html.I(className='fas fa-info-circle ms-2', id='map-style-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.Dropdown(
                                id='map-style-dropdown',
                                options=[
                                    {'label': 'Clair', 'value': 'clair'},
                                    {'label': 'Sombre', 'value': 'sombre'},
                                    {'label': 'Standard', 'value': 'standard'}
                                ],
                                value=DEFAULT_MAP_STYLE,
                                clearable=False,
                                searchable=False,
                                className='modern-dropdown'
                            ),
                        ], className='mb-4'),
                        
                        # Paramètres avancés de la carte
                        html.Div([
                            html.Label([
                                'Paramètres avancés',
                                html.I(className='fas fa-info-circle ms-2', id='advanced-params-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            html.Div([
                                # Opacité des points
                                html.Div([
                                    html.Label([
                                        'Opacité des points',
                                        html.I(className='fas fa-info-circle ms-2', id='point-opacity-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Slider(
                                        id='point-opacity',
                                        min=0.1,
                                        max=1,
                                        step=0.1,
                                        value=0.7,
                                        marks={i/10: str(i/10) for i in range(1, 11)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], className='mb-3'),
                                
                                # Taille des points
                                html.Div([
                                    html.Label([
                                        'Taille des points',
                                        html.I(className='fas fa-info-circle ms-2', id='point-size-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Slider(
                                        id='point-size',
                                        min=1,
                                        max=10,
                                        step=0.5,
                                        value=5,
                                        marks={i: str(i) for i in range(1, 11)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], className='mb-3'),
                                
                                # Rayon de la carte de chaleur
                                html.Div([
                                    html.Label([
                                        'Rayon de la carte de chaleur',
                                        html.I(className='fas fa-info-circle ms-2', id='heatmap-radius-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Slider(
                                        id='heatmap-radius',
                                        min=5,
                                        max=30,
                                        step=1,
                                        value=15,
                                        marks={i: str(i) for i in range(5, 31, 5)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], className='mb-3'),
                                
                                # Échelle de couleur de la carte de chaleur
                                html.Div([
                                    html.Label([
                                        'Échelle de couleur',
                                        html.I(className='fas fa-info-circle ms-2', id='heatmap-colorscale-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Dropdown(
                                        id='heatmap-colorscale',
                                        options=[
                                            {'label': 'Inferno', 'value': 'Inferno'},
                                            {'label': 'Plasma', 'value': 'Plasma'},
                                            {'label': 'Viridis', 'value': 'Viridis'},
                                            {'label': 'Magma', 'value': 'Magma'},
                                            {'label': 'RdBu', 'value': 'RdBu'},
                                            {'label': 'YlOrRd', 'value': 'YlOrRd'}
                                        ],
                                        value='Inferno',
                                        clearable=False,
                                        searchable=False,
                                        className='modern-dropdown'
                                    ),
                                ], className='mb-3'),
                                
                                # Interactivité entre les cartes
                                html.Div([
                                    html.Label([
                                        'Interactivité',
                                        html.I(className='fas fa-info-circle ms-2', id='map-interactivity-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Checklist(
                                        id='map-interactivity',
                                        options=[
                                            {'label': 'Synchroniser les vues', 'value': 'sync_views'},
                                            {'label': 'Synchroniser les sélections', 'value': 'sync_selections'},
                                            {'label': 'Afficher les points sélectionnés', 'value': 'show_selected'}
                                        ],
                                        value=['sync_views'],
                                        className='ps-2'
                                    ),
                                ], className='mb-3'),
                            ], className='p-3 bg-light rounded')
                        ], className='mb-4'),
                        
                        # Statistics
                        html.Div([
                            html.Div([
                                html.Div(id='stats-output')
                            ], className='rounded p-3 mb-3')
                        ], className='mb-3'),
                        
                    ], className='card-body', style={'position': 'sticky', 'top': '0'})
                ], className='card h-100 shadow-sm')
            ], className='col-md-3'),
            
            # Colonne droite - Graphiques
            html.Div([
                # Carte du monde
                html.Div([
                    html.Div([
                        html.H5('Distribution Mondiale des Météorites', 
                            className='card-header py-3', 
                            style={
                                'fontWeight': '500',
                                'fontSize': '24px',
                                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                                'color': '#1d1d1f',
                                'borderBottom': '1px solid #f5f5f7'
                            }
                        ),
                        html.Div([
                            # Remplacer les tabs par des boutons
                            html.Div([
                                html.Button(
                                    "Distribution",
                                    id='btn-distribution-map',
                                    className="btn btn-outline-primary me-2"
                                ),
                                html.Button(
                                    "Carte de Chaleur",
                                    id='btn-heatmap',
                                    className="btn btn-outline-primary"
                                ),
                            ], className="d-flex mb-3"),
                            
                            # Conteneurs pour les cartes
                            html.Div([
                                dcc.Loading(
                                    id="loading-map",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='world-map',
                                        config={
                                            'displayModeBar': True, 
                                            'scrollZoom': True,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                            'doubleClick': 'reset'
                                        },
                                        style={'height': '650px'},
                                        responsive=True,
                                        loading_state={'is_loading': False, 'component_name': 'world-map'},
                                        clear_on_unhover=True
                                    )
                                )
                            ], id='distribution-map-content', style={'display': 'block'}),
                            
                            html.Div([
                                dcc.Loading(
                                    id="loading-heatmap",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='heatmap',
                                        config={
                                            'displayModeBar': True,
                                            'scrollZoom': True,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                            'doubleClick': 'reset'
                                        },
                                        style={'height': '650px'},
                                        responsive=True,
                                        loading_state={'is_loading': False, 'component_name': 'heatmap'},
                                        clear_on_unhover=True
                                    )
                                )
                            ], id='heatmap-content', style={'display': 'none'})
                        ], className='card-body')
                    ], className='card h-100 shadow-sm')
                ], className='mb-4'),
                
                # Graphiques d'analyse
                html.Div([
                    html.Div([
                        html.H5('Analyse Statistique', 
                            className='card-header py-3', 
                            style={
                                'fontWeight': '500',
                                'fontSize': '24px',
                                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                                'color': '#1d1d1f',
                                'borderBottom': '1px solid #f5f5f7'
                            }
                        ),
                        html.Div([
                            # Boutons de navigation pour les analyses
                            html.Div([
                                html.Div([
                                    html.Button(
                                        [html.I(className="fas fa-chart-bar me-2"), "Distributions"],
                                        id="btn-distributions",
                                        className="btn btn-outline-primary me-2 mb-2"
                                    ),
                                    html.Button(
                                        [html.I(className="fas fa-chart-line me-2"), "Séries Chronologiques"],
                                        id="btn-time-series",
                                        className="btn btn-outline-primary me-2 mb-2"
                                    ),
                                    html.Button(
                                        [html.I(className="fas fa-project-diagram me-2"), "Corrélations"],
                                        id="btn-correlations",
                                        className="btn btn-outline-primary mb-2"
                                    ),
                                ], className="d-flex flex-wrap mb-3"),
                                
                                # Section Distributions
                                html.Div([
                                    html.Div([
                                        html.H6([
                                            'Distribution des masses', 
                                            html.I(className='fas fa-info-circle ms-2', id='mass-hist-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-mass-hist",
                                            type="circle",
                                            children=dcc.Graph(
                                                id='mass-hist', 
                                                config={'displayModeBar': False},
                                                # Éviter des mises à jour inutiles pendant le zoom/déplacement
                                                loading_state={'is_loading': False, 'component_name': 'mass-hist'}
                                            )
                                        )
                                    ], className="col-md-6"),
                                    html.Div([
                                        html.H6([
                                            'Distribution des classes', 
                                            html.I(className='fas fa-info-circle ms-2', id='class-distribution-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-class-dist",
                                            type="circle",
                                            children=dcc.Graph(
                                                id='class-distribution', 
                                                config={'displayModeBar': False},
                                                # Éviter des mises à jour inutiles pendant le zoom/déplacement
                                                loading_state={'is_loading': False, 'component_name': 'class-distribution'}
                                            )
                                        )
                                    ], className="col-md-6"),
                                    # Space for two more distributions that can be added later
                                    html.Div([
                                        html.H6([
                                            'Distribution géographique', 
                                            html.I(className='fas fa-info-circle ms-2', id='geo-distribution-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-geo-dist",
                                            type="circle",
                                            children=dcc.Graph(id='geo-distribution', figure=px.density_mapbox(
                                                pd.DataFrame({'lat': [0], 'lon': [0]}),
                                                lat='lat', lon='lon', zoom=1,
                                                mapbox_style="carto-positron"
                                            ).update_layout(margin={"r":0,"t":0,"l":0,"b":0}), 
                                            config={'displayModeBar': False})
                                        )
                                    ], className="col-md-6"),
                                    html.Div([
                                        html.H6([
                                            'Distribution des années', 
                                            html.I(className='fas fa-info-circle ms-2', id='year-distribution-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-year-dist",
                                            type="circle",
                                            children=dcc.Graph(id='year-distribution', figure=px.histogram(
                                                pd.DataFrame({'year': np.random.normal(1950, 30, 500)}),
                                                x='year', nbins=50,
                                                height=300  # Réduire la hauteur
                                            ).update_layout(margin={"r":20,"t":20,"l":20,"b":20}),
                                            config={'displayModeBar': False})
                                        )
                                    ], className="col-md-6"),
                                ], id="panel-distributions", className="row"),
                                
                                # Section Séries Chronologiques
                                html.Div([
                                    html.Div([
                                        html.H6([
                                            'Évolution temporelle des découvertes', 
                                            html.I(className='fas fa-info-circle ms-2', id='time-series-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-time-series",
                                            type="circle",
                                            children=dcc.Graph(
                                                id='time-series', 
                                                config={'displayModeBar': False},
                                                # Éviter des mises à jour inutiles pendant le zoom/déplacement
                                                loading_state={'is_loading': False, 'component_name': 'time-series'}
                                            )
                                        )
                                    ], className="col-md-6"),
                                    html.Div([
                                        html.H6([
                                            'Tendance annuelle', 
                                            html.I(className='fas fa-info-circle ms-2', id='annual-trend-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-annual-trend",
                                            type="circle",
                                            children=dcc.Graph(id='annual-trend', figure=px.line(
                                                pd.DataFrame({
                                                    'month': range(1, 13),
                                                    'count': np.random.randint(10, 100, 12)
                                                }),
                                                x='month', y='count', markers=True,
                                                labels={'count': 'Nombre de météorites', 'month': 'Mois'}
                                            ).update_layout(margin={"r":20,"t":20,"l":20,"b":20}),
                                            config={'displayModeBar': False})
                                        )
                                    ], className="col-md-6"),
                                    html.Div([
                                        html.H6([
                                            'Évolution des masses au fil du temps', 
                                            html.I(className='fas fa-info-circle ms-2', id='mass-time-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-mass-time",
                                            type="circle",
                                            children=dcc.Graph(id='mass-time', figure=px.scatter(
                                                pd.DataFrame({
                                                    'year': np.random.normal(1950, 30, 200),
                                                    'mass': np.random.exponential(5000, 200)
                                                }),
                                                x='year', y='mass', opacity=0.7,
                                                labels={'mass': 'Masse (g)', 'year': 'Année'}
                                            ).update_layout(margin={"r":20,"t":20,"l":20,"b":20}),
                                            config={'displayModeBar': False})
                                        )
                                    ], className="col-md-6"),
                                    html.Div([
                                        html.H6([
                                            'Prévisions', 
                                            html.I(className='fas fa-info-circle ms-2', id='forecast-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-forecast",
                                            type="circle",
                                            children=dcc.Graph(id='forecast', figure=px.line(
                                                pd.DataFrame({
                                                    'year': range(1990, 2030),
                                                    'count': [np.random.randint(10, 100) for _ in range(1990, 2020)] + [None] * 10,
                                                    'forecast': [None] * 30 + [np.random.randint(50, 150) for _ in range(2020, 2030)]
                                                }),
                                                x='year', y=['count', 'forecast'],
                                                labels={'value': 'Nombre de météorites', 'year': 'Année'}
                                            ).update_layout(margin={"r":20,"t":20,"l":20,"b":20}),
                                            config={'displayModeBar': False})
                                        )
                                    ], className="col-md-6"),
                                ], id="panel-time-series", className="row d-none"),
                                
                                # Section Corrélations
                                html.Div([
                                    html.Div([
                                        html.H6([
                                            'Matrice de corrélation', 
                                            html.I(className='fas fa-info-circle ms-2', id='correlation-heatmap-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-corr-heatmap",
                                            type="circle",
                                            children=dcc.Graph(id='correlation-heatmap', config={'displayModeBar': False})
                                        )
                                    ], className="col-md-6"),
                                    html.Div([
                                        html.H6([
                                            'Importance des variables', 
                                            html.I(className='fas fa-info-circle ms-2', id='feature-importance-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-feature-importance",
                                            type="circle",
                                            children=dcc.Graph(id='feature-importance', config={'displayModeBar': False})
                                        )
                                    ], className="col-md-6"),
                                    html.Div([
                                        html.H6([
                                            'Nuage de points interactif', 
                                            html.I(className='fas fa-info-circle ms-2', id='scatter-matrix-info', style={'cursor': 'pointer'})
                                        ], className='mt-3 mb-2'),
                                        dcc.Loading(
                                            id="loading-scatter-matrix",
                                            type="circle",
                                            children=dcc.Graph(id='scatter-matrix', figure=px.scatter_matrix(
                                                pd.DataFrame({
                                                    'mass': np.random.exponential(5000, 500),
                                                    'year': np.random.normal(1950, 30, 500),
                                                    'lat': np.random.normal(0, 45, 500),
                                                    'lon': np.random.normal(0, 90, 500)
                                                }),
                                                dimensions=['mass', 'year', 'lat', 'lon'],
                                                opacity=0.5
                                            ).update_layout(height=500, margin={"r":20,"t":20,"l":20,"b":20}),
                                            config={'displayModeBar': False})
                                        )
                                    ], className="col-12 mt-3"),
                                ], id="panel-correlations", className="row d-none"),
                            ])
                        ], className='card-body')
                    ], className='card h-100 shadow-sm border-0')
                ], className='mb-4'),
                
                # Section de prédiction
                html.Div([
                    html.Div([
                        html.H5('Prédiction de Météorites', 
                            className='card-header py-3', 
                            style={
                                'fontWeight': '500',
                                'fontSize': '24px',
                                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                                'color': '#1d1d1f',
                                'borderBottom': '1px solid #f5f5f7'
                            }
                        ),
                        html.Div([
                            # Premier panneau - Carte de prédiction
                            html.Div([
                                html.H6([
                                    html.I(className="fas fa-map-marked-alt me-2"), 
                                    "Carte de Sélection"
                                ], className="mt-1 mb-3"),
                                html.P([
                                    html.I(className="fas fa-info-circle me-2"), 
                                    "Les points colorés représentent les météorites historiques (classées par type). Cliquez sur la carte pour sélectionner un point d'intérêt ou une zone."
                                ], className="mb-2 text-muted"),
                                dcc.Graph(
                                    id='prediction-map',
                                    config={
                                        'displayModeBar': True, 
                                        'scrollZoom': True,
                                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                                        'modeBarButtonsToRemove': ['autoScale2d'],
                                        'toImageButtonOptions': {
                                            'format': 'png',
                                            'filename': 'prediction_map',
                                            'height': 800,
                                            'width': 1200,
                                            'scale': 2
                                        }
                                    },
                                    style={
                                        'height': '650px',
                                        'border': '1px solid #e2e2e2',
                                        'border-radius': '5px'
                                    },
                                    className='shadow-sm'
                                ),
                                html.Div([
                                    html.Div([
                                        html.H6("Coordonnées sélectionnées", className="mb-2"),
                                        html.P(id='selected-coordinates', className='mb-3 p-2 bg-light rounded border')
                                    ], className="col-md-6"),
                                    
                                    # Ajouter les inputs de prédiction dans une colonne
                                    html.Div([
                                        html.H6("Paramètres de prédiction", className="mb-2"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Type de chute", className="form-label"),
                                                dcc.Dropdown(
                                                    id='pred-fall',
                                                    options=[
                                                        {'label': 'Trouvée', 'value': 'Found'},
                                                        {'label': 'Tombée', 'value': 'Fell'}
                                                    ],
                                                    value='Found',
                                                    clearable=False,
                                                    className="form-select"
                                                ),
                                                html.P("Utilisez le paramètre Horizon de prévision ci-dessous pour définir la période de prédiction", 
                                                      className="text-muted small mt-2")
                                            ], className="mb-2"),
                                        ], className="d-flex flex-wrap mb-3"),
                                        
                                        # Boutons d'action déplacés ici pour être plus proches des paramètres
                                        html.Div([
                                            html.Button([
                                                html.I(className="fas fa-magic me-2"),
                                                "Prédire"
                                            ], id='predict-button', n_clicks=0, 
                                            className='btn btn-success me-2'),
                                            html.Button([
                                                html.I(className="fas fa-search-location me-2"),
                                                "Analyser la zone"
                                            ], id='analyze-zone-button', n_clicks=0, 
                                            className='btn btn-primary me-2'),
                                        ], className='d-flex'),
                                    ], className="col-md-6"),
                                ], className='row mt-3'),
                                
                                # Indicateur de fiabilité
                                html.Div([
                                    html.Div([
                                        html.P([
                                            html.I(className="fas fa-info-circle me-2 text-primary"),
                                            html.Span("Indice de fiabilité: ", className="fw-bold"),
                                            html.Span(id='reliability-index', className="badge bg-secondary")
                                        ], className="mt-2 mb-0")
                                    ], className="col-12"),
                                ], className="row mt-1"),
                            ], className='col-12 mb-4'),  # Carte en pleine largeur
                            
                            # Panneau d'analyse étendu
                            html.Div([
                                html.H5([
                                    html.I(className="fas fa-chart-line me-2"),
                                    "Analyses Prédictives"
                                ], className="mb-3"),
                                
                                # Onglets pour naviguer entre les différentes analyses
                                html.Div([
                                    html.Div([
                                        html.Button(
                                            [html.I(className="fas fa-chart-bar me-2"), "Prédiction Simple"],
                                            id='btn-prediction-results',
                                            className="btn btn-outline-primary me-2"
                                        ),
                                        html.Button(
                                            [html.I(className="fas fa-search me-2"), "Analyse de Zone"],
                                            id='btn-zone-analysis',
                                            className="btn btn-outline-primary me-2"
                                        ),
                                        html.Button(
                                            [html.I(className="fas fa-calendar-alt me-2"), "Prévision Temporelle"],
                                            id='btn-temporal-prediction',
                                            className="btn btn-outline-primary me-2"
                                        ),
                                        html.Button(
                                            [html.I(className="fas fa-globe-americas me-2"), "Probabilité Spatiale"],
                                            id='btn-spatial-prediction',
                                            className="btn btn-outline-primary"
                                        ),
                                    ], className="d-flex flex-wrap mb-3"),
                                    
                                    # Conteneurs pour les résultats
                                    html.Div([
                                        dcc.Loading(
                                            id="loading-prediction",
                                            type="circle",
                                            children=html.Div(id='prediction-output', className='p-3')
                                        )
                                    ], id='prediction-results-content', style={'min-height': '350px', 'overflow': 'auto', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}),
                                    
                                    html.Div([
                                        dcc.Loading(
                                            id="loading-zone",
                                            type="circle",
                                            children=html.Div(id='zone-analysis-output', className='p-3')
                                        )
                                    ], id='zone-analysis-content', style={'min-height': '350px', 'overflow': 'auto', 'display': 'none', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}),
                                    
                                    html.Div([
                                        dcc.Loading(
                                            id="loading-temporal",
                                            type="circle",
                                            children=html.Div([
                                                html.H5("Prévisions temporelles", className="mb-3"),
                                                html.P("Cette visualisation montre la probabilité d'impact de météorites dans le temps pour l'emplacement sélectionné."),
                                                html.Div(id='temporal-prediction-chart', className='mt-3'),
                                                html.Div(id='temporal-prediction-output', className='mt-3')
                                            ], className='p-3')
                                        )
                                    ], id='temporal-prediction-content', style={'min-height': '350px', 'overflow': 'auto', 'display': 'none', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}),
                                    
                                    html.Div([
                                        dcc.Loading(
                                            id="loading-spatial",
                                            type="circle",
                                            children=html.Div([
                                                html.H5("Carte de probabilité spatiale", className="mb-3"),
                                                html.P("Cette carte montre la densité de probabilité d'impact de météorites dans la région environnante."),
                                                html.Div(id='spatial-heatmap', className='mt-3'),
                                                html.Div(id='spatial-prediction-output', className='mt-3')
                                            ], className='p-3')
                                        )
                                    ], id='spatial-prediction-content', style={'min-height': '350px', 'overflow': 'auto', 'display': 'none', 'border': '1px solid #f0f0f0', 'border-radius': '5px'}),
                                ]),
                                
                                # Paramètres de prédiction avancés
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-sliders-h me-2"),
                                        "Paramètres Avancés"
                                    ], className="mt-4 mb-3"),
                                    
                                    html.Div([
                                        # Colonne 1: Paramètres temporels
                                        html.Div([
                                            html.Label("Horizon de prévision", className="form-label"),
                                            html.Div([
                                                dcc.Slider(
                                                    id='forecast-horizon',
                                                    min=1,
                                                    max=50,
                                                    step=1,
                                                    value=10,
                                                    marks={1: '1 an', 10: '10 ans', 25: '25 ans', 50: '50 ans'},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                )
                                            ], className="mb-3"),
                                            
                                            html.Label("Plage de masses à prévoir", className="form-label"),
                                            html.Div([
                                                dcc.RangeSlider(
                                                    id='mass-prediction-range',
                                                    min=0,
                                                    max=6,
                                                    step=0.5,
                                                    value=[1, 4],
                                                    marks={
                                                        0: '1g', 
                                                        1: '10g', 
                                                        2: '100g', 
                                                        3: '1kg', 
                                                        4: '10kg',
                                                        5: '100kg',
                                                        6: '1T+'
                                                    },
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                )
                                            ], className="mb-3"),
                                        ], className="col-md-6"),
                                        
                                        # Colonne 2: Paramètres spatiaux
                                        html.Div([
                                            html.Label("Rayon d'analyse (degrés)", className="form-label"),
                                            html.Div([
                                                dcc.Slider(
                                                    id='analysis-radius',
                                                    min=0.5,
                                                    max=10,
                                                    step=0.5,
                                                    value=2.5,
                                                    marks={0.5: '0.5°', 2.5: '2.5°', 5: '5°', 10: '10°'},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                )
                                            ], className="mb-3"),
                                            
                                            html.Label("Sensibilité de détection", className="form-label"),
                                            html.Div([
                                                dcc.Slider(
                                                    id='detection-sensitivity',
                                                    min=0,
                                                    max=100,
                                                    step=5,
                                                    value=50,
                                                    marks={0: 'Min', 50: 'Mod', 100: 'Max'},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                )
                                            ], className="mb-3"),
                                        ], className="col-md-6"),
                                    ], className="row"),
                                ]),
                            ], className='col-12'),  # Panneau d'analyse en pleine largeur
                        ], className='row card-body'),
                    ], className='card h-100 shadow-sm')
                ], className='mb-4'),
                
            ], className='col-md-9')
        ], className='row g-4'),
        
        # Footer
        html.Footer([
            html.Div([
                html.P([
                    'Dashboard Météorites © 2023 ',
                    html.Button([
                        html.I(className="fas fa-bug me-1"),
                        "Debug"
                    ], id="toggle-debug", className="btn btn-sm btn-outline-secondary ms-2", 
                       style={"fontSize": "0.8rem", "padding": "2px 5px"})
                ], className='mb-0')
            ], className='text-center py-3')
        ], className='mt-4 bg-light rounded-bottom'),
        
        # Ajout des stores pour le fonctionnement de l'application
        html.Div([
            dcc.Interval(
                id='interval',
                interval=1*1000,  # en millisecondes, déclenché une fois au début
                n_intervals=0,
                max_intervals=1
            ),
            dcc.Store(id='last-relayout-data', storage_type='memory'),
            dcc.Store(id='selected-location', storage_type='memory'),
            dcc.Store(id='selected-zone', storage_type='memory'),
            dcc.Store(id='last-update-time', storage_type='memory', data=0),
            dcc.Store(id='debug-data', storage_type='memory', data='')  # Pour stocker les messages de debug
        ], style={'display': 'none'}),
        
        # Conteneur pour les messages de debug (caché par défaut, peut être affiché en mode développement)
        html.Div([
            html.Div(id='debug-output', className='p-2 bg-dark text-light small')
        ], id='debug-container', style={'display': 'none', 'position': 'fixed', 'bottom': '0', 'left': '0', 'right': '0', 'max-height': '200px', 'overflow': 'auto', 'z-index': '1000'}),
        
        # Tooltips pour tous les paramètres
        dbc.Tooltip(
            "Filtrez les météorites par leur masse en grammes. L'échelle est logarithmique (base 10).",
            target='mass-slider-info'
        ),
        dbc.Tooltip(
            "Choisissez le mode de coloration des points sur la carte : par classe, masse, type de chute ou année.",
            target='color-mode-info'
        ),
        dbc.Tooltip(
            "Sélectionnez une ou plusieurs classes de météorites. Les classes sont basées sur leur composition chimique.",
            target='class-dropdown-info'
        ),
        dbc.Tooltip(
            "Filtrez par type de chute : 'Trouvé' pour les météorites découvertes après leur chute, 'Observé' pour celles vues tomber.",
            target='fall-checklist-info'
        ),
        dbc.Tooltip(
            "Sélectionnez la période de découverte des météorites. Les données couvrent de 1800 à 2020.",
            target='decade-slider-info'
        ),
        dbc.Tooltip(
            "Choisissez le style de fond de carte pour une meilleure visualisation des données.",
            target='map-style-info'
        ),
        dbc.Tooltip(
            "Ajustez les paramètres avancés de visualisation pour personnaliser l'apparence des cartes.",
            target='advanced-params-info'
        ),
        dbc.Tooltip(
            "Contrôlez la transparence des points sur la carte. Une opacité plus faible permet de mieux voir les points superposés.",
            target='point-opacity-info'
        ),
        dbc.Tooltip(
            "Ajustez la taille maximale des points sur la carte. La taille réelle est proportionnelle à la masse de la météorite.",
            target='point-size-info'
        ),
        dbc.Tooltip(
            "Définissez le rayon d'influence de chaque point sur la carte de chaleur. Un rayon plus grand crée des zones plus diffuses.",
            target='heatmap-radius-info'
        ),
        dbc.Tooltip(
            "Choisissez l'échelle de couleur pour la carte de chaleur. Chaque échelle met en évidence différents aspects des données.",
            target='heatmap-colorscale-info'
        ),
        dbc.Tooltip(
            "Activez différentes options d'interactivité entre les cartes pour une meilleure exploration des données.",
            target='map-interactivity-info'
        ),
        dbc.Tooltip(
            "Ce graphique montre la distribution des masses des météorites sur une échelle logarithmique.",
            target='mass-hist-info'
        ),
        dbc.Tooltip(
            "Ce graphique montre l'évolution des découvertes de météorites au fil du temps.",
            target='time-series-info'
        ),
        dbc.Tooltip(
            "Ce graphique montre la répartition des différentes classes de météorites dans le jeu de données.",
            target='class-distribution-info'
        ),
        dbc.Tooltip(
            "Cette matrice de corrélation montre les relations entre les différentes variables numériques du jeu de données.",
            target='correlation-heatmap-info'
        ),
        dbc.Tooltip(
            "Ce graphique montre la distribution géographique des météorites à travers le monde.",
            target='geo-distribution-info'
        ),
        dbc.Tooltip(
            "Ce graphique montre la distribution des météorites par année de découverte.",
            target='year-distribution-info'
        ),
        dbc.Tooltip(
            "Ce graphique montre la tendance annuelle des découvertes de météorites par mois.",
            target='annual-trend-info'
        ),
        dbc.Tooltip(
            "Ce graphique montre l'évolution des masses des météorites au fil des années.",
            target='mass-time-info'
        ),
        dbc.Tooltip(
            "Ce graphique présente des prévisions sur le nombre de météorites qui pourraient être découvertes dans les années à venir.",
            target='forecast-info'
        ),
        dbc.Tooltip(
            "Ce graphique montre l'importance relative des différentes variables pour la classification et la prédiction.",
            target='feature-importance-info'
        ),
        dbc.Tooltip(
            "Cette matrice de nuages de points permet d'explorer les relations entre plusieurs variables simultanément.",
            target='scatter-matrix-info'
        )
    ], className='container-fluid py-4', style={'backgroundColor': '#f5f5f7'})

def create_prediction_tab():
    """
    Crée l'onglet de prédiction
    """
    return html.Div([
        # Titre principal avec badge de version
        html.Div([
            html.H2("Prédiction de météorites", className="tab-title"),
            html.Span("Beta", className="version-badge")
        ], className="title-container"),

        # Section des contrôles de prédiction (carte et paramètres)
        html.Div([
            # Colonne gauche: carte de prédiction
            html.Div([
                html.H3("Carte de prédiction"),
                dcc.Graph(
                    id='prediction-map',
                    config={
                        'displayModeBar': True,
                        'scrollZoom': True,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                    },
                    className="prediction-map"
                ),
                
                # Bouton de mise à jour en temps réel
                html.Div([
                    dbc.Checkbox(id="realtime-updates", value=False, className="realtime-checkbox"),
                    html.Label("Mise à jour en temps réel", htmlFor="realtime-updates", className="realtime-label"),
                    html.Span(
                        html.I(className="fas fa-info-circle"),
                        id="tooltip-realtime",
                        className="info-icon"
                    ),
                    dbc.Tooltip(
                        "Lorsque cette option est activée, les graphiques se mettent à jour automatiquement à chaque modification des paramètres, sans cliquer sur le bouton Calculer.",
                        target="tooltip-realtime",
                        placement="top"
                    )
                ], className="realtime-control")
            ], className="map-container"),
            
            # Colonne droite: paramètres et contrôles
            html.Div([
                # Section des informations de localisation
                html.Div([
                    html.H3("Localisation et paramètres"),
                    
                    html.Div([
                        html.Div([
                            html.Label("Latitude"),
                            html.Div(id='selected-lat', className="coordinate-display")
                        ], className="coordinate-container"),
                        
                        html.Div([
                            html.Label("Longitude"),
                            html.Div(id='selected-lon', className="coordinate-display")
                        ], className="coordinate-container")
                    ], className="coordinates-row"),
                    
                    html.Hr(),
                
                    # Tous les paramètres regroupés en haut
                    html.Div([
                        # Paramètres de base dans une rangée
                        html.Div([
                            # Rayon d'analyse
                            html.Div([
                                html.Label([
                                    "Rayon d'analyse (°) ",
                                    html.Span(
                                        html.I(className="fas fa-info-circle"),
                                        id="tooltip-radius",
                                        className="info-icon"
                                    )
                                ]),
                                dbc.Tooltip(
                                    "Définit la zone circulaire autour du point sélectionné qui sera analysée pour la prédiction. Un rayon plus grand prend en compte plus de données mais peut diluer les tendances locales.",
                                    target="tooltip-radius",
                                    placement="top"
                                ),
                                dcc.Slider(
                                    id='analysis-radius',
                                    min=0.5,
                                    max=10,
                                    step=0.5,
                                    value=2.5,
                                    marks={i: f'{i}°' for i in range(1, 11, 2)},
                                    className="parameter-slider"
                                )
                            ], className="parameter-container half-width"),
                            
                            # Horizon de prévision
                            html.Div([
                                html.Label([
                                    "Horizon de prévision (années) ",
                                    html.Span(
                                        html.I(className="fas fa-info-circle"),
                                        id="tooltip-horizon",
                                        className="info-icon"
                                    )
                                ]),
                                dbc.Tooltip(
                                    "Détermine combien d'années dans le futur la prédiction sera faite. Des horizons plus longs ont généralement une fiabilité plus faible.",
                                    target="tooltip-horizon",
                                    placement="top"
                                ),
                                dcc.Slider(
                                    id='forecast-horizon',
                                    min=1,
                                    max=50,
                                    step=1,
                                    value=10,
                                    marks={i: f'{i} ans' for i in range(0, 51, 10)},
                                    className="parameter-slider"
                                )
                            ], className="parameter-container half-width")
                        ], className="parameters-row"),
                        
                        # Paramètres avancés dans une rangée
                        html.Div([
                            # Type de détection
                            html.Div([
                                html.Label([
                                    "Type de détection ",
                                    html.Span(
                                        html.I(className="fas fa-info-circle"),
                                        id="tooltip-detection",
                                        className="info-icon"
                                    )
                                ]),
                                dbc.Tooltip(
                                    "Choisissez entre la probabilité de trouver une météorite (Found) ou d'observer sa chute (Fell). Les météorites 'trouvées' ont généralement un âge inconnu.",
                                    target="tooltip-detection",
                                    placement="top"
                                ),
                                dcc.RadioItems(
                                    id='detection-type',
                                    options=[
                                        {'label': 'Météorite trouvée (Found)', 'value': 'found'},
                                        {'label': 'Chute observée (Fell)', 'value': 'fell'}
                                    ],
                                    value='found',
                                    className="detection-radio"
                                )
                            ], className="parameter-container"),
                        ], className="parameters-row"),
                        
                        # Deuxième rangée de paramètres avancés
                        html.Div([
                            # Facteur environnemental
                            html.Div([
                                html.Label([
                                    "Facteur environnemental ",
                                    html.Span(
                                        html.I(className="fas fa-info-circle"),
                                        id="tooltip-env-factor",
                                        className="info-icon"
                                    )
                                ]),
                                dbc.Tooltip(
                                    "Ajuste l'influence des facteurs environnementaux comme l'atmosphère et le climat sur la prédiction. Des valeurs plus élevées donnent plus d'importance à ces facteurs.",
                                    target="tooltip-env-factor",
                                    placement="top"
                                ),
                                dcc.Slider(
                                    id='environmental-factor',
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=0.5,
                                    marks={i/10: f'{i/10}' for i in range(0, 11, 2)},
                                    className="parameter-slider"
                                )
                            ], className="parameter-container third-width"),
                            
                            # Poids historique
                            html.Div([
                                html.Label([
                                    "Poids des données historiques ",
                                    html.Span(
                                        html.I(className="fas fa-info-circle"),
                                        id="tooltip-hist-weight",
                                        className="info-icon"
                                    )
                                ]),
                                dbc.Tooltip(
                                    "Détermine l'importance accordée aux données historiques par rapport aux tendances récentes. Une valeur plus élevée donne plus d'importance aux données anciennes.",
                                    target="tooltip-hist-weight",
                                    placement="top"
                                ),
                                dcc.Slider(
                                    id='historical-weight',
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=0.7,
                                    marks={i/10: f'{i/10}' for i in range(0, 11, 2)},
                                    className="parameter-slider"
                                )
                            ], className="parameter-container third-width"),
                            
                            # Complexité du modèle
                            html.Div([
                                html.Label([
                                    "Complexité du modèle ",
                                    html.Span(
                                        html.I(className="fas fa-info-circle"),
                                        id="tooltip-complexity",
                                        className="info-icon"
                                    )
                                ]),
                                dbc.Tooltip(
                                    "Ajuste la complexité des algorithmes de prédiction. Une complexité plus élevée peut capturer des motifs plus subtils mais risque de surajuster les données.",
                                    target="tooltip-complexity",
                                    placement="top"
                                ),
                                dcc.Slider(
                                    id='model-complexity',
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in range(1, 11, 2)},
                                    className="parameter-slider"
                                )
                            ], className="parameter-container third-width")
                        ], className="parameters-row"),
                        
                        # Bouton de calcul
                        html.Div([
                            html.Button(
                                "Calculer",
                                id='calculate-prediction',
                                className="calculate-button"
                            )
                        ], className="button-container")
                    ], className="prediction-parameters"),
                ], className="location-container"),
                
                # Section des résultats de prédiction
                html.Div([
                    html.H3("Résultats de prédiction"),
                    
                    # Indice de confiance
                    html.Div([
                        html.Label("Indice de confiance"),
                        daq.Gauge(
                            id='confidence-gauge',
                            color={"gradient":True, "ranges":{"red":[0,30],"yellow":[30,70],"green":[70,100]}},
                            value=0,
                            min=0,
                            max=100,
                            showCurrentValue=True,
                            units="%",
                            className="confidence-gauge"
                        ),
                        html.Div(id="confidence-explanation", className="confidence-explanation")
                    ], className="confidence-container"),
                    
                    html.Hr(),
                    
                    # Statistiques principales
                    html.Div([
                        html.Div([
                            html.Label("Probabilité (Found)"),
                            html.Div(id='found-probability', className="stat-value")
                        ], className="stat-container"),
                        
                        html.Div([
                            html.Label("Probabilité (Fell)"),
                            html.Div(id='fell-probability', className="stat-value")
                        ], className="stat-container"),
                        
                        html.Div([
                            html.Label("Masse estimée"),
                            html.Div(id='estimated-mass', className="stat-value")
                        ], className="stat-container"),
                        
                        html.Div([
                            html.Label("Classification probable"),
                            html.Div(id='probable-class', className="stat-value")
                        ], className="stat-container")
                    ], className="statistics-container")
                ], className="results-container")
            ], className="controls-container")
        ], className="prediction-controls-row"),
        
        # Section des graphiques de prédiction temporelle
        html.Div([
            html.H3("Analyse temporelle des prédictions"),
            
            # Prédictions temporelles
            dcc.Graph(
                id='time-prediction-chart',
                config={'displayModeBar': False},
                className="time-chart"
            )
        ], className="time-prediction-container"),
        
        # Section d'analyse de zone (réaménagée avec graphiques)
        html.Div([
            html.H3("Analyse comparative de la zone"),
            
            # Rangée de petits graphiques
            html.Div([
                # Graphique 1: Comparaison avec région
                html.Div([
                    html.H4("Comparaison régionale"),
                    dcc.Graph(id='region-comparison-chart', className="zone-chart")
                ], className="zone-chart-container"),
                
                # Graphique 2: Distribution des masses
                html.Div([
                    html.H4("Distribution des masses"),
                    dcc.Graph(id='zone-mass-distribution', className="zone-chart")
                ], className="zone-chart-container"),
                
                # Graphique 3: Types de météorites
                html.Div([
                    html.H4("Types de météorites"),
                    dcc.Graph(id='zone-type-distribution', className="zone-chart")
                ], className="zone-chart-container")
            ], className="zone-charts-row"),
            
            # Deuxième rangée de graphiques
            html.Div([
                # Graphique 4: Chronologie des impacts
                html.Div([
                    html.H4("Chronologie des impacts"),
                    dcc.Graph(id='zone-timeline', className="zone-chart")
                ], className="zone-chart-container wide"),
                
                # Graphique 5: Densité locale
                html.Div([
                    html.H4("Densité locale"),
                    dcc.Graph(id='zone-density-map', className="zone-chart")
                ], className="zone-chart-container wide")
            ], className="zone-charts-row")
        ], className="zone-analysis-container"),
        
        # Section des analyses supplémentaires
        html.Div([
            html.H3("Analyses supplémentaires"),
            
            # Onglets pour différentes analyses
            dcc.Tabs([
                # Analyse 1: Distribution des masses
                dcc.Tab(label="Distribution des masses", children=[
                    dcc.Graph(
                        id='mass-distribution-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 2: Évolution des classifications
                dcc.Tab(label="Évolution des classifications", children=[
                    dcc.Graph(
                        id='class-evolution-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 3: Carte de densité de probabilité
                dcc.Tab(label="Carte de densité", children=[
                    dcc.Graph(
                        id='density-map-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 4: Facteurs d'influence
                dcc.Tab(label="Facteurs d'influence", children=[
                    dcc.Graph(
                        id='influence-factors-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 5: Tendances saisonnières
                dcc.Tab(label="Tendances saisonnières", children=[
                    dcc.Graph(
                        id='seasonal-trends-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 6: Comparaison historique
                dcc.Tab(label="Comparaison historique", children=[
                    dcc.Graph(
                        id='historical-comparison-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 7: Scénarios d'impact
                dcc.Tab(label="Scénarios d'impact", children=[
                    dcc.Graph(
                        id='impact-scenarios-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 8: Analyse de composition
                dcc.Tab(label="Analyse de composition", children=[
                    dcc.Graph(
                        id='composition-analysis-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 9: Trajectoires orbitales
                dcc.Tab(label="Trajectoires orbitales", children=[
                    dcc.Graph(
                        id='orbital-trajectories-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ]),
                
                # Analyse 10: Évaluation des risques
                dcc.Tab(label="Évaluation des risques", children=[
                    dcc.Graph(
                        id='risk-assessment-chart',
                        config={'displayModeBar': False},
                        className="analysis-chart"
                    )
                ])
            ], className="analysis-tabs")
        ], className="additional-analyses-container"),
        
        # Store pour les données de localisation sélectionnée
        dcc.Store(id='selected-location')
    ], className="prediction-tab")