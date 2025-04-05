import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.config import PREDICTION_MAP_HEIGHT, DEFAULT_MAP_STYLE, COLOR_SCHEMES, ZONE_RADIUS_DEGREES

dash.register_page(__name__, path='/predictions', title='Prédictions - Dashboard Météorites')

def layout(**kwargs):
    return dbc.Container([
        # En-tête
        dbc.Row([
            dbc.Col([
                html.H1('Prédictions de Météorites',
                    className='mb-0',
                    style={
                        'fontWeight': '600',
                        'fontSize': '42px',
                        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                        'letterSpacing': '-0.5px',
                        'color': '#1d1d1f'
                    }
                ),
                html.P('Analysez et prédisez les futures chutes de météorites par zone géographique',
                    className='text-muted',
                    style={
                        'fontWeight': '400',
                        'fontSize': '21px',
                        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                        'color': '#86868b',
                        'marginBottom': '20px'
                    }
                ),
            ], width=12, className='mb-4 ps-3 py-4')
        ], className='bg-white shadow-sm rounded-3'),

        # Carte de prédiction et panneau de contrôle
        dbc.Row([
            # Colonne gauche - Carte et instructions
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5('Carte de Prédiction',
                            className='py-2',
                            style={
                                'fontWeight': '500',
                                'fontSize': '24px',
                                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                                'color': '#1d1d1f'
                            }
                        )
                    ]),
                    dbc.CardBody([
                        html.P("Cliquez sur la carte pour sélectionner une zone d'analyse et de prédiction.", className="mb-3"),
                        # Style de carte (caché mais nécessaire pour le callback)
                        html.Div([
                            dcc.Dropdown(
                                id='map-style-dropdown',
                                options=[
                                    {'label': 'Standard', 'value': 'standard'},
                                    {'label': 'Clair', 'value': 'clair'},
                                    {'label': 'Sombre', 'value': 'sombre'},
                                    {'label': 'Satellite', 'value': 'satellite'}
                                ],
                                value=DEFAULT_MAP_STYLE,
                                clearable=False
                            )
                        ], style={'display': 'none'}),

                        dcc.Loading(
                            id="loading-prediction-map",
                            children=[
                                dcc.Graph(
                                    id='prediction-map',
                                    figure={},
                                    style={'height': f'{PREDICTION_MAP_HEIGHT}px'},
                                    config={
                                        'displayModeBar': True,
                                        'scrollZoom': True,
                                        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                                    }
                                )
                            ],
                            type="circle"
                        ),
                        html.Div([
                            html.Strong("Coordonnées sélectionnées: "),
                            html.Span(id="selected-coordinates", className="text-muted"),
                            dcc.Store(id="selected-location")
                        ], className="mt-3")
                    ])
                ], className="shadow-sm h-100 mb-4")
            ], md=7),

            # Colonne droite - Panneau de contrôle
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5('Paramètres de Prédiction',
                            className='py-2',
                            style={
                                'fontWeight': '500',
                                'fontSize': '24px',
                                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                                'color': '#1d1d1f'
                            }
                        )
                    ]),
                    dbc.CardBody([
                        # Type de détection
                        html.Div([
                            html.Label([
                                'Type de détection',
                                html.I(className='fas fa-info-circle ms-2', id='detection-type-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.Dropdown(
                                id='detection-type',
                                options=[
                                    {'label': 'Tous les types', 'value': 'all'},
                                    {'label': 'Observation directe (Fell)', 'value': 'Fell'},
                                    {'label': 'Découverte ultérieure (Found)', 'value': 'Found'}
                                ],
                                value='all',
                                clearable=False,
                                className='modern-dropdown mb-3'
                            ),
                        ]),

                        # Rayon d'analyse
                        html.Div([
                            html.Label([
                                "Rayon d'analyse (degrés)",
                                html.I(className='fas fa-info-circle ms-2', id='analysis-radius-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.Slider(
                                id='analysis-radius',
                                min=0.5,
                                max=10,
                                step=0.5,
                                value=ZONE_RADIUS_DEGREES,
                                marks={i: str(i) for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], className='mb-3'),

                        # Horizon de prévision
                        html.Div([
                            html.Label([
                                'Horizon de prévision (années)',
                                html.I(className='fas fa-info-circle ms-2', id='forecast-horizon-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.Slider(
                                id='forecast-horizon',
                                min=1,
                                max=50,
                                step=1,
                                value=10,
                                marks={i: str(i) for i in range(0, 51, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], className='mb-3'),

                        # Plage de masse prédite
                        html.Div([
                            html.Label([
                                'Plage de masse (g)',
                                html.I(className='fas fa-info-circle ms-2', id='mass-prediction-range-info', style={'cursor': 'pointer'})
                            ], className='form-label'),
                            dcc.RangeSlider(
                                id='mass-prediction-range',
                                min=0,
                                max=6,
                                step=0.5,
                                marks={i: f'10^{i}' for i in range(7)},
                                value=[1, 4],
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], className='mb-3'),

                        # Paramètres avancés
                        dbc.Accordion([
                            dbc.AccordionItem([
                                # Facteur environnemental
                                html.Div([
                                    html.Label([
                                        'Facteur environnemental',
                                        html.I(className='fas fa-info-circle ms-2', id='environmental-factor-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Slider(
                                        id='environmental-factor',
                                        min=0,
                                        max=1,
                                        step=0.1,
                                        value=0.5,
                                        marks={i/10: str(i/10) for i in range(0, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], className='mb-3'),

                                # Poids historique
                                html.Div([
                                    html.Label([
                                        'Poids historique',
                                        html.I(className='fas fa-info-circle ms-2', id='historical-weight-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Slider(
                                        id='historical-weight',
                                        min=0,
                                        max=1,
                                        step=0.1,
                                        value=0.7,
                                        marks={i/10: str(i/10) for i in range(0, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], className='mb-3'),

                                # Complexité du modèle
                                html.Div([
                                    html.Label([
                                        'Complexité du modèle',
                                        html.I(className='fas fa-info-circle ms-2', id='model-complexity-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Slider(
                                        id='model-complexity',
                                        min=1,
                                        max=10,
                                        step=1,
                                        value=5,
                                        marks={i: str(i) for i in range(1, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ]),

                                # Mises à jour en temps réel
                                html.Div([
                                    dbc.Checkbox(
                                        id="realtime-updates",
                                        className="form-check-input me-2",
                                        value=True
                                    ),
                                    dbc.Label(
                                        "Mise à jour automatique des prédictions",
                                        html_for="realtime-updates",
                                        className="form-check-label"
                                    ),
                                ], className="form-check mt-3"),
                            ], title="Paramètres avancés")
                        ], start_collapsed=True, className="mb-4"),

                        # Bouton de prédiction
                        dbc.Button(
                            [
                                html.I(className="fas fa-rocket me-2"),
                                "Calculer les prédictions"
                            ],
                            id="calculate-prediction",
                            color="success",
                            className="w-100 py-2 mb-3"
                        ),

                        # Indice de fiabilité
                        html.Div([
                            html.Label("Indice de fiabilité:", className="form-label"),
                            html.Div(id="reliability-index", className="mt-2")
                        ])
                    ])
                ], className="shadow-sm h-100 mb-4")
            ], md=5),
        ]),

        # Menu de navigation des sections de prédiction
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Résultats de prédiction", id="btn-prediction-results", color="primary", outline=True, className="active me-1"),
                    dbc.Button("Analyse de zone", id="btn-zone-analysis", color="primary", outline=True, className="me-1"),
                    dbc.Button("Prédiction temporelle", id="btn-temporal-prediction", color="primary", outline=True, className="me-1"),
                    dbc.Button("Prédiction spatiale", id="btn-spatial-prediction", color="primary", outline=True),
                ], className="mb-4 w-100")
            ], width=12)
        ]),

        # Conteneur de résultats de prédiction
        dbc.Row([
            # Résultats de prédiction
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Résultats de la Prédiction", className="m-0")
                    ]),
                    dbc.CardBody(id="prediction-results-content", children=[
                        dbc.Row([
                            # Probabilité de chute (Found)
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("Probabilité de découverte (Found)", className="card-title text-center mb-3"),
                                        html.Div(id="found-probability", className="text-center fs-1 fw-bold", style={"color": COLOR_SCHEMES["primary"]})
                                    ])
                                ], className="shadow-sm text-center")
                            ], md=6, className="mb-3"),

                            # Probabilité de chute (Fell)
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("Probabilité d'observation (Fell)", className="card-title text-center mb-3"),
                                        html.Div(id="fell-probability", className="text-center fs-1 fw-bold", style={"color": COLOR_SCHEMES["warning"]})
                                    ])
                                ], className="shadow-sm text-center")
                            ], md=6, className="mb-3"),

                            # Masse estimée
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("Masse estimée (g)", className="card-title text-center mb-3"),
                                        html.Div(id="estimated-mass", className="text-center fs-1 fw-bold", style={"color": COLOR_SCHEMES["success"]})
                                    ])
                                ], className="shadow-sm text-center")
                            ], md=6, className="mb-3"),

                            # Classe probable
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("Classe la plus probable", className="card-title text-center mb-3"),
                                        html.Div(id="probable-class", className="text-center fs-1 fw-bold", style={"color": COLOR_SCHEMES["info"]})
                                    ])
                                ], className="shadow-sm text-center")
                            ], md=6, className="mb-3"),

                            # Sortie de prédiction détaillée
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Détails de la prédiction"),
                                    dbc.CardBody(id="prediction-output")
                                ], className="shadow-sm h-100")
                            ], width=12, className="mb-3")
                        ])
                    ])
                ], className="shadow-sm h-100 mb-4")
            ], width=12),

            # Analyse de zone (initialement caché)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Analyse de la Zone", className="m-0")
                    ]),
                    dbc.CardBody(id="zone-analysis-content", style={"display": "none"}, children=[
                        html.Div(id="zone-analysis-output")
                    ])
                ], className="shadow-sm h-100 mb-4")
            ], width=12),

            # Prédiction temporelle (initialement caché)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Prédiction Temporelle", className="m-0")
                    ]),
                    dbc.CardBody(id="temporal-prediction-content", style={"display": "none"}, children=[
                        html.Div(id="temporal-prediction-output"),
                        html.Div(id="temporal-prediction-chart")
                    ])
                ], className="shadow-sm h-100 mb-4")
            ], width=12),

            # Prédiction spatiale (initialement caché)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Prédiction Spatiale", className="m-0")
                    ]),
                    dbc.CardBody(id="spatial-prediction-content", style={"display": "none"}, children=[
                        html.Div(id="spatial-prediction-output")
                    ])
                ], className="shadow-sm h-100 mb-4")
            ], width=12)
        ]),

        # Popover pour les infobulles
        html.Div(id='info-popovers-prediction'),

        # Intervalle pour l'actualisation périodique des données
        dcc.Interval(id='interval', interval=60000)  # 1 minute
    ], fluid=True)