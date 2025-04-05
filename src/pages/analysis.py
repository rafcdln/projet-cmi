import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.config import GRAPH_HEIGHT, MASS_RANGE, YEAR_RANGE, DEFAULT_FALLS, COLOR_SCHEMES

dash.register_page(__name__, path='/analysis', title='Analyse de Données - Dashboard Météorites')

def layout(**kwargs):
    return dbc.Container([
        # En-tête
        dbc.Row([
            dbc.Col([
                html.H1('Analyse des Données Météoritiques',
                    className='mb-0',
                    style={
                        'fontWeight': '600',
                        'fontSize': '42px',
                        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                        'letterSpacing': '-0.5px',
                        'color': '#1d1d1f'
                    }
                ),
                html.P('Visualisez les distributions, tendances et corrélations des données de météorites',
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

        # Menu de navigation des sections d'analyse
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Distributions", id="btn-distributions", color="primary", outline=True, className="active me-1"),
                    dbc.Button("Séries Temporelles", id="btn-time-series", color="primary", outline=True, className="me-1"),
                    dbc.Button("Corrélations", id="btn-correlations", color="primary", outline=True),
                ], className="mb-4 w-100")
            ], width=12)
        ]),

        # Filtres communs pour toutes les analyses
        dbc.Row([
            dbc.Col([
                # Composant caché pour résoudre l'erreur de callback
                html.Div([
                    dcc.Dropdown(
                        id='heatmap-colorscale',
                        options=[
                            {'label': 'Inferno', 'value': 'Inferno'},
                            {'label': 'Viridis', 'value': 'Viridis'},
                            {'label': 'Cividis', 'value': 'Cividis'},
                            {'label': 'Plasma', 'value': 'Plasma'},
                            {'label': 'Turbo', 'value': 'Turbo'}
                        ],
                        value='Inferno',
                        clearable=False
                    )
                ], style={'display': 'none'}),

                dbc.Card([
                    dbc.CardHeader("Filtres communs"),
                    dbc.CardBody([
                        dbc.Row([
                            # Filtre de masse
                            dbc.Col([
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
                                    value=MASS_RANGE,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], md=3, className='mb-3'),

                            # Sélection de classe
                            dbc.Col([
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
                            ], md=3, className='mb-3'),

                            # Filtre de chute
                            dbc.Col([
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
                                    value=DEFAULT_FALLS,
                                    inline=True,
                                    className='ps-2'
                                ),
                            ], md=3, className='mb-3'),

                            # Filtre de décennie
                            dbc.Col([
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
                                    value=YEAR_RANGE,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], md=3, className='mb-3'),
                        ]),
                    ])
                ], className="shadow-sm mb-4")
            ], width=12),
        ]),

        # Panneau de distributions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Distributions des Données", className="m-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            # Distribution de Masse
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Distribution des Masses"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-mass-hist",
                                            children=[
                                                dcc.Graph(
                                                    id='mass-hist',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),

                            # Distribution de Classe
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Distribution des Classes"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-class-distribution",
                                            children=[
                                                dcc.Graph(
                                                    id='class-distribution',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),

                            # Distribution d'Année
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Distribution Temporelle"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-year-distribution",
                                            children=[
                                                dcc.Graph(
                                                    id='year-distribution',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),

                            # Distribution Géographique
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Distribution Géographique"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-geo-distribution",
                                            children=[
                                                dcc.Graph(
                                                    id='geo-distribution',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),
                        ]),
                    ])
                ], className="shadow-sm", id="panel-distributions")
            ], width=12),
        ], id="row-distributions"),

        # Panneau de séries temporelles (initialement caché)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Séries Temporelles", className="m-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            # Série temporelle simple
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Évolution dans le Temps"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-time-series",
                                            children=[
                                                dcc.Graph(
                                                    id='time-series',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),

                            # Tendance annuelle
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Tendance Annuelle"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-annual-trend",
                                            children=[
                                                dcc.Graph(
                                                    id='annual-trend',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),

                            # Évolution de la masse
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Évolution de la Masse"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-mass-time",
                                            children=[
                                                dcc.Graph(
                                                    id='mass-time',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),

                            # Prévision future
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Prévision des Tendances"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-forecast",
                                            children=[
                                                dcc.Graph(
                                                    id='forecast',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),
                        ]),
                    ])
                ], className="shadow-sm", id="panel-time-series", style={'display': 'none'})
            ], width=12),
        ], id="row-time-series"),

        # Panneau de corrélations (initialement caché)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Corrélations et Relations", className="m-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            # Heatmap de corrélation
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Matrice de Corrélation"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-correlation-heatmap",
                                            children=[
                                                dcc.Graph(
                                                    id='correlation-heatmap',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),

                            # Importance des caractéristiques
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Importance des Caractéristiques"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-feature-importance",
                                            children=[
                                                dcc.Graph(
                                                    id='feature-importance',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=6, className='mb-4'),

                            # Autre importance des caractéristiques
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Importance des Caractéristiques (Classe)"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-feature-importance-2",
                                            children=[
                                                dcc.Graph(
                                                    id='feature-importance-2',
                                                    figure={},
                                                    style={'height': f'{GRAPH_HEIGHT}px'},
                                                    config={'displayModeBar': False}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ])
                                ], className="shadow-sm h-100")
                            ], width=12, className='mb-4'),
                        ]),
                    ])
                ], className="shadow-sm", id="panel-correlations", style={'display': 'none'})
            ], width=12),
        ], id="row-correlations"),

        # Popover pour les infobulles
        html.Div(id='info-popovers-analysis'),

        # Intervalle pour l'actualisation périodique des données (si nécessaire)
        dcc.Interval(id='interval', interval=600000)  # 10 minutes
    ], fluid=True)