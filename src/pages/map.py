import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.config import MAP_HEIGHT, DEFAULT_MAP_STYLE, DEFAULT_COLOR_MODE, MASS_RANGE, YEAR_RANGE, DEFAULT_FALLS

# Les callbacks sont déjà importés dans controller.py
# Ne pas importer map_callbacks ici pour éviter les duplications

dash.register_page(__name__, path='/map', title='Carte Mondiale - Dashboard Météorites')

def layout(**kwargs):
    return dbc.Container([
        # En-tête
        dbc.Row([
            dbc.Col([
                html.H1('Distribution Mondiale des Météorites',
                    className='mb-0',
                    style={
                        'fontWeight': '600',
                        'fontSize': '42px',
                        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                        'letterSpacing': '-0.5px',
                        'color': '#1d1d1f'
                    }
                ),
                html.P('Explorez la répartition géographique des météorites à travers le monde',
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

        # Contenu principal - Filtres et carte
        dbc.Row([
            # Colonne gauche - Filtres
            dbc.Col([
                dbc.Card([
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
                    dbc.CardBody([
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
                                value=MASS_RANGE,
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
                                value=DEFAULT_COLOR_MODE,
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
                                value=DEFAULT_FALLS,
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
                                value=YEAR_RANGE,
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

                        # Paramètres avancés
                        dbc.Accordion([
                            dbc.AccordionItem([
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
                                        step=1,
                                        value=5,
                                        marks={i: str(i) for i in range(1, 11)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], className='mb-3'),

                                # Mode d'interaction carte
                                html.Div([
                                    html.Label([
                                        'Interactivité de la carte',
                                        html.I(className='fas fa-info-circle ms-2', id='map-interactivity-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.RadioItems(
                                        id='map-interactivity',
                                        options=[
                                            {'label': ' Performance (moins de détails)', 'value': 'low'},
                                            {'label': ' Équilibré', 'value': 'medium'},
                                            {'label': ' Détaillé (peut ralentir)', 'value': 'high'}
                                        ],
                                        value='medium',
                                        labelStyle={'display': 'block', 'marginBottom': '5px'},
                                        className='ps-2'
                                    ),
                                ]),
                            ], title="Paramètres avancés")
                        ], start_collapsed=True, className="mb-3"),

                        # Options de visualisation
                        dbc.Tabs([
                            dbc.Tab([
                                html.P("Affiche toutes les météorites individuellement", className="mt-3 small text-muted"),
                                dcc.Loading(id="loading-map", children=[html.Div(id="map-loading-placeholder")], type="circle")
                            ], label="Points", tab_id="tab-points", className="p-2"),

                            dbc.Tab([
                                html.P("Affiche la densité des météorites par zone", className="mt-3 small text-muted"),
                                html.Div([
                                    html.Label([
                                        'Rayon de densité',
                                        html.I(className='fas fa-info-circle ms-2', id='heatmap-radius-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
                                    dcc.Slider(
                                        id='heatmap-radius',
                                        min=1,
                                        max=20,
                                        step=1,
                                        value=8,
                                        marks={i: str(i) for i in range(1, 21, 4)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], className='mb-3 mt-3'),

                                html.Div([
                                    html.Label([
                                        'Palette de couleurs',
                                        html.I(className='fas fa-info-circle ms-2', id='heatmap-colorscale-info', style={'cursor': 'pointer'})
                                    ], className='form-label small'),
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
                                        clearable=False,
                                        className='modern-dropdown'
                                    ),
                                ], className='mb-3'),
                                dcc.Loading(id="loading-heatmap", children=[html.Div(id="heatmap-loading-placeholder")], type="circle")
                            ], label="Densité", tab_id="tab-heatmap", className="p-2"),
                        ], id="map-tabs", active_tab="tab-points")
                    ])
                ], className="shadow-sm")
            ], md=3, className="mb-4"),

            # Colonne droite - Carte
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5('Carte des Météorites',
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
                        # Onglets pour les différents modes de carte
                        dbc.Tabs([
                            dbc.Tab([
                                dcc.Loading(
                                    id="loading-world-map",
                                    children=[
                                        dcc.Graph(
                                            id='world-map',
                                            figure={},
                                            style={'height': f'{MAP_HEIGHT}px'},
                                            config={
                                                'displayModeBar': True,
                                                'scrollZoom': True,
                                                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                                            }
                                        )
                                    ],
                                    type="circle"
                                )
                            ], label="Carte des Points", tab_id="map-tab-1"),

                            dbc.Tab([
                                dcc.Loading(
                                    id="loading-heatmap-tab",
                                    children=[
                                        dcc.Graph(
                                            id='heatmap',
                                            figure={},
                                            style={'height': f'{MAP_HEIGHT}px'},
                                            config={
                                                'displayModeBar': True,
                                                'scrollZoom': True,
                                                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                                            }
                                        )
                                    ],
                                    type="circle"
                                )
                            ], label="Carte de Densité", tab_id="map-tab-2"),
                        ], id="map-view-tabs", active_tab="map-tab-1")
                    ])
                ], className="shadow-sm h-100")
            ], md=9, className="mb-4"),
        ]),

        # Section des statistiques sommaires - Maintenant sous la carte avec un design moderne
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Statistiques sur la Sélection",
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
                        dbc.Row([
                            # Nombre de météorites
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Div([
                                            html.I(className="fas fa-meteor fa-2x text-primary mb-2"),
                                            html.H4(id="stats-count", className="mb-0 mt-1 fs-2 fw-bold"),
                                            html.P("Météorites", className="text-muted small mb-0")
                                        ], className="text-center")
                                    ])
                                ], className="shadow-sm h-100 stat-card")
                            ], width=12, sm=6, md=3, className="mb-3"),

                            # Masse totale
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Div([
                                            html.I(className="fas fa-weight-hanging fa-2x text-success mb-2"),
                                            html.H4(id="stats-mass", className="mb-0 mt-1 fs-2 fw-bold"),
                                            html.P("Masse totale", className="text-muted small mb-0")
                                        ], className="text-center")
                                    ])
                                ], className="shadow-sm h-100 stat-card")
                            ], width=12, sm=6, md=3, className="mb-3"),

                            # Période couverte
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Div([
                                            html.I(className="fas fa-calendar-alt fa-2x text-warning mb-2"),
                                            html.H4(id="stats-years", className="mb-0 mt-1 fs-2 fw-bold"),
                                            html.P("Période", className="text-muted small mb-0")
                                        ], className="text-center")
                                    ])
                                ], className="shadow-sm h-100 stat-card")
                            ], width=12, sm=6, md=3, className="mb-3"),

                            # Classes principales
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Div([
                                            html.I(className="fas fa-tags fa-2x text-info mb-2"),
                                            html.H4(id="stats-classes", className="mb-0 mt-1 fs-2 fw-bold"),
                                            html.P("Classes principales", className="text-muted small mb-0")
                                        ], className="text-center")
                                    ])
                                ], className="shadow-sm h-100 stat-card")
                            ], width=12, sm=6, md=3, className="mb-3"),
                        ]),

                        # Répartition supplémentaire
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Distribution Temporelle", className="p-2 text-center fw-bold"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-timeline-mini",
                                            children=[
                                                dcc.Graph(
                                                    id='timeline-mini',
                                                    figure={},
                                                    config={'displayModeBar': False},
                                                    style={"height": "120px"}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ], className="p-1")
                                ], className="shadow-sm h-100")
                            ], width=12, md=6, className="mb-3"),

                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Répartition des Classes", className="p-2 text-center fw-bold"),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-pie-mini",
                                            children=[
                                                dcc.Graph(
                                                    id='class-pie-mini',
                                                    figure={},
                                                    config={'displayModeBar': False},
                                                    style={"height": "120px"}
                                                )
                                            ],
                                            type="circle"
                                        )
                                    ], className="p-1")
                                ], className="shadow-sm h-100")
                            ], width=12, md=6, className="mb-3"),
                        ]),

                        # Bouton pour exporter les résultats
                        dbc.Row([
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="fas fa-download me-2"),
                                    "Exporter les résultats"
                                ], id="btn-export", color="primary", className="w-100")
                            ], width=12, className="mt-2 text-center")
                        ])
                    ])
                ], className="shadow-sm h-100")
            ], width=12, className="mb-4")
        ]),

        # Popover pour les infobulles (ajoutés dynamiquement)
        html.Div(id='info-popovers'),

        # Store pour les données intermédiaires
        dcc.Store(id='filtered-data'),

        # Intervalle pour l'actualisation périodique des données (si nécessaire)
        dcc.Interval(id='interval', interval=600000),  # 10 minutes

        # Style CSS pour les cartes de statistiques avec animation au survol
        html.Div([
            dcc.Markdown("""
            <style>
            .stat-card {
                transition: all 0.3s ease;
            }
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
            }
            </style>
            """, dangerously_allow_html=True)
        ], id="map-styles", style={'display': 'none'})
    ], fluid=True)