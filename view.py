from dash import dcc, html
import dash_bootstrap_components as dbc

def create_layout():
    return html.Div([
        dcc.Store(id='memory'),
        dcc.Interval(id='interval', interval=10000),
        
        dbc.Row([
            dbc.Col([
                html.H1("Meteorite Analytics Dashboard", className="text-center mb-4"),
                html.Hr(),
                
                # Filtres
                html.H4("Filtres interactifs"),
                dcc.RangeSlider(
                    id='mass-slider',
                    min=0,
                    max=100000,
                    step=100,
                    value=[0, 10000],
                    marks={i: f'{i/1000}k' for i in range(0, 100001, 10000)}
                ),
                dcc.Dropdown(
                    id='class-dropdown',
                    multi=True,
                    placeholder="Sélectionner des classifications"
                ),
                dcc.Checklist(
                    id='fall-checklist',
                    options=[{'label': t, 'value': t} for t in ['Fell', 'Found']],
                    value=['Fell', 'Found']
                ),
                dcc.RangeSlider(
                    id='decade-slider',
                    min=1700,
                    max=2020,
                    step=10,
                    value=[1900, 2020],
                    marks={i: str(i) for i in range(1700, 2021, 50)}
                )
            ], md=3, className="sidebar"),
            
            # Graphiques
            dbc.Col([
                dbc.Row([
                    dbc.Col(dcc.Loading(dcc.Graph(id='world-map')), md=12),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Loading(dcc.Graph(id='mass-hist')), md=6),
                    dbc.Col(dcc.Loading(dcc.Graph(id='time-series')), md=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Loading(dcc.Graph(id='class-bar')), md=12)
                ])
            ], md=9)
        ])
    ], className="container-fluid")