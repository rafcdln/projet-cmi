from dash import html, dcc

class DashboardView:
    def __init__(self):
        self.layout = self._create_layout()

    def _create_layout(self):
        """Crée la mise en page du dashboard."""
        return html.Div([
            html.H1("Analyse des Météorites NASA", style={'textAlign': 'center'}),
            
            # Filtres
            html.Div([
                html.Label("Filtrer par année :"),
                dcc.RangeSlider(
                    id='year-slider',
                    min=1900,
                    max=2023,
                    step=1,
                    marks={i: str(i) for i in range(1900, 2024, 10)},
                    value=[1900, 2023]
                ),
                html.Label("Filtrer par type de chute :"),
                dcc.Dropdown(
                    id='fall-type-dropdown',
                    options=[
                        {'label': 'Fell', 'value': 'Fell'},
                        {'label': 'Found', 'value': 'Found'}
                    ],
                    value=['Fell', 'Found'],
                    multi=True
                )
            ], style={'margin': '20px'}),
            
            # Graphiques
            html.Div([
                dcc.Graph(id='world-map'),
                dcc.Graph(id='mass-dist'),
                dcc.Graph(id='class-dist'),
                dcc.Graph(id='year-timeline')
            ])
        ])