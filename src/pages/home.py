import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.config import COLOR_SCHEMES

dash.register_page(__name__, path='/', title='Accueil - Dashboard Météorites')

def layout(**kwargs):
    return dbc.Container([
        # En-tête de style Apple
        html.Div([
            html.H1('Bienvenue sur le Dashboard des Météorites', 
                className='display-4 fw-bold', 
                style={
                    'fontWeight': '600',
                    'fontSize': '48px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                    'letterSpacing': '-0.5px',
                    'color': '#1d1d1f',
                    'marginBottom': '24px'
                }
            ),
            html.P('Explorez et analysez de manière interactive les données mondiales sur les chutes de météorites', 
                className='lead fs-3 mb-4', 
                style={
                    'fontWeight': '400',
                    'fontSize': '24px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                    'color': '#86868b',
                    'lineHeight': '1.5'
                }
            ),
        ], className='py-5 text-center'),
        
        # Présentation des fonctionnalités principales en utilisant des cartes avec animation au survol
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.I(className="fas fa-globe-americas fa-3x", style={'color': COLOR_SCHEMES['primary']}),
                    ], className="text-center mt-4"),
                    dbc.CardBody([
                        html.H3("Carte Mondiale", className="card-title text-center fw-bold"),
                        html.P([
                            "Explorez la distribution globale des météorites avec notre carte interactive. ",
                            "Filtrez par classe, masse, et période pour visualiser les modèles de chute à travers le monde."
                        ], className="card-text")
                    ]),
                    dbc.CardFooter([
                        dbc.Button("Explorer la carte", color="primary", href="/map", className="w-100")
                    ])
                ], className="h-100 shadow-sm hover-card")
            ], width=12, md=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-3x", style={'color': COLOR_SCHEMES['success']}),
                    ], className="text-center mt-4"),
                    dbc.CardBody([
                        html.H3("Analyse de Données", className="card-title text-center fw-bold"),
                        html.P([
                            "Découvrez des statistiques approfondies, des distributions, des séries chronologiques et des corrélations ",
                            "pour mieux comprendre les phénomènes météoritiques."
                        ], className="card-text")
                    ]),
                    dbc.CardFooter([
                        dbc.Button("Analyser les données", color="success", href="/analysis", className="w-100")
                    ])
                ], className="h-100 shadow-sm hover-card")
            ], width=12, md=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.I(className="fas fa-rocket fa-3x", style={'color': COLOR_SCHEMES['warning']}),
                    ], className="text-center mt-4"),
                    dbc.CardBody([
                        html.H3("Prédictions", className="card-title text-center fw-bold"),
                        html.P([
                            "Utilisez nos modèles avancés pour prédire les chutes de météorites. ",
                            "Analyses spatiales et temporelles pour estimer les futures zones d'impact."
                        ], className="card-text")
                    ]),
                    dbc.CardFooter([
                        dbc.Button("Voir les prédictions", color="warning", href="/predictions", className="w-100")
                    ])
                ], className="h-100 shadow-sm hover-card")
            ], width=12, md=4, className="mb-4")
        ], className="mb-5"),
        
        # Statistiques et chiffres clés
        dbc.Row([
            html.Div([
                html.H2("Chiffres Clés", className="text-center mb-4", 
                       style={'fontWeight': '500', 'fontSize': '32px', 'color': '#1d1d1f'}),
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("45,716", className="text-center text-primary mb-0", style={'fontSize': '42px', 'fontWeight': '600'}),
                        html.P("Météorites analysées", className="text-center text-muted")
                    ])
                ], className="text-center shadow-sm")
            ], width=12, md=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("220+", className="text-center text-success mb-0", style={'fontSize': '42px', 'fontWeight': '600'}),
                        html.P("Années de données", className="text-center text-muted")
                    ])
                ], className="text-center shadow-sm")
            ], width=12, md=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("1,107", className="text-center text-warning mb-0", style={'fontSize': '42px', 'fontWeight': '600'}),
                        html.P("Chutes observées", className="text-center text-muted")
                    ])
                ], className="text-center shadow-sm")
            ], width=12, md=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("235", className="text-center text-info mb-0", style={'fontSize': '42px', 'fontWeight': '600'}),
                        html.P("Pays concernés", className="text-center text-muted")
                    ])
                ], className="text-center shadow-sm")
            ], width=12, md=3, className="mb-4")
        ], className="mb-5"),
        
        # Une section "Commencez maintenant" avec un call-to-action
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("Commencez votre exploration", className="text-center mb-3"),
                        html.P("Découvrez dès maintenant les mystères des météorites à travers nos outils interactifs.", 
                               className="text-center mb-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Explorer la carte mondiale", color="primary", size="lg", href="/map", className="me-md-2 mb-2 mb-md-0 w-100")
                            ], width=12, md=6),
                            dbc.Col([
                                dbc.Button("Voir les analyses", color="success", size="lg", href="/analysis", className="w-100")
                            ], width=12, md=6)
                        ])
                    ])
                ], className="text-center shadow-sm bg-light")
            ], width=12, className="mb-4")
        ])
    ], fluid=True, className="py-3")

# CSS pour l'effet d'animation au survol
dash.get_app().index_string = dash.get_app().index_string.replace('</head>', '''
<style>
.hover-card {
    transition: transform 0.3s, box-shadow 0.3s;
}
.hover-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
}
</style>
</head>
''') 