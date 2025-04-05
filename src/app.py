"""
Point d'entrée de l'application d'analyse de météorites.
Lance le serveur Dash pour le tableau de bord interactif multi-pages.
"""
import dash
from dash import Dash, html, page_container
import dash_bootstrap_components as dbc
import traceback
import sys
from flask import Flask, request, Response
import json
import time
from utils.config import DATA_PATH, PORT, DEBUG_MODE, STOP_ON_ERROR, VERBOSE_ERRORS

def format_error(e, environ=None):
    """
    Formate une erreur pour l'affichage et le logging
    """
    error_class = e.__class__.__name__
    error_msg = str(e)

    # Tracer l'erreur
    trace = traceback.format_exc()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Informations sur la requête si disponible
    request_info = ""
    if environ:
        path = environ.get('PATH_INFO', 'Chemin inconnu')
        method = environ.get('REQUEST_METHOD', 'Méthode inconnue')
        request_info = f"Requête: {method} {path}\n"

    # Formater le message complet
    error_text = f"""
=============== ERREUR APPLICATION [{timestamp}] ===============
{request_info}
Type: {error_class}
Message: {error_msg}

Traceback:
{trace}
================================================================
"""

    return error_text

# Fonction pour capturer les erreurs non gérées
class ErrorCatchingMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        try:
            return self.app(environ, start_response)
        except Exception as e:
            # Capturer l'erreur et la tracer
            error_text = format_error(e, environ)
            print(error_text, file=sys.stderr)

            # Si configuré pour arrêter le programme en cas d'erreur
            if STOP_ON_ERROR:
                print("Arrêt du programme demandé après erreur.")
                sys.exit(1)

            # Sinon, continuer et afficher une réponse d'erreur
            if request.headers.get('Content-Type') == 'application/json':
                # Réponse JSON pour les requêtes AJAX
                error_response = json.dumps({
                    'error': str(e),
                    'type': e.__class__.__name__
                })
                return Response(error_response, status=500, mimetype='application/json')
            else:
                # Réponse HTML pour les autres requêtes
                response_body = html.Div([
                    html.H1("Erreur Application", style={'color': 'red'}),
                    html.Hr(),
                    html.Pre(f"Type: {e.__class__.__name__}"),
                    html.Pre(f"Message: {str(e)}"),
                    html.Hr(),
                    html.Pre(traceback.format_exc() if VERBOSE_ERRORS else "Les détails de l'erreur ont été enregistrés.",
                             style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word'})
                ]).to_string()

                response_headers = [
                    ('Content-Type', 'text/html'),
                    ('Content-Length', str(len(response_body)))
                ]

                start_response('500 Internal Server Error', response_headers)
                return [response_body.encode('utf-8')]

# Initialisation de l'application Dash avec support multi-pages
server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
        'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
    ],
    use_pages=True,  # Activer le système de pages
    suppress_callback_exceptions=True,  # Supprimer les exceptions de callback
    assets_folder='src/assets'  # Spécifier explicitement le dossier des assets
)

# Configuration pour Plotly Express
import plotly.express as px
import plotly.graph_objs as go
# Monkey patching pour les erreurs Plotly
from plotly.express._core import make_figure as original_make_figure
from plotly.express._core import build_dataframe

# Patch pour afficher plus d'informations en cas d'erreur
def verbose_make_figure(*args, **kwargs):
    try:
        return original_make_figure(*args, **kwargs)
    except Exception as e:
        if VERBOSE_ERRORS:
            print(f"\n{'='*80}")
            print(f"Erreur dans plotly.express.make_figure: {str(e)}")
            print("Arguments:")
            for i, arg in enumerate(args):
                print(f"  Arg {i}: {type(arg)}")
                # Limiter l'affichage pour éviter des sorties trop grandes
                if hasattr(arg, 'shape'):  # Pour les DataFrames
                    print(f"     Shape: {arg.shape}")
                elif hasattr(arg, 'keys'):  # Pour les dictionnaires
                    print(f"     Keys: {list(arg.keys())}")
                elif isinstance(arg, (list, tuple)) and len(arg) > 10:
                    print(f"     Length: {len(arg)}, First 5: {arg[:5]}")
                else:
                    try:
                        print(f"     Value: {arg}")
                    except:
                        print("     <Valeur non affichable>")
            print("Kwargs:")
            for k, v in kwargs.items():
                print(f"  {k}: {type(v)}")
            print(f"{'='*80}\n")
        raise

# Patch de sécurité pour build_dataframe
original_build_dataframe = build_dataframe
def safe_build_dataframe(*args, **kwargs):
    try:
        return original_build_dataframe(*args, **kwargs)
    except ValueError as e:
        if "columns of different type" in str(e) and VERBOSE_ERRORS:
            print("\n" + "="*80)
            print("ERREUR DE TYPES COLONNES DÉTECTÉE DANS PLOTLY EXPRESS")
            print("Message d'erreur:", str(e))

            # Analyse des args pour trouver le dataframe
            if len(args) > 0 and isinstance(args[0], dict) and 'args' in args[0]:
                data_args = args[0]['args']
                if 'data_frame' in data_args and hasattr(data_args['data_frame'], 'dtypes'):
                    df = data_args['data_frame']
                    print("\nColonnes et types:")
                    for col, dtype in df.dtypes.items():
                        print(f"  - {col}: {dtype}")

                    # Tenter de corriger automatiquement pour les prochains appels
                    print("\nConseil: Convertissez toutes les colonnes au même type avant de les passer à Plotly Express.")
            print("="*80 + "\n")
        raise

# Appliquer les patches
import plotly.express._core
plotly.express._core.make_figure = verbose_make_figure
plotly.express._core.build_dataframe = safe_build_dataframe

# Ajouter le middleware de capture d'erreurs
server.wsgi_app = ErrorCatchingMiddleware(server.wsgi_app)

# Configuration de l'application
app.title = 'Dashboard Météorites'

# Définir le layout principal avec le conteneur de pages
app.layout = html.Div([
    # Barre de navigation commune à toutes les pages
    dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("Analyse des Météorites", href="/", className="ms-2",
                                style={'fontWeight': '500', 'fontSize': '22px'}),
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Accueil", href="/")),
                        dbc.NavItem(dbc.NavLink("Carte Mondiale", href="/map")),
                        dbc.NavItem(dbc.NavLink("Analyse de Données", href="/analysis")),
                        dbc.NavItem(dbc.NavLink("Prédictions", href="/predictions")),
                    ],
                    className="ms-auto",
                    navbar=True,
                ),
            ]
        ),
        color="light",
        className="mb-4",
    ),

    # Conteneur qui affichera le contenu de la page active
    page_container,

    # Pied de page
    html.Footer(
        dbc.Container([
            html.Hr(),
            html.P("Dashboard d'analyse de météorites © 2023", className="text-center text-muted")
        ]),
        className="mt-5"
    )
])

# Initialisation des modèles et enregistrement des callbacks
from components.controller import register_callbacks
register_callbacks(app, DATA_PATH)

# Lancement du serveur
if __name__ == '__main__':
    print(f"Lancement du tableau de bord d'analyse de météorites sur http://127.0.0.1:{PORT}/")
    print(f"Mode débogage: {'Activé' if DEBUG_MODE else 'Désactivé'}")
    print(f"Arrêt sur erreur: {'Activé' if STOP_ON_ERROR else 'Désactivé'}")
    print(f"Verbosité des erreurs: {'Activée' if VERBOSE_ERRORS else 'Désactivée'}")
    print("\nPour changer ces paramètres, modifiez le fichier 'src/utils/config.py'")

    app.run_server(debug=DEBUG_MODE, port=PORT)