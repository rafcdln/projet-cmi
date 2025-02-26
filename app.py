from dash import Dash
import dash_bootstrap_components as dbc
from view import create_layout
from controller import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = create_layout()
register_callbacks(app, 'Meteorite_Landings.csv')

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)