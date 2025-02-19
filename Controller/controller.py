from dash import Dash, Input, Output
import plotly.express as px
from ..Model.model import MeteoriteData
from ..View.view import DashboardView


class DashboardController:
    def __init__(self, filepath):
        self.app = Dash(__name__)
        self.data_model = MeteoriteData(filepath)
        self.view = DashboardView()
        self.app.layout = self.view.layout
        self._register_callbacks()

    def _register_callbacks(self):
        """Enregistre les callbacks pour les interactions utilisateur."""
        @self.app.callback(
            [Output('world-map', 'figure'),
             Output('mass-dist', 'figure'),
             Output('class-dist', 'figure'),
             Output('year-timeline', 'figure')],
            [Input('year-slider', 'value'),
             Input('fall-type-dropdown', 'value')]
        )
        def update_graphs(selected_years, selected_fall_types):
            df = self.data_model.get_data()
            filtered_df = df[(df['year'] >= selected_years[0]) & 
                              (df['year'] <= selected_years[1]) & 
                              (df['fall'].isin(selected_fall_types))]
            
            # Carte mondiale
            world_map = px.scatter_geo(
                filtered_df,
                lat='reclat',
                lon='reclong',
                size='mass (g)',
                hover_name='name',
                title='Localisation des impacts',
                projection='natural earth',
                size_max=20
            )
            
            # Histogramme des masses
            mass_dist = px.histogram(
                filtered_df,
                x='mass (g)',
                nbins=50,
                title='Distribution des masses',
                log_x=True
            )
            
            # Top 10 des classifications
            class_dist = px.bar(
                filtered_df['recclass'].value_counts().head(10).reset_index(),
                x='count',
                y='recclass',
                title='Top 10 des classifications'
            )
            
            # Timeline des chutes
            year_timeline = px.line(
                filtered_df.groupby('year').size().reset_index(name='count'),
                x='year',
                y='count',
                title='Chutes par année'
            )
            
            return world_map, mass_dist, class_dist, year_timeline

    def run(self):
        """Lance l'application Dash."""
        self.app.run_server(debug=True)

if __name__ == '__main__':
    filepath = 'dashboard/Meteorite_Landings.csv'  # Remplacez par le chemin de votre fichier
    controller = DashboardController(filepath)
    controller.run()