from dash import Input, Output, callback
import plotly.express as px
from model import MeteoriteData
import pandas as pd

def register_callbacks(app, data_path):
    meteor_data = MeteoriteData(data_path)
    
    @app.callback(
        Output('memory', 'data'),
        [Input('mass-slider', 'value'),
         Input('class-dropdown', 'value'),
         Input('fall-checklist', 'value'),
         Input('decade-slider', 'value')]
    )
    def update_filtered_data(mass_range, classification, fall_type, decade_range):
        filtered = meteor_data.get_filtered_data(mass_range, classification, fall_type, decade_range)
        return filtered.to_dict('records')
    
    @app.callback(
        [Output('world-map', 'figure'),
         Output('mass-hist', 'figure'),
         Output('time-series', 'figure'),
         Output('class-bar', 'figure'),
         Output('class-dropdown', 'options')],
        [Input('memory', 'data')]
    )
    def update_plots(data):
        df = pd.DataFrame(data)
        
        # Carte interactive avec animation
        map_fig = px.scatter_geo(df.iloc[::10], 
                                lat='reclat', 
                                lon='reclong',
                                animation_frame='decade',
                                color='fall',
                                hover_name='name',
                                projection='natural earth',
                                title='Distribution géographique par décennie',
                                width=1000,
                                height=680) 
        # Histogramme avec sélection de plage
        hist_fig = px.histogram(df, x='mass (g)', nbins=50, 
                               log_x=True, 
                               title='Distribution des masses',
                               color_discrete_sequence=['#2A9D8F'])
        
        # Séries temporelles avec agrégation
        time_fig = px.line(df.groupby('year', as_index=False).size(),
                          x='year', y='size',
                          title='Chutes par année',
                          color_discrete_sequence=['#E76F51'])
        
        # Top 10 des classifications
        class_fig = px.bar(df['recclass'].value_counts().head(10).reset_index(),
                          x='count', y='recclass',
                          title='Top 10 des classifications',
                          color='recclass')
        
        # Options du dropdown
        class_options = [{'label': c, 'value': c} 
                        for c in df['recclass'].unique()]
        
        return (map_fig, hist_fig, time_fig, class_fig, class_options)