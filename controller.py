from dash import Input, Output, callback
import plotly.express as px
from model import MeteoriteData
import pandas as pd
from datetime import datetime

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
        empty_fig = px.scatter(title="Aucune donnée disponible")
        
        # Vérification des données critiques
        if df.empty or 'year' not in df.columns or df['year'].isnull().all():
            return empty_fig, empty_fig, empty_fig, empty_fig, []

        try:
            # Correction 1: Conversion numérique de l'année
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[df['year'].between(1000, datetime.now().year)]
            
            # Correction 2: Création des intervalles de 5 ans
            df['year_interval'] = (df['year'] // 5) * 5
            time_df = df.groupby('year_interval', as_index=False).size()
            time_df = time_df.rename(columns={'size': 'count'})

            # Carte interactive
            map_fig = px.scatter_geo(
                df.iloc[::10],
                lat='reclat',
                lon='reclong',
                color='fall',
                size='mass (g)',
                hover_name='name',
                projection='natural earth',
                title='Localisations des météorites'
            )

            # Histogramme des masses
            hist_fig = px.histogram(
                df[df['mass (g)'] > 0],
                x='mass (g)',
                nbins=50,
                log_x=True,
                title='Distribution des masses',
                color_discrete_sequence=['#2A9D8F']
            )

            # Histogramme temporel corrigé
            time_fig = px.bar(
                time_df,
                x='year_interval',
                y='count',
                title='Chutes par tranche de 5 ans',
                labels={'year_interval': 'Période', 'count': 'Nombre'},
                color_discrete_sequence=['#E76F51']
            ).update_xaxes(
                type='category',
                tickangle=45,
                title_text='Années'
            )

            # Top des classifications
            class_fig = px.bar(
                df['recclass'].value_counts().head(10).reset_index(),
                x='count',
                y='recclass',
                title='Top 10 des classifications',
                color='recclass'
            )

            # Options du dropdown
            class_options = [{'label': c, 'value': c} for c in df['recclass'].unique()]

            return map_fig, hist_fig, time_fig, class_fig, class_options

        except Exception as e:
            error_fig = px.scatter(title=f"Erreur: {str(e)}")
            return error_fig, error_fig, error_fig, error_fig, []