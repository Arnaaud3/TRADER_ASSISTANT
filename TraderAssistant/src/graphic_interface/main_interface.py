import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from datetime import datetime
import random

# Définition de la classe pour le tableau de bord des statistiques de trading
class DashboardTraderStatistics:
    def __init__(self):
        self.csv_file = "trader_statistics.csv"
        self.df = self.load_initial_data()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.create_layout()
        self.create_callbacks()

    # Charger les données initiales pour les graphiques
    def load_initial_data(self):
        initial_data = {
            "Date": pd.date_range(start="2024-01-01", end="2024-05-24", freq="D"),
            "Trades": [random.randint(0, 10) for _ in range(0, 145)],
            "Trades_Won": [random.randint(0, 10) for _ in range(0, 145)],
            "Gains": [random.randint(0, 1000) for _ in range(0, 145)],
            "Losses": [random.randint(0, 500) for _ in range(0, 145)]
        }
        df = pd.DataFrame(initial_data)
        df.to_csv(self.csv_file, index=False)
        return df
    
    # Save the updated data to a CSV file
    def save_data(self):
        self.df.to_csv(self.csv_file, index=False)
        
    def display_initial_data(self):
        total_return_text = f"Rendement Total: {self.df['Gains'].sum() - self.df['Losses'].sum()} €"
        volatility_text = f"Volatilité: {self.df['Gains'].std():.2f} €"
        sharpe_ratio_text = f"Ratio de Sharpe: {(self.df['Gains'].mean() / self.df['Gains'].std()):.2f}"

        # Créer les graphiques initiaux
        trades_fig = px.line(self.df, x="Date", y="Trades", title="Nombre de Trades par Jour")
        gains_fig = px.line(self.df, x="Date", y="Gains", title="Gains par Jour")

        return total_return_text, volatility_text, sharpe_ratio_text, trades_fig, gains_fig
    
    # Créer la mise en page du tableau de bord
    def create_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Tableau de Bord des Statistiques de Trading"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    dbc.Button("Ajouter Info Trade du Jour", id="add-trade-info-button", color="primary", className="mt-2", outline=True)
                ]), width=4),
                dbc.Col(html.Div([
                    html.H4("Statistiques de Trading"),
                    html.P(id="output-total-return", children=f"{self.display_initial_data()[0]}"),
                    html.P(id="output-volatility", children=f"{self.display_initial_data()[1]}"),
                    html.P(id="output-sharpe-ratio", children=f"{self.display_initial_data()[2]}")
                ]), width=8)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="trades-per-day-chart",figure=self.display_initial_data()[3]), width=6),
                dbc.Col(dcc.Graph(id="gains-per-day-chart",figure=self.display_initial_data()[4]), width=6)
            ]),
            dbc.Modal([
                dbc.ModalHeader("Nouvelles Informations sur les Trades du Jour"),
                dbc.ModalBody([
                    dbc.Form([
                        dbc.CardGroup([
                            dbc.Label("Nombre de Trades Réalisés"),
                            dbc.Input(type="number", id="new-input-trades", placeholder="Entrer le nombre de trades réalisés")
                        ]),
                        dbc.CardGroup([
                            dbc.Label("Nombre de Trades Gagnés"),
                            dbc.Input(type="number", id="new-input-trades-won", placeholder="Entrer le nombre de trades gagnés")
                        ]),
                        dbc.CardGroup([
                            dbc.Label("Gains Réalisés (€)"),
                            dbc.Input(type="number", id="new-input-gains", placeholder="Entrer les gains réalisés")
                        ]),
                        dbc.CardGroup([
                            dbc.Label("Pertes (€)"),
                            dbc.Input(type="number", id="new-input-losses", placeholder="Entrer les pertes")
                        ]),
                    ]),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Annuler", id="close-modal-button", className="mr-auto"),
                    dbc.Button("Soumettre", id="submit-modal-button", color="primary")
                ]),
            ], id="modal"),
        ])

        # Callback pour ouvrir la fenêtre modale
    def create_callbacks(self):
        # Callback to open and close the modal
        @self.app.callback(
            Output("modal", "is_open"),
            [Input("add-trade-info-button", "n_clicks"),
            Input("close-modal-button", "n_clicks")],
            [State("modal", "is_open")]
        )
        
        def toggle_modal(add_clicks, close_clicks, is_open):
            if add_clicks or close_clicks:
                return not is_open
            return is_open

    # Callback pour mettre à jour les graphiques après avoir soumis de nouvelles informations sur les trades du jour
        @self.app.callback(
            [Output("output-total-return", "children"),
            Output("output-volatility", "children"),
            Output("output-sharpe-ratio", "children"),
            Output("trades-per-day-chart", "figure"),
            Output("gains-per-day-chart", "figure")],
            [Input("submit-modal-button", "n_clicks")],
            [State("new-input-trades", "value"),
            State("new-input-trades-won", "value"),
            State("new-input-gains", "value"),
            State("new-input-losses", "value")]
        )
        def update_statistics(n_clicks, new_trades, new_trades_won, new_gains, new_losses):
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate

            # Mettre à jour les données
            new_data = {
                "Date": [datetime.now().date()],
                "Trades": [new_trades],
                "Trades_Won": [new_trades_won],
                "Gains": [new_gains],
                "Losses": [new_losses]
            }
            new_df = pd.DataFrame(new_data)

            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.save_data()
            
            total_return_text = f"Rendement Total: {self.df['Gains'].sum() - self.df['Losses'].sum()} €"
            volatility_text = f"Volatilité: {self.df['Gains'].std():.2f} €"
            sharpe_ratio_text = f"Ratio de Sharpe: {(self.df['Gains'].mean() / self.df['Gains'].std()):.2f}"

            # Mettre a jour les graphiques
            trades_fig = px.line(self.df, x="Date", y="Trades", title="Nombre de Trades par Jour")
            gains_fig = px.line(self.df, x="Date", y="Gains", title="Gains par Jour")

            return total_return_text, volatility_text, sharpe_ratio_text, trades_fig, gains_fig
            
    # Exécuter l'application
    def run(self):
        self.app.run_server(debug=True)

# Créer une instance de DashboardTraderStatistics et exécuter l'application
dashboard = DashboardTraderStatistics()
dashboard.run()
