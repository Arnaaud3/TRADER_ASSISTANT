import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from datetime import datetime

# Fonction pour convertir la date au format "dd/mm/yyyy" en un objet datetime
def parse_date(date_string):
    try:
        return datetime.strptime(date_string, '%d/%m/%Y')
    except ValueError:
        return None
    
# Données initiales pour les graphiques
initial_data = {
    "Date": pd.date_range(start="2023-01-01", periods=30, freq="D"),
    "Trades": [1, 5, 2, 6, 8, 10, 3, 7, 6, 9, 4, 5, 3, 6, 8, 7, 5, 9, 6, 7, 5, 4, 6, 8, 5, 6, 7, 4, 8, 6],
    "Trades_Lost": [0, 1, 0, 2, 1, 3, 1, 2, 1, 2, 1, 1, 1, 2, 1, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2],
    "Trades_Won": [1, 4, 2, 4, 7, 7, 2, 5, 5, 7, 3, 4, 2, 4, 7, 4, 3, 7, 5, 5, 4, 3, 5, 6, 4, 4, 6, 3, 7, 4],
    "Gains": [100, 200, 150, 300, 250, 350, 100, 200, 250, 300, 150, 200, 100, 250, 300, 200, 150, 300, 250, 200, 150, 100, 200, 250, 150, 200, 250, 100, 300, 250],
    "Losses": [0, 50, 0, 100, 50, 150, 50, 100, 50, 100, 50, 50, 50, 100, 50, 150, 100, 100, 50, 100, 50, 50, 50, 100, 50, 100, 50, 50, 50, 100]
}

df = pd.DataFrame(initial_data)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Tableau de Bord des Statistiques de Trading"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dbc.Form([
                dbc.Row([
                    dbc.Col(dbc.Label("Date (dd-mm-yyyy)"), width=4),
                    dbc.Col(dbc.Input(id="input-date", type="text", placeholder="Entrer la date"), width=8)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Label("Nombre de Trades Réalisés"), width=4),
                    dbc.Col(dbc.Input(id="input-trades", type="number", placeholder="Entrer le nombre de trades réalisés"), width=8)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Label("Nombre de Trades Perdus"), width=4),
                    dbc.Col(dbc.Input(id="input-trades-lost", type="number", placeholder="Entrer le nombre de trades perdus"), width=8)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Label("Nombre de Trades Gagnés"), width=4),
                    dbc.Col(dbc.Input(id="input-trades-won", type="number", placeholder="Entrer le nombre de trades gagnés"), width=8)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Label("Gains Réalisés €"), width=4),
                    dbc.Col(dbc.Input(id="input-gains", type="number", placeholder="Entrer les gains réalisés"), width=8)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Label("Pertes €"), width=4),
                    dbc.Col(dbc.Input(id="input-losses", type="number", placeholder="Entrer les pertes"), width=8)
                ], className="mb-3"),
                dbc.Button("Soumettre", id="submit-button", color="primary", className="mt-2")
            ])
        ]), width=4),
        dbc.Col(html.Div([
            html.H4("Statistiques de Trading"),
            html.P(id="output-total-return", children="Rendement Total: "),
            html.P(id="output-volatility", children="Volatilité: "),
            html.P(id="output-sharpe-ratio", children="Ratio de Sharpe: "),
        ]), width=8)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="trades-per-day-chart"), width=6),
        dbc.Col(dcc.Graph(id="gains-per-day-chart"), width=6)
    ])
])

@app.callback(
    [Output("output-total-return", "children"),
     Output("output-volatility", "children"),
     Output("output-sharpe-ratio", "children"),
     Output("trades-per-day-chart", "figure"),
     Output("gains-per-day-chart", "figure")],
    [Input("submit-button", "n_clicks")],
    [State("input-date", "value"),
     State("input-trades", "value"),
     State("input-trades-lost", "value"),
     State("input-trades-won", "value"),
     State("input-gains", "value"),
     State("input-losses", "value")]
)
def update_statistics(n_clicks, date, trades, trades_lost, trades_won, gains, losses):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Convertir la date au format "dd/mm/yyyy" en objet datetime
    parsed_date = parse_date(date)
    if parsed_date is None:
        raise dash.exceptions.PreventUpdate
    # Mettre à jour les données
    new_data = {
        "Date": [parsed_date],
        "Trades": [trades],
        "Trades_Lost": [trades_lost],
        "Trades_Won": [trades_won],
        "Gains": [gains],
        "Losses": [losses]
    }
    new_df = pd.DataFrame(new_data)
    
    global df
    df = pd.concat([df, new_df], ignore_index=True)
    
    total_return_text = f"Rendement Total: {df['Gains'].sum() - df['Losses'].sum()} €"
    volatility_text = f"Volatilité: {df['Gains'].std():.2f} €"
    sharpe_ratio_text = f"Ratio de Sharpe: {(df['Gains'].mean() / df['Gains'].std()):.2f}"

    # Mettre à jour les graphiques
    trades_fig = px.line(df, x="Date", y="Trades", title="Nombre de Trades par Jour")
    gains_fig = px.line(df, x="Date", y="Gains", title="Gains par Jour")

    return total_return_text, volatility_text, sharpe_ratio_text, trades_fig, gains_fig

if __name__ == "__main__":
    app.run_server(debug=True)
