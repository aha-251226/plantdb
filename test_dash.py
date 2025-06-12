# test_dash.py
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Test Dash App"),
    dbc.Button("Test Button", id="test-btn", color="primary"),
    html.Div(id="output")
])

@app.callback(
    Output("output", "children"),
    [Input("test-btn", "n_clicks")]
)
def test_callback(n_clicks):
    print(f"🔥 Button clicked! n_clicks: {n_clicks}")
    if n_clicks:
        return f"Button was clicked {n_clicks} times!"
    return "Click the button above"

if __name__ == "__main__":
    print("🚀 Starting test app...")
    app.run(debug=True, port=8051)