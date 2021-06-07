"""imports"""
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_apps.apps.myapp import app
from dash_apps.apps import main, explore, train, eval
from dash_apps.shared_styles import *
from dash_apps.shared_components import navbar, sidebar, sidebar_btn

# visit http://127.0.0.1:8050/ in your web browser.

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    sidebar,
    sidebar_btn,
    html.Div(id='page')
])

@app.callback(Output('page', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    """displays dash page"""
    if pathname == '/':
        return main.layout
    elif pathname == '/explore':
        return explore.layout
    elif pathname == '/eval':
        return eval.layout
    elif pathname == '/train':
        return train.layout
    else:
        return html.Div(dbc.Col(dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognized..."),
        ]
    ), width = 9), style = CONTENT_STYLE
)

if __name__ == '__main__':
    app.run_server(debug = True)
