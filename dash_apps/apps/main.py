import dash_bootstrap_components as dbc
import dash_html_components as html
from dash_apps.shared_styles import *
from dash_apps.apps.myapp import app

# layout all the components to be displayed
content = html.Div(
    [
        dbc.Row([
            dbc.Col(html.H1(children='Surrodash'), width = 9),
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='by JUST a Capstone Project'), width = 9),
        ]),
        dbc.Row([
            dbc.Col(html.H1(children=''), width = 9),
        ]),
    ],
    id="page-content",
    style = CONTENT_STYLE
)

layout = html.Div(
    [
        content,
    ]
)