import dash_bootstrap_components as dbc
import dash_core_components as dcc
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
            html.Hr(),
            dcc.Markdown('''
                
Many biotechnology companies have moved from experimental probes to simulated **mechanistic models to identify the molecular and process parameters that give the highest yield and purity** for affordable manufacturing of biologic targets.
While these mechanistic models have sped up the process of molecular and process design compared to bench-top science, **these models can be slow and computationally expensive** to consistently run.
Our team has developed a **surrogate modeling Python package** that cleans data generated from Just--Evotec Biologics’ mechanistic model and trains ML models to **predict yield and purity for a set of molecular interaction parameters and operating conditions in a faster and less computationally expensive way**.
This package loads our pretrained models, explores the input data, provides visualizations and comparisons of the model accuracy, and validates the training process using an interactive Dash app in order to speed the prediction of ideal molecular and process design parameters.

            ''',style={
                        'padding-left' : 25,
                        'padding-right' : 50,
                        'text-align':'justify'
                    }),
            html.Img(src=app.get_asset_url('intro.png'), 
                    style={
                        'height' : '22%',
                        'width' : '66%',
                        'float' : 'middle',
                        'position' : 'relative',
                        'padding-top' : 10,
                        'padding-left' : 400
                    }),
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