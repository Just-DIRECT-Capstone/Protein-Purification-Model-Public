"""imports"""
import copy
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_apps.shared_styles import *
from dash_apps.apps.myapp import app

# https://stackoverflow.com/questions/62732631/how-to-collapsed-sidebar-in-dash-plotly-dash-bootstrap-components
navbar = dbc.NavbarSimple(
    brand="Surrodash",
    brand_href="/",
    color="dark",
    dark=True,
    fluid=True,
    sticky='top'
)

sidebar = html.Div(
    [
        html.H2("Navigate", className="display-4"),
        html.Hr(),
        html.P(
            "tools to interact with the datasets and surrogate models", className="lead"
        ),
        dbc.Nav(
            [
                dcc.Link("Explore Dataset\n", href="/explore"),
                dcc.Link("Evaluate Models\n", href="/eval"),
                dcc.Link("Train Models\n", href="/train"),
                dcc.Link("Compare\n", href='/compare'),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.Img(src=app.get_asset_url('Surrodash_Logo.JPG'), 
                style={
                    'height' : '22%',
                    'width' : '66%',
                    'float' : 'middle',
                    'position' : 'relative',
                    'padding-top' : 0,
                    'padding-right' : 0
                }),
        html.H4("Check us out!"),
        html.A([
            html.Img(
                src=app.get_asset_url('github.png'),
                style={
                    'height' : '25%',
                    'width' : '90%',
                    'float' : 'middle',
                    'position' : 'relative',
                    'padding-top' : 0,
                    'padding-right' : 0
                })
            ], href='https://github.com/Just-DIRECT-Capstone/Protein-Purification-Model-Public'),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

sidebar_btn = dbc.Button(children = "<", outline=True, size = "sm", color="secondary",
        n_clicks =0, className="mr-1", id="btn_sidebar", style = SIDEBAR_BTN_STYLE)

def NamedSlider(name, **kwargs):
    """name slider"""
    other = kwargs.pop('other')

    if other['type'] == 'slider-input':
        slider = dcc.Slider(**copy.deepcopy(kwargs))
    else:
        slider = html.Div()

    kwargs['id']['type'] = kwargs['id']['type']+'-input'

    try:
        step = kwargs['step']
    except:
        step = None

    input_field = dbc.Input(id=kwargs['id'], type="number", value = kwargs['value'], step = step)

    try:
        label = dbc.Label(f"{name}\n ({other['units']})", width=8)
    except:
        label = dbc.Label(f"{name})", width=8)
    return html.Div(
        children=[
            dbc.FormGroup(
            [
                label,
                dbc.Col(input_field,width=4,
                ),
            ],
            row=True,
        ),
        slider,
        ],
    )


def generate_table(dataframe, max_rows=20):
    """generate table"""
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def collapse(child, bttn_label, id):
    """collapse"""
    return [
            dbc.Button(
                bttn_label,
                id=id+'-button',
                className="mb-3",
                color="primary",
            ),
            dbc.Collapse(
                child,
                id=id,
            ),
        ]

def train_setting(n_clicks):
    """train setting"""
    return dbc.Col(
            children=[
        dbc.InputGroup(
            [
                dbc.InputGroupAddon("Model Name", addon_type="prepend"),
                dbc.Input(
                    id={'type':'dynamic-model-name',
                        'index':n_clicks
                    },
                    value="mymodel_mydataset",debounce = True),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupAddon("Hidden Units", addon_type="prepend"),
                dbc.Input(
                    id={'type':'dynamic-hidden-units',
                        'index':n_clicks
                    },
                    value="[16,8,4]",debounce = True),
            ],
            className="mb-3",
        ),
         dbc.InputGroup(
            [
                dbc.InputGroupAddon("Model Output", addon_type="prepend"),
                dbc.Select(
                    id={'type':'dynamic-model-output',
                        'index':n_clicks
                    },
                    options=[
                        {"label": "deterministc", "value": 'D'},
                        {"label": "probabilistic", "value": 'P'},
                    ],
                    value = 'D',
                ),
            ],
            className="mb-3"
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupAddon("Learning Rate", addon_type="prepend"),
                dbc.Input(
                    id={'type':'dynamic-learning-rate',
                        'index':n_clicks
                    },
                    value="0.01", type = 'numeric',debounce = True),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupAddon("Epochs", addon_type="prepend"),
                dbc.Input(
                    id={'type':'dynamic-epochs',
                        'index':n_clicks
                    },
                    value="100", type = 'numeric',debounce = True),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupAddon("Cross Validation", addon_type="prepend"),
                dbc.Input(
                    id={'type':'dynamic-cv',
                        'index':n_clicks
                    },
                    value="1", type = 'numeric',debounce = True),
            ],
            className="mb-3",
        ),
        dbc.Button(children = "Save Model", outline=True, color="primary",
                className="mb-3", n_clicks = 0,
                    id={'type':'dynamic-start',
                        'index':n_clicks
                    },)
        ],width = 3)
