import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_apps.shared_styles import *
import copy

# https://stackoverflow.com/questions/62732631/how-to-collapsed-sidebar-in-dash-plotly-dash-bootstrap-components
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 1", href="/")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="/app2"),
                dbc.DropdownMenuItem("Page 3", href="/app2"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Brand",
    brand_href="#",
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
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dcc.Link("Explore", href="/"),
                dcc.Link("Compare", href='/compare'),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

sidebar_btn = dbc.Button(children = "<", outline=True, size = "sm", color="secondary", n_clicks =0, className="mr-1", id="btn_sidebar", style = SIDEBAR_BTN_STYLE)

def NamedSlider(name, **kwargs):
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


