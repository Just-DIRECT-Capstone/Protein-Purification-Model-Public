import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash_apps.shared_styles import *
from dash_apps.apps.myapp import app
import dash, dash_table
import os
import plotly.express as px

import utils

path = os.getcwd()
# get all data in the private data directory
data_files = [o[:-4] for o in sorted(os.listdir(os.path.join('sample_datasets')))]
# make a Dropdown Menu to select a dataset
dropdown_data = lambda pick: [dbc.DropdownMenuItem(m, id = m, active = True) if i is pick else dbc.DropdownMenuItem(m, id = m,  active = False) for i,m in enumerate(data_files)]

DATA = None
pDATA = None

# get all saved models
model_names = [o for o in sorted(os.listdir(os.path.join('surrogate_models','saved_models'))) if os.path.isdir(os.path.join('surrogate_models','saved_models',o))]
# make a Dropdown Menu to select a dataset
dropdown_models = lambda pick: [dbc.DropdownMenuItem(m, id = m, active = True) if i is pick else dbc.DropdownMenuItem(m, id = m,  active = False) for i,m in enumerate(model_names)]

MODEL = None

# make a button for plots
plot_btn = dbc.Button(children = "Add Graph", outline=True, size = "lg", color="primary", className="mb-3", id="btn_plot", n_clicks = 0)



# layout all the components to be displayed
content = html.Div(
    [
        dbc.Row([
            dbc.Col(html.H1(children='Explore Dataset'), width = 9),
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='visualize generated data based on mechanistic model'), width = 9),
            dbc.Col([
                dbc.DropdownMenu(
                    label = "Select a dataset",
                    children = dropdown_data(0),
                    right=False,
                    id = 'dd_data'
                ),           
            ])
        ]),
        dbc.Row([
            dbc.Col(html.H1(children=''), width = 9),
            dbc.Col(
                dash_table.DataTable(
                    id = 'data_table',
                    page_size=50,
                    style_header={'height': '50px','textAlign': 'center'},
                    style_table={'height': '500px','overflowX': 'auto','overflowY': 'auto'},
                    style_cell={'textAlign': 'left','overflow': 'hidden','textOverflow': 'ellipsis','maxWidth': 0,
                    'minWidth': '90px', 'width': '90px', 'maxWidth': '90px'},
                    fixed_rows={'headers': True},
                    tooltip_duration=None
                ),
            )
        ]),
        dbc.Row(dbc.Col(plot_btn)),
        dbc.Row(id = 'container', children = []),
    ],
    id="page-content",
    style = CONTENT_STYLE
)

layout = html.Div(
    [
        content,
        html.Div(id='dummy-output4'),
        html.Div(id='dummy-output-models')
    ]
)

# callback to update the dataset
@app.callback(
    [Output("dd_data", "children")],
    [Output("dd_data", "label")],
    [Output("data_table", "columns")],
    [Output("data_table", "data")],
    [Output("data_table", "tooltip_data")],
    [Output('dummy-output4','children')],
    [Input(m, "n_clicks") for m in data_files],
)
def update_dataset(*args):
    ctx = dash.callback_context
    # this gets the id of the button that triggered the callback
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        new_pick = data_files.index(button_id)
    except:
        new_pick = 0

    global DATA, pDATA
    DATA = utils.load_data(path,data_files[new_pick]+'.csv')
    pDATA = utils.data_pipeline([DATA,])

    columns = [{"name": i, "id": i} for i in DATA.columns]
    ttip = [{column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()} for row in DATA.to_dict('records')]
    return dropdown_data(new_pick), data_files[new_pick], columns, DATA.to_dict('records'), ttip, []

# Takes the n-clicks of the add-chart button and the state of the container children.
@app.callback(
   Output('container','children'),
   [Input('btn_plot','n_clicks'),
   Input('dummy-output4','children'),
   Input('dummy-output-models','children')],
   [State('container','children')]
)
#This function is triggered when the add-chart clicks changes. This function is not triggered by changes in the state of the container. If children changes, state saves the change in the callback.
def display_graphs(n_clicks, dummy,dummy_models, div_children):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'dummy-output-models':
        div_children = []

    elif button_id == 'btn_plot' and DATA is not None:
        new_child = dbc.Col(
            children=[
                dbc.Row([
                dbc.Col(dbc.RadioItems(
                    id={'type':'dynamic-choice',
                        'index':n_clicks
                    },
                    options=[{'label': 'Yield', 'value': 'yield'},
                            {'label':'Purity', 'value': 'purity'},{'label':'Histogram', 'value': 'count'}
                            ],
                    value='yield',
                ), width = 2),
                dbc.Col(dcc.Dropdown(
                    id={
                        'type': 'dynamic-dpn-var1',
                        'index': n_clicks
                    },
                    options=[{'label': var, 'value': var} for var in DATA.columns if var not in ['cut 1','cut 2','yield','purity']],
                    multi=False,
                    value = [],
                    placeholder='Select variable to plot...',
                    clearable=False
                ), width = 8)
                ]),
                dcc.Graph(
                    id={
                        'type':'dynamic-graph',
                        'index':n_clicks
                    },
                    figure={}
                )
            ],
        width = 4)
        div_children.append(new_child)

    return div_children

# callback to update the graphs with the selected variables and graph types
@app.callback(
    Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure'),
    [Input(component_id={'type': 'dynamic-dpn-var1', 'index': MATCH}, component_property='value'),
     Input(component_id={'type': 'dynamic-choice', 'index': MATCH}, component_property='value')],
     State({'type': 'dynamic-graph', 'index': MATCH}, 'figure')
)
def new_graph(var, chart_var, old_fig):
    ctx = dash.callback_context

    if ctx.triggered[0]["prop_id"] != '.':

        if len(var) == 0:
            fig = old_fig
        else:
            if chart_var == 'count':
                fig = px.histogram(DATA, x=var)
            else:
                fig = px.scatter(DATA, x=var, y=chart_var)

        return fig

    else:
        return old_fig