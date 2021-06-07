import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash_apps.shared_styles import *
from dash_apps.apps.myapp import app
import dash
import os
import plotly.express as px
from plotly.tools import mpl_to_plotly as pltwrap

import utils
import visualization.simple_data_vis as vis

path = os.getcwd()
# get all data in the private data directory
data_files = [o[:-4] for o in sorted(os.listdir(os.path.join('just-private','data')))]
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
plot_btn = dbc.Button(children = "Add Graph", outline=True, size = "lg", color="primary", className="mb-3", id="btn_plot_train", n_clicks = 0)



# layout all the components to be displayed
content = html.Div(
    [
        dbc.Row([
            dbc.Col(html.H1(children='Train Surrogate Models'), width = 9),
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='define network topology and training schedule'), width = 9),
            dbc.Col([
                dbc.DropdownMenu(
                    label = "Select a dataset",
                    children = dropdown_data(0),
                    right=False,
                    id = 'dd_data_train'
                ),
                dbc.DropdownMenu(
                    label = "Select a pretrained model",
                    children = dropdown_models(0),
                    right=False,
                    id = 'dd_model_train'
                ),            
            ])
        ]),
        dbc.Row([
            dbc.Col(html.H1(children=''), width = 9),
        ]),
        # dbc.Row([
        #     dbc.Col(dbc.Col([
        #         dbc.Row(dsc.collapse([], 'Manipulated Variables', 'mvars-collapse')),
        #         dbc.Row(dsc.collapse([], 'Control Variables', 'cvars-collapse')),
        #     ],
        #         id = 'slidersL'),width = 2),

        #     dbc.Col([],id = 'diagram1', width = 6),

        #     dbc.Col([
        #         dbc.Row(dsc.collapse([], 'Model Parameters', 'mparams-collapse')),
        #         dbc.Row(dsc.collapse([], 'Simulation Settings', 'sparams-collapse')),
        #     ],
        #         id = 'slidersR', width = 4),
        # ]),

        # dbc.Row(dbc.Col(run_btn)),
        dbc.Row(dbc.Col(plot_btn)),
        dbc.Row(dcc.Graph(
                    id='mygraph_train',
                    figure={}
                )
        ),
        dbc.Row(id = 'container', children = []),
    ],
    id="page-content",
    style = CONTENT_STYLE
)

layout = html.Div(
    [
        content,
        html.Div(id='dummy-output4_train'),
        html.Div(id='dummy-output_train')
    ]
)

# callback to update the dataset
@app.callback(
    [Output("dd_data_train", "children")],
    [Output("dd_data_train", "label")],
    [Output('dummy-output4_train','children')],
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

    return dropdown_data(new_pick), data_files[new_pick], []

# callback to update the model
@app.callback(
    [Output("dd_model_train", "children")],
    [Output("dd_model_train", "label")],
    [Output('dummy-output_train','children')],
    [Output('mygraph_train','figure')],
    [Input(m, "n_clicks") for m in model_names],
)
def update_model(*args):
    ctx = dash.callback_context
    # this gets the id of the button that triggered the callback
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    try:
        new_pick = model_names.index(button_id)
    except:
        new_pick = 0

    global MODEL
    MODEL, settings = utils.load_model(os.path.join(path,'surrogate_models','saved_models',model_names[new_pick]))

    if pDATA is not None:
        f = pltwrap(vis.scatter_hats([MODEL],pDATA[0][0][0], settings=settings, display_info=False))
    else:
        f = {}
    return dropdown_models(new_pick), model_names[new_pick],[],f
