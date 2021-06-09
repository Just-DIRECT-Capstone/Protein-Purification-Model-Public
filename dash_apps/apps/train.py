import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from h5py._hl import dataset
from dash_apps.shared_styles import *
from dash_apps.shared_components import train_setting
from dash_apps.apps.myapp import app
import dash
import os
import plotly.express as px
from plotly.tools import mpl_to_plotly as pltwrap
import ast

import utils
import visualization.simple_data_vis as vis
import surrogate_models.nn_defs as engine
import tensorflow as tf

path = os.getcwd()
# get all data in the private data directory
data_files = [o[:-4] for o in sorted(os.listdir(os.path.join('sample_datasets')))]
# make a Dropdown Menu to select a dataset
dropdown_data = lambda pick: [dbc.DropdownMenuItem(m, id = m, active = True) if i is pick else dbc.DropdownMenuItem(m, id = m,  active = False) for i,m in enumerate(data_files)]

DATA = None
pDATA = None

# make a button for plots
model_btn = dbc.Button(children = "Add Model", outline=True, size = "lg", color="primary", className="mb-3", id="btn_model_train", n_clicks = 0)

# make a button for plots
start_btn = dbc.Button(children = "Start Training", outline=True, size = "lg", color="primary", className="mb-3", id="btn_start_train", n_clicks = 0)


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
            ])
        ]),
        dbc.Row([dbc.Col(html.H1(children=''), width = 9),]),
        dbc.Row(dbc.Col(model_btn)),
        dbc.Row(id = 'model-inputs', children = []),
        dbc.Row(dbc.Col(start_btn)),
        dbc.Row(id = 'fitness_train', children = []),
        dbc.Row([dbc.Col(html.H1(children=''), width = 9),]),
        dbc.Row(id = 'tcurve_train', children = []),

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

    global DATA, dataset
    dataset = data_files[new_pick]
    DATA = utils.load_data(path,dataset+'.csv')

    return dropdown_data(new_pick), dataset, []

# Takes the n-clicks of the add-chart button and the state of the container_eval children.
@app.callback(
   Output('model-inputs','children'),
   [Input('btn_model_train','n_clicks')],
   [State('model-inputs','children')]
)
#This function is triggered when the add-chart clicks changes. This function is not triggered by changes in the state of the container_eval. If children changes, state saves the change in the callback.
def display_graphs(n_clicks, div_children):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == 'model-inputs':
        div_children = []

    elif button_id == 'btn_model_train' and DATA is not None:
        div_children.append(train_setting(n_clicks))

    return div_children

inputs = ['dynamic-model-name',
'dynamic-hidden-units',
'dynamic-model-output',
'dynamic-learning-rate',
'dynamic-epochs',
'dynamic-cv']

# callback to make models and train
@app.callback(
    [Output('fitness_train','children'),
    Output('tcurve_train','children')],
    Input('btn_start_train','n_clicks'),
    [State('btn_model_train','n_clicks'),
    *[State({'type': i, 'index': ALL}, 'value') for i in inputs]]
)

def compile_and_train(n_start,n_models,*args):
    args = [list(i) for i in zip(*args)] # orgnaize by model
    children_fitness = []
    children_tcurve = []

    if DATA is not None:
        x = [*DATA.columns[:2],*DATA.columns[4:]]
        y = DATA.columns[2:4]
        data2split, validation = utils.chroma_train_test_split(DATA, test_size=0.20)

        global all_models, all_settings, all_histories
        all_models = []
        all_settings = []
        all_histories = []

        for arg in args:
            CV = ast.literal_eval(arg[5])
            out = arg[2]
            trains, tests = utils.data_pipeline([data2split,], x_data=x, y_data=y, cross_val = CV)
            model = engine.create_deterministic_nn(
                feature_names = x, 
                target_names = y,
                hidden_units = ast.literal_eval(arg[1]),
                name = arg[0],
                out = out
            )

            learning_rate = ast.literal_eval(arg[3])
            epochs = ast.literal_eval(arg[4])
            if out == 'D': loss = 'mean_squared_error'
            if out == 'P': loss = engine.negative_loglikelihood

            loss_weights = (1/trains[0][0][1].mean().div(trains[0][0][1].mean().max())).round(2).to_dict()
            hist = {}

            for i in range(CV):
                hist[utils.get_model_name(model,dataset)+'_'+str(i)] = engine.run_experiment(
                    model = model, 
                    loss = {y[0]:loss,y[1]:loss},
                    loss_weights = loss_weights,
                    optimizer = tf.keras.optimizers.Adam,
                    learning_rate = learning_rate,
                    num_epochs = epochs,
                    train_dataset = trains[0][i], 
                    test_dataset = tests[0][i],
                    verbose = 0,
                    log = 0
                    )

            sets = {'learning_rate' : learning_rate,
                        'epochs' : epochs,
                        'optimizer': 'Adam',
                        'loss_weights': loss_weights,
                        'dataset' : dataset}

            all_models.append(model)
            all_histories.append(hist)
            all_settings.append(sets)
            
            # figure out best run from histories
            i = 0
            bhist = {}
            bhist[utils.get_model_name(model,dataset)] = hist[utils.get_model_name(model,dataset)+'_'+str(i)]

            f = pltwrap(vis.scatter_hats([model],trains[0][i],tests[0][i],settings=sets, display_info=False))
            children_fitness.append(dcc.Graph(figure = f))

            f = pltwrap(vis.training_curves([model], y, sets, bhist, smoothing = int(sets['epochs']*0.1)))
            children_tcurve.append(dcc.Graph(figure = f))

    return children_fitness, children_tcurve



#     global MODEL
#     MODEL, settings = utils.load_model(os.path.join(path,'surrogate_models','saved_models',model_names[new_pick]))

#     if pDATA is not None:
#         f = pltwrap(vis.scatter_hats([MODEL],pDATA[0][0][0], settings=settings, display_info=False))
#     else:
#         f = {}
#     return dropdown_models(new_pick), model_names[new_pick],[],f
