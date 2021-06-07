import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash_apps.shared_styles import *
from dash_apps.apps.myapp import app
import dash, dash_table
import os
import plotly.express as px
from plotly.tools import mpl_to_plotly as pltwrap
import plotly.graph_objects as go

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
plot_btn = dbc.Button(children = "Add Graph", outline=True, size = "lg", color="primary", className="mb-3", id="btn_plot_eval", n_clicks = 0)



# layout all the components to be displayed
content = html.Div(
    [
        dbc.Row([
            dbc.Col(html.H1(children='Evaluate Surrogate Models'), width = 9),
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='compare model predictions to generated data'), width = 9),
            dbc.Col([
                dbc.DropdownMenu(
                    label = "Select a dataset",
                    children = dropdown_data(0),
                    right=False,
                    id = 'dd_data_eval'
                ),
                dbc.DropdownMenu(
                    label = "Select a pretrained model",
                    children = dropdown_models(0),
                    right=False,
                    id = 'dd_model_eval'
                ),            
            ])
        ]),

        dbc.Row(dbc.Col(plot_btn)),
        dbc.Row(id = 'container_eval', children = []),
        dbc.Row(id = 'container_analysis', children = []),

    ],
    id="page-content",
    style = CONTENT_STYLE
)

layout = html.Div(
    [
        content,
        html.Div(id='dummy-output4_eval'),
        html.Div(id='dummy-output_eval')
    ]
)

# callback to update the dataset
@app.callback(
    [Output("dd_data_eval", "children")],
    [Output("dd_data_eval", "label")],
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

    return dropdown_data(new_pick), data_files[new_pick]

# callback to update the model
@app.callback(
    [Output("dd_model_eval", "children")],
    [Output("dd_model_eval", "label")],
    [Output('dummy-output_eval','children')],
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

    global MODEL, settings
    MODEL, settings = utils.load_model(os.path.join(path,'surrogate_models','saved_models',model_names[new_pick]))

    return dropdown_models(new_pick), model_names[new_pick],[]

# Takes the n-clicks of the add-chart button and the state of the container_eval children.
@app.callback(
   Output('container_eval','children'),
   [Input('btn_plot_eval','n_clicks')],
   [State('container_eval','children')]
)
#This function is triggered when the add-chart clicks changes. This function is not triggered by changes in the state of the container_eval. If children changes, state saves the change in the callback.
def display_graphs(n_clicks, div_children):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == 'dummy-output_eval':
        div_children = []

    elif button_id == 'btn_plot_eval' and DATA is not None:
        new_child = dbc.Col(
            children=[
                dbc.Row([
                dbc.Col(dbc.RadioItems(
                    id={'type':'dynamic-choice_eval_eval',
                        'index':n_clicks
                    },
                    options=[{'label': 'Yield', 'value': 'yield'},
                            {'label':'Purity', 'value': 'purity'},
                            ],
                    value='yield',
                ), width = 2),
                dbc.Col(dcc.Input(
                    id={
                        'type': 'dynamic-var_eval',
                        'index': n_clicks
                    },
                    type="number", placeholder="samples",
                    min=1, max=int(len(DATA)/2), step=1,
                ),width = 9)
                ]),
                dcc.Graph(
                    id={
                        'type':'dynamic-graph_eval',
                        'index':n_clicks
                    },
                    figure = {}
                )
            ],
        width = 4)
        div_children.append(new_child)

    # elif button_id == 'dummy-output4_eval':
    #     for c,child in enumerate(div_children):
    #         try:
    #             for l,line in enumerate(child['props']['children'][1]['props']['figure']['data']):
    #                 old_data = div_children[c]['props']['children'][1]['props']['figure']['data'][l]['y']
    #                 var = div_children[c]['props']['children'][1]['props']['figure']['data'][l]['legendgroup']
    #                 div_children[c]['props']['children'][1]['props']['figure']['data'][l]['y'] = data[var].iloc[:len(old_data)].tolist()
    #         except:
    #             pass

    return div_children

# callback to update the graphs with the selected variables and graph types
@app.callback(
    Output({'type': 'dynamic-graph_eval', 'index': MATCH}, 'figure'),
    [Input(component_id={'type': 'dynamic-var_eval', 'index': MATCH}, component_property='value'),
     Input(component_id={'type': 'dynamic-choice_eval_eval', 'index': MATCH}, component_property='value')],
     State({'type': 'dynamic-graph_eval', 'index': MATCH}, 'figure')
)
def new_graph(var, chart_var, old_fig):
    ctx = dash.callback_context

    if ctx.triggered[0]["prop_id"] != '.':

        if len([var]) == 0:
            fig = old_fig
        else:
            f, ids = vis.scatter_hats([MODEL],pDATA[0][0][0],
                    settings=settings,n_points = var, display_info=False, plot = chart_var.lower(),index=True)
            fig = pltwrap(f)
            fig.update_traces(customdata=ids)
        return fig

    else:
        return old_fig

@app.callback(
    Output('container_analysis','children'),
    [Input(component_id={'type': 'dynamic-graph_eval', 'index': ALL}, component_property='selectedData')],
    [State('container_analysis','children')]
)
def display_analysis_graph(selectedData, div_children):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(button_id)
    if button_id == 'dummy-output_eval':
        div_children = []

    elif (selectedData is not None) and (DATA is not None):
        print(button_id)
        new_child = dbc.Col(
            children=[
                dbc.Row([
                dbc.Col(dbc.RadioItems(
                    id={'type':'dynamic-choice_analysis',
                        'index':len(div_children)
                    },
                    options=[{'label': 'Yield', 'value': 'yield'},
                            {'label':'Purity', 'value': 'purity'},{'label':'Histogram', 'value': 'count'}
                            ],
                    value='yield',
                ), width = 2),
                dbc.Col(dcc.Dropdown(
                    id={
                        'type': 'dynamic-var_analysis',
                        'index': len(div_children)
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
                        'type':'dynamic-graph_analysis',
                        'index':len(div_children)
                    },
                    figure={}
                )
            ],
        width = 4)
        div_children.append(new_child)

    return div_children

# callback to update the graphs with the selected variables and graph types
@app.callback(
    Output({'type': 'dynamic-graph_analysis', 'index': MATCH}, 'figure'),
    [Input(component_id={'type': 'dynamic-var_analysis', 'index': MATCH}, component_property='value'),
     Input(component_id={'type': 'dynamic-choice_analysis', 'index': MATCH}, component_property='value'),
     Input(component_id={'type': 'dynamic-graph_eval', 'index': MATCH}, component_property='selectedData')],
     State({'type': 'dynamic-graph_analysis', 'index': MATCH}, 'figure')
)
def new_graph(var, chart_var, selectedData, old_fig):
    ctx = dash.callback_context
    if selectedData is not None: ids = selectedData['points'][0]['customdata']
    else: ids = DATA.index

    if ctx.triggered[0]["prop_id"] != '.':

        if len([var]) == 0:
            fig = old_fig
        else:
            if chart_var == 'count':
                fig = px.histogram(DATA, x=var)
                fig.add_trace(px.histogram(DATA.iloc[ids], x=var, color = 'red'))
                fig.update_traces(opacity=0.75)
            else:
                fig = px.scatter(DATA, x=var, y=chart_var)

        return fig

    else:
        return old_fig