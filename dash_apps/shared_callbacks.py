"""imports"""
from dash_apps.apps.myapp import app
from dash_apps.shared_styles import *
from dash.dependencies import Input, Output, State

# Change Content Style
@app.callback(
    [
        Output("page-content", "style"),
    ],
    [
        Input("btn_sidebar", "n_clicks"),
        Input("url", "pathname")
    ],
)
def change_style(n, pathname):
    """change style"""
    if n%2: # nav_var just closed
        content_style = CONTENT_STYLE1
    else:
        content_style = CONTENT_STYLE
    return [content_style]

# Toggle Sidebar
@app.callback(
    [
        Output("sidebar", "style"),
        Output("btn_sidebar", "style"),
        Output("btn_sidebar", "children"),
    ],
    [   Input("btn_sidebar", "n_clicks")],
)

def toggle_sidebar(n):
    """toggle sidebar"""
    if n%2: # nav_var just closed
        sidebar_style = SIDEBAR_HIDEN
        sidebar_btn_style = SIDEBAR_BTN_HIDEN
        sidebar_btn_children = ">"
    elif not n%2:
        sidebar_style = SIDEBAR_STYLE
        sidebar_btn_style = SIDEBAR_BTN_STYLE
        sidebar_btn_children = "<"

    return sidebar_style, sidebar_btn_style, sidebar_btn_children
