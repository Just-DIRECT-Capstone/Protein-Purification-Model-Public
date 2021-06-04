# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "1rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_BTN_STYLE = {
    "position": "fixed",
    "top": "77rem",
    "left": "13rem",
    "bottom": "2rem",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
}

SIDEBAR_BTN_HIDEN = {
    "position": "fixed",
    "top": "77rem",
    "left": "1rem",
    "bottom": "2rem",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "5rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "5rem",
    "margin-right": "5rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}