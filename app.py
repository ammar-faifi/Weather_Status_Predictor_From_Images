"""
    Main entry of Dash app
"""

import io
import time
import pickle
import base64

import keras
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from PIL import Image
from sklearn.model_selection import RandomizedSearchCV
from dash import Dash, dcc, html, Input, Output, State


ML_PIXELS = 50
CNN_PIXELS = 200

# Load ML and DL models
with open("./code/tunned_xgb_random_result.pickle", "rb") as file:
    tunned_xgb_random_result: RandomizedSearchCV = pickle.load(file)


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(
    __name__,
    # external_stylesheets=external_stylesheets,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server

upload_style = {
    "width": "100%",
    "height": "60px",
    "lineHeight": "60px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "10px",
    "textAlign": "center",
    "margin": "10px",
}


app.layout = dbc.Container(
    [
        html.H1("online weather status predictor from images".title()),
        html.P("To start select the way you want to predict."),
        # Loading indicator
        dcc.Loading(
            id="loading",
            type="default",
        ),
        # tabs
        dcc.Tabs(
            id="main_tabs",
            value="binary_tab",
            children=[
                dcc.Tab(label="Binary Predictor", value="binary_tab"),
                dcc.Tab(label="CNN Predictor", value="cnn_tab"),
            ],
        ),
        # tabs content
        html.Div(id="tabs_content"),
    ]
)


def construct_html_image(contents, filename):
    return html.Div(
        [
            html.H5(filename),
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html.Img(src=contents, style={"width": "50%"}),
            html.Hr(),
        ]
    )


@app.callback(Output("tabs_content", "children"), Input("main_tabs", "value"))
def render_main_tabs(tab):
    """Callback to render main_tabs"""

    if tab == "binary_tab":
        return [
            html.H3("Binary Classification: sunny vs. cloudy"),
            dcc.Upload(
                id="upload1",
                children=html.Div(
                    [
                        "Drag and Drop or ",
                        html.A("Select an Image"),
                    ]
                ),
                style=upload_style,
                accept="image/jpg,image/jpeg",
            ),
            html.Div(
                dbc.Button("Predict", id="predict_btn1", color="primary"),
                className="d-grid gap-2 col-4 mx-auto",
            ),
            html.Div(id="output_image", className="text-center"),
            dbc.Row(
                html.Div(id="output_table", className="text-center"),
                class_name="col-4",
            ),
        ]

    return html.Div([html.H3("5-Class Predictor")])


table_header = [
    html.Thead(html.Tr([html.Th("Class"), html.Th("Probability")]))
]


@app.callback(
    Output("output_image", "children"),
    Output("output_table", "children"),
    Output("loading", "children"),
    Input("predict_btn1", "n_clicks"),
    State("upload1", "contents"),
    State("upload1", "filename"),
    running=[
        (Output("predict_btn1", "disabled"), True, False),
    ],
)
def upload_image(n_clicks, content, filename):

    if content is not None:
        # decode base64 image into IOByte
        text = content.removeprefix("data:image/jpeg;base64,")
        text = base64.b64decode(text)
        img = (
            Image.open(io.BytesIO(text))
            .convert("L")
            .resize((ML_PIXELS, ML_PIXELS))
        )
        img_array = np.asarray(img).flatten() / 255
        prob = tunned_xgb_random_result.predict_proba([img_array]).flatten()
        prob = np.round(prob * 100, 1)

        row1 = html.Tr([html.Td("Sunny"), html.Td(str(prob[0]))])
        row2 = html.Tr([html.Td("Cloudy"), html.Td(str(prob[1]))])

        table = dbc.Table(
            table_header + [html.Tbody([row1, row2])], bordered=True
        )

        return construct_html_image(content, filename), table, None

    return (None,) * 3


if __name__ == "__main__":
    app.run_server(debug=True)
