"""
    Main entry of Dash app
"""

import io
import pickle
import base64

import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import dash_bootstrap_components as dbc
from PIL import Image
from sklearn.model_selection import RandomizedSearchCV
from keras.models import load_model, Sequential

from dash import Dash, dcc, html, Input, Output, State


ML_PIXELS = 50
CNN_PIXELS = 200
CLASSES = {
    "sunny": 0,
    "cloudy": 1,
    "foggy": 2,
    "rainy": 3,
    "snowy": 4,
}

# Load ML and DL models
with open("./code/tunned_xgb_random_result.pickle", "rb") as file:
    tunned_xgb_random_result: RandomizedSearchCV = pickle.load(file)

cnn_model: Sequential = load_model("./code/CNN")

# Setup Dash
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

row_content = [
    dbc.Col(html.Div("One of two columns"), width=4),
    dbc.Col(html.Div("One of two columns"), width=4),
]

row = html.Div(
    [
        dbc.Row(
            row_content,
            justify="center",
        ),
        dbc.Row(
            row_content,
            justify="end",
        ),
    ]
)

app.layout = dbc.Container(
    [
        # row,
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


def construct_html_image(image, filename):
    return html.Div(
        [
            html.H5("Original Image"),
            html.P(filename),
            dcc.Graph(figure=px.imshow(image)),
            html.Hr(),
        ]
    )


@app.callback(Output("tabs_content", "children"), Input("main_tabs", "value"))
def render_main_tabs(tab):
    """Callback to render main_tabs"""

    main_body = [
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
            [
                dbc.Col(width=5),
                dbc.Col(id="output_table", width=2),
                dbc.Col(width=5),
            ]
        ),
        html.H5("Processed Image"),
        html.Div(id="output_fig"),
    ]

    if tab == "binary_tab":
        return [html.H3("Binary Classification: sunny vs. cloudy")] + main_body

    return [html.H3("5-Class Predictor")] + main_body


table_header = [
    html.Thead(html.Tr([html.Th("Class"), html.Th("Probability")]))
]


@app.callback(
    Output("loading", "children"),
    Output("output_image", "children"),
    Output("output_table", "children"),
    Output("output_fig", "children"),
    Input("predict_btn1", "n_clicks"),
    State("upload1", "contents"),
    State("upload1", "filename"),
    State("main_tabs", "value"),
    running=[
        (Output("predict_btn1", "disabled"), True, False),
    ],
)
def upload_process_image(n_clicks, content, filename, tab):
    print(tab)

    if content is not None:
        # decode base64 image into IOByte
        text = content.removeprefix("data:image/jpeg;base64,")
        text = base64.b64decode(text)
        pil_img = Image.open(io.BytesIO(text))

        if tab == "binary_tab":
            img = (
                np.asarray(pil_img.convert("L").resize((ML_PIXELS, ML_PIXELS)))
                / 255
            )

            prob = tunned_xgb_random_result.predict_proba(
                [img.flatten()]
            ).flatten()
            prob = np.round(prob * 100, 1)

            row1 = html.Tr([html.Td("Sunny"), html.Td(str(prob[0]))])
            row2 = html.Tr([html.Td("Cloudy"), html.Td(str(prob[1]))])
            table = dbc.Table(
                table_header + [html.Tbody([row1, row2])], bordered=True
            )

            fig = px.imshow(np.asarray(img), color_continuous_scale="gray")

            return (
                None,
                construct_html_image(pil_img, filename),
                table,
                dcc.Graph(figure=fig),
            )

        if tab == "cnn_tab":
            img = (
                np.asarray(
                    pil_img.convert("RGB").resize((CNN_PIXELS, CNN_PIXELS))
                )
                / 255
            )
            prob = cnn_model.predict(
                img.reshape((1, CNN_PIXELS, CNN_PIXELS, 3))
            ).flatten()

            df = pd.DataFrame(
                {
                    "Class": CLASSES.keys(),
                    "Probability": [str(x) for x in np.round(prob * 100, 1).flatten()],
                },
            ).sort_values("Probability", ascending=False)
            print(df)

            fig = px.imshow(np.asarray(img))

            return (
                None,
                construct_html_image(pil_img, filename),
                dbc.Table.from_dataframe(
                    df, striped=True, bordered=True, hover=True,
                ),
                dcc.Graph(figure=fig),
            )

    return (None,) * 4


if __name__ == "__main__":
    app.run_server(debug=True)
