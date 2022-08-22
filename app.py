"""
    Main entry of Dash app
"""

import io
import time
import pickle
import base64

import keras
import numpy as np
from PIL import Image
from dash import Dash, dcc, html, Input, Output, State


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
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

app.layout = html.Div(
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
            html.Img(src=contents, style={'width': '50%'}),
            html.Hr(),
        ]
    )


@app.callback(Output("tabs_content", "children"), Input("main_tabs", "value"))
def render_main_tabs(tab):
    """Callback to render main_tabs"""

    if tab == "binary_tab":
        return html.Div(
            [
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
                    accept='image/png,image/jpg,image/jpeg'
                ),
                html.Button('Predict', id='predict_btn1'),
                html.Hr(),
                html.Div(id="output_image"),
            ]
        )

    return html.Div([html.H3("5-Class Predictor")])


@app.callback(
    Output("output_image", "children"),
    Output('loading', 'children'),
    Input('predict_btn1', 'n_clicks'),
    State("upload1", "contents"),
    State("upload1", "filename"),
    running=[
        (Output("predict_btn1", "disabled"), True, False),
    ],
)
def upload_image(n_clicks, content, filename):
    print(n_clicks)
    # time.sleep(1)

    if content is not None:
        # decode base64 imge into IOByte
        print(content[:100])

        text = content.removeprefix('data:image/jpeg;base64,')
        text = base64.b64decode(text)
        img = Image.open(io.BytesIO(text))

        return construct_html_image(content, filename), None

    return None, None


if __name__ == "__main__":
    app.run_server(debug=True)
