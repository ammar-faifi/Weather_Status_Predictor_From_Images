"""
The main app entry
"""

import io
import json
import base64

import numpy as np
from PIL import Image
from flask import Flask, request
from keras.models import load_model, Sequential

CNN_PIXELS = 200
model: Sequential = load_model("./code/CNN")

# create the flask object
app = Flask(__name__)


@app.route("/")
def index():
    return "Index Page"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.json

    if data is None:
        return "Got None"

    text = base64.b64decode(data["image"])
    pil_img = Image.open(io.BytesIO(text))
    pil_img.show()

    img = (
        np.asarray(pil_img.convert("RGB").resize((CNN_PIXELS, CNN_PIXELS)))
        / 255
    )

    prediction = model.predict(img.reshape((1, CNN_PIXELS, CNN_PIXELS, 3)))
    return json.dumps({"result": prediction.flatten()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
