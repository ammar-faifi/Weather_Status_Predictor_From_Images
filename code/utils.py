"""
    Some utilities support  the ML and DL proccessing
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

CLASSES = {
    "sunny": 0,
    "cloudy": 1,
    "foggy": 2,
    "rainy": 3,
    "snowy": 4,
}
DATA_DIR = Path("../data")


def get_class_name(class_number: int) -> str:
    """Find the class name from given number.

    Args:
        class_number (int): class number

    Raises:
        KeyError: if `class_number` was not found.

    Returns:
        str: the class' name
    """

    for key, val in CLASSES.items():
        if val == class_number:
            return key
    raise KeyError(f"`class_number = {class_number}` not found")


def load_all_images(
    classes: list, pixels: int = 50
) -> Tuple[List[np.ndarray], List[int]]:
    """To read all images from `data` folder
    proccessing them then return them as array data

    Arguments:
        - `PIXELS`: int max size of image's hight & width

    return
        (images, labels)
        `images`: list is images as array
        `labels`: list is the corresponding classes int `images`
    """

    images = []
    labels = []

    for class_ in classes:
        assert class_ in os.listdir(
            DATA_DIR
        ), "The class didn't match any folder"

        files = filter(
            lambda x: not x.startswith("._"), os.listdir(DATA_DIR / class_)
        )
        for file in files:
            img = Image.open(DATA_DIR / class_ / file)

            # convert into gray and resize
            img = img.convert("L").resize((pixels, pixels))
            # scale pixel values out of 256 values
            img_array = np.asarray(img).flatten() / 255

            # append `img_array` and its class number
            images.append(img_array)
            labels.append(CLASSES[class_])

    return images, labels


def load_all_images_3channel(
    classes: list, pixels: int = 50
) -> Tuple[List[np.ndarray], List[int]]:
    """Same as `loa_all_images` but keeps
    it as RGB and no flattenning.
    """

    images = []
    labels = []
    i = 0

    for class_ in classes:
        assert class_ in os.listdir(
            DATA_DIR
        ), "The class didn't match any folder"

        files = filter(
            lambda x: not x.startswith("._"), os.listdir(DATA_DIR / class_)
        )
        for file in files:
            i += 1
            sys.stdout.write(f"\r has processed {i}")
            sys.stdout.flush()

            img = Image.open(DATA_DIR / class_ / file)

            # convert into gray and resize
            img = img.convert("RGB").resize((pixels, pixels))
            # scale pixel values out of 256 values
            img_array = np.asarray(img) / 255

            # append `img_array` and its class number
            images.append(img_array)
            labels.append(CLASSES[class_])

    return images, labels


def predict_image(
    model, file: str, pixels: int = 50, show: bool = False
) -> str:
    """Return the prediction of the `file` image"""

    img = Image.open(file)

    # convert into gray and resize
    img = img.convert("L").resize((pixels, pixels))
    # scale pixel values out of 256 values
    img_array = np.asarray(img).flatten() / 255

    # predict
    pred = get_class_name(model.predict([img_array])[0])
    prob = model.predict_proba([img_array])[0]

    if show:
        ax = plt.gca()
        ax.imshow(
            img_array.reshape((pixels, pixels)), cmap=plt.get_cmap("gray")
        )
        ax.set_title(f"{prob[0]*100:0.1f}% sunny & {prob[1]*100:0.1f}% cloudy")

    return prob

def predict_image_3c(
    model, file: str, pixels: int = 50, show: bool = False
) -> str:
    """Return the prediction of the `file` image
    with 3 channels, for CNN."""

    img = Image.open(file)

    # convert into gray and resize
    img = img.convert("RGB").resize((pixels, pixels))
    # scale pixel values out of 256 values
    img_array = np.asarray(img) / 255
    # reashape to feed it in the CNN
    img_array = img_array.reshape((1, pixels, pixels, 3))

    return model.predict(img_array)
