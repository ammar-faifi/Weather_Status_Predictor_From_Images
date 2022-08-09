"""
    Some utilities support  the ML and DL proccessing
"""

import os
from pathlib import Path

from PIL import Image
import numpy as np

CLASSES = {
    'sunny': 0,
    'foggy':1,
}
DATA_DIR = Path('../data')

def load_all_images(classes: list, pixels: int = 50) -> np.ndarray:
    """To read all images from `data` folder
    proccessing them then return them as array data

    Arguments:
        - `PIXELS`: int max size of image's hight & width

    return
        (images, labels)
        `images`: list is images as array
        `labels`: list is the corresponding classes int `images
    """

    images = []
    labels = []

    for class_ in classes:
        assert class_ in os.listdir('../data/'), "The class didn't match any folder"

        for file in os.listdir(DATA_DIR / class_):
            img = Image.open("../data/")
            # convert into gray and resize
            img = img.convert("L").resize((pixels, pixels))
            # scale out of 256 values
            img_array = np.asarray(img).flatten() / 255

            # append img_array
            images.append(images)
            labels.append()

    return images, labels
