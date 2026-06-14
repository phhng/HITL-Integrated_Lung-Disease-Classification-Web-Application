import os
from keras import models
from tensorflow.keras.layers import Conv2D

def load_model(model_path, num_classes=2):
    model = models.load_model(
        model_path
    )
    return model

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, Conv2D):
            return layer

        if hasattr(layer, "layers"):
            result = find_last_conv_layer(layer)
            if result is not None:
                return result

    return None

import re

def normalize_class_name(class_name):
    class_name = class_name.strip().lower()

    class_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        class_name
    )

    class_name = re.sub(
        r"_+",
        "_",
        class_name
    ).strip("_")

    return class_name