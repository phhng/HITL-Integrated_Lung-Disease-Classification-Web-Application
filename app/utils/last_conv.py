from tensorflow.keras.layers import Conv2D

def find_last_conv_layer(model):
    """
    Finds the last Conv2D layer in a Keras model.
    Returns the layer itself.
    """
    last_conv = None

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            last_conv = layer

    if last_conv is None:
        raise ValueError("No Conv2D layer found in model")

    return last_conv