import tensorflow as tf
import numpy as np

def make_saliency_map(
    model,
    image,
    pred_index=None
):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)

        predictions = model(image)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, image)

    saliency = tf.reduce_max(
        tf.abs(grads),
        axis=-1
    )[0]

    saliency = saliency.numpy()

    saliency = (
        saliency - saliency.min()
    ) / (
        saliency.max() - saliency.min() + 1e-8
    )

    return saliency