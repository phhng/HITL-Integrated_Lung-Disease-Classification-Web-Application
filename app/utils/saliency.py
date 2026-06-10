import tensorflow as tf
import numpy as np

def compute_saliency(model, image):
    """
    image shape: (1,H,W,C)
    """

    image = tf.convert_to_tensor(image)

    with tf.GradientTape() as tape:
        tape.watch(image)

        preds = model(image, training=False)

        class_idx = tf.argmax(preds[0])

        loss = preds[:, class_idx]

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
'''
usage

saliency = compute_saliency(
    model,
    np.expand_dims(img, 0)
)
'''