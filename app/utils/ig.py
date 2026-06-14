import tensorflow as tf
import numpy as np

def integrated_gradients(model, image, baseline=None, steps=32, class_idx=None):
    image = tf.cast(image, tf.float32)

    if baseline is None:
        # baseline = tf.zeros_like(image)
        # baseline = tf.ones_like(image) * tf.reduce_mean(image)
        baseline = tf.ones_like(image) * 127.0

    if class_idx is None:
        preds = model(image, training=False)
        class_idx = tf.argmax(preds[0])

    # interpolate images
    interpolated = [
        baseline + (float(i) / steps) * (image - baseline)
        for i in range(steps + 1)
    ]

    interpolated = tf.concat(interpolated, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)
        selected = preds[:, class_idx]

    grads = tape.gradient(selected, interpolated)

    avg_grads = tf.reduce_mean(
        (grads[:-1] + grads[1:]) / 2.0,
        axis=0
    )

    ig = (image - baseline) * avg_grads

    heatmap = tf.reduce_mean(
        tf.nn.relu(ig),
        axis=-1
    )

    heatmap = tf.squeeze(heatmap)

    heatmap = heatmap.numpy()

    p99 = np.percentile(heatmap, 99)

    heatmap = np.clip(
        heatmap / p99,
        0,
        1
    )
    return heatmap