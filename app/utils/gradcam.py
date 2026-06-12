import tensorflow as tf
import numpy as np
def make_gradcam_heatmap(
    model,
    image,
    last_conv_layer_name,
    pred_index=None
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.outputs[0]
        ]
    )
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    print("conv_outputs:",
        tf.reduce_min(conv_outputs).numpy(),
        tf.reduce_max(conv_outputs).numpy())

    print("grads:",
        tf.reduce_min(grads).numpy(),
        tf.reduce_max(grads).numpy())

    print("pooled_grads:",
        tf.reduce_min(pooled_grads).numpy(),
        tf.reduce_max(pooled_grads).numpy())

    print("grads mean:", tf.reduce_mean(grads).numpy())
    print("grads abs mean:", tf.reduce_mean(tf.abs(grads)).numpy())
    print("Before:", np.min(heatmap), np.max(heatmap))

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = heatmap.numpy()

    print("After:", heatmap.min(), heatmap.max())
    return heatmap