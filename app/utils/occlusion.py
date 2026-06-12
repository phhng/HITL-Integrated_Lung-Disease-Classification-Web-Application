import tensorflow as tf
import numpy as np

def occlusion_map(model, image, patch_size=64, stride=16, baseline=0.0, class_idx=None):
    """
    image: (1, H, W, 3)
    """

    image = tf.convert_to_tensor(image)
    _, H, W, C = image.shape

    # original prediction
    preds = model(image, training=False)
    if class_idx is None:
        class_idx = tf.argmax(preds[0]).numpy()

    orig_score = preds[0, class_idx].numpy()

    heatmap = np.zeros((H, W))

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):

            occluded = tf.identity(image).numpy()

            occluded[0, y:y+patch_size, x:x+patch_size, :] = baseline

            pred = model(tf.convert_to_tensor(occluded), training=False)
            score = pred[0, class_idx].numpy()

            drop = orig_score - score

            heatmap[y:y+patch_size, x:x+patch_size] += drop

    # normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap