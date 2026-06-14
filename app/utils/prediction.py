import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from .gradcam import make_gradcam_heatmap
from .saliency import make_saliency_map
from .occlusion import occlusion_map
from .ig import integrated_gradients
from io import BytesIO
from PIL import Image
from scipy.ndimage import gaussian_filter
from .utils import find_last_conv_layer



def predict_and_visualize(
    model,
    image=None,
    url=None,
    class_names=None,
    method="gradcam",
    last_conv_layer_name=None,
    topk=5,
    alpha=0.3
):

    try:
        IMAGE_SIZE = 224

        last_conv_layer_name = "activation_3"
        # last_conv_layer_name = find_last_conv_layer(model).name

        if image is not None:

            if not isinstance(image, Image.Image):
                img_pil = Image.open(image).convert("RGB")
            else:
                img_pil = image

        else:
            if image is None:
                if not url:
                    raise ValueError(
                        "Either image or url must be provided"
                    )

                response = requests.get(url)
            response.raise_for_status()

            img_pil = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

        original_np = np.array(img_pil).astype(np.float32)
        original_viz = original_np / 255.0

        orig_h, orig_w = original_np.shape[:2]

        img_resized = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))

        img_np = np.array(img_resized).astype(np.float32)

        img_viz = img_np / 255.0

        img_input = np.expand_dims(img_np.copy(),axis=0)

        # -----------------------------
        # Prediction
        # -----------------------------
        preds = model.predict(
            img_input,
            verbose=0
        )

        if np.max(preds[0]) > 1:
            probs = tf.nn.softmax(preds[0]).numpy()
        else:
            probs = preds[0]

        prob_dict = {
            class_names[i]: float(probs[i])
            for i in range(len(probs))
        }

        top_idxs = np.argsort(probs)[::-1][:topk]
        top_probs = probs[top_idxs]

        topk_table = []

        for idx, prob in zip(top_idxs, top_probs):
            topk_table.append([
                class_names[idx] if class_names else str(idx),
                round(float(prob) * 100, 2)
            ])

        top1_idx = int(top_idxs[0])

        pred_class = (
            class_names[top1_idx]
            if class_names
            else str(top1_idx)
        )

        confidence = float(top_probs[0])

        # -----------------------------
        # Visualizer
        # -----------------------------
        if method == "gradcam":
            heatmap = make_gradcam_heatmap(
                model,
                img_input,
                last_conv_layer_name,
                pred_index=top1_idx
            )

        elif method == "saliency":
            heatmap = make_saliency_map(
                model,
                img_input,
                pred_index=top1_idx
            )

        elif method == "integrated":
            heatmap = integrated_gradients(
                model,
                img_input,
                class_idx=top1_idx
            )
        
        elif method == "occlusion":
            heatmap = occlusion_map(
                model,
                img_input,
                class_idx=top1_idx
            )

        # heatmap = gaussian_filter(heatmap,sigma=5)

        heatmap = np.clip(heatmap,0,1)

        # heatmap = cv2.resize(heatmap,(IMAGE_SIZE, IMAGE_SIZE))

        heatmap = cv2.resize(heatmap,(orig_w, orig_h))

        heatmap_rgb = plt.get_cmap("jet")(
            heatmap
        )[..., :3]

        # superimposed = (
        #     heatmap_rgb * alpha
        #     + img_viz * (1 - alpha)
        # )

        superimposed = (heatmap_rgb * alpha + original_viz * (1 - alpha))

        superimposed = np.clip(superimposed,0,1)
        superimposed = (superimposed * 255).astype(np.uint8)

        superimposed = Image.fromarray(superimposed)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(range(len(probs)),probs)

        if class_names:
            ax.set_xticks(
                range(len(class_names))
            )
            ax.set_xticklabels(
                class_names,
                rotation=45,
                ha="right"
            )

        ax.set_title(
            # "Class Probabilities"
            "Tỷ lệ theo lớp"
        )

        plt.tight_layout()

        return (
            superimposed,#.resize((224, 224)),
            pred_class,
            confidence,
            # f"{confidence * 100:.2f}%",
            topk_table,
            fig
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(
            f"Failed to process image: {e}"
        )

def visualize_all_classes_from_image(
    model,
    image,
    class_names,
    true_class=None,
    method="gradcam",
    last_conv_layer_name=None,
    alpha=0.3,
):

    IMAGE_SIZE = 224

    # Load image
    if isinstance(image, Image.Image):
        img_pil = image.convert("RGB")
    else:
        img_pil = Image.open(image).convert("RGB")

    # Original image for display
    original_np = np.array(img_pil).astype(np.float32)
    original_viz = original_np / 255.0

    orig_h, orig_w = original_np.shape[:2]

    # Model input
    img_resized = img_pil.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    img_np = np.array(img_resized).astype(np.float32)

    img_input = np.expand_dims(
        img_np,
        axis=0
    )

    # Prediction
    preds = model.predict(
        img_input,
        verbose=0
    )

    if np.max(preds[0]) > 1:
        probs = tf.nn.softmax(
            preds[0]
        ).numpy()
    else:
        probs = preds[0]

    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]

    num_classes = len(class_names)

    fig, axes = plt.subplots(
        1,
        num_classes + 1,
        figsize=(4 * (num_classes + 1), 4)
    )

    fig.suptitle(
        f"Method: {method.upper()}\n"
        f"True Class: {true_class}\n"
        f"Predicted Class: {pred_class} "
        f"({probs[pred_idx] * 100:.2f}%)",
        fontsize=14,
        y=1.05
    )

    # Original image
    axes[0].imshow(original_viz)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Generate explanation for each class
    for class_idx in range(num_classes):

        if method == "gradcam":

            heatmap = make_gradcam_heatmap(
                model,
                img_input,
                last_conv_layer_name,
                pred_index=class_idx
            )

        elif method == "saliency":

            heatmap = make_saliency_map(
                model,
                img_input,
                pred_index=class_idx
            )

        elif method == "integrated":

            heatmap = integrated_gradients(
                model,
                img_input,
                class_idx=class_idx
            )

        elif method == "occlusion":

            heatmap = occlusion_map(
                model,
                img_input,
                class_idx=class_idx
            )

        else:
            raise ValueError(
                f"Unknown method: {method}"
            )

        heatmap = np.clip(
            heatmap,
            0,
            1
        )

        heatmap = cv2.resize(
            heatmap,
            (orig_w, orig_h)
        )

        heatmap_rgb = plt.get_cmap(
            "jet"
        )(heatmap)[..., :3]

        overlay = (
            heatmap_rgb * alpha
            + original_viz * (1 - alpha)
        )

        overlay = np.clip(
            overlay,
            0,
            1
        )

        axes[class_idx + 1].imshow(
            overlay
        )

        title = (
            f"{class_names[class_idx]}\n"
            f"{probs[class_idx] * 100:.2f}%"
        )

        tags = []

        if true_class == class_names[class_idx]:
            tags.append("TRUE")

        if pred_idx == class_idx:
            tags.append("PRED")

        if tags:
            title += "\n[" + " | ".join(tags) + "]"

        axes[class_idx + 1].set_title(
            title,
            fontsize=10
        )

        axes[class_idx + 1].axis("off")

    plt.tight_layout()

    return fig