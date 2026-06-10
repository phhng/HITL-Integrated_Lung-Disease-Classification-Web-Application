import os
import base64
from flask import Flask, request, jsonify, make_response
import requests
import json
import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
from .prediction import predict_and_visualize
import torch
from utils.last_conv import find_last_conv_layer
from utils.utils import load_model
from utils.grad_cam import GradCAM
import matplotlib.pyplot as plt
import traceback

app = Flask(__name__)
public_url = "http://127.0.0.1:5000"

IMAGE_SIZE = (224, 224)
current_dir = os.path.dirname(__file__)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

MODEL_CONFIGS = {
    "Kermany": {
        "filename": "resnet50_kermany.pth",
        "num_classes": 2,
        "class_names": ["Bình thường", "Viêm phổi"]
    },
    "Rahman": {
        "filename": "rahman_4class.pth",
        "num_classes": 4,
        "class_names": ["COVID-19", "Mờ phổi", "Bình thường", "Viêm phổi"]
    }
}

# Load all models ahead of time (only once)
AVAILABLE_MODELS = {}
    
for key, cfg in MODEL_CONFIGS.items():
    path = os.path.join(current_dir, "model", "weights", cfg["filename"])
    model, device = load_model(path, cfg["num_classes"])
    layer = find_last_conv_layer(model)
    cam = GradCAM(model, layer)

    AVAILABLE_MODELS[key] = {
        "model": model,
        "device": device,
        "grad_cam": cam,
        "class_names": cfg["class_names"],
        "num_classes": cfg["num_classes"]
    }

# Default active model
ACTIVE_MODEL_KEY = "cohen"

def decode_base64_image(base64_str):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

@app.route("/switch_model", methods=["POST"])
def switch_model():
    global ACTIVE_MODEL_KEY
    data = request.get_json()

    if not data or "model" not in data:
        return jsonify({"error": "Missing 'model' parameter"}), 400

    key = data["model"]

    if key not in AVAILABLE_MODELS:
        return jsonify({"error": f"Model '{key}' does not exist"}), 400

    ACTIVE_MODEL_KEY = key
    return jsonify({"message": f"Active model switched to {key}"}), 200


def call_flask_api( model_key=ACTIVE_MODEL_KEY, image_filepath=None, url=None):
    if model_key:
        try:
            requests.post(
                public_url + "/switch_model",
                json={"model": model_key}
            )
        except Exception as e:
            return None, f"Switch model failed: {e}", "", None
    try:
        # ----------------------------------------
        # 1. Validate input
        # ----------------------------------------
        if not image_filepath and not url:
            return "Error: Provide image_filepath or url.", None

        if image_filepath and url:
            return "Error: Provide only one input, not both.", None

        # ----------------------------------------
        # 2. Prepare request payload
        # ----------------------------------------
        payload = {}

        if image_filepath:  # image_filepath is actually a PIL Image
            try:
                buffered = io.BytesIO()
                image_filepath.save(buffered, format="PNG")  # save PIL Image to bytes
                img_bytes = buffered.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode()
                payload = {"image_data": img_b64}
            except Exception as e:
                return f"Error reading image: {e}", None

        # Image URL
        elif url:
            payload = {"image_url": url.strip()}

        # ----------------------------------------
        # 3. Make request
        # ----------------------------------------
        response = requests.post(
            public_url + "/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        response.raise_for_status()

        # ----------------------------------------
        # 4. Parse Response
        # ----------------------------------------
        try:
            response.raise_for_status()
            data = response.json()
        except json.JSONDecodeError:
            return "Error: Invalid JSON response from API.", None

        if "error" in data:
            return data["error"], None

        heatmap = decode_base64_image(data.get("heatmap"))
        fig_base64 = data.get("fig")
        
        # Optionally convert fig base64 to matplotlib figure
        fig_img = decode_base64_image(fig_base64)
        return heatmap, data.get("prediction"), data.get("confidence"), fig_img
    
    except Exception as e:
        return str(e), None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -----------------------------
        # 1. Validate request body
        # -----------------------------
        data = request.json if request.is_json else None
        image = None
        image_url = None

        # --- image from URL ---
        if data and "image_url" in data:
            image_url = data["image_url"]

        # --- image from base64 ---
        elif data and "image_data" in data:
            try:
                img_bytes = base64.b64decode(data["image_data"])
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400

        # --- image from file upload ---
        elif "file" in request.files:
            try:
                file = request.files["file"]
                image = Image.open(file).convert("RGB")
            except Exception as e:
                return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        else:
            return jsonify({"error": "Provide image_url, image_data, or file"}), 400

        # -----------------------------
        # 2. Make prediction
        # -----------------------------
        try:
            active = AVAILABLE_MODELS[ACTIVE_MODEL_KEY]

            heatmap, pred_class, confidence, fig = predict_and_visualize(
                active["model"],
                active["grad_cam"],
                active["device"],
                image=image,
                url=image_url,
                class_names=active["class_names"],
                topk=active["num_classes"]
            )
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        def pil_to_base64(img):
            if img is None:
                return None
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        def fig_to_base64(fig):
            if fig is None:
                return None
            buf = io.BytesIO()
            fig.savefig(buf, format="PNG")  # use savefig for matplotlib figures
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode()

        # -----------------------------
        # 3. Return JSON response
        # -----------------------------
        return jsonify({
            "prediction": pred_class,
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap": pil_to_base64(heatmap),
            "fig": fig_to_base64(fig)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
