import os
import gradio as gr
import io
from .prediction import predict_and_visualize
from .utils import find_last_conv_layer, normalize_class_name, load_model
import matplotlib.pyplot as plt
import traceback
import uuid
from datetime import datetime

IMAGE_SIZE = (224, 224)
current_dir = os.path.dirname(__file__)

DATASET_DIR = "dataset"

MODEL_CONFIGS = {
    "Kermany": {
        "filename": "kermanymodel_42.keras",
        "num_classes": 2,
        "classes": {
            "normal": "Bình thường",
            "pneumonia": "Viêm phổi"
        }
    },

    "Rahman": {
        "filename": "cov_model_42.keras",
        "num_classes": 4,
        "classes": {
            "covid": "COVID-19",
            "lung_opacity": "Mờ phổi",
            "normal": "Bình thường",
            "pneumonia": "Viêm phổi"
        }
    },
    "Rahman với lớp cắt": {
        "filename": "cov_masked_42.keras",
        "num_classes": 4,
        "classes": {
            "covid": "COVID-19",
            "lung_opacity": "Mờ phổi",
            "normal": "Bình thường",
            "pneumonia": "Viêm phổi"
        }
    },
}

# Load all models ahead of time (only once)
AVAILABLE_MODELS = {}
    
for key, cfg in MODEL_CONFIGS.items():
    path = os.path.join(current_dir, ".." , "models", cfg["filename"])
    model = load_model(path, cfg["num_classes"])
    
    AVAILABLE_MODELS[key] = {
        "model": model,
        "class_ids": list(cfg["classes"].keys()),
        "class_names": list(cfg["classes"].values()),
        "classes": cfg["classes"]
    }

# Default active model
ACTIVE_MODEL_KEY = "Kermany"

def make_confidence_bar(conf):

    color = "#22c55e"

    if conf < 70:
        color = "#f59e0b"

    if conf < 50:
        color = "#ef4444"

    return f"""
    <div>
        <b>{conf:.2f}%</b>
        <div style="
            width:100%;
            height:22px;
            background:#eee;
            border-radius:12px;
            overflow:hidden;
        ">
            <div style="
                width:{conf}%;
                height:100%;
                background:{color};
            ">
            </div>
        </div>
    </div>
    """

def run_prediction(model_key,image,method):
    active = AVAILABLE_MODELS[model_key]

    return predict_and_visualize(
        model=active["model"],
        image=image,
        class_names=active["class_names"],
        method=method,
        topk=len(active["class_names"])
    )

def predict_ui(model_name,image,method):
    heatmap, pred, conf, topk_table, fig = run_prediction(
        model_name,
        image,
        method
    )

    return (
        heatmap,
        pred,
        # f"{conf*100:.2f}%",
        make_confidence_bar(conf * 100),
        topk_table,
        fig
    )


def save_feedback(image,class_name):
    class_name = normalize_class_name(class_name)
    class_dir = os.path.join(DATASET_DIR,class_name)

    os.makedirs(class_dir,exist_ok=True)
    filename = datetime.now().strftime("img_%y%m%d%H%M%S.png")
    save_path = os.path.join(class_dir,filename)

    image.save(save_path)

    return save_path

def submit_feedback(image,correct_class):
    path = save_feedback(
        image,
        correct_class
    )

    # return f"Saved to {path}"
    return f"Lưu về {path}"