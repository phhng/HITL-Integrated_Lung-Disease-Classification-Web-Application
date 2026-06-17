import os
import gradio as gr
import io
from .prediction import predict_and_visualize
from .utils import find_last_conv_layer, normalize_class_name, load_model
import matplotlib.pyplot as plt
import traceback
import uuid
from datetime import datetime
import csv 

IMAGE_SIZE = (224, 224)
current_dir = os.path.dirname(__file__)

DATASET_DIR = "dataset"
UNCERTAIN_DIR = os.path.join(
    "dataset",
    "uncertain"
)
UNCERTAIN_DIR = os.path.abspath(UNCERTAIN_DIR)
FEEDBACK_DIR = os.path.join(
    "dataset",
    "feedback"
)
FEEDBACK_DIR = os.path.abspath(FEEDBACK_DIR)
PENDING_DIR = os.path.join(
    "dataset",
    "pending"
)
PENDING_DIR = os.path.abspath(PENDING_DIR)
PENDING_FEEDBACK = None

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
            "covid-19": "COVID-19",
            "lung_opacity": "Mờ phổi",
            "normal": "Bình thường",
            "pneumonia": "Viêm phổi"
        }
    },
    "Rahman với lớp cắt": {
        "filename": "cov_masked_42.keras",
        "num_classes": 4,
        "classes": {
            "covid-19": "COVID-19",
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

def predict_ui(model_name,image,method,pending_path):
    global PENDING_FEEDBACK
    active = AVAILABLE_MODELS[model_name]

    save_pending_if_abandoned()

    heatmap, pred, conf, topk_table, fig = run_prediction(
        model_name,
        image,
        method
    )
    display_to_id = {
        v: k for k, v in active["classes"].items()
    }

    pred_key = display_to_id[pred]
    # pred_key = normalize_class_name(pred)

    PENDING_FEEDBACK = {
        "image": image,
        "prediction": pred_key,
        "confidence": conf
    }

    if conf < 0.70:
        save_prediction(
            image,
            pred_key,
            UNCERTAIN_DIR,
            conf * 100
        )

    return (
        heatmap,
        pred,
        # f"{conf*100:.2f}%",
        make_confidence_bar(conf * 100),
        topk_table,
        fig
    )


def submit_feedback(image,correct_class):
    global PENDING_FEEDBACK

    path = save_prediction(
        image,
        correct_class,
        FEEDBACK_DIR
    )
    PENDING_FEEDBACK = None
    # return f"Saved to {path}"
    return f"Lưu về {path}"

def save_prediction(image,pred_class,root_dir,confidence=None):
    class_dir = os.path.join(root_dir, pred_class)

    os.makedirs(class_dir, exist_ok=True)

    uid = uuid.uuid4().hex[:8]

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    if confidence is not None:
        filename += f"_conf{confidence:.1f}"

    filename += f"_{uid}.png"

    save_path = os.path.join(class_dir, filename)

    image.save(save_path)
    return save_path

def save_pending_if_abandoned():
    global PENDING_FEEDBACK

    if not PENDING_FEEDBACK:
        return

    required_keys = ["image", "prediction", "confidence"]

    if not all(k in PENDING_FEEDBACK for k in required_keys):
        return
    save_prediction(
        image=PENDING_FEEDBACK["image"],
        pred_class=PENDING_FEEDBACK["prediction"],
        confidence=PENDING_FEEDBACK["confidence"] * 100,
        root_dir=PENDING_DIR
    )

    PENDING_FEEDBACK = None