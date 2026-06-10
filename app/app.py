import gradio as gr
from utils.api import call_flask_api, AVAILABLE_MODELS, app as flask_app
from threading import Thread

def run_flask():
    flask_app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

Thread(target=run_flask, daemon=True).start()

# Wait a bit for Flask to start
import time
time.sleep(1)

model_selector = gr.Dropdown(
    choices=list(AVAILABLE_MODELS.keys()),
    value="Kermany",
    label="Chọn mô hình"
)

# Gradio Interface
iface = gr.Interface(
    fn=call_flask_api,
    inputs=[
        model_selector,
        gr.Image(type="pil", label="Tải hình ảnh"),
        gr.Textbox(label="Đường dẫn URL"),
    ],
    outputs=[
        gr.Image(type="pil", label="Chẩn đoán từ Grad-CAM"),
        gr.Textbox(label="Chẩn đoán"),
        gr.Textbox(label="Tỷ lệ"),
        gr.Image(type="pil", label="Chẩn đoán khác"),
    ],
    title="Hệ thống chẩn đoán bệnh viêm phổi bằng hình ảnh X-quang",
).launch()