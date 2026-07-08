import gradio as gr
from .utils.api import (
    AVAILABLE_MODELS,
    predict_ui,
    submit_feedback
)

with gr.Blocks(
    title="Hệ thống chẩn đoán bệnh phổi",
    theme=gr.themes.Ocean()
) as demo:

    gr.Markdown("# Hệ thống chẩn đoán bệnh phổi")

    with gr.Row():

        # -------------------------
        # Left: Inputs
        # -------------------------

        with gr.Column(scale=1):

            model_selector = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value="Kermany",
                label="Chọn mô hình"
            )

            image_input = gr.Image(
                type="pil",
                label="Tải hình ảnh"
            )

            visualize_method = gr.Dropdown(
                choices=[
                    ("Grad-CAM", "gradcam"),
                    ("Saliency Map", "saliency"),
                    ("Integrated Gradients", "integrated"),
                    ("Occlusion Map", "occlusion"),
                ],
                value="gradcam",
                label="Phương pháp giải thích"
            )

            predict_btn = gr.Button(
                "Chẩn đoán",
                variant="primary"
            )

        # -------------------------
        # Right: Results
        # -------------------------

        with gr.Column(scale=1):

            with gr.Tabs():

                with gr.Tab("Kết quả"):

                    prediction_output = gr.Textbox(
                        label="Chẩn đoán"
                    )

                    confidence_bar = gr.HTML(
                        label="Độ tin cậy"
                    )

                    # confidence_output = gr.Textbox(
                    #     label="Độ tin cậy chi tiết"
                    # )

                # with gr.Tab("Giải thích"):

                    heatmap_output = gr.Image(
                        label="Giải thích bằng hình ảnh"
                    )

                with gr.Tab("Phân tích"):

                    topk_output = gr.Dataframe(
                        headers=["Lớp", "Tỷ lệ (%)"],
                        label="Các chẩn đoán khác"
                    )

                    plot_output = gr.Plot(
                        label="Biểu đồ tỷ lệ"
                    )

            # -------------------------
            # Feedback
            # -------------------------

            with gr.Accordion(
                "Phản hồi",
                open=False
            ):

                default_model = "Kermany"

                default_choices = [
                    (display_name, class_id)
                    for class_id, display_name
                    in AVAILABLE_MODELS[
                        default_model
                    ]["classes"].items()
                ]

                feedback_class = gr.Dropdown(
                    choices=default_choices,
                    value=default_choices[0][1],
                    allow_custom_value=True,
                    label="Chẩn đoán đúng"
                )

                feedback_btn = gr.Button(
                    "Gửi phản hồi"
                )

                feedback_status = gr.Textbox(
                    label="Trạng thái phản hồi"
                )

            def update_feedback_choices(model_key):
                classes = AVAILABLE_MODELS[model_key]["classes"]

                choices = [
                    (display_name, class_id)
                    for class_id, display_name
                    in classes.items()
                ]

                return gr.Dropdown(
                    choices=choices,
                    value=choices[0][1]
                )

            model_selector.change(
                fn=update_feedback_choices,
                inputs=model_selector,
                outputs=feedback_class
            )
            
            predict_btn.click(
                fn=predict_ui,
                inputs=[
                    model_selector,
                    image_input,
                    visualize_method
                ],
                outputs=[
                    heatmap_output,
                    prediction_output,
                    confidence_bar,
                    topk_output,
                    plot_output
                ]
            )

            feedback_btn.click(
                fn=submit_feedback,
                inputs=[
                    image_input,
                    feedback_class
                ],
                outputs=feedback_status
            )

demo.launch()