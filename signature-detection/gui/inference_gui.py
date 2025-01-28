import argparse
import os

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List

from detector import SignatureDetector
from inference.predictors import TritonClientPredictor
from metrics_storage import DATABASE_DIR, DATABASE_PATH

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Signature Detection GUI")
parser.add_argument(
    "--triton-url",
    type=str,
    default="grpc://localhost:8001/yolov8_ensemble",
    help="URL of the Triton server, e.g., grpc://localhost:8001/yolov8_ensemble",
)
args = parser.parse_args()

# Get the Triton server URL from the command-line arguments
TRITON_SERVER_URL = args.triton_url

if not os.path.exists(DATABASE_PATH):
    os.makedirs(DATABASE_DIR, exist_ok=True)

predictor = TritonClientPredictor(url=TRITON_SERVER_URL)
detector = SignatureDetector(predictor)


def create_gradio_interface():

    css = """
    .custom-button {
        background-color: #b0ffb8 !important;
        color: black !important;
    }
    .custom-button:hover {
        background-color: #b0ffb8b3 !important;
    }
    .container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-container {
        gap: 20px !important;
    }
    .metrics-container {
        padding: 1.5rem !important;
        border-radius: 0.75rem !important;
        background-color: #1f2937 !important;
        margin: 1rem 0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    .metrics-title {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #1f2937 !important;
        margin-bottom: 1rem !important;
    }
    .metrics-row {
        display: flex !important;
        gap: 1rem !important;
        margin-top: 0.5rem !important;
    }
    """
    example_dir = os.path.join(os.getcwd(), "signature-detection", "gui", "examples")

    def process_image(
        image: Image.Image, conf_thres: float, iou_thres: float
    ) -> Tuple[Image.Image, str, plt.Figure, plt.Figure, str, str]:
        if image is None:
            return None, None, None, None, None, None

        output_image, metrics = detector.detect(image, conf_thres, iou_thres)

        # Create plots data
        hist_data = pd.DataFrame({"Time (ms)": metrics["times"]})
        indices = range(
            metrics["start_index"], metrics["start_index"] + len(metrics["times"])
        )

        line_data = pd.DataFrame(
            {
                "Inference": indices,
                "Time (ms)": metrics["times"],
                "Mean": [metrics["avg_time"]] * len(metrics["times"]),
            }
        )

        hist_fig, line_fig = detector.create_plots(hist_data, line_data)

        return (
            output_image,
            gr.update(
                value=f"{metrics['total_inferences']}",
                container=True,
            ),
            hist_fig,
            line_fig,
            f"{metrics['avg_time']:.2f}",
            f"{metrics['times'][-1]:.2f}",
        )

    def process_folder(files_paths: List[str], conf_thres: float, iou_thres: float):
        if not files_paths:
            return None, None, None, None, None, None

        valid_extensions = [".jpg", ".jpeg", ".png"]
        image_files = [
            f for f in files_paths if os.path.splitext(f.lower())[1] in valid_extensions
        ]

        if not image_files:
            return None, None, None, None, None, None

        for img_file in image_files:
            image = Image.open(img_file)

            yield process_image(image, conf_thres, iou_thres)

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo", secondary_hue="gray", neutral_hue="gray"
        ),
        css=css,
    ) as iface:
        gr.HTML(
            """
            <h1>Tech4Humans - Signature Detector</h1>
    
            <div style="display: flex; align-items: center; gap: 10px;">
                <a href="https://huggingface.co/tech4humans/yolov8s-signature-detector">
                    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg" alt="Model on HF">
                </a>
                <a href="https://huggingface.co/datasets/tech4humans/signature-detection">
                    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" alt="Dataset on HF">
                </a>
                <a href="https://github.com/tech4ai/t4ai-signature-detect-server">
                    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
                </a>
            </div>
            """
        )
        gr.Markdown(
            """
            This system uses the [**YOLOv8s**](https://huggingface.co/tech4humans/yolov8s-signature-detector) model, specially fine-tuned for detecting handwritten signatures in document images.
           
            With this detector, it is possible to identify signatures in digital documents with high accuracy in real time, making it ideal for applications involving validation, organization, and document processing.
            
            ---
            """
        )

        with gr.Row(equal_height=True, elem_classes="main-container"):
            # Left column for controls and information
            with gr.Column(scale=1):
                with gr.Tab("Single Image"):
                    input_image = gr.Image(label="Upload your document", type="pil")
                    with gr.Row():
                        clear_single_btn = gr.ClearButton([input_image], value="Clear")
                        detect_single_btn = gr.Button(
                            "Detect", elem_classes="custom-button"
                        )

                with gr.Tab("Image Folder"):
                    input_folder = gr.File(
                        label="Upload a folder with images",
                        file_count="directory",
                        type="filepath",
                    )
                    with gr.Row():
                        clear_folder_btn = gr.ClearButton([input_folder], value="Clear")
                        detect_folder_btn = gr.Button(
                            "Detect", elem_classes="custom-button"
                        )

                with gr.Group():
                    confidence_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Adjust the minimum confidence score required for detection.",
                    )
                    iou_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="IoU Threshold",
                        info="Adjust the Intersection over Union threshold for Non-Maximum Suppression (NMS).",
                    )

            with gr.Column(scale=1):
                output_image = gr.Image(label="Detection Results")

                with gr.Accordion("ExExamplesemplos", open=True):
                    gr.Examples(
                        label="Image Examples",
                        examples=[
                            [f"{example_dir}/example_{i}.jpg".format(i=i)]
                            for i in range(
                                0,
                                len(os.listdir(example_dir)),
                            )
                        ],
                        inputs=input_image,
                        outputs=output_image,
                        fn=detector.detect_example,
                        cache_examples=True,
                        cache_mode="lazy",
                    )

        with gr.Row(elem_classes="metrics-container"):
            with gr.Column(scale=1):
                total_inferences = gr.Textbox(
                    label="Total Inferences", show_copy_button=True, container=True
                )
                hist_plot = gr.Plot(label="Time Distribution", container=True)

            with gr.Column(scale=1):
                line_plot = gr.Plot(label="Time History", container=True)
                with gr.Row(elem_classes="metrics-row"):
                    avg_inference_time = gr.Textbox(
                        label="Average Inference Time (ms)",
                        show_copy_button=True,
                        container=True,
                    )
                    last_inference_time = gr.Textbox(
                        label="Last Inference Time (ms)",
                        show_copy_button=True,
                        container=True,
                    )

        with gr.Row(elem_classes="container"):

            gr.Markdown(
                """
                ---
                ## About the Project

                This project uses the YOLOv8s model fine-tuned for detecting handwritten signatures in document images. It was trained with data from the [Tobacco800](https://paperswithcode.com/dataset/tobacco-800) and [signatures-xc8up](https://universe.roboflow.com/roboflow-100/signatures-xc8up) datasets, undergoing preprocessing and data augmentation processes.

                ### Key Metrics:
                - **Precision:** 94.74%
                - **Recall:** 89.72%
                - **mAP@50:** 94.50%
                - **mAP@50-95:** 67.35%
                - **Inference Time (CPU):** 171.56 ms

                Complete details on the training process, hyperparameter tuning, model evaluation, dataset creation, and inference server can be found in the links below.
                
                ---

                **Developed by [Tech4Humans](https://www.tech4h.com.br/)** | **Model:** [YOLOv8s](https://huggingface.co/tech4humans/yolov8s-signature-detector) | **Dataset:** [Tobacco800 + signatures-xc8up](https://huggingface.co/datasets/tech4humans/signature-detection)
                """
            )

        clear_single_btn.add([output_image])
        clear_folder_btn.add([output_image])

        detect_single_btn.click(
            fn=process_image,
            inputs=[input_image, confidence_threshold, iou_threshold],
            outputs=[
                output_image,
                total_inferences,
                hist_plot,
                line_plot,
                avg_inference_time,
                last_inference_time,
            ],
        )

        detect_folder_btn.click(
            fn=process_folder,
            inputs=[input_folder, confidence_threshold, iou_threshold],
            outputs=[
                output_image,
                total_inferences,
                hist_plot,
                line_plot,
                avg_inference_time,
                last_inference_time,
            ],
        )

        # Carregar métricas iniciais ao carregar a página
        iface.load(
            fn=detector.load_initial_metrics,
            inputs=None,
            outputs=[
                output_image,
                total_inferences,
                hist_plot,
                line_plot,
                avg_inference_time,
                last_inference_time,
            ],
        )

    return iface


if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch()
