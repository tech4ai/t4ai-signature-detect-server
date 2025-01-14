import os

import gradio as gr
import pandas as pd
from PIL import Image

from detector import SignatureDetector
from metrics_storage import DATABASE_DIR, DATABASE_PATH
from inference.predictors import TritonClientPredictor


def create_gradio_interface():
    # Seleção do servidor Triton e configuração inicial
    TRITON_SERVER_URL = "grpc://t4ai-signature-detector-100881400340.us-central1.run.app/yolov8_ensemble"
    predictor = TritonClientPredictor(url=TRITON_SERVER_URL)
    detector = SignatureDetector(predictor)

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

    def process_image(image, conf_thres, iou_thres):
        if image is None:
            return None, None, None, None, None, None

        output_image, metrics = detector.detect(image, conf_thres, iou_thres)

        # Create plots data
        hist_data = pd.DataFrame({"Tempo (ms)": metrics["times"]})
        indices = range(
            metrics["start_index"], metrics["start_index"] + len(metrics["times"])
        )

        line_data = pd.DataFrame(
            {
                "Inferência": indices,
                "Tempo (ms)": metrics["times"],
                "Média": [metrics["avg_time"]] * len(metrics["times"]),
            }
        )

        # Criar plots
        hist_fig, line_fig = detector.create_plots(hist_data, line_data)

        return (
            output_image,
            gr.update(
                value=f"Total de Inferências: {metrics['total_inferences']}",
                container=True,
            ),
            hist_fig,
            line_fig,
            f"{metrics['avg_time']:.2f}",
            f"{metrics['times'][-1]:.2f}",
        )

    def process_folder(files_path, conf_thres, iou_thres):
        if not files_path:
            return None, None, None, None, None, None

        valid_extensions = [".jpg", ".jpeg", ".png"]
        image_files = [
            f for f in files_path if os.path.splitext(f.lower())[1] in valid_extensions
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
        gr.Markdown(
            """
            # Tech4Humans - Detector de Assinaturas
            
            Este sistema utiliza o modelo [**YOLOv8s**](https://huggingface.co/tech4humans/yolov8s-signature-detector), especialmente ajustado para a detecção de assinaturas manuscritas em imagens de documentos. 
           
            Com este detector, é possível identificar assinaturas em documentos digitais com elevada precisão em tempo real, sendo ideal para
            aplicações que envolvem validação, organização e processamento de documentos.
            
            ---
            """
        )

        with gr.Row(equal_height=True, elem_classes="main-container"):
            # Coluna da esquerda para controles e informações
            with gr.Column(scale=1):
                with gr.Tab("Imagem Única"):
                    input_image = gr.Image(
                        label="Faça o upload do seu documento", type="pil"
                    )
                    with gr.Row():
                        clear_single_btn = gr.ClearButton([input_image], value="Limpar")
                        detect_single_btn = gr.Button(
                            "Detectar", elem_classes="custom-button"
                        )

                with gr.Tab("Pasta de Imagens"):
                    input_folder = gr.File(
                        label="Faça o upload de uma pasta com imagens",
                        file_count="directory",
                        type="filepath",
                    )
                    with gr.Row():
                        clear_folder_btn = gr.ClearButton(
                            [input_folder], value="Limpar"
                        )
                        detect_folder_btn = gr.Button(
                            "Detectar", elem_classes="custom-button"
                        )

                with gr.Group():
                    confidence_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="Limiar de Confiança",
                        info="Ajuste a pontuação mínima de confiança necessária para detecção.",
                    )
                    iou_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Limiar de IoU",
                        info="Ajuste o limiar de Interseção sobre União para Non Maximum Suppression (NMS).",
                    )

            with gr.Column(scale=1):
                output_image = gr.Image(label="Resultados da Detecção")

                with gr.Accordion("Exemplos", open=True):
                    gr.Examples(
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
                    label="Total de Inferências", show_copy_button=True, container=True
                )
                hist_plot = gr.Plot(label="Distribuição dos Tempos", container=True)

            with gr.Column(scale=1):
                line_plot = gr.Plot(label="Histórico de Tempos", container=True)
                with gr.Row(elem_classes="metrics-row"):
                    avg_inference_time = gr.Textbox(
                        label="Tempo Médio de Inferência (ms)",
                        show_copy_button=True,
                        container=True,
                    )
                    last_inference_time = gr.Textbox(
                        label="Último Tempo de Inferência (ms)",
                        show_copy_button=True,
                        container=True,
                    )

        with gr.Row(elem_classes="container"):

            gr.Markdown(
                """
                ---
                ## Sobre o Projeto

                Este projeto utiliza o modelo YOLOv8s ajustado para detecção de assinaturas manuscritas em imagens de documentos. Ele foi treinado com dados provenientes dos conjuntos [Tobacco800](https://paperswithcode.com/dataset/tobacco-800) e [signatures-xc8up](https://universe.roboflow.com/roboflow-100/signatures-xc8up), passando por processos de pré-processamento e aumentação de dados.

                ### Principais Métricas:
                - **Precisão (Precision):** 94,74%
                - **Revocação (Recall):** 89,72%
                - **mAP@50:** 94,50%
                - **mAP@50-95:** 67,35%
                - **Tempo de Inferência (CPU):** 171,56 ms

                O processo completo de treinamento, ajuste de hiperparâmetros, e avaliação do modelo pode ser consultado em detalhes no repositório abaixo.

                [Leia o README completo no Hugging Face Models](https://huggingface.co/tech4humans/yolov8s-signature-detector)

                ---

                **Desenvolvido por [Tech4Humans](https://www.tech4h.com.br/)** | **Modelo:** [YOLOv8s](https://huggingface.co/tech4humans/yolov8s-signature-detector) | **Datasets:** [Tobacco800](https://paperswithcode.com/dataset/tobacco-800), [signatures-xc8up](https://universe.roboflow.com/roboflow-100/signatures-xc8up)
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
    if not os.path.exists(DATABASE_PATH):
        os.makedirs(DATABASE_DIR, exist_ok=True)

    iface = create_gradio_interface()
    iface.launch()
