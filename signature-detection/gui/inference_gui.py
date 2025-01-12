import pandas as pd
import matplotlib.pyplot as plt

import gradio as gr
import os
import sqlite3
import random
from PIL import Image, ImageDraw

from inference.inference_pipeline import InferencePipeline
from inference.predictors import TritonClientPredictor


class MetricsStorage:
    def __init__(self, db_path="metrics.db"):
        self.db_path = os.path.join(
            os.getcwd(), "signature-detection", "gui", "db", db_path
        )
        self.setup_database()

    def setup_database(self):
        """Initialize the SQLite database and create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inference_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def add_metric(self, inference_time):
        """Add a new inference time measurement to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO inference_metrics (inference_time) VALUES (?)",
                (inference_time,),
            )
            conn.commit()

    def get_recent_metrics(self, limit=50):
        """Get the most recent metrics from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT inference_time FROM inference_metrics ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            results = cursor.fetchall()
            return [r[0] for r in reversed(results)]

    def get_total_inferences(self):
        """Get the total number of inferences recorded"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inference_metrics")
            return cursor.fetchone()[0]

    def get_average_time(self, limit=50):
        """Get the average inference time from the most recent entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT AVG(inference_time) FROM (SELECT inference_time FROM inference_metrics ORDER BY timestamp DESC LIMIT ?)",
                (limit,),
            )
            result = cursor.fetchone()[0]
            return result if result is not None else 0


class SignatureDetector(InferencePipeline):
    def __init__(self, predictor):
        super().__init__(predictor)

        self.temp_path = os.path.join(
            os.getcwd(), "signature-detection", "gui", "tmp", "temp.jpg"
        )
        self.metrics_storage = MetricsStorage()

    def update_metrics(self, inference_time):
        """Update metrics in persistent storage"""
        self.metrics_storage.add_metric(inference_time)

    def get_metrics(self):
        """Get current metrics from storage"""
        times = self.metrics_storage.get_recent_metrics()
        total = self.metrics_storage.get_total_inferences()
        avg = self.metrics_storage.get_average_time()

        start_index = max(0, total - len(times))

        return {
            "times": times,
            "total_inferences": total,
            "avg_time": avg,
            "start_index": start_index,  # Adicionar índice inicial
        }

    def load_initial_metrics(self):
        """Load initial metrics for display"""
        metrics = self.get_metrics()

        if not metrics["times"]:  # Se não houver dados
            return None, None, None, None

        # Criar plots data
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
        hist_fig, line_fig = self.create_plots(hist_data, line_data)

        return (
            None,
            f"Total de Inferências: {metrics['total_inferences']}",
            hist_fig,
            line_fig,
        )

    def create_plots(self, hist_data, line_data):
        """Helper method to create plots"""
        plt.style.use("dark_background")

        # Histograma
        hist_fig, hist_ax = plt.subplots(figsize=(8, 4), facecolor="#f0f0f5")
        hist_ax.set_facecolor("#f0f0f5")
        hist_data.hist(
            bins=20, ax=hist_ax, color="#4F46E5", alpha=0.7, edgecolor="white"
        )
        hist_ax.set_title(
            "Distribuição dos Tempos de Inferência",
            pad=15,
            fontsize=12,
            color="#1f2937",
        )
        hist_ax.set_xlabel("Tempo (ms)", color="#374151")
        hist_ax.set_ylabel("Frequência", color="#374151")
        hist_ax.tick_params(colors="#4b5563")
        hist_ax.grid(True, linestyle="--", alpha=0.3)

        # Gráfico de linha
        line_fig, line_ax = plt.subplots(figsize=(8, 4), facecolor="#f0f0f5")
        line_ax.set_facecolor("#f0f0f5")
        line_data.plot(
            x="Inferência",
            y="Tempo (ms)",
            ax=line_ax,
            color="#4F46E5",
            alpha=0.7,
            label="Tempo",
        )
        line_data.plot(
            x="Inferência",
            y="Média",
            ax=line_ax,
            color="#DC2626",
            linestyle="--",
            label="Média",
        )
        line_ax.set_title(
            "Tempo de Inferência por Execução", pad=15, fontsize=12, color="#1f2937"
        )
        line_ax.set_xlabel("Número da Inferência", color="#374151")
        line_ax.set_ylabel("Tempo (ms)", color="#374151")
        line_ax.tick_params(colors="#4b5563")
        line_ax.grid(True, linestyle="--", alpha=0.3)
        line_ax.legend(frameon=True, facecolor="#f0f0f5", edgecolor="none")

        hist_fig.tight_layout()
        line_fig.tight_layout()

        # Fechar as figuras para liberar memória
        plt.close(hist_fig)
        plt.close(line_fig)

        return hist_fig, line_fig

    def detect(self, image, conf_thres=0.25, iou_thres=0.5):
        # Salvar imagem temporariamente
        image.save(self.temp_path)

        response = self.run(self.temp_path, conf=conf_thres, iou=iou_thres)

        output_image = self.draw_result(self.temp_path, response["result"])

        self.update_metrics(response["inference_time"] * 1000)

        return output_image, self.get_metrics()

    def detect_example(self, image, conf_thres=0.25, iou_thres=0.5):
        """Wrapper method for examples that returns only the image"""
        output_image, _ = self.detect(image, conf_thres, iou_thres)
        return output_image


def create_gradio_interface():
    # Seleção do servidor Triton e configuração inicial
    TRITON_SERVER_URL = "grpc://localhost:8001/yolov8_ensemble"
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
    """
    example_dir = os.path.join(os.getcwd(), "signature-detection", "gui", "examples")

    def process_image(image, conf_thres, iou_thres):
        if image is None:
            return None, None, None, None

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
        )

    def process_folder(files_path, conf_thres, iou_thres):
        if not files_path:
            return None, None, None, None

        valid_extensions = [".jpg", ".jpeg", ".png"]
        image_files = [
            f for f in files_path if os.path.splitext(f.lower())[1] in valid_extensions
        ]

        if not image_files:
            return None, None, None, None

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
            outputs=[output_image, total_inferences, hist_plot, line_plot],
        )

        detect_folder_btn.click(
            fn=process_folder,
            inputs=[input_folder, confidence_threshold, iou_threshold],
            outputs=[output_image, total_inferences, hist_plot, line_plot],
        )

        # Carregar métricas iniciais ao carregar a página
        iface.load(
            fn=detector.load_initial_metrics,
            inputs=None,
            outputs=[output_image, total_inferences, hist_plot, line_plot],
        )

    return iface


if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch()
