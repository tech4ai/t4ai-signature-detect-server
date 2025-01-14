import os

import matplotlib.pyplot as plt
import pandas as pd

from metrics_storage import MetricsStorage

from inference.inference_pipeline import InferencePipeline


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
            return None, None, None, None, None, None

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
            f"{metrics['avg_time']:.2f}",
            f"{metrics['times'][-1]:.2f}",
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
