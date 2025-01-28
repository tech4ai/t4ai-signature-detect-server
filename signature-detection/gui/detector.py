import os

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pandas import DataFrame
from typing import Tuple

from inference.inference_pipeline import InferencePipeline, BasePredictor
from metrics_storage import MetricsStorage


class SignatureDetector(InferencePipeline):
    def __init__(self, predictor: BasePredictor):
        """
        Initializes the SignatureDetector with a predictor.

        Args:
            predictor: The predictor object for inference.
        """
        super().__init__(predictor)

        self.temp_dir = os.path.join(os.getcwd(), "signature-detection", "gui", "tmp")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.temp_path = os.path.join(self.temp_dir, "temp.jpg")

        self.metrics_storage = MetricsStorage()

    def update_metrics(self, inference_time: float) -> None:
        """
        Updates metrics in persistent storage.

        Args:
            inference_time (float): The time taken for inference in milliseconds.
        """
        self.metrics_storage.add_metric(inference_time)

    def get_metrics(self) -> dict:
        """
        Retrieves current metrics from storage.

        Returns:
            dict: A dictionary containing times, total inferences, average time, and start index.
        """
        times = self.metrics_storage.get_recent_metrics()
        total = self.metrics_storage.get_total_inferences()
        avg = self.metrics_storage.get_average_time()

        start_index = max(0, total - len(times))

        return {
            "times": times,
            "total_inferences": total,
            "avg_time": avg,
            "start_index": start_index,
        }

    def load_initial_metrics(
        self,
    ) -> Tuple[None, str, plt.Figure, plt.Figure, str, str]:
        """
        Loads initial metrics for display.

        Returns:
            tuple: A tuple containing None, total inferences, histogram figure, line figure, average time, and last time.
        """
        metrics = self.get_metrics()

        if not metrics["times"]:
            return None, None, None, None, None, None

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

        hist_fig, line_fig = self.create_plots(hist_data, line_data)

        return (
            None,
            f"{metrics['total_inferences']}",
            hist_fig,
            line_fig,
            f"{metrics['avg_time']:.2f}",
            f"{metrics['times'][-1]:.2f}",
        )

    def create_plots(
        self, hist_data: DataFrame, line_data: DataFrame
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Helper method to create plots.

        Args:
            hist_data (pd.DataFrame): Data for histogram plot.
            line_data (pd.DataFrame): Data for line plot.

        Returns:
            tuple: A tuple containing histogram figure and line figure.
        """
        plt.style.use("dark_background")

        # Histograma
        hist_fig, hist_ax = plt.subplots(figsize=(8, 4), facecolor="#f0f0f5")
        hist_ax.set_facecolor("#f0f0f5")
        hist_data.hist(
            bins=20, ax=hist_ax, color="#4F46E5", alpha=0.7, edgecolor="white"
        )
        hist_ax.set_title(
            "Distribution of Inference Times",
            pad=15,
            fontsize=12,
            color="#1f2937",
        )
        hist_ax.set_xlabel("Time (ms)", color="#374151")
        hist_ax.set_ylabel("Frequency", color="#374151")
        hist_ax.tick_params(colors="#4b5563")
        hist_ax.grid(True, linestyle="--", alpha=0.3)

        # Gráfico de linha
        line_fig, line_ax = plt.subplots(figsize=(8, 4), facecolor="#f0f0f5")
        line_ax.set_facecolor("#f0f0f5")
        line_data.plot(
            x="Inference",
            y="Time (ms)",
            ax=line_ax,
            color="#4F46E5",
            alpha=0.7,
            label="Time",
        )
        line_data.plot(
            x="Inference",
            y="Mean",
            ax=line_ax,
            color="#DC2626",
            linestyle="--",
            label="Mean",
        )
        line_ax.set_title(
            "Inference Time per Execution", pad=15, fontsize=12, color="#1f2937"
        )
        line_ax.set_xlabel("Inference Number", color="#374151")
        line_ax.set_ylabel("Time (ms)", color="#374151")
        line_ax.tick_params(colors="#4b5563")
        line_ax.grid(True, linestyle="--", alpha=0.3)
        line_ax.legend(
            frameon=True, facecolor="#f0f0f5", edgecolor="white", labelcolor="black"
        )
        hist_fig.tight_layout()
        line_fig.tight_layout()

        plt.close(hist_fig)
        plt.close(line_fig)

        return hist_fig, line_fig

    def detect(
        self, image: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.5
    ) -> Tuple[Image.Image, dict]:
        """
        Detects signatures in the given image.

        Args:
            image: The image to process.
            conf_thres (float): Confidence threshold for detection.
            iou_thres (float): Intersection over Union threshold for detection.

        Returns:
            tuple: A tuple containing the output image and metrics.
        """
        image.save(self.temp_path)

        response = self.run(self.temp_path, conf=conf_thres, iou=iou_thres)

        output_image = self.draw_result(self.temp_path, response["result"])

        self.update_metrics(response["inference_time"] * 1000)

        return output_image, self.get_metrics()

    def detect_example(
        self, image: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.5
    ) -> Image.Image:
        """
        Wrapper method for examples that returns only the image.

        Args:
            image: The image to process.
            conf_thres (float): Confidence threshold for detection.
            iou_thres (float): Intersection over Union threshold for detection.

        Returns:
            The output image.
        """
        output_image, _ = self.detect(image, conf_thres, iou_thres)
        return output_image
