import os
import cv2
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from inference.predictors import (
    BasePredictor,
    HttpPredictor,
    TritonClientPredictor,
    VertexAIPredictor,
    encode_image,
)


class InferencePipeline:
    """Manages the complete inference pipeline."""

    def __init__(self, predictor: BasePredictor):
        """
        Initializes the pipeline with a predictor.

        Args:
            predictor (BasePredictor): The inference predictor instance.
        """
        self.predictor = predictor

    def run(self, image_path: str, conf: float = 0.25, iou: float = 0.5) -> Dict:
        """
        Executes the inference pipeline for a single image.

        Args:
            image_path (str): Path to the input image.
            conf (float): Confidence threshold for predictions. Default is 0.25.
            iou (float): IOU threshold for predictions. Default is 0.5.

        Returns:
            dict: Results containing detections and inference time.
        """
        image_data = encode_image(image_path)
        response, inference_time = self.predictor.predict(image_data, conf, iou)
        result = self._process_response(response)
        return {"result": result, "inference_time": inference_time}

    def _process_response(self, response: np.ndarray) -> Dict:
        """
        Processes the prediction response.

        Args:
            response (np.ndarray): Raw response from the predictor.

        Returns:
            dict: Processed detection boxes and scores.
        """
        return {
            "detection_boxes": response[:, :4],
            "detection_scores": response[:, 4],
        }

    def draw_result(self, image_path, result):
        """
        Draws bounding boxes on the image based on the results.

        Args:
            image_path (str): Path to the input image.
            result (dict): Contains detection boxes and scores.

        Returns:
            np.ndarray: Annotated image in OpenCV format.
        """
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size

        for box, score in zip(result["detection_boxes"], result["detection_scores"]):
            if score >= 0.2:  # Filter low-confidence detections
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h

                # Scale coordinates back to original image size
                x1 = int(x1 * img_width / 640)
                y1 = int(y1 * img_height / 640)
                x2 = int(x2 * img_width / 640)
                y2 = int(y2 * img_height / 640)

                color = tuple(np.random.randint(0, 256, size=3))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, y1 - 10), f"{score:.2f}", fill=color)

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def get_image_paths(dataset_dir: str, split: str = "test") -> List[str]:
    """
    Retrieves image file paths from a dataset directory.

    Args:
        dataset_dir (str): Root directory of the dataset.
        split (str): Dataset split (e.g., 'train', 'test'). Default is 'test'.

    Returns:
        list: List of image file paths.
    """
    images_dir = os.path.join(dataset_dir, split, "images")
    return [
        os.path.join(images_dir, img)
        for img in os.listdir(images_dir)
        if img.endswith(".jpg")
    ]


def select_predictor() -> Tuple[type, str]:
    """
    Prompts the user to select a predictor type and provides the endpoint URL.

    Returns:
        tuple: Selected predictor class and its endpoint URL.
    """
    predictor_classes = {
        "1": (
            HttpPredictor,
            "HTTP Predictor",
            "http://<host>/v2/models/<model_name>/infer\n - http://localhost:8000/v2/models/yolov8_ensemble/infer (local)",
        ),
        "2": (
            TritonClientPredictor,
            "Triton Client Predictor",
            "http://<host>/<model_name>\n - http://localhost:8000/yolov8_ensemble (local)",
        ),
        "3": (VertexAIPredictor, "VertexAI Predictor", "https://<vertex_endpoint>"),
    }

    print("Select predictor type:")
    for key, (cls, name, _) in predictor_classes.items():
        print(f"{key}: {name}")

    predictor_type = input("Enter the number: ").strip()
    predictor_info = predictor_classes.get(predictor_type)

    if not predictor_info:
        raise ValueError("Invalid predictor type.")

    _, name, example = predictor_info

    print(f"Enter the endpoint URL for {name}\n Exemplo: {example}")
    url = input("URL: ").strip()

    return predictor_info[0], url


def run_pipeline(pipeline: InferencePipeline, image_paths: List[str]) -> None:
    """
    Executes the pipeline on a list of images and displays results.

    Args:
        pipeline (InferencePipeline): The inference pipeline instance.
        image_paths (list): List of image paths to process.
    """
    print("Warming up with the first image...")
    pipeline.run(image_paths[0])

    print("Processing all images...")
    inference_times = []

    cv2.namedWindow("Inference Results", cv2.WINDOW_NORMAL)

    # Criando barra de progresso
    progress_bar = tqdm(
        image_paths,
        desc="Processing images",
        unit="img",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        dynamic_ncols=True,
    )

    for image_path in progress_bar:
        result = pipeline.run(image_path)
        current_time = result["inference_time"]
        inference_times.append(current_time)

        progress_bar.set_description(f"Current Time: {current_time:.3f}s")

        annotated_image = pipeline.draw_result(image_path, result["result"])

        cv2.imshow("Inference Results", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Terminated by user.")
            break

    cv2.destroyAllWindows()

    # Metrics
    inference_times_ms = [time * 1000 for time in inference_times]
    total_time_ms = sum(inference_times_ms)

    minutes = int(total_time_ms // (1000 * 60))
    seconds = int((total_time_ms % (1000 * 60)) // 1000)
    milliseconds = int(total_time_ms % 1000)
    formatted_total_time = f"{minutes:02}:{seconds:02}:{milliseconds:03}"

    metrics = {
        "Metric": [
            "Mean time (ms)",
            "Standard deviation (ms)",
            "Max time (ms)",
            "Min time (ms)",
            "Total time (min)",
            "Number of inferences",
        ],
        "Value": [
            np.mean(inference_times_ms),
            np.std(inference_times_ms),
            np.max(inference_times_ms),
            np.min(inference_times_ms),
            formatted_total_time,
            len(inference_times),
        ],
    }

    # Imprime tabela com métricas
    print("\nInference Metrics:")
    print(tabulate(metrics, headers="keys", tablefmt="grid"))


def main():
    dotenv_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    )
    load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)

    # Paths
    HOME = os.getcwd()

    DATASET_DIR = os.path.join(HOME, "signature-detection", "data", "datasets")

    train_image_paths = get_image_paths(DATASET_DIR, "train")
    test_image_paths = get_image_paths(DATASET_DIR, "test")
    val_image_paths = get_image_paths(DATASET_DIR, "valid")
    image_paths = test_image_paths + train_image_paths + val_image_paths

    # Predictor selection
    predictor_class, url = select_predictor()
    predictor = predictor_class(url=url)

    # Run pipeline
    pipeline = InferencePipeline(predictor)
    run_pipeline(pipeline, image_paths)


if __name__ == "__main__":
    main()
