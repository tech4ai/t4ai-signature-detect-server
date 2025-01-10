import os
from dotenv import load_dotenv

import numpy as np
from tqdm import tqdm
from inference.predictors import (
    BasePredictor,
    HttpPredictor,
    TritonClientPredictor,
    VertexAIPredictor,
    encode_image,
)


## Inference Pipeline
class InferencePipeline:
    """Orchestrates the entire inference pipeline."""

    def __init__(self, predictor: BasePredictor):
        self.predictor = predictor

    def run(self, image_path: str, conf: float = 0.25, iou: float = 0.5):
        """Run the inference pipeline."""
        image_data = encode_image(image_path)

        response, mean_time = self.predictor.predict(image_data, conf, iou)

        result = self._process_response(response)

        return {"result": result, "inference_time": mean_time}

    def _process_response(self, response):
        """Extract predictions from response."""
        # Separar bounding boxes e scores
        return {
            "detection_boxes": response[:, :4],
            "detection_scores": response[:, 4],
        }


def get_image_paths(dataset_dir, split="test"):
    """Retrieve all image paths from the dataset directory."""
    root_dir_images = os.path.join(dataset_dir, split, "images")
    return [
        os.path.join(root_dir_images, img)
        for img in os.listdir(root_dir_images)
        if img.endswith(".jpg")
    ]


def select_predictor():
    """Display a menu to select the predictor type and return the selected predictor class and URL."""
    predictor_classes = {
        "1": (
            HttpPredictor,
            "HTTP Predictor",
            "http://<host>/v2/models/<model_name>/infer\n - http://localhost:8000/v2/models/yolov8_ensemble/infer (local)\n - REDACTED_TRITON_URL (Cloud Run)",
        ),
        "2": (
            TritonClientPredictor,
            "Triton Client Predictor",
            "http://<host>/<model_name>\n - http://localhost:8000/yolov8_ensemble (local)\n - REDACTED_TRITON_URL (Cloud Run)",
        ),
        "3": (VertexAIPredictor, "VertexAI Predictor", "https://<vertex_endpoint>"),
    }

    print("Selecione o tipo de predictor:")
    for key, (cls, name, _) in predictor_classes.items():
        print(f"{key}: {name}")

    predictor_type = input("Digite o número do tipo de predictor: ").strip()
    predictor_info = predictor_classes.get(predictor_type)

    if not predictor_info:
        raise ValueError("Tipo de predictor inválido.")

    _, name, example = predictor_info
    print(f"Usando {name}")
    print(f"Digite a URL do endpoint\n Exemplo: {example}")
    url = input("URL: ").strip()

    return predictor_info[0], url


def run_pipeline(pipeline: InferencePipeline, image_paths):
    """Run the inference pipeline on all images and print results."""
    print("Executando inferência na primeira imagem para aquecimento...")
    pipeline.run(image_paths[0])

    print("Executando inferência em todas as imagens...")
    inference_times = []

    # Criando barra de progresso
    progress_bar = tqdm(
        image_paths,
        desc="Processando imagens",
        unit="img",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        dynamic_ncols=True,
    )

    for image_path in progress_bar:
        result = pipeline.run(image_path)
        current_time = result["inference_time"]
        # Atualiza a descrição da barra com o tempo de inferência atual
        progress_bar.set_description(f"Tempo atual: {current_time:.3f}s")
        inference_times.append(current_time)

    print(f"\nTempo médio de inferência: {np.mean(inference_times):.3f} s")


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
    image_paths = train_image_paths + test_image_paths + val_image_paths

    # Predictor selection
    predictor_class, url = select_predictor()
    predictor = predictor_class(url=url)
    print(f"Usando o predictor: {predictor.__class__.__name__}")

    # Run pipeline
    pipeline = InferencePipeline(predictor)
    run_pipeline(pipeline, image_paths)


if __name__ == "__main__":
    main()
