import os
from pprint import pprint
from dotenv import load_dotenv

import numpy as np
from predictors import (BasePredictor, HttpPredictor, TritonClientPredictor,
                        VertexAIPredictor)


def encode_image(image_path):
    """
    Encode an image file as a numpy array in uint8 format and add a batch dimension.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        image_data (numpy.ndarray): Image data as a numpy array with shape (1, H, W, C).
    """
    image_data = np.fromfile(image_path, dtype="uint8")
    image_data = np.expand_dims(image_data, axis=0)
    return image_data

## Inference Pipeline
class InferencePipeline:
    """Orchestrates the entire inference pipeline."""

    def __init__(self, predictor: BasePredictor):
        self.predictor = predictor

    def run(self, image_path):
        """Run the inference pipeline."""
        image_data = encode_image(image_path)

        response, mean_time = self.predictor.predict(image_data)
        
        result = self._process_response(response)
   
        return {'result' : result, 'inference_time': mean_time}

    def _process_response(self, response):
        """Extract predictions from response."""
        # Separar bounding boxes e scores
        return {
            "detection_boxes": response[:, :4],
            "detection_scores" : response[:, 4],
        }
            
def get_image_paths(dataset_dir):
    """Retrieve all image paths from the dataset directory."""
    root_dir_images = os.path.join(dataset_dir, "test", "images")
    return [
        os.path.join(root_dir_images, img)
        for img in os.listdir(root_dir_images) if img.endswith(".jpg")
    ]

def select_predictor():
    """Display a menu to select the predictor type and return the selected predictor class and URL."""
    predictor_classes = {
        '1': (HttpPredictor, "HTTP Predictor", "http://<host>/v2/models/<model_name>/infer\n - http://localhost:8000/v2/models/yolov8_ensemble/infer (local)\n - REDACTED_TRITON_URL (Cloud Run)"),
        '2': (TritonClientPredictor, "Triton Client Predictor", "http://<host>/<model_name>\n - http://localhost:8000/yolov8_ensemble (local)\n - REDACTED_TRITON_URL (Cloud Run)"),
        '3': (VertexAIPredictor, "VertexAI Predictor", "https://<vertex_endpoint>")
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

def run_pipeline(pipeline, image_paths):
    """Run the inference pipeline on all images and print results."""
    print("Executando inferência na primeira imagem para aquecimento...")
    pipeline.run(image_paths[0])

    print("Executando inferência em todas as imagens...")
    inference_times = []
    for image_path in image_paths:
        result = pipeline.run(image_path)
        pprint(result)
        inference_times.append(result['inference_time'])

    print(f"Tempo médio de inferência: {np.mean(inference_times)} ms")

def main():
    load_dotenv()

    # Paths
    HOME = os.getcwd()
    DATASET_DIR = os.path.join(HOME, "signature-detection", "data", "datasets")
    image_paths = get_image_paths(DATASET_DIR)

    # Predictor selection
    predictor_class, url = select_predictor()
    predictor = predictor_class(url=url)
    print(f"Usando o predictor: {predictor.__class__.__name__}")

    # Run pipeline
    pipeline = InferencePipeline(predictor)
    run_pipeline(pipeline, image_paths)

if __name__ == "__main__":
    main()
