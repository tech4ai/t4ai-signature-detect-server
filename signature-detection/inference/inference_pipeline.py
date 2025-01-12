import os
import cv2
from PIL import Image, ImageDraw
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

    def draw_result(self, image_path, result):
        """
        Desenha as bounding boxes na imagem.

        Args:
            image_path (str): Caminho da imagem original.
            result (dict): Dicionário contendo `detection_boxes` e `detection_scores`.

        Returns:
            np.ndarray: Imagem com as bounding boxes desenhadas (formato OpenCV).
        """
        # Carregar a imagem
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Obter dimensões originais
        img_width, img_height = image.size

        # Desenhar as bounding boxes
        boxes = result["detection_boxes"]
        scores = result["detection_scores"]

        for box, score in zip(boxes, scores):
            if score >= 0.2:  # Filtrar por confiança mínima
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h

                # Reescalar para o tamanho original da imagem
                x1 = int(x1 * img_width / 640)
                y1 = int(y1 * img_height / 640)
                x2 = int(x2 * img_width / 640)
                y2 = int(y2 * img_height / 640)

                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )

                # Desenhar bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Adicionar label com score
                label = f"{score:.2f}"
                draw.text((x1, y1 - 10), label, fill=color)

        # Converter a imagem PIL para formato OpenCV
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


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

    window_name = "Resultados de Inferência"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
        inference_times.append(current_time)

        # Atualiza a descrição da barra com o tempo de inferência atual
        progress_bar.set_description(f"Tempo atual: {current_time:.3f}s")

        annotated_image = pipeline.draw_result(image_path, result["result"])
        cv2.imshow(window_name, annotated_image)

        # Esperar por 1ms para permitir a interação
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Interrompido pelo usuário.")
            break

    # Fechar janela ao final
    cv2.destroyAllWindows()
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
    image_paths = test_image_paths + train_image_paths + val_image_paths

    # Predictor selection
    predictor_class, url = select_predictor()
    predictor = predictor_class(url=url)
    print(f"Usando o predictor: {predictor.__class__.__name__}")

    # Run pipeline
    pipeline = InferencePipeline(predictor)
    run_pipeline(pipeline, image_paths)


if __name__ == "__main__":
    main()
