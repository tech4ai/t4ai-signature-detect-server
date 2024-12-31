import os
import gradio as gr
from PIL import Image, ImageDraw
import requests
from pprint import pprint
from inference_pipeline import InferencePipeline, HttpPredictor, TritonClientPredictor, VertexAIPredictor
import numpy as np

# Variável global para armazenar o preditor atual e sua escolha
current_predictor = None
current_choice = None

def draw_result(image_path, result):
    """
    Desenha as bounding boxes na imagem.
    
    Args:
        image_path (str): Caminho da imagem original.
        result (dict): Dicionário contendo `detection_boxes` e `detection_scores`.
    
    Returns:
        PIL.Image: Imagem com as bounding boxes desenhadas.
    """
    # Carregar a imagem
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Obter dimensões originais
    img_width, img_height = image.size

    # Desenhar as bounding boxes
    boxes = result["detection_boxes"]
    scores = result["detection_scores"]
    
    pprint(f"Boxes: {boxes}")
    pprint(f"Scores: {scores}")

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

            # Desenhar bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Adicionar label com score
            label = f"{score:.2f}"
            draw.text((x1, y1 - 10), label, fill="red")

    return image

def initialize_predictor(choice):
    """Inicializa dinamicamente o preditor escolhido."""
    global current_predictor, current_choice

    if choice == current_choice:
        return current_predictor

    try:
        if choice == "Requests HTTP":
            current_predictor = HttpPredictor(url="https://t4ai-signature-detector-100881400340.us-central1.run.app/v2/models/yolov8_ensemble/infer")
        elif choice == "Triton Client":
            current_predictor = TritonClientPredictor(url="https://t4ai-signature-detector-100881400340.us-central1.run.app/yolov8_ensemble")
        elif choice == "Vertex Endpoint":
            current_predictor = VertexAIPredictor(url="https://8605861017928335360.us-central1-100881400340.prediction.vertexai.goog/v1/projects/100881400340/locations/us-central1/endpoints/8605861017928335360:rawPredict")
        else:
            raise ValueError("Preditor inválido.")

        current_choice = choice
        return current_predictor

    except Exception as e:
        raise RuntimeError(f"Erro ao inicializar o preditor '{choice}': {e}")

def save_image(image, url):
    """Salva uma imagem localmente a partir de um upload ou URL."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, 'tmp')
    os.makedirs(root_path, exist_ok=True)
    image_path = os.path.join(root_path, "temp_image.jpg")

    if image is not None:
        pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
        pil_image.save(image_path, format="JPEG")
    elif url:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            im = Image.open(response.raw)
            im.save(image_path)
        else:
            raise ValueError("Failed to fetch image from URL.")
    else:
        raise ValueError("No image or URL provided.")

    return image_path

def inference(image, url, predictor_choice):
    """Realiza a inferência usando o preditor escolhido."""
    try:
        image_path = save_image(image, url)
    except Exception as e:
        return f"Erro ao salvar ou baixar a imagem: {e}"

    try:
        predictor = initialize_predictor(predictor_choice)
        pipeline = InferencePipeline(predictor)
        resp = pipeline.run(image_path)
    except Exception as e:
        return f"Erro durante a inferência: {e}"
    
    return draw_result(image_path, resp['result'])

def create_demo():
    """Cria a interface gráfica com Gradio."""
    return gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(label="Upload Image"),
            gr.Textbox(label="Image URL"),
            gr.Radio(choices=["Requests HTTP", "Triton Client", "Vertex Endpoint"], label="Escolha o Preditor", value="Triton Client")
        ],
        outputs=[
            gr.Image(label="Resultado da Inferência")
        ],
        title="Pipeline de Inferência de Imagens",
        description="Faça upload de uma imagem ou insira um URL para realizar a inferência usando um dos preditores disponíveis."
    )

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
