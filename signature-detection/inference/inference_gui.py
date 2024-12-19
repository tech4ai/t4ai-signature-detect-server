import os
import gradio as gr
from PIL import Image, ImageDraw
import requests
from inference_pipeline import InferencePipeline, LocalPredictor, VertexAIPredictor
import numpy as np

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

    for box, score in zip(boxes, scores):
        if score >= 0.5:  # Filtrar por confiança mínima
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h

            # Rescale se necessário
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            # Desenhar bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Adicionar label com score
            label = f"{score:.2f}"
            draw.text((x1, y1 - 10), label, fill="red")

    return image

# Gradio Interface
def inference(image, url, use_vertex):
    if image is None and not url:
        return "Please upload an image or provide a URL."

    # Handle file input or URL
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, 'tmp')
    if not os.path.exists(root_path):
        os.makedirs(root_path, exist_ok=True)
    image_path = os.path.join(root_path, "temp_image.jpg")
        
    if image is not None and isinstance(image, np.ndarray):
        print("Image received:", type(image), image.shape)
        try:
            pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
            pil_image.save(image_path, format="JPEG")
            print(f"Image saved at: {image_path}")
        except Exception as e:
            print("Error saving image:", e)
            return "Failed to save image."
    else:
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                print("Image downloaded successfully.")
            else:
                return "Failed to fetch image from URL."
        except Exception as e:
            print("Error fetching image:", e)
            return "Error downloading image."

    pipeline = InferencePipeline(predictors["local"] if not use_vertex else predictors["vertex"])
    print(f"Running inference on image: {image_path}")

    try:
        resp = pipeline.run(image_path)
    except Exception as e:
        print("Error during inference:", e)
        return "Failed to process image."

    # Remove temporary file if created
    # os.remove(image_path)

    return draw_result(image_path, resp['result'])

predictors = {
    "local":  LocalPredictor(url="http://localhost:8000/v2/models/yolov8_ensemble/infer"),
    "vertex": VertexAIPredictor(url="REDACTED_VERTEX_ENDPOINT")
}

# Define Gradio Interface
demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(label="Upload Image"),
        gr.Textbox(label="Image URL"),
        gr.Checkbox(label="Use Vertex AI", value=False)
    ],
    outputs=gr.Image(label="Inference Result"),
    title="Image Inference Pipeline",
    description="Upload an image or enter a URL to perform inference using a Local or Vertex AI Predictor."
)

demo.launch()



