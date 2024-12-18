import os
import gradio as gr
from PIL import Image
import requests
from inference_pipeline import InferencePipeline, LocalPredictor, VertexAIPredictor
import numpy as np

# Gradio Interface
def inference(image, url, use_vertex):
    if image is None and not url:
        return "Please upload an image or provide a URL."

    # Handle file input or URL
    if image is not None and isinstance(image, np.ndarray):
        print("Image received:", type(image), image.shape)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(current_dir, 'tmp')
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)
            
        image_path = os.path.join(root_path, "temp_image.jpg")
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
                image_path = "temp_image.jpg"
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
    os.remove(image_path)

    return resp['result'].plot()

predictors = {
    "local":  LocalPredictor(url="http://localhost:8000/v2/models/yolov8s/infer"),
    "vertex": VertexAIPredictor(url="https://8605861017928335360.us-central1-100881400340.prediction.vertexai.goog/v1/projects/100881400340/locations/us-central1/endpoints/8605861017928335360:rawPredict")
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



