import os
import time
from typing import Optional
import subprocess

import gradio as gr
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64

def encode_image(image_path):
    image_data = np.fromfile(image_path, dtype="uint8")
    image_data = np.expand_dims(image_data, axis=0)
    return image_data

## Strategy Pattern for Inference
class BasePredictor:
    """Base class for predictors."""

    def __init__(self):
        pass

    def predict(self, payload):
        """To be implemented by concrete classes."""
        raise NotImplementedError("Predict method not implemented.")

class LocalPredictor(BasePredictor):
    """Inference predictor for local models."""

    def __init__(self, url):
        super().__init__()
        self.url = url

    def predict(self, payload):
        tic = time.time()
        response = requests.post(self.url, headers={"Content-Type": "application/json"}, json=payload)
        mean_time = time.time() - tic
        return response.json(), mean_time

class VertexAIPredictor(BasePredictor):
    """Inference predictor for Vertex AI models."""

    def __init__(self, url, access_token: Optional[str] = None):
        super().__init__()
        self.url = url
        if access_token is None:
            try:
                self.access_token = self.get_google_access_token()
            except Exception as e:
                print("Error fetching access token:", e)
                return
                
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def get_google_access_token(self):
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()

    def predict(self, payload):
        tic = time.time()
        response = requests.post(self.url, headers=self.headers, json=payload)
        mean_time = time.time() - tic
        return response.json(), mean_time

## Inference Pipeline
class InferencePipeline:
    """Orchestrates the entire inference pipeline."""

    def __init__(self, predictor: BasePredictor):
        self.predictor = predictor

    def run(self, image_path):
        """Run the inference pipeline."""
        image_data = encode_image(image_path)
        
        payload = self._create_payload(image_data)

        response, mean_time = self.predictor.predict(payload)
        
        result = self._process_response(response)
        print(result)
        
        print(f"Inference time: {mean_time}")
        return {'result' : result, 'inference_time': mean_time}

    def _create_payload(self, image):
        """Create the payload for the model."""
        return {
            "id": "0",
            "inputs": [
                {
                    "name": "raw_image",
                    "shape": image.shape,
                    "datatype": "UINT8",
                    "data": image.tolist()
                }
            ]
        }

    def _process_response(self, response):
        """Extract predictions from response."""
        data = np.array(response['outputs'][0]['data']).astype(np.float32)
        # Número de detecções
        num_detections = data.shape[0] // 5
        # Reorganize o array em uma matriz de N x 5
        data = data.reshape((num_detections, 5))
        # Separar bounding boxes e scores
        return {
            "detection_boxes": data[:, :4],
            "detection_scores" : data[:, 4],
        }
            

## Main Execution
def main():
    # Paths
REDACTED_PATH
    image_paths = [os.path.join(root_dir_images, img) for img in os.listdir(root_dir_images) if img.endswith(".jpg")]

    # Select predictor: Local or Vertex AI
    use_vertex_ai = False

    if use_vertex_ai:
        predictor = VertexAIPredictor(
            url="REDACTED_VERTEX_ENDPOINT",
        )
    else:
        predictor = LocalPredictor(url="http://localhost:8000/v2/models/yolov8_ensemble/infer")

    # Run Pipeline
    pipeline = InferencePipeline(predictor)
    i_times = []
    for image_path in image_paths:
        r = pipeline.run(image_path)
        i_times.append(r['inference_time'])
    
    print(f"Average inference time: {np.mean(i_times)}")
    
if __name__ == "__main__":
    main()
    
# 0.45890190170079775