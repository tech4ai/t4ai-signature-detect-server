import os
import numpy as np
from pprint import pprint
from predictors import BasePredictor, HttpPredictor, VertexAIPredictor, TritonClientPredictor

def encode_image(image_path):
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
            
## Main Execution
def main():
    # Paths
    root_dir_images = "/home/samuel/Downloads/Signature Detection.v3i.yolov8/test/images"
    image_paths = [os.path.join(root_dir_images, img) for img in os.listdir(root_dir_images) if img.endswith(".jpg")]

    predictors = {
        '1': (HttpPredictor, {"url": "https://t4ai-signature-detector-100881400340.us-central1.run.app/v2/models/yolov8_ensemble/infer"}),
        '2': (TritonClientPredictor, {"url": "https://t4ai-signature-detector-100881400340.us-central1.run.app/yolov8_ensemble"}),
        '3': (VertexAIPredictor, {"url": "https://8605861017928335360.us-central1-100881400340.prediction.vertexai.goog/v1/projects/100881400340/locations/us-central1/endpoints/8605861017928335360:rawPredict"}),
    }

    # Menu for selecting predictor
    print("Select predictor:")
    for key, value in predictors.items():
        print(f"{key}: {value[0].__name__}")
        
    selected = input("Enter number: ")
    predictor = predictors[selected][0](**predictors[selected][1])

    print(f"Using predictor: {predictor.__class__.__name__}")
    
    # Run Pipeline
    pipeline = InferencePipeline(predictor)
    
    # First inference for warm-up
    print("Running inference on first image...")
    pipeline.run(image_paths[0])
    
    print("Running inference on all images...")
    i_times = []
    for image_path in image_paths:
        r = pipeline.run(image_path)
        pprint(r)
        i_times.append(r['inference_time'])
        pprint(r)
    
    print(f"Average inference time: {np.mean(i_times)}")
    
if __name__ == "__main__":
    main()
    