import cv2
import numpy as np

# Load and preprocess image
def preprocess(image_path):
    img = cv2.imread(image_path)  # Load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (640, 640))  # Resize to model input size
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # Change to [C, H, W]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
