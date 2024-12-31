from ultralytics import YOLO
from google.cloud import storage
import os

# Initialize GCP client
storage_client = storage.Client()

# Define bucket and file names
bucket_name = 'REDACTED_BUCKET_NAME'
source_blob_name = 'models/image/signature-detection/yolov8/yolov8s/train/weights/best.pt'
destination_blob_name = 'models/image/signature-detection/yolov8/yolov8s/train/weights/test.onnx'
local_pt_path = '/tmp/model.pt'
local_onnx_path = '/tmp/model.onnx'

# Download .pt file from GCP bucket
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(source_blob_name)
print(f"Downloading {source_blob_name} to {local_pt_path}")
blob.download_to_filename(local_pt_path)

# Load model
model = YOLO(local_pt_path)

# Export model to ONNX format
print(f"Exporting model to {local_onnx_path}")
model.export(format="onnx", dynamic=True)

# Upload .onnx file to GCP bucket
print(f"Uploading {destination_blob_name} to {bucket_name}")
onnx_blob = bucket.blob(destination_blob_name)
onnx_blob.upload_from_filename(local_onnx_path)

# Clean up local files
os.remove(local_pt_path)
os.remove(local_onnx_path)