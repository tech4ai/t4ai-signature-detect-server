import argparse
import os

from google.cloud import storage
from ultralytics import YOLO


def init_gcp():
    return storage.Client()

def download_from_gcp(storage_client, bucket_name, source_blob_name, local_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    print(f"Downloading {source_blob_name} to {local_path}")
    blob.download_to_filename(local_path)

def upload_to_gcp(storage_client, bucket_name, destination_blob_name, local_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    print(f"Uploading {local_path} to {destination_blob_name}")
    blob.upload_from_filename(local_path)

def export_to_onnx(model_path, output_path):
    model = YOLO(model_path)
    print(f"Exporting model to ONNX: {output_path}")
    model.export(format="onnx", dynamic=True)

def export_to_tensorrt(model_path, output_path):
    model = YOLO(model_path)
    print(f"Exporting model to TensorRT: {output_path}")
    model.export(format="engine", dynamic=True)

def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 model to ONNX or TensorRT')
    parser.add_argument('--format', choices=['onnx', 'tensorrt'], required=True,
                      help='Export format (onnx or tensorrt)')
    args = parser.parse_args()

    # GCP configurations
    bucket_name = 'iag-training'
    base_path = 'models/image/signature-detection/yolov8/yolov8s/train/weights'
    source_blob_name = f'{base_path}/best.pt'
    
    # Set local and destination paths based on format
    local_pt_path = '/tmp/model.pt'
    local_export_path = '/tmp/model.onnx' if args.format == 'onnx' else '/tmp/model.engine'
    destination_blob_name = f'{base_path}/model.{"onnx" if args.format == "onnx" else "engine"}'

    try:
        # Initialize GCP client
        storage_client = init_gcp()

        # Download model
        download_from_gcp(storage_client, bucket_name, source_blob_name, local_pt_path)

        # Export model
        if args.format == 'onnx':
            export_to_onnx(local_pt_path, local_export_path)
        else:
            export_to_tensorrt(local_pt_path, local_export_path)

        # Upload exported model
        upload_to_gcp(storage_client, bucket_name, destination_blob_name, local_export_path)

    finally:
        # Clean up local files
        if os.path.exists(local_pt_path):
            os.remove(local_pt_path)
        if os.path.exists(local_export_path):
            os.remove(local_export_path)

if __name__ == "__main__":
    main()