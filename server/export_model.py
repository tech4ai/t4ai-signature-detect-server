from ultralytics import YOLO
from google.cloud import storage
import os

# Initialize GCP client
storage_client = storage.Client()

# Define bucket and file names
bucket_name = 'iag-training'
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


'''
signature-detection/
├── analyzer
│   ├── output-repo
│   │   └── outdir
│   │       ├── yolov8s_config_0
│   │       │   └── config.pbtxt
│   │       ├── ...
│   │       └── yolov8s_config_default
│   │           └── config.pbtxt
│   ├── perf.yaml
│   └── profile_results
│       ├── perf_analyzer_error.log
│       ├── plots
│       │   ├── detailed
│       │   │   ├── yolov8s_config_0
│       │   │   │   └── latency_breakdown.png
│       │   │   ├── ....
│       │   │   └── yolov8s_config_default
│       │   │       └── latency_breakdown.png
│       │   └── simple
│       │       ├── yolov8s
│       │       │   ├── cpu_mem_v_latency.png
│       │       │   ├── gpu_mem_v_latency.png
│       │       │   └── throughput_v_latency.png
│       │       ├── ....
│       │       └── yolov8s_config_default
│       │           ├── cpu_mem_v_latency.png
│       │           ├── gpu_mem_v_latency.png
│       │           ├── gpu_power_v_latency.png
│       │           └── gpu_util_v_latency.png
│       ├── reports
│       │   ├── detailed
│       │   │   ├── yolov8s_config_0
│       │   │   │   └── detailed_report.pdf
│       │   │   ├── ...
│       │   │   └── yolov8s_config_default
│       │   │       └── detailed_report.pdf
│       │   └── summaries
│       │       └── yolov8s
│       │           └── result_summary.pdf
│       └── results
│           └── metrics-model-inference.csv
├── inference
│   ├── inference_gui.py
│   ├── inference_pipeline.py
│   ├── inference_triton_client.py
│   ├── inference_yolo.py
│   └── utils
│       ├── postprocessing.py
│       └── preprocessing.py
├── models
│   ├── postprocessing
│   │   ├── 1
│   │   │   └── model.py
│   │   └── config.pbtxt
│   ├── preprocessing
│   │   ├── 1
│   │   │   └── model.py
│   │   └── config.pbtxt
│   └── yolov8s
│       ├── 1
│       │   └── model.onnx
│       └── config.pbtxt
├── __pycache__
│   └── inference_pipeline.cpython-310.pyc
└── server
    ├── export_model.py
    ├── local
    │   ├── prometheus.yml
    │   └── serve_triton_local.py
    └── vertexai
        ├── deploy_vertex_ai.sh
        ├── nvidia_triton_custom_container_prediction.ipynb
        └── undeploy_model_vertex_ai.sh

'''