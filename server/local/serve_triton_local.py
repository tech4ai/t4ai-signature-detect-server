import contextlib
import subprocess
import time

from tritonclient.http import InferenceServerClient

model_name = "yolov8_ensemble"
REDACTED_PATH

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "triton-signature-server:latest" 

# Pull the image
#subprocess.call(f"docker pull {tag}", shell=True)

# Run the Triton server and capture the container ID
container_id = (
    subprocess.check_output(
        f'''
        docker run \
            -p 8000:8000 \
            -d  \
            --name=local_object_detector \
            -v {triton_repo_path}:/models \
            {tag}
        ''',
        shell=True,
    )
    .decode("utf-8")
    .strip()
)

# Wait for the Triton server to start
triton_client = InferenceServerClient(url="localhost:8000", verbose=True, ssl=False)

# Wait until model is ready
for _ in range(100):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)
    

'''
docker run -t -d -p 8000:8000 --rm \
    -v REDACTED_CREDENTIALS_PATH:/gcloud-creds.json \
    -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud-creds.json -e AIP_MODE=True \
    REDACTED_CONTAINER_URL \
    --model-repository REDACTED_BUCKET_PATH \
    --strict-model-config=false
'''  

# Model Analyzer
'''
docker run -it  \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v $(pwd)/signature-detection/models:$(pwd)/signature-detection/models \
      --net=host nvcr.io/nvidia/tritonserver:24.11-py3-sdk 
'''

'''
REDACTED_PATH

model-analyzer report --report-model-configs yolov8s_config_0,yolov8s_config_12,yolov8s_config_4,yolov8s_config_8,yolov8s_config_1,yolov8s_config_13,yolov8s_config_5,yolov8s_config_default,yolov8s_config_10,yolov8s_config_2,yolov8s_config_6,yolov8s_config_11,yolov8s_config_3,yolov8s_config_7 --export-path /workspace --config-file perf.yaml 

'''