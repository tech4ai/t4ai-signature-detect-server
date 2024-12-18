import contextlib
import subprocess
import time

from tritonclient.http import InferenceServerClient

model_name = "yolov8s"
triton_repo_path = "gs://iag-training/triton-server/image/signature-detection/models"

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "avante" 

# Pull the image
#subprocess.call(f"docker pull {tag}", shell=True)

# Run the Triton server and capture the container ID
container_id = (
    subprocess.check_output(
        f'''
        docker run \
            -t -d \
            -p 8000:8000 -p 8001:8001 -p 8002:8002 \
            --name=local_object_detector \
            -e AIP_MODE=True \
            -v /home/samuel/.config/gcloud/application_default_credentials.json:/gcloud-creds.json \
            -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud-creds.json \
            {tag} \
            --model-repository {triton_repo_path} \
            --model-control-mode=explicit \
            --load-model=* \
            --log-verbose=1
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
    -v /home/samuel/.config/gcloud/application_default_credentials.json:/gcloud-creds.json \
    -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud-creds.json -e AIP_MODE=True \
    us-central1-docker.pkg.dev/tech4ai-develop/t4ai-nvidia-triton/vertex-triton-inference \
    --model-repository gs://iag-training/triton-server/image/signature-detection/models \
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
model-analyzer profile  --model-repository home/samuel/codes/tech4/tests/triton-server/signature-detection/models  -f perf.yaml --triton-launch-mode=remote --triton-http-endpoint=localhost:8000     --output-model-repository-path /home/samuel/codes/tech4/tests/triton-server/signature-detection/output-repo/outdir      --export-path profile_results --override-output-model-repository --collect-cpu-metrics --client-protocol=http --monitoring-interval=5

model-analyzer report --report-model-configs yolov8s_config_0,yolov8s_config_12,yolov8s_config_4,yolov8s_config_8,yolov8s_config_1,yolov8s_config_13,yolov8s_config_5,yolov8s_config_default,yolov8s_config_10,yolov8s_config_2,yolov8s_config_6,yolov8s_config_11,yolov8s_config_3,yolov8s_config_7 --export-path /workspace --config-file perf.yaml 

'''