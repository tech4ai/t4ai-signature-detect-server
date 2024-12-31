import contextlib
import subprocess
import time
import os

from tritonclient.http import InferenceServerClient

model_name = "yolov8_ensemble"
triton_repo_path = "gs://iag-training/triton-server/image/signature-detection/models"
tag = "triton-signature-server:latest" 

# Obtém o diretório onde o script está
base_dir = os.path.dirname(os.path.abspath(__file__))

dockerfile_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'Dockerfile'))

# Verifica se a variável de ambiente GOOGLE_APPLICATION_CREDENTIALS está configurada
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    raise EnvironmentError(
        "A variável de ambiente 'GOOGLE_APPLICATION_CREDENTIALS' não está configurada. "
        "Configure-a apontando para o caminho do arquivo JSON de credenciais do Google Cloud."
    )

# Caminho para o arquivo de credenciais
google_creds_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

# Build the Triton server image
# subprocess.call(f"docker build -t {tag} -f {dockerfile_path} {base_dir}", shell=True)


# Run the Triton server and capture the container ID
container_id = (
    subprocess.check_output(
        f'''
            docker run \
            -p 8000:8000 -p 8001:8001 -p 8002:8002 \
            -d  \
            --name=local_object_detector \
            -v {google_creds_path}:/gcloud-creds.json \
            -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud-creds.json \
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
        if triton_client.is_model_ready(model_name):
            print(f"Modelo {model_name} está pronto.")
            break
    time.sleep(1)
else:
    raise RuntimeError(f"O modelo {model_name} não ficou pronto a tempo.")
