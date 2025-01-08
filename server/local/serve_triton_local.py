import contextlib
import os
import subprocess
import time

from tritonclient.http import InferenceServerClient

BUILD = True

model_name = "yolov8_ensemble"
tag = "triton-signature-server:latest" 

# Obtém o diretório onde o script está
base_dir = os.path.dirname(os.path.abspath(__file__))

compose_file = os.path.abspath(os.path.join(base_dir, "..", "..", 'docker-compose.yml'))

# Inicia o serviço com Docker Compose
compose_cmd = f"docker compose -f {compose_file} up -d"
if BUILD:
    compose_cmd += " --build"
    
subprocess.call(compose_cmd, shell=True)

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
