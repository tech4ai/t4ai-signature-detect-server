FROM nvcr.io/nvidia/tritonserver:24.11-py3

# Instale as dependências
RUN apt-get update && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Instale o ultralytics
RUN pip install ultralytics==8.3.50

# Configure as portas do Triton Server
EXPOSE 8000 8001 8002

# Comando para iniciar o Triton Server
CMD ["tritonserver", \
     "--model-repository=REDACTED_BUCKET_PATH", \
     "--model-control-mode=explicit", \
     "--load-model=*", \
     "--log-verbose=1"]