#!/bin/bash

# Verificar se as variáveis necessárias estão definidas
if [ -z "$TRITON_ADMIN_KEY" ]; then
  echo "Erro: TRITON_ADMIN_KEY não está definida."
  exit 1
fi

# Executar o Triton com as variáveis interpoladas
exec tritonserver \
  --model-repository=${TRITON_MODEL_REPOSITORY} \
  --model-control-mode=explicit \
  --load-model=* \
  --log-verbose=1 \
  --allow-metrics=false \
  --allow-grpc=false \
  --http-restricted-api=model-repository,model-config,shared-memory,statistics,trace:admin-key=${TRITON_ADMIN_KEY}
