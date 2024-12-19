#
# Multistage build.
#
ARG TRITON_VERSION=2.52.0
ARG TRITON_CONTAINER_VERSION=24.11

FROM nvcr.io/nvidia/tritonserver:24.11-py3 AS full
FROM nvcr.io/nvidia/tritonserver:24.11-py3-min

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}
# Remove once https://github.com/openucx/ucx/pull/9148 is available
# in the min container.
ENV UCX_MEM_EVENTS no

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    1

# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true \
      && userdel ubuntu > /dev/null 2>&1 || true \
      && if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
          useradd $TRITON_SERVER_USER; \
        fi \
      && [ `id -u $TRITON_SERVER_USER` -eq 1000 ] \
      && [ `id -g $TRITON_SERVER_USER` -eq 1000 ]

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Common dependencies. FIXME (can any of these be conditional? For
# example libcurl only needed for GCS?)
RUN apt-get update \
      && apt-get install -y --no-install-recommends \
              clang \
              curl \
              dirmngr \
              git \
              gperf \
              libb64-0d \
              libcurl4-openssl-dev \
              libgoogle-perftools-dev \
              libjemalloc-dev \
              libnuma-dev \
              software-properties-common \
              wget \
              libgomp1 \
              python3-pip \
      && rm -rf /var/lib/apt/lists/*

# Set TCMALLOC_RELEASE_RATE for users setting LD_PRELOAD with tcmalloc
ENV TCMALLOC_RELEASE_RATE 200

ENV DCGM_VERSION 3.3.6
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN curl -o /tmp/cuda-keyring.deb \
          https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
      && apt install /tmp/cuda-keyring.deb \
      && rm /tmp/cuda-keyring.deb \
      && apt-get update \
      && apt-get install -y datacenter-gpu-manager=1:3.3.6

# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
    && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
    && ldconfig \
    && rm -f ${_CUDA_COMPAT_PATH}/lib

ENV PIP_BREAK_SYSTEM_PACKAGES=1
# python3, python3-pip and some pip installs required for the python backend
RUN apt-get update \
      && apt-get install -y --no-install-recommends \
            python3 \
            libarchive-dev \
            python3-pip \
            python3-wheel \
            python3-setuptools \
            libpython3-dev \
      && pip3 install --upgrade \
            "numpy<2" \
            opencv-python-headless \
            virtualenv \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
ENV NVIDIA_PRODUCT_NAME="Triton Server"
# COPY /entrypoint.d/ /opt/nvidia/entrypoint.d/

ENV NVIDIA_BUILD_ID 124543091
LABEL com.nvidia.build.id=124543091
LABEL com.nvidia.build.ref=500dce76552ace5128a42607944640c78ff468d7

WORKDIR /opt/tritonserver
COPY --chown=1000:1000 --from=full /opt/tritonserver/LICENSE .
COPY --chown=1000:1000 --from=full /opt/tritonserver/TRITON_VERSION .
COPY --chown=1000:1000 --from=full /opt/tritonserver/NVIDIA_Deep_Learning_Container_License.pdf .
COPY --chown=1000:1000 --from=full /opt/tritonserver/bin bin/
COPY --chown=1000:1000 --from=full /opt/tritonserver/lib lib/
COPY --chown=1000:1000 --from=full /opt/tritonserver/include include/
# Copying over backends 
COPY --chown=1000:1000 --from=full /opt/tritonserver/backends/onnxruntime /opt/tritonserver/backends/onnxruntime
COPY --chown=1000:1000 --from=full /opt/tritonserver/backends/openvino /opt/tritonserver/backends/openvino
COPY --chown=1000:1000 --from=full /opt/tritonserver/backends/python /opt/tritonserver/backends/python

# Top-level /opt/tritonserver/backends not copied so need to explicitly set permissions here
RUN chown triton-server:triton-server /opt/tritonserver/backends
#  Copying over repoagents 
#  Copying over caches 

COPY --chown=1000:1000 --from=full /usr/bin/serve /usr/bin/.

# Configure as portas do Triton Server
EXPOSE 8000 8001

# Comando para iniciar o Triton Server
CMD ["tritonserver", \
     "--model-repository=gs://iag-training/triton-server/image/signature-detection/models", \
     "--model-control-mode=explicit", \
     "--load-model=*", \
     "--log-verbose=1", \
     "--allow-metrics=false"]