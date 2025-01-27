# Object Detection with Triton Inference Server

<table>
  <tr>
    <td>
      <a href="https://github.com/triton-inference-server/server"><img src="https://img.shields.io/badge/Triton-Inference%20Server-76B900?style=for-the-badge&labelColor=black&logo=nvidia" alt="Triton Badge" /></a>
    </td>
    <td>
      <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/-Docker-2496ED?style=for-the-badge&labelColor=black&logo=docker&logoColor=2496ED" alt="Docker Badge" /></a>
    </td>
    <td>
      <a href="https://www.python.org/"><img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&labelColor=black&logo=python&logoColor=3776AB" alt="Python Badge" /></a>
    </td>
    <td>
      <a href="https://www.apache.org/licenses/LICENSE-2.0.html"><img src="https://img.shields.io/badge/Apache-2.0-D22128?style=for-the-badge&labelColor=black&logo=apache" alt="Apache Badge" /></a>
    </td>
    <td>
      <a href="https://opencv.org/"><img src="https://img.shields.io/badge/-Opencv-5C3EE8?style=for-the-badge&labelColor=black&logo=opencv&logoColor=5C3EE8" alt="Opencv Badge" /></a>
    </td>
  </tr>
</table>

This project provides a  pipeline for deploying and performing inference with a YOLOv8 object detection model using [Triton Inference Server](https://github.com/triton-inference-server/server) on Google Cloud's Vertex AI, locally or Docker based systems. The repository includes scripts for automating the deployment process, a graphical user interface for inference, and performance analysis tools for optimizing the model's performance.

## Table of Contents

- [Project Structure](#-project-structure)
- [Features](#%EF%B8%8F-features)
- [Installation](#-installation)
- [Ensemble Model](#--ensemble-model)
- [Inference](#-inference)
  - [Available Methods](#available-methods)
  - [How To Use](#how-to-use)
    - [Graphical User Interface (GUI)](#1-graphical-user-interface-gui)
    - [Command-Line Interface (CLI)](#2-command-line-interface-cli)
    - [ONNX Runtime](#3-onnx-runtime)
- [Limit Endpoint Access](#-limit-endpoint-access)
- [Model Analyzer](#-model-analyzer)
- [Model & Dataset Resources](#-model--dataset-resources)
- [Utils](#-utils)
  - [Downloading Models from Cloud Storage](#1-downloading-models-from-cloud-storage)
  - [Uploading Models to Cloud Storage](#2-uploading-models-to-cloud-storage)
  - [Exporting Models](#3-exporting-models)
- [Contributors](#-contributors)
- [Contributing](#contributing)
- [License](#license)


## 📁 Project Structure

### Key Files
- **[`requirements.txt`](requirements.txt)**: Lists the external libraries and dependencies required for the project.
- **[`server/`](./server/)**: Contains scripts for deploying the model to Triton Inference Server.
  - **[`local/`](./server/local/)**: Scripts for running the Triton Inference Server locally.
  - **[`vertexai/`](./server/vertexai/)**: Scripts for deploying the model to Vertex AI Endpoint.
- **[`signature-detection/`](./signature-detection/)**: Contains scripts for performing inference with the YOLOv8 model.
   - **[`analyzer/`](./signature-detection/analyzer/)**: Contains results and configuration for performance analysis using Triton Model Analyzer.
   - **[`inference/`](./signature-detection/inference/)**: Scripts for performing inference using Triton Client, Vertex AI, or locally and GUI for visualization.
      - **[`inference_onnx.py`](./signature-detection/inference/inference_onnx.py)**: Script for performing inference with ONNX runtime locally.
      - **[`inference_pipeline.py`](./signature-detection/inference/inference_pipeline.py)**: Script for performing inference on images using different methods.
      - **[`predictors.py`](./signature-detection/inference/predictors.py)**: Contains the predictor classes for different inference methods. You can add new predictors for custom inference methods.
  - **[`gui/`](./signature-detection/gui/)**: Contains the Gradio interface for interacting with the deployed model. The [`inference_gui.py`](./signature-detection/gui/inference_gui.py) script can be used to test the model in real time. The UI has built-in examples and plots of results and performance.
   - **[`models/`](./signature-detection/models/)**: Contains the Model Repository for Triton Server, including the YOLOv8 model and pre/post-processing scripts in a Ensemble Model.
   - **[`data/`](./signature-detection/data/)**: Contains the datasets and data processing scripts.
   - **[`utils/`](./signature-detection/utils/)**: Scripts for uploading/download the model to/from Google Cloud Storage or Azure Stoage and exporting the model to ONNX/TensorRT format.
- **[`Dockerfile`](Dockerfile)**: Contains the configuration for building the Docker image for Triton Inference Server. 
  - **[`Dockerfile.dev`](Dockerfile.dev)**: Contains the configuration for building the Docker image for local development.
  - **[`docker-compose.yml`](docker-compose.yml)**: Contains the configuration for running Dockerfile.dev.

- **[`entrypoint.sh`](entrypoint.sh)**: Script for initializing the Triton Inference Server with the required configurations.
- **[`LICENSE`](LICENSE)**: The license for the project.

## 🛠️ Features

- **Seamless Model Deployment**: Automates the deployment of the YOLOv8 model using Triton Inference Server.
- **Multi-Backend Support**: Allows inference locally, on Vertex AI, or directly with Triton Client.
- **Optimized Performance**: Utilizes Triton's features like dynamic batching, OpenVINO backend and Ensemble Model for efficient inference.
- **GUI for Easy Inference**: Provides an intuitive Gradio interface for interacting with the deployed model.
- **Automated Scripts**: Includes scripts for model uploading, server startup, and resource cleanup.

## 💻 Installation 

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/t4ai-triton-server.git
   ```
2. **Install dependencies** (Optional: Create a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure your environment**: Set up Google Cloud credentials and env file (See [.env.example](.env.example)).
4. **Build and deploy**: 
   - **Vertex AI:** Follow the instructions in [`deploy_vertex_ai.sh`](server/vertexai/deploy_vertex_ai.sh) to deploy the model to Vertex AI Endpoint. Or programmatically using [`nvidia_triton_custom_container_prediction.ipynb`](server/vertexai/nvidia_triton_custom_container_prediction.ipynb).
   - **Docker:** Run the Triton Inference Server using the provided [Dockerfile](Dockerfile.dev). The [`serve_triton_local_.py`](server/local/serve_triton_local.py) script can be used to start the server locally.
    - **docker compose:** You can use the provided [`docker-compose.yml`](docker-compose.yml).
5. **Run inference**: The scripts in signature-detection/inference can be used to perform inference on images using differents methods (requests, triton client, vertex ai).
   - **GUI:** Use the [`inference_gui.py`](signature-detection/gui/inference_gui.py) to test the deployed model and visualize the results.
   - **CLI:** Use the [`inference_pipeline.py`](signature-detection/inference/inference_pipeline.py) script to select predictor and perform inference on test dataset images.
   - **ONNX:** Use the [`inference_onnx.py`](signature-detection/inference/inference_onnx.py) script to perform inference with the ONNX runtime locally.

## 🧩  Ensemble Model

The repository includes an [Ensemble Model](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models) for the YOLOv8 object detection model. The Ensemble Model combines the YOLOv8 model with pre and post-processing scripts to perform inference on images. The model repository is located in the [`models/`](signature-detection/models) directory.

```mermaid
flowchart TB
    subgraph "Triton Inference Server"
        direction TB
        subgraph "Ensemble Model Pipeline"
            direction TB
            subgraph Input
                raw["raw_image
                 (UINT8, [-1])"]
                conf["confidence_threshold
                 (FP16, [1])"]
                iou["iou_threshold
                 (FP16, [1])"]
            end

            subgraph "Preprocess Py-Backend"
                direction TB
                pre1["Decode Image
                    BGR to RGB"]
                pre2["Resize (640x640)"]
                pre3["Normalize (/255.0)"]
                pre4["Transpose
                [H,W,C]->[C,H,W]"]
                pre1 --> pre2 --> pre3 --> pre4
            end

            subgraph "YOLOv8 Model ONNX Backend"
                yolo["Inference YOLOv8s"]
            end

            subgraph "Postproces Python Backend"
                direction TB
                post1["Transpose
                   Outputs"]
                post2["Filter Boxes (confidence_threshold)"]
                post3["NMS (iou_threshold)"]
                post4["Format Results [x,y,w,h,score]"]
                post1 --> post2 --> post3 --> post4
            end

            subgraph Output
                result["detection_result
                    (FP16, [-1,5])"]
            end

            raw --> pre1
            pre4 --> |"preprocessed_image (FP32, [3,-1,-1])"| yolo
            yolo --> |"output0"| post1
            conf --> post2
            iou --> post3
            post4 --> result
        end
    end

    subgraph Client
        direction TB
        client_start["Client Application"]
        response["Detections Result
                [x,y,w,h,score]"]
    end

    client_start -->|"HTTP/gRPC Request
          with raw image
          confidence_threshold
          iou_threshold"| raw
    result -->|"HTTP/gRPC Response with detections"| response

    style Client fill:#e6f3ff,stroke:#333
    style Input fill:#f9f,stroke:#333
    style Output fill:#9ff,stroke:#333
```

---

## ⚡ Inference

The inference module allows you to perform image analysis using different methods, leveraging both local and cloud-based solutions. The pipeline is designed to be flexible and supports multiple prediction methods, making it easy to experiment and deploy in different environments.

### Available Methods

The pipeline supports the following inference methods:

1. **Triton Client**: Inference using the Triton Inference Server SDK.
2. **Vertex AI**: Inference using Google Cloud's Vertex AI Endpoint.
3. **HTTP**: Inference using HTTP requests to the Triton Inference Server.

### How To Use

The inference module provides both a graphical user interface (GUI) and command-line tools for performing inference.

#### 1. **Graphical User Interface (GUI)**

The GUI allows you to interactively test the deployed model and visualize the results in real-time.

- **Script**: [`inference_gui.py`](signature-detection/gui/inference_gui.py)
- **Usage**: Run the script to launch the GUI interface.

```bash
python signature-detection/gui/inference_gui.py --triton-url {triton_url} 
```

https://github.com/user-attachments/assets/d41a45a1-8783-41a6-b963-b315d0e994b4

#### 2. **Command-Line Interface (CLI)**

The CLI tool provides a flexible way to perform inference on a dataset using different predictors.

- **Script**: [`inference_pipeline.py`](signature-detection/inference/inference_pipeline.py)
- **Usage**: The script will show a menu to select a predictor and perform inference on the test dataset.

```bash
python signature-detection/inference/inference_pipeline.py
```

<details> 
<summary><strong>💡</strong></summary>
<div style="background-color:rgb(38, 35, 41); border-radius: 4px; padding: 1rem; border-left: 4px solid #2ecc71; margin: 1rem 0;">
  <div style="display: flex; align-items: center;">
    <div>
      <p style="color: #ffffff;  font-size: 1.0em;">
        This script calculates metrics of inference time and gives you a tabulated final report like this:
      </p>
      <pre style="background-color: rgb(38, 35, 41); padding: 1rem; border-radius: 4px; margin: 1rem 0;">
        <code style="color: #fff;">
+-----------------------+----------------------+
| Métrica               | Valor                |
+=======================+======================+
| <span style="color: #2ecc71">Tempo médio (ms)</span>      | 141.20447635650635   |
+----------------------------+-----------------+
| <span style="color: #2ecc71">Desvio padrão (ms)</span>    | 17.0417248165512     |
+----------------------------+-----------------+
| <span style="color: #2ecc71">Tempo máximo (ms)</span>     | 175.67205429077148   |
+----------------------------+-----------------+
| <span style="color: #2ecc71">Tempo mínimo (ms)</span>     | 125.48470497131348   |
+----------------------------+-----------------+
| <span style="color: #2ecc71">Tempo total (min)</span>     | 00:02:541            |
+----------------------------+-----------------+
| <span style="color: #2ecc71">Número de inferências</span> | 18                   |
+----------------------------+-----------------+
        </code>
      </pre>
    </div>
  </div>
</div>
</details>

#### 3. **ONNX Runtime**

For local inference without relying on external services, you can use the ONNX runtime.

- **Script**: [`inference_onnx.py`](signature-detection/inference/inference_onnx.py)
- **Usage**: Perform inference with the ONNX runtime locally.

```bash
python signature-detection/inference/inference_onnx.py \
  --model_path {onnx_model_path} \
  --img './input/test_image.jpg' \
  --conf-thres 0.5 \
  --iou-thres 0.5
```

- All arguments are `optional`, the default values are:
  - `--model_path`: `signature-detection/models/yolov8s.onnx`
  - `--img`: Random image from the test dataset
  - `--conf-thres`: `0.5`
  - `--iou-thres`: `0.5`

### Extending the Pipeline

If you need to extend the inference pipeline or add custom prediction methods, you can:

1. Create a new predictor class that inherits from `BasePredictor`.
2. Implement the required methods (`request`, `format_response`, etc.).
3. Update the `InferencePipeline` to support the new predictor.

#### Class Diagram

The inference pipeline is built around a modular class structure that allows for easy extension and customization. Here's the class hierarchy:

```mermaid
classDiagram
    class ABC {
    }
    class BasePredictor {
        +__init__()
        +request(input)
        +format_response(response)
        +predict(input)
    }
    class HttpPredictor {
        +__init__(url)
        ~_create_payload(image)
        +request(input)
        +format_response(response)
    }
    class VertexAIPredictor {
        +__init__(url, access_token)
        ~_get_google_access_token()
    }
    class TritonClientPredictor {
        +__init__(url, endpoint, scheme)
        +request()
        +format_response(response)
    }
    class InferencePipeline {
        +__init__(predictor)
        +run(image_path)
        ~_process_response(response)
    }
  
    ABC <|-- BasePredictor
    BasePredictor <|-- HttpPredictor
    HttpPredictor <|-- VertexAIPredictor
    BasePredictor <|-- TritonClientPredictor
    InferencePipeline --> BasePredictor : uses
```


## 🔒 Limit Endpoint Access

To control access to specific server protocols, the server uses the `--http-restricted-api` and `--grpc-restricted-protocol` flags. These flags ensure that only requests containing the required `admin-key` header with the correct value will have access to restricted endpoints.

- Checkout the triton documentation for more information on [Inference Protocols and APIs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html#limit-endpoint-access-beta)

In this project, the entrance configuration restricts access to the following endpoints via both HTTP and GRPC protocols:

### Restricted Endpoints:
- model-repository
- model-config
- shared-memory
- statistics
- trace

### Entry Point Configuration

The [**`entrypoint.sh`**](entrypoint.sh) script is configured to restrict access to the server's administrative endpoints. The access control is enforced via both HTTP and GRPC protocols, ensuring that only requests containing the `admin-key` header with the correct value will be allowed.

```bash
tritonserver \
  --model-repository=${TRITON_MODEL_REPOSITORY} \
  --model-control-mode=explicit \
  --load-model=* \
  --log-verbose=1 \
  --allow-metrics=false \
  --allow-grpc=true \
  --grpc-restricted-protocol=model-repository,model-config,shared-memory,statistics,trace:admin-key=${TRITON_ADMIN_KEY} \
  --http-restricted-api=model-repository,model-config,shared-memory,statistics,trace:admin-key=${TRITON_ADMIN_KEY}
```

### Key Points:
1. **Inference Access**: The server allows inference requests from any user.
2. **Admin Access**: Access to the restricted endpoints (model-repository, model-config, etc.) is limited to requests that include the `admin-key` header with the correct value defined in the `.env` file.
3. **GRPC Protocol**: The GRPC protocol is enabled and restricted in the same way as HTTP, providing consistent security across both protocols.

This configuration ensures that sensitive operations and configurations are protected, while still allowing regular inference requests to proceed without restrictions.


## 📊 Model Analyzer

The Triton Model Analyzer can be used to profile the model and generate performance reports. The [`metrics-model-inference.csv`](signature-detection/analyzer/profile_results/results/metrics-model-inference.csv) file contains performance metrics for various configurations of the YOLOv8 model.

You can run the Model Analyzer using the following command:
```bash
docker run -it  \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd)/signature-detection/models:/signature-detection/models \
    --net=host nvcr.io/nvidia/tritonserver:24.11-py3-sdk 
```

```bash
model-analyzer profile -f perf.yaml \
    --triton-launch-mode=remote --triton-http-endpoint=localhost:8000  \
    --output-model-repository-path /signature-detection/analyzer/configs  \
    --export-path profile_results --override-output-model-repository \
    --collect-cpu-metrics --monitoring-interval=5
```

```bash
model-analyzer report --report-model-configs yolov8s_config_0,yolov8s_config_12,yolov8s_config_4,yolov8s_config_8 ... --export-path /workspace --config-file perf.yaml 
```

You can modify the [`perf.yaml`](signature-detection/analyzer/config/perf.yaml) file to experiment with different configurations and analyze the performance of the model in your deployment environment. See the [Triton Model Analyzer documentation](https://github.com/triton-inference-server/model_analyzer) for more details.

## 🤗 Model & Dataset Resources

This project uses a custom-trained YOLOv8 model for signature detection. All model weights, training artifacts, and the dataset are hosted on Hugging Face to comply with Ultralytics' YOLO licensing requirements and to ensure proper versioning and documentation.

- **Model Repository**: Contains the trained model weights, ONNX exports, and comprehensive model card detailing the training process, performance metrics, and usage guidelines.

  [![Model Card](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/tech4humans/yolov8s-signature-detector)

- **Dataset Repository**: Includes the training dataset, validation splits, and detailed documentation about data collection and preprocessing steps.

  [![Dataset Card](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/tech4humans/signature-detection)

- **Demo Space**: Provides a live demo space for testing the model and dataset using the Hugging Spaces.

  [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/tech4humans/signature-detection)

## 🧰 Utils

The [`utils/`](./signature-detection/utils/) folder contains scripts designed to simplify interactions with cloud storage providers and the process of exporting machine learning models. Below is an overview of the available scripts and their usage examples.

#### 1. **Downloading Models from Cloud Storage**

The [`download_from_cloud.py`](./signature-detection/utils/download_from_cloud.py) script allows you to download models or other files from Google Cloud Storage (GCP) or Azure Blob Storage. Use the appropriate arguments to specify the provider, storage credentials, and paths.

- **Google Cloud Storage (GCP):**
  ```bash
  python signature-detection/utils/download_from_cloud.py --provider gcp --bucket-name <your-bucket-name>
  ```

- **Azure Blob Storage:**
  ```bash
  python signature-detection/utils/download_from_cloud.py --provider az --container-name <your-container-name> --connection-string "<your-connection-string>"
  ```

**Arguments:**
- `--provider`: Specify the cloud provider (`gcp` or `az`).
- `--bucket-name`: GCP bucket name (required for `gcp`).
- `--container-name`: Azure container name (required for `az`).
- `--connection-string`: Azure connection string (required for `az`).
- `--local-folder`: Local folder to save downloaded files (default: `models` folder).
- `--remote-folder`: Remote folder path in the cloud (default: `triton-server/image/signature-detection/models`).


#### 2. **Uploading Models to Cloud Storage**

The [`upload_models_to_cloud.py`](./signature-detection/utils/upload_models_to_cloud.py) script allows you to upload models or files from a local directory to either GCP or Azure storage. 

- **Google Cloud Storage (GCP):**
  ```bash
  python signature-detection/utils/upload_models_to_cloud.py --provider gcp --bucket-name <your-bucket-name>
  ```

- **Azure Blob Storage:**
  ```bash
  python signature-detection/utils/upload_models_to_cloud.py --provider az --container-name <your-container-name> --connection-string "<your-connection-string>"
  ```

**Arguments:**
- `--provider`: Specify the cloud provider (`gcp` or `az`).
- `--bucket-name`: GCP bucket name (required for `gcp`).
- `--container-name`: Azure container name (required for `az`).
- `--connection-string`: Azure connection string (required for `az`).
- `--local-folder`: Local folder containing files to upload (default: `models` folder).
- `--remote-folder`: Remote folder path in the cloud (default: `triton-server/image/signature-detection/models`).

#### 3. **Exporting Models**

The [`export_model.py`](./signature-detection/utils/export_model.py) script simplifies the process of exporting YOLOv8 models to either ONNX or TensorRT formats. This is useful for deploying models in environments requiring specific formats.

- **Export to ONNX:**
  ```bash
  python signature-detection/utils/export_model.py --model-path /path/to/yolov8s.pt --output-path model.onnx --format onnx
  ```

- **Export to TensorRT:**
  ```bash
  python signature-detection/utils/export_model.py --model-path /path/to yolov8s.pt --output-path model.engine --format tensorrt
  ```

**Arguments:**
- `--model-path`: Path to the input model file (e.g., YOLOv8 `.pt` file).
- `--output-path`: Path to save the exported model.
- `--format`: Export format (`onnx` or `tensorrt`).

## 🤝 Contributors 


<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/samuellimabraz"><img src="https://avatars.githubusercontent.com/u/115582014?v=4?s=100" width="100px;" alt="Samuel Lima Braz"/><br /><sub><b>Samuel Lima Braz</b></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jwillians"><img src="https://avatars.githubusercontent.com/u/299830?v=4?s=100" width="100px;" alt="Jorge Willians"/><br /><sub><b>Jorge Willians</b></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/NixonMSilva"><img src="https://avatars.githubusercontent.com/u/15185532?v=4?s=100" width="100px;" alt="Nixon Silva"/><br /><sub><b>Nixon Silva</b></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ronaldobalzi-tech4h"><img src="https://avatars.githubusercontent.com/u/136820259?v=4?s=100" width="100px;" alt="ronaldobalzi-tech4h"/><br /><sub><b>ronaldobalzi-tech4h</b></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="7">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Contributing

First off, thanks for taking the time to contribute! Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make will benefit everybody else and are **greatly appreciated**.


Please read [our contribution guidelines](docs/CONTRIBUTING.md), and thank you for being involved!

## License

This project is licensed under the **Apache Software License 2.0**.

See [LICENSE](LICENSE) for more information.
