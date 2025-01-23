import gzip
import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urlsplit

import numpy as np
import requests


def encode_image(image_path: str) -> np.ndarray:
    """
    Encode an image file as a numpy array in uint8 format and add a batch dimension.

    Args:
        image_path (str): Path to the image file.

    Returns:
        image_data (numpy.ndarray): Image data as a numpy array with shape (1, H, W, C).
    """
    image_data = np.fromfile(image_path, dtype="uint8")
    image_data = np.expand_dims(image_data, axis=0)
    return image_data


class BasePredictor(ABC):
    """Abstract base class for predictors."""

    @abstractmethod
    def request(
        self, input: np.ndarray, confidence_threshold: float, iou_threshold: float
    ) -> tuple[dict, float]:
        """Send a request for inference. Must be implemented by subclasses.

        Args:
            input (np.ndarray): Input data for inference.
            confidence_threshold (float): Confidence threshold for predictions.
            iou_threshold (float): IoU threshold for predictions.

        Returns:
            tuple[dict, float]: Response dictionary and inference time.
        """
        raise NotImplementedError("Request method not implemented.")

    @abstractmethod
    def format_response(self, response: dict) -> np.ndarray:
        """Format the response from inference. Must be implemented by subclasses.

        Args:
            response (dict): Raw response from the server.

        Returns:
            np.ndarray: Formatted inference results.
        """
        raise NotImplementedError("Format response method not implemented.")

    def predict(
        self,
        input: np.ndarray,
        confidence_threshold: float = None,
        iou_threshold: float = None,
    ) -> tuple[np.ndarray, float]:
        """Perform inference and format the result.

        Args:
            input (np.ndarray): Input data for inference.
            confidence_threshold (float, optional): Confidence threshold. Defaults to None.
            iou_threshold (float, optional): IoU threshold. Defaults to None.

        Returns:
            tuple[np.ndarray, float]: Formatted inference results and inference time.
        """
        response, inference_time = self.request(
            input, confidence_threshold, iou_threshold
        )
        return self.format_response(response), inference_time


class HttpPredictor(BasePredictor):
    """Predictor for making HTTP requests to a Triton server."""

    def __init__(self, url: str):
        """Initialize the HTTP predictor.

        Args:
            url (str): URL of the Triton server.
        """
        super().__init__()
        self.url = url
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Content-Encoding": "gzip",
            "Connection": "keep-alive",
        }

    def _create_payload(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        iou_threshold: float = None,
    ) -> dict:
        """Create payload for the HTTP request.

        Args:
            image (np.ndarray): Input image(s) for inference.
            confidence_threshold (float, optional): Confidence threshold. Defaults to None.
            iou_threshold (float, optional): IoU threshold. Defaults to None.

        Returns:
            dict: Payload for the HTTP request.
        """
        batch_size = image.shape[0] if len(image.shape) == 4 else 1
        payload = {
            "inputs": [
                {
                    "name": "raw_image",
                    "shape": list(image.shape),
                    "datatype": "UINT8",
                    "data": image.tolist(),
                    "compression": "deflate",
                }
            ]
        }

        if confidence_threshold is not None:
            payload["inputs"].append(
                {
                    "name": "confidence_threshold",
                    "shape": [batch_size, 1],
                    "datatype": "FP32",
                    "data": [[confidence_threshold]] * batch_size,
                    "compression": "deflate",
                }
            )

        if iou_threshold is not None:
            payload["inputs"].append(
                {
                    "name": "iou_threshold",
                    "shape": [batch_size, 1],
                    "datatype": "FP32",
                    "data": [[iou_threshold]] * batch_size,
                    "compression": "deflate",
                }
            )

        return payload

    def request(
        self,
        input: np.ndarray,
        confidence_threshold: float = None,
        iou_threshold: float = None,
    ) -> tuple[dict, float]:
        """Send an HTTP request to the Triton server.

        Args:
            input (np.ndarray): Input data for inference.
            confidence_threshold (float, optional): Confidence threshold. Defaults to None.
            iou_threshold (float, optional): IoU threshold. Defaults to None.

        Returns:
            tuple[dict, float]: Server response and inference time.
        """
        try:
            payload = self._create_payload(input, confidence_threshold, iou_threshold)
            compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))

            start_time = time.time()
            response = self.session.post(
                self.url, headers=self.headers, data=compressed_payload
            )
            inference_time = time.time() - start_time

            response.raise_for_status()
            return response.json(), inference_time

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request failed: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    def format_response(self, response: dict) -> np.ndarray:
        """Format the server response into a numpy array.

        Args:
            response (dict): Raw server response.

        Returns:
            np.ndarray: Formatted inference results.
        """
        output_data = np.array(response["outputs"][0]["data"], dtype=np.float32)
        num_detections = output_data.shape[0] // 5
        return output_data.reshape((num_detections, 5))


class VertexAIPredictor(HttpPredictor):
    """Predictor for Vertex AI models using HTTP."""

    def __init__(self, url: str, access_token: Optional[str] = None):
        """
        Initialize the VertexAIPredictor.

        Args:
            url (str): The endpoint URL of the Vertex AI model.
            access_token (Optional[str]): Access token for authentication.
        """
        super().__init__(url=url)

        if access_token is None:
            try:
                access_token = self._fetch_google_access_token()
            except Exception as e:
                raise RuntimeError(f"Failed to fetch access token: {e}")

        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _fetch_google_access_token() -> str:
        """
        Retrieve a Google Cloud access token using the gcloud CLI.

        Returns:
            str: The access token.

        Raises:
            RuntimeError: If the access token retrieval fails.
        """
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error retrieving access token: {e}")


class TritonClientPredictor(BasePredictor):
    """
    Client for interacting with a Triton Inference Server.
    """

    # Static model configuration
    MODEL_CONFIG = {
        "input": [
            {"data_type": "TYPE_UINT8", "dims": ["-1"], "name": "raw_image"},
            {"data_type": "TYPE_FP16", "dims": ["1"], "name": "confidence_threshold"},
            {"data_type": "TYPE_FP16", "dims": ["1"], "name": "iou_threshold"},
        ],
        "output": [
            {"data_type": "TYPE_FP16", "dims": ["-1", "5"], "name": "detection_result"}
        ],
    }

    def __init__(self, url: str, endpoint: str = "", scheme: str = ""):
        """
        Initialize the TritonRemoteModel.

        Arguments may be provided individually or parsed from a collective 'url' argument of the form
            <scheme>://<netloc>/<endpoint>/<task_name>

        Args:
            url (str): The URL of the Triton server.
            endpoint (str): The name of the model on the Triton server.
            scheme (str): The communication scheme ('http' or 'grpc').
        """
        if not endpoint and not scheme:
            splits = urlsplit(url)
            endpoint = splits.path.strip("/").split("/")[0]
            scheme = splits.scheme
            url = splits.netloc

        self.endpoint = endpoint
        self.url = url
        self.scheme = scheme

        print(
            f"Connecting to Triton server at {self.url} with endpoint {self.endpoint} using {scheme}..."
        )

        if scheme[:4] == "http":
            import tritonclient.http as client

            self.triton_client = client.InferenceServerClient(
                url=self.url,
                verbose=False,
                ssl=True if scheme == "https" else False,
            )
        else:
            import tritonclient.grpc as client

            self.triton_client = client.InferenceServerClient(
                url=self.url, verbose=False, ssl=False
            )

        # Initialize model-related attributes based on the static configuration
        config = self.MODEL_CONFIG
        config["output"] = sorted(config["output"], key=lambda x: x["name"])

        self.np_type_map = {
            "TYPE_FP32": np.float32,
            "TYPE_FP16": np.float16,
            "TYPE_UINT8": np.uint8,
        }
        self.type_map = {
            "TYPE_FP32": "FP32",
            "TYPE_FP16": "FP16",
            "TYPE_UINT8": "UINT8",
        }
        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput
        self.input_formats = [x["data_type"] for x in config["input"]]
        self.np_input_formats = [self.np_type_map[x] for x in self.input_formats]
        self.input_names = [x["name"] for x in config["input"]]
        self.output_names = [x["name"] for x in config["output"]]
        self.output_formats = [x["data_type"] for x in config["output"]]
        self.np_output_formats = [self.np_type_map[x] for x in self.output_formats]

    def request(
        self,
        input: np.ndarray,
        confidence_threshold: float = None,
        iou_threshold: float = None,
    ) -> tuple[np.ndarray, float]:
        """
        Make an inference request to the Triton server.

        Args:
            input (np.ndarray): Input image tensor
            confidence_threshold (float, optional): Confidence threshold for detection
            iou_threshold (float, optional): IOU threshold for NMS

        Returns:
            tuple[np.ndarray, float]: Detection results and inference time
        """
        infer_inputs = []

        # # Get batch size from input
        batch_size = input.shape[0] if len(input.shape) == 4 else 1

        # Create inputs based on MODEL_CONFIG
        for input_config in self.MODEL_CONFIG["input"]:
            input_name = input_config["name"]
            data_type = self.type_map[input_config["data_type"]]

            if input_name == "raw_image":
                # Handle image input
                infer_input = self.InferInput(input_name, input.shape, data_type)
                # Convert to appropriate type using np_input_formats mapping
                input_idx = self.input_names.index(input_name)
                infer_input.set_data_from_numpy(
                    input.astype(self.np_input_formats[input_idx])
                )
                infer_inputs.append(infer_input)

            elif (
                input_name == "confidence_threshold"
                and confidence_threshold is not None
            ):
                # Handle confidence threshold
                infer_input = self.InferInput(input_name, [batch_size, 1], data_type)
                input_idx = self.input_names.index(input_name)
                infer_input.set_data_from_numpy(
                    np.array(
                        [[confidence_threshold]] * batch_size,
                        dtype=self.np_input_formats[input_idx],
                    )
                )
                infer_inputs.append(infer_input)

            elif input_name == "iou_threshold" and iou_threshold is not None:
                # Handle IOU threshold
                infer_input = self.InferInput(input_name, [batch_size, 1], data_type)
                input_idx = self.input_names.index(input_name)
                infer_input.set_data_from_numpy(
                    np.array(
                        [[iou_threshold]] * batch_size,
                        dtype=self.np_input_formats[input_idx],
                    )
                )
                infer_inputs.append(infer_input)

        # Configurar saídas
        infer_outputs = [
            self.InferRequestedOutput(output_name) for output_name in self.output_names
        ]

        infer_args = {
            "model_name": self.endpoint,
            "inputs": infer_inputs,
            "outputs": infer_outputs,
            "priority": 2,
        }

        if self.scheme[:4] == "http":
            infer_args["request_compression_algorithm"] = "deflate"
            infer_args["response_compression_algorithm"] = "deflate"
        else:
            infer_args["compression_algorithm"] = "deflate"

        try:
            tic = time.time()
            outputs = self.triton_client.infer(**infer_args)
            latency = time.time() - tic
            return (
                outputs.as_numpy(self.output_names[0]).astype(
                    self.np_output_formats[0]
                ),
                latency,
            )
        except Exception as e:
            raise RuntimeError(f"Inference request failed: {e}")

    def format_response(self, response):
        """
        Format the response from the model.
            [-1, 5] -> [num_detections, 5]

        Args:
            response (numpy.ndarray): The model response.

        Returns:
            numpy.ndarray: Formatted detection results.
        """
        num_detections = response.shape[0]
        return response.reshape((num_detections, 5))

    # Restricted operations
    def _get_headers(self, admin_key: str = None) -> dict:
        """Generate headers for restricted server operations."""
        if not admin_key:
            admin_key = os.getenv("TRITON_ADMIN_KEY", "")
        if not admin_key:
            raise ValueError("Admin key is required for this operation.")
        if self.scheme[:4] == "http":
            return {"admin-key": admin_key}
        else:
            return {"triton-grpc-protocol-admin-key": admin_key}

    def get_model_config(self, admin_key: str = None) -> dict:
        """Fetch model configuration from the server."""
        headers = self._get_headers(admin_key)
        return self.triton_client.get_model_config(self.endpoint, headers=headers)

    def get_model_repository_index(self, admin_key: str = None) -> list:
        """Fetch the model repository index from the server."""
        headers = self._get_headers(admin_key)
        return self.triton_client.get_model_repository_index(headers=headers)

    def get_inference_statistics(self, admin_key: str = None) -> dict:
        """Fetch inference statistics from the server."""
        headers = self._get_headers(admin_key)
        return self.triton_client.get_inference_statistics(
            model_name=self.endpoint, headers=headers
        )
