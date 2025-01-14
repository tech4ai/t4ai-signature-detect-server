import os
import json
import subprocess
import time
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Optional
from urllib.parse import urlsplit

import gevent
import numpy as np
import requests


def encode_image(image_path):
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


## Strategy Pattern for Inference
class BasePredictor(ABC):
    """Base class for predictors."""

    def __init__(self):
        pass

    @abstractmethod
    def request(
        self, input: np.ndarray, confidence_threshold: float, iou_threshold: float
    ) -> tuple[dict, float]:
        """To be implemented by concrete classes."""
        raise NotImplementedError("Request method not implemented.")

    @abstractmethod
    def format_response(self, response) -> np.ndarray:
        """To be implemented by concrete classes."""
        raise NotImplementedError("Format response method not implemented.")

    def predict(
        self, input: np.ndarray, confidence_threshold: float, iou_threshold: float
    ) -> tuple[np.ndarray, float]:
        """To be implemented by concrete classes."""
        response, mean_time = self.request(input, confidence_threshold, iou_threshold)
        return self.format_response(response), mean_time


class HttpPredictor(BasePredictor):
    """Inference predictor for local models."""

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def _create_payload(
        self, image: np.ndarray, confidence_threshold: float, iou_threshold: float
    ) -> dict:
        """
        Create the payload for the HTTP request.

        Args:
            image (numpy.ndarray): The input image.
            confidence_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IOU threshold for NMS.

        Returns:
            dict: The payload for the HTTP request.
        """
        return {
            "inputs": [
                {
                    "name": "raw_image",
                    "shape": image.shape,
                    "datatype": "UINT8",
                    "data": image.tolist(),
                },
                {
                    "name": "confidence_threshold",
                    "shape": [1],
                    "datatype": "FP16",
                    "data": [np.float16(confidence_threshold)],
                },
                {
                    "name": "iou_threshold",
                    "shape": [1],
                    "datatype": "FP16",
                    "data": [np.float16(iou_threshold)],
                },
            ]
        }

    def request(
        self, input: np.ndarray, confidence_threshold: float, iou_threshold: float
    ) -> tuple[dict, float]:
        tic = time.time()
        response = requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(
                self._create_payload(input, confidence_threshold, iou_threshold)
            ),
        )
        latency = time.time() - tic
        return response.json(), latency

    def format_response(self, response: dict) -> np.ndarray:
        out = np.array(response["outputs"][0]["data"]).astype(np.float32)
        num_detections = out.shape[0] // 5
        return out.reshape((num_detections, 5))


class VertexAIPredictor(HttpPredictor):
    """Inference predictor for Vertex AI models."""

    def __init__(self, url, access_token: Optional[str] = None):
        super().__init__(url=url)

        if access_token is None:
            try:
                self.access_token = self._get_google_access_token()
            except Exception as e:
                print("Error fetching access token:", e)
                return

        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    def _get_google_access_token(self):
        """
        Fetch the Google access token.

        Returns:
            str: The Google access token.
        """
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"], stdout=subprocess.PIPE, text=True
        )
        return result.stdout.strip()


class TritonClientPredictor(BasePredictor):
    """
    Client for interacting with a remote Triton Inference Server model.

    Attributes:
        endpoint (str): The name of the model on the Triton server.
        url (str): The URL of the Triton server.
        triton_client: The Triton client (either HTTP or gRPC).
        InferInput: The input class for the Triton client.
        InferRequestedOutput: The output request class for the Triton client.
        input_formats (List[str]): The data types of the model inputs.
        np_input_formats (List[type]): The numpy data types of the model inputs.
        input_names (List[str]): The names of the model inputs.
        output_names (List[str]): The names of the model outputs.
    """

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

            ssl = False
            ssl_context_factory = None
            if scheme == "https":
                ssl = True
                ssl_context_factory = None

            self.triton_client = client.InferenceServerClient(
                url=self.url,
                verbose=False,
                ssl=ssl,
                ssl_context_factory=ssl_context_factory,
            )

            self.headers = {"admin-key": os.getenv("TRITON_ADMIN_KEY", "")}
            config = self.triton_client.get_model_config(endpoint, headers=self.headers)
        else:
            import tritonclient.grpc as client

            self.triton_client = client.InferenceServerClient(
                url=self.url, verbose=False, ssl=True
            )

            self.headers = {
                "triton-grpc-protocol-admin-key": os.getenv("TRITON_ADMIN_KEY", "")
            }
            config = self.triton_client.get_model_config(
                endpoint, headers=self.headers, as_json=True
            )["config"]

        # Sort output names alphabetically, i.e. 'output0', 'output1', etc.
        config["output"] = sorted(config["output"], key=lambda x: x.get("name"))

        pprint(
            self.triton_client.get_inference_statistics(
                model_name=self.endpoint, headers=self.headers
            )
        )

        # Define model attributes
        type_map = {
            "TYPE_FP32": np.float32,
            "TYPE_FP16": np.float16,
            "TYPE_UINT8": np.uint8,
        }
        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput
        self.input_formats = [x["data_type"] for x in config["input"]]
        self.np_input_formats = [type_map[x] for x in self.input_formats]
        self.input_names = [x["name"] for x in config["input"]]
        self.output_names = [x["name"] for x in config["output"]]
        self.output_formats = [x["data_type"] for x in config["output"]]
        self.np_output_formats = [type_map[x] for x in self.output_formats]

    def request(
        self,
        input: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> tuple[np.ndarray, float]:
        infer_inputs = []

        # Input image tensor (com dimensão de batch)
        batch_size = (
            input.shape[0] if len(input.shape) == 4 else 1
        )  # Suporta inputs com e sem batch explícito
        infer_input = self.InferInput("raw_image", input.shape, "UINT8")
        infer_input.set_data_from_numpy(input.astype(np.uint8))
        infer_inputs.append(infer_input)

        # Confidence threshold tensor (com dimensão de batch)
        infer_input = self.InferInput("confidence_threshold", [batch_size, 1], "FP16")
        infer_input.set_data_from_numpy(
            np.array([[confidence_threshold]] * batch_size, dtype=np.float16)
        )
        infer_inputs.append(infer_input)

        # IOU threshold tensor (com dimensão de batch)
        infer_input = self.InferInput("iou_threshold", [batch_size, 1], "FP16")
        infer_input.set_data_from_numpy(
            np.array([[iou_threshold]] * batch_size, dtype=np.float16)
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
        }

        if self.scheme[:4] == "http":
            infer_args["request_compression_algorithm"] = "deflate"
            infer_args["response_compression_algorithm"] = "deflate"
        else:
            infer_args["compression_algorithm"] = "deflate"

        tic = time.time()
        outputs = self.triton_client.infer(
            **infer_args,
        )
        latency = time.time() - tic

        return (
            outputs.as_numpy(self.output_names[0]).astype(self.np_output_formats[0]),
            latency,
        )

    def format_response(self, response):
        """
        Format the response from the model.
            [-1, 5] -> [num_detections, 5]

        Args:
            response (numpy.ndarray): The model response.

        Returns:
            numpy.ndarray: The formatted response.
        """
        num_detections = response.shape[0]
        return response.reshape((num_detections, 5))
