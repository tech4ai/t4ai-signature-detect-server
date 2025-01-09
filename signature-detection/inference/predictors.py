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
    Encode image data from file path.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Image data.
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
    def request(self, input: np.ndarray) -> tuple[dict, float]:
        """To be implemented by concrete classes."""
        raise NotImplementedError("Request method not implemented.")

    @abstractmethod
    def format_response(self, response) -> np.ndarray:
        """To be implemented by concrete classes."""
        raise NotImplementedError("Format response method not implemented.")

    def predict(self, input: np.ndarray) -> tuple[np.ndarray, float]:
        """To be implemented by concrete classes."""
        response, mean_time = self.request(input)
        return self.format_response(response), mean_time
    
class HttpPredictor(BasePredictor):
    """Inference predictor for local models."""

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def _create_payload(self, image: np.ndarray) -> dict:
        """
        Create the payload for the HTTP request.
        
        Args:
            image (numpy.ndarray): The input image.
            
        Returns:
            dict: The payload for the HTTP request.     
        """
        return {
            "inputs": [
                {
                    "name": "raw_image",
                    "shape": image.shape,
                    "datatype": "UINT8",
                    "data": image.tolist()
                }
            ]
        }

    def request(self, input: np.ndarray) -> tuple[dict, float]:
        tic = time.time()
        response = requests.post(self.url, headers=self.headers, data=json.dumps(self._create_payload(input)))
        latency = time.time() - tic
        return response.json(), latency
    
    def format_response(self, response: dict) -> np.ndarray:
        out =  np.array(response['outputs'][0]['data']).astype(np.float32)
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
            "Content-Type": "application/json"
        }
    
    def _get_google_access_token(self):
        """
        Fetch the Google access token.
        
        Returns:
            str: The Google access token.
        """
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], stdout=subprocess.PIPE, text=True)
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
        if not endpoint and not scheme:  # Parse all args from URL string
            splits = urlsplit(url)
            endpoint = splits.path.strip("/").split("/")[0]
            scheme = splits.scheme
            url = splits.netloc

        self.endpoint = endpoint
        self.url = url
        self.headers = {"admin-key": os.getenv("TRITON_ADMIN_KEY", "")}
        
        print(f"Connecting to Triton server at {self.url} with endpoint {self.endpoint} using {scheme}...")

        # Choose the Triton client based on the communication scheme
        if scheme == "grpc":
            import tritonclient.grpc as client  # noqa

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint, as_json=True)["config"]
        else:
            import tritonclient.http as client  # noqa
            
            if scheme == "https":
                ssl = True
                ssl_context_factory = gevent.ssl.create_default_context
            else:
                ssl = False
                ssl_context_factory = None

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=ssl, ssl_context_factory=ssl_context_factory)
            config = self.triton_client.get_model_config(endpoint, headers=self.headers)

        # Sort output names alphabetically, i.e. 'output0', 'output1', etc.
        config["output"] = sorted(config["output"], key=lambda x: x.get("name"))

        # pprint(self.triton_client.get_inference_statistics(model_name=self.endpoint, headers=self.headers))
        # pprint(self.triton_client.get_model_repository_index())
        
        # Define model attributes
        type_map = {"TYPE_FP32": np.float32, "TYPE_FP16": np.float16, "TYPE_UINT8": np.uint8}
        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput
        self.input_formats = [x["data_type"] for x in config["input"]]
        self.np_input_formats = [type_map[x] for x in self.input_formats]
        self.input_names = [x["name"] for x in config["input"]]
        self.output_names = [x["name"] for x in config["output"]]
        self.output_formats = [x["data_type"] for x in config["output"]]
        self.np_output_formats = [type_map[x] for x in self.output_formats]
        self.metadata = eval(config.get("parameters", {}).get("metadata", {}).get("string_value", "None"))

    def request(self, *inputs: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Call the model with the given inputs.

        Args:
            *inputs (List[np.ndarray]): Input data to the model.

        Returns:
            (List[np.ndarray]): Model outputs.
        """
        infer_inputs = []
        for i, x in enumerate(inputs):
            if x.dtype != self.np_input_formats[i]:
                x = x.astype(self.np_input_formats[i])
            infer_input = self.InferInput(self.input_names[i], [*x.shape], self.input_formats[i].replace("TYPE_", ""))
            infer_input.set_data_from_numpy(x)
            infer_inputs.append(infer_input)

        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]
        
        tic = time.time()
        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs, request_compression_algorithm=None, response_compression_algorithm='gzip', headers=self.headers)
        latency = time.time() - tic

        return outputs.as_numpy(self.output_names[0]).astype(self.np_output_formats[0]), latency
    
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
 