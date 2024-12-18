from typing import List
from urllib.parse import urlsplit
import time
import pprint
import os
import cv2
import numpy as np
from utils.preprocessing import preprocess
from utils.postprocessing import postprocess

 
class TritonRemoteModel:
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

        # Choose the Triton client based on the communication scheme
        if scheme == "http":
            import tritonclient.http as client  # noqa

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint)
        else:
            import tritonclient.grpc as client  # noqa

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint, as_json=True)["config"]

        pprint.pprint(config)
        # Sort output names alphabetically, i.e. 'output0', 'output1', etc.
        config["output"] = sorted(config["output"], key=lambda x: x.get("name"))

        # Define model attributes
        type_map = {"TYPE_FP32": np.float32, "TYPE_FP16": np.float16, "TYPE_UINT8": np.uint8}
        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput
        self.input_formats = [x["data_type"] for x in config["input"]]
        self.np_input_formats = [type_map[x] for x in self.input_formats]
        self.input_names = [x["name"] for x in config["input"]]
        self.output_names = [x["name"] for x in config["output"]]
        self.metadata = eval(config.get("parameters", {}).get("metadata", {}).get("string_value", "None"))

    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:
        """
        Call the model with the given inputs.

        Args:
            *inputs (List[np.ndarray]): Input data to the model.

        Returns:
            (List[np.ndarray]): Model outputs.
        """
        infer_inputs = []
        input_format = inputs[0].dtype
        for i, x in enumerate(inputs):
            if x.dtype != self.np_input_formats[i]:
                x = x.astype(self.np_input_formats[i])
            infer_input = self.InferInput(self.input_names[i], [*x.shape], self.input_formats[i].replace("TYPE_", ""))
            infer_input.set_data_from_numpy(x)
            infer_inputs.append(infer_input)

        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]
        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs)

        return [outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names]

if __name__ == "__main__":
    model = TritonRemoteModel(url="localhost:8000", endpoint="yolov8s", scheme='http')

REDACTED_PATH
    paths = [root_dir_images + image for image in os.listdir(root_dir_images) if image.endswith(".jpg")][:1]
    np_orig_images = [cv2.imread(path) for path in paths]

    # Run inference on the server
    times = []
    i=0
    total_tic = time.time()

    for path in paths:
        print(f"Example: {path}")
        orig_image = cv2.imread(path)
        np_image = preprocess(path)
        tic = time.time()
        preds = model(np_image)
        times.append(time.time()-tic)
        
        results = postprocess(preds[0], np_image, [orig_image], [path])
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            print(boxes)
            result.show()
        
            
    print(f"Total inference time: {time.time()-total_tic}")
    print(f"Average inference time: {sum(times)/len(times)}")
