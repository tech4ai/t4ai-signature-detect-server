import json
import logging
import time

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    A Python model class for preprocessing input data for Triton Inference Server.
    This class must be named `TritonPythonModel` and implements the methods required to
    initialize, execute, and finalize the model.
    """

    def initialize(self, args):
        """
        Initializes the model when it is being loaded.
        This method is called once during the model's lifetime and is used to set up any
        necessary configurations or state required for preprocessing.

        Parameters
        ----------
        args : dict
            A dictionary containing initialization arguments, with keys and values as strings:
            - model_config: JSON string containing the model configuration.
            - model_instance_kind: String indicating the model instance kind.
            - model_instance_device_id: String indicating the model instance device ID.
            - model_repository: Path to the model repository.
            - model_version: Version of the model.
            - model_name: Name of the model.
        """
        # Parse the model configuration from JSON
        self.model_config = model_config = json.loads(args["model_config"])

        # Retrieve the output configuration for "preprocessed_image"
        preprocessed_image_config = pb_utils.get_output_config_by_name(
            model_config, "preprocessed_image"
        )

        # Convert Triton data types to NumPy data types
        self.preprocessed_image_dtype = pb_utils.triton_string_to_numpy(
            preprocessed_image_config["data_type"]
        )

    def execute(self, requests):
        """
        Handles inference requests and returns preprocessed data.

        This method processes one or more requests, decodes and preprocesses images,
        and constructs responses with the preprocessed tensors.

        Parameters
        ----------
        requests : list
            A list of `pb_utils.InferenceRequest` objects containing input tensors for preprocessing.

        Returns
        -------
        list
            A list of `pb_utils.InferenceResponse` objects, one for each request.
        """
        responses = []

        for request in requests:
            # Retrieve the input tensor "raw_image"
            raw_image = pb_utils.get_input_tensor_by_name(request, "raw_image")

            # Decode and preprocess the input image
            img = raw_image.as_numpy()
            image = cv2.imdecode(
                np.frombuffer(img.tobytes(), np.uint8), cv2.IMREAD_COLOR
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = cv2.resize(image, (640, 640))  # Resize to 640x640
            image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
            image = np.transpose(
                image, (2, 0, 1)
            )  # Rearrange dimensions [H, W, C] -> [C, H, W]
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Convert the preprocessed image to a Triton tensor
            # Create an inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "preprocessed_image",
                        image.astype(self.preprocessed_image_dtype),
                    )
                ]
            )

            responses.append(inference_response)

        return responses

    def finalize(self):
        """
        Finalizes the model when it is being unloaded.
        This method is called once during the model's lifetime and can be used to
        clean up resources or perform other teardown tasks.
        """
        pass
