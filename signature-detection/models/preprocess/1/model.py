import numpy as np
import json
import triton_python_backend_utils as pb_utils
import cv2
import time
import logging


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        preprocessed_image_config = pb_utils.get_output_config_by_name(
            model_config, "preprocessed_image"
        )

        # Convert Triton types to numpy types
        self.preprocessed_image_dtype = pb_utils.triton_string_to_numpy(
            preprocessed_image_config['data_type']
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        times = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            raw_image = pb_utils.get_input_tensor_by_name(request, "raw_image")

            tic = time.time_ns()
            img = raw_image.as_numpy()
            image = cv2.imdecode(np.frombuffer(img.tobytes(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 640))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # [H, W, C] -> [C, H, W]
            image = np.expand_dims(image, axis=0)  # Batch dimension
            times.append(time.time_ns() - tic)

            preprocessed_image_tensor = pb_utils.Tensor(
                'preprocessed_image',
                image.astype(self.preprocessed_image_dtype),
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[preprocessed_image_tensor]
            )
            
            responses.append(inference_response)

        logging.info(f"Average [Pre-process] time: {np.mean(times)}ns")
        
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass