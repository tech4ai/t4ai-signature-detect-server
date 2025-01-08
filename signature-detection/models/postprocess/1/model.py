import json
import logging
import time

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    A Python model class for Triton Inference Server.
    This class must be named `TritonPythonModel` and implements the methods
    required to initialize, execute, and finalize the model.
    """

    def initialize(self, args):
        """
        Initializes the model when it is being loaded.
        This method is called once during the model's lifetime and is used to set up any
        necessary configurations or state required for inference.

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
        self.model_config = model_config = json.loads(args['model_config'])

        # Retrieve the output configuration for "detection_result"
        detection_result_config = pb_utils.get_output_config_by_name(
            model_config, "detection_result")

        # Convert Triton data types to NumPy data types
        self.detection_result_dtype = pb_utils.triton_string_to_numpy(
            detection_result_config['data_type'])

        # Set thresholds for score and Non-Maximum Suppression (NMS)
        self.score_threshold = 0.1  # Minimum confidence score for a detection to be valid
        self.nms_threshold = 0.45  # NMS overlap threshold

    def execute(self, requests):
        """
        Handles inference requests and returns the corresponding responses.

        This method processes one or more requests, performs post-processing
        (Filtering and Non-Maximum Suppression), and constructs responses.

        Parameters
        ----------
        requests : list
            A list of `pb_utils.InferenceRequest` objects containing input tensors for inference.

        Returns
        -------
        list
            A list of `pb_utils.InferenceResponse` objects, one for each request.
        """
        responses = []
        times = []

        for request in requests:
            tic = time.time_ns()

            # Retrieve the input tensor "postprocess_input"
            in_0 = pb_utils.get_input_tensor_by_name(request, "postprocess_input")

            # Extract NumPy array from the input tensor
            outputs = in_0.as_numpy()

            # Transpose and process the input tensor
            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []

            # Process each row to extract bounding boxes and scores
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                _, maxScore, _, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)

                if maxScore >= self.score_threshold:
                    # Compute bounding box coordinates and add them to the list
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2],
                        outputs[0][i][3]
                    ]
                    boxes.append(box)
                    scores.append(maxScore)

            # Apply Non-Maximum Suppression to filter overlapping boxes
            result_boxes = cv2.dnn.NMSBoxes(
                boxes, scores, self.score_threshold, self.nms_threshold, 0.5)

            # Construct output boxes with scores
            output_boxes_with_scores = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box_with_score = boxes[index] + [scores[index]]  # [x_min, y_min, width, height, score]
                output_boxes_with_scores.append(box_with_score)

            times.append(time.time_ns() - tic)

            # Convert the result to a Triton tensor and prepare the response
            detection_boxes_scores = np.array(output_boxes_with_scores)
            detection_boxes_scores_tensor = pb_utils.Tensor(
                "detection_result", detection_boxes_scores.astype(self.detection_result_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[detection_boxes_scores_tensor])

            responses.append(inference_response)

        # Log the average processing time
        logging.info(f"Average [Postprocess] time: {np.mean(times)} ns")

        return responses

    def finalize(self):
        """
        Finalizes the model when it is being unloaded.
        This method is called once during the model's lifetime and can be used to
        clean up resources or perform other teardown tasks.
        """
        pass
