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
        detection_result_config = pb_utils.get_output_config_by_name(
            model_config, "detection_result")

        # Convert Triton types to numpy types
        self.detection_result_dtype = pb_utils.triton_string_to_numpy(
            detection_result_config['data_type'])

        self.score_threshold = 0.6
        self.nms_threshold = 0.45

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
            tic = time.time_ns()
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "postprocess_input")

            # Get the output arrays from the results
            outputs = in_0.as_numpy()

            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)
                 ) = cv2.minMaxLoc(classes_scores)
                if maxScore >= self.score_threshold:
                    box = [outputs[0][i][0] -
                           (0.5 *
                            outputs[0][i][2]), outputs[0][i][1] -
                           (0.5 *
                            outputs[0][i][3]), outputs[0][i][2], outputs[0][i][3]]
                    boxes.append(box)
                    scores.append(maxScore)

            result_boxes = cv2.dnn.NMSBoxes(boxes, scores,
                                            self.score_threshold,
                                            self.nms_threshold,
                                            0.5)


            output_boxes_with_scores = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                # [x_min, y_min, width, height, score]
                box_with_score = boxes[index] + [scores[index]]
                output_boxes_with_scores.append(box_with_score)
                
            times.append(time.time_ns() - tic)

            detection_boxes_scores = np.array(output_boxes_with_scores)
            detection_boxes_scores = pb_utils.Tensor(
                "detection_result", detection_boxes_scores.astype(self.detection_result_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    detection_boxes_scores,
                ])
            
            responses.append(inference_response)

        logging.info(f"Average [Postprocess] time: {np.mean(times)}ns")
        
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass