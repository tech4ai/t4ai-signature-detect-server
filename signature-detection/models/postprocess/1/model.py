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
        self.model_config = model_config = json.loads(args["model_config"])

        # Retrieve the output configuration for "detection_result"
        detection_result_config = pb_utils.get_output_config_by_name(
            model_config, "detection_result"
        )

        # Convert Triton data types to NumPy data types
        self.detection_result_dtype = pb_utils.triton_string_to_numpy(
            detection_result_config["data_type"]
        )

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

        for request in requests:
            # Obter tensores de entrada de forma otimizada
            postprocess_input = pb_utils.get_input_tensor_by_name(
                request, "postprocess_input"
            ).as_numpy()

            confidence_threshold = (
                pb_utils.get_input_tensor_by_name(request, "confidence_threshold")
                .as_numpy()
                .astype(np.float16)[0]
            )

            iou_threshold = (
                pb_utils.get_input_tensor_by_name(request, "iou_threshold")
                .as_numpy()
                .astype(np.float16)[0]
            )

            # Transposição e processamento vetorizado
            outputs = postprocess_input.transpose(
                0, 2, 1
            )  

            # Extração vetorizada de scores e boxes
            class_scores = outputs[0, :, 4:]
            max_scores = np.max(class_scores, axis=1)
            max_class_indices = np.argmax(class_scores, axis=1)

            # Filtragem vetorizada por confidence threshold
            valid_mask = max_scores >= confidence_threshold

            if np.any(valid_mask):
                # Cálculo vetorizado das boxes
                boxes = np.zeros((np.sum(valid_mask), 4), dtype=np.float32)
                boxes[:, 0] = outputs[0, valid_mask, 0] - (
                    0.5 * outputs[0, valid_mask, 2]
                )  # x_min
                boxes[:, 1] = outputs[0, valid_mask, 1] - (
                    0.5 * outputs[0, valid_mask, 3]
                )  # y_min
                boxes[:, 2] = outputs[0, valid_mask, 2]  # width
                boxes[:, 3] = outputs[0, valid_mask, 3]  # height

                filtered_scores = max_scores[valid_mask]

                # Aplicar NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(),
                    filtered_scores.tolist(),
                    float(confidence_threshold),
                    float(iou_threshold),
                    0.5,
                )

                if len(indices) > 0:
                    # Construir resultado final de forma vetorizada
                    final_boxes = boxes[indices]
                    final_scores = filtered_scores[indices]
                    detection_boxes_scores = np.hstack(
                        (final_boxes, final_scores.reshape(-1, 1))
                    )
                else:
                    detection_boxes_scores = np.zeros((0, 5), dtype=np.float32)
            else:
                detection_boxes_scores = np.zeros((0, 5), dtype=np.float32)

            # Criar tensor de resposta
            detection_boxes_scores_tensor = pb_utils.Tensor(
                "detection_result",
                detection_boxes_scores.astype(self.detection_result_dtype),
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[detection_boxes_scores_tensor]
                )
            )

        return responses

    def finalize(self):
        """
        Finalizes the model when it is being unloaded.
        This method is called once during the model's lifetime and can be used to
        clean up resources or perform other teardown tasks.
        """
        pass
