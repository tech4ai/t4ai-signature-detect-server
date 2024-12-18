import numpy as np
import cv2
import time
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.metric_family = pb_utils.MetricFamily(
          name="preprocess_latency_ns",
          description="Cumulative time spent pre-processing requests",
          kind=pb_utils.MetricFamily.COUNTER
        )

        # Create a Metric object under the MetricFamily object. The 'labels'
        # is a dictionary of key-value pairs.
        self.metric = self.metric_family.Metric(
            labels={"model" : "preprocessing", "version" : "1"}
        )
            
    def execute(self, requests):
        responses = []
        for request in requests:
            if request.is_cancelled():
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Message", pb_utils.TritonError.CANCELLED)))
            
            else:
                # Extract the input image tensor
                input_tensor = pb_utils.get_input_tensor_by_name(request, "input_image")
                image_data = input_tensor.as_numpy()
                
                # Preprocess image
                start_ns = time.time_ns()
                img = cv2.imdecode(np.frombuffer(image_data.tobytes(), np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 640))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)
                end_ns = time.time_ns()
                
                self.metric.increment_by(end_ns - start_ns)
                
                # Prepare the output tensor
                preprocessed_tensor = pb_utils.Tensor("preprocessed_image", img.astype(np.float32))
                responses.append(pb_utils.InferenceResponse(output_tensors=[preprocessed_tensor]))

                print("Cumulative pre-processing latency:", self.metric.value())
                
        return responses

    def finalize(self):
        pass
