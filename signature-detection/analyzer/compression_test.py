import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(base_dir)

import concurrent.futures
import logging
import statistics
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from inference.predictors import TritonClientPredictor


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


class CompressionAlgorithm(Enum):
    NONE = None
    GZIP = "gzip"
    DEFLATE = "deflate"


@dataclass
class CompressionConfig:
    request: CompressionAlgorithm
    response: CompressionAlgorithm

    def __str__(self):
        return f"Request: {self.request.name}, Response: {self.response.name}"


class ConcurrentTritonClientPredictor(TritonClientPredictor):
    def __init__(self, url: str, endpoint: str = "", scheme: str = ""):
        super().__init__(url=url, endpoint=endpoint, scheme=scheme)

    def request_with_compression(
        self, inputs: np.ndarray, compression_config: CompressionConfig
    ) -> Tuple[np.ndarray, float]:
        infer_inputs = []
        x = inputs
        if x.dtype != self.np_input_formats[0]:
            x = x.astype(self.np_input_formats[0])
        infer_input = self.InferInput(
            self.input_names[0], [*x.shape], self.input_formats[0].replace("TYPE_", "")
        )
        infer_input.set_data_from_numpy(x)
        infer_inputs.append(infer_input)

        infer_outputs = [
            self.InferRequestedOutput(output_name) for output_name in self.output_names
        ]

        tic = time.time()
        outputs = self.triton_client.infer(
            model_name=self.endpoint,
            inputs=infer_inputs,
            outputs=infer_outputs,
            request_compression_algorithm=compression_config.request.value,
            response_compression_algorithm=compression_config.response.value,
            headers=self.headers,
        )
        latency = time.time() - tic

        return (
            outputs.as_numpy(self.output_names[0]).astype(self.np_output_formats[0]),
            latency,
        )


class CompressionTestPipeline:
    def __init__(self, predictor_url: str):
        self.predictor_url = predictor_url  # Armazena a URL ao invés do predictor
        self.compression_configs = [
            CompressionConfig(CompressionAlgorithm.NONE, CompressionAlgorithm.NONE),
            CompressionConfig(CompressionAlgorithm.GZIP, CompressionAlgorithm.NONE),
            CompressionConfig(CompressionAlgorithm.NONE, CompressionAlgorithm.GZIP),
            CompressionConfig(CompressionAlgorithm.GZIP, CompressionAlgorithm.GZIP),
            CompressionConfig(CompressionAlgorithm.DEFLATE, CompressionAlgorithm.NONE),
            CompressionConfig(CompressionAlgorithm.NONE, CompressionAlgorithm.DEFLATE),
            CompressionConfig(
                CompressionAlgorithm.DEFLATE, CompressionAlgorithm.DEFLATE
            ),
        ]
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def make_prediction_with_retry(
        self,
        predictor: ConcurrentTritonClientPredictor,
        image_data: np.ndarray,
        config: CompressionConfig,
    ) -> Tuple[np.ndarray, float]:
        try:
            return predictor.request_with_compression(image_data, config)
        except Exception as e:
            self.logger.warning(f"Error in prediction: {str(e)}, retrying...")
            raise

    def test_config(self, image_paths: List[str], config: CompressionConfig) -> Dict:
        thread_id = threading.current_thread().name
        # Criar um predictor específico para esta thread
        predictor = ConcurrentTritonClientPredictor(url=self.predictor_url)

        self.logger.info(
            f"Thread {thread_id} initialized with new predictor for config {config}"
        )

        latencies = []
        failed_images = []

        pbar = tqdm(
            total=len(image_paths),
            desc=f"Thread {thread_id} - {config}",
            position=self.compression_configs.index(config),
            leave=True,
        )

        for image_path in image_paths:
            try:
                image_data = encode_image(image_path)
                _, latency = self.make_prediction_with_retry(
                    predictor, image_data, config
                )
                latencies.append(latency)

                if latencies:
                    current_mean = statistics.mean(latencies)
                    pbar.set_description(
                        f"Thread {thread_id} - {config} (Mean: {current_mean:.4f}s)"
                    )
            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {str(e)}")
                failed_images.append(image_path)
            finally:
                pbar.update(1)

        pbar.close()

        if not latencies:
            self.logger.error(f"No successful predictions for configuration {config}")
            return {
                "config": str(config),
                "thread_id": thread_id,
                "status": "failed",
                "error": "No successful predictions",
            }

        return {
            "config": str(config),
            "thread_id": thread_id,
            "status": "success",
            "mean_latency": statistics.mean(latencies),
            "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "num_images": len(latencies),
            "failed_images": len(failed_images),
            "success_rate": (len(latencies) / len(image_paths)) * 100,
        }

    def run_concurrent_tests(self, image_paths: List[str]) -> List[Dict]:
        print(
            f"Starting concurrent compression tests with {len(self.compression_configs)} configurations"
        )
        print(f"Total images to process per configuration: {len(image_paths)}\n")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.compression_configs), thread_name_prefix="CompTest"
        ) as executor:
            futures = [
                executor.submit(self.test_config, image_paths, config)
                for config in self.compression_configs
            ]
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Thread failed: {str(e)}")

        # Filter out failed configurations
        successful_results = [r for r in results if r.get("status") == "success"]
        return sorted(successful_results, key=lambda x: x["mean_latency"])


def get_image_paths(dataset_dir: str, split: str = "test") -> List[str]:
    root_dir_images = os.path.join(dataset_dir, split, "images")
    return [
        os.path.join(root_dir_images, img)
        for img in os.listdir(root_dir_images)
        if img.endswith(".jpg")
    ]


def run_compression_tests(predictor_url: str, dataset_dir: str):
    pipeline = CompressionTestPipeline(predictor_url)

    # Get all image paths
    train_paths = get_image_paths(dataset_dir, "train")
    test_paths = get_image_paths(dataset_dir, "test")
    val_paths = get_image_paths(dataset_dir, "valid")
    all_paths = train_paths + test_paths + val_paths

    print(f"Running tests on {len(all_paths)} images...")
    results = pipeline.run_concurrent_tests(all_paths)

    print("\nCompression Test Results:")
    print("-" * 80)
    for result in results:
        print(f"\nConfiguration: {result['config']}")
        print(f"Mean Latency: {result['mean_latency']:.4f} s")
        print(f"Std Latency:  {result['std_latency']:.4f} s")
        print(f"Min Latency:  {result['min_latency']:.4f} s")
        print(f"Max Latency:  {result['max_latency']:.4f} s")
        print(f"Images Processed: {result['num_images']}")


if __name__ == "__main__":
    HOME = os.getcwd()
    DATASET_DIR = os.path.join(HOME, "signature-detection", "data", "datasets")

    if not os.path.exists(DATASET_DIR):
        print("Dataset not found at:", DATASET_DIR)
        print("Check the data folder and scripts for download the dataset")
        exit(1)

    # http://localhost:8000/yolov8_ensemble  REDACTED_TRITON_URL
    predictor_url = "REDACTED_TRITON_URL"  # Replace with your Triton server URL

    run_compression_tests(predictor_url, DATASET_DIR)
