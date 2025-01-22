import argparse
import os
from ultralytics import YOLO


def export_to_onnx(model_path: str, output_path: str):
    """Exports a YOLO model to ONNX format.

    Args:
        model_path (str): Path to the YOLO model.
        output_path (str): Path where the ONNX model will be saved.
    """
    model = YOLO(model_path)
    print(f"Exporting model to ONNX: {output_path}")
    model.export(format="onnx", dynamic=True)


def export_to_tensorrt(model_path: str, output_path: str):
    """Exports a YOLO model to TensorRT format.

    Args:
        model_path (str): Path to the YOLO model.
        output_path (str): Path where the TensorRT model will be saved.
    """
    model = YOLO(model_path)
    print(f"Exporting model to TensorRT: {output_path}")
    model.export(format="engine", dynamic=True)


def main():
    """Main function to handle model exportation based on user input."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX or TensorRT"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the YOLO model file (.pt)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "models",
                "yolov8s",
                "1",
                "model.onnx",
            )
        ),
        help="Path where the exported model will be saved",
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "tensorrt"],
        required=True,
        help="Export format (onnx or tensorrt)",
    )

    args = parser.parse_args()

    # Export model based on the chosen format
    try:
        if args.format == "onnx":
            export_to_onnx(args.model_path, args.output_path)
        else:
            export_to_tensorrt(args.model_path, args.output_path)
    except Exception as e:
        print(f"An error occurred during model export: {e}")


if __name__ == "__main__":
    main()
