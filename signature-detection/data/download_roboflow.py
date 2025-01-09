import os

from roboflow import Roboflow

HOME = os.getcwd()
DATASET_DIR = f"{HOME}/signature-detection/data/datasets"

def setup_roboflow(n_version=3):
    """Initialize Roboflow and create necessary directories"""
    os.makedirs(f"{HOME}/datasets/", exist_ok=True)

    if "ROBOFLOW_API_KEY" not in os.environ:
        raise ValueError("ROBOFLOW_API_KEY environment variable not set")

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    workspace = rf.workspace("tech-ysdkk")
    project = workspace.project("signature-detection-hlx8j")
    version = project.version(n_version)
    return project, version

def download_dataset(version, format: str = "yolov8"):
    """
    Download the dataset
    Args:
        version: Roboflow version object
        format: Format of the dataset to download
    Returns:
        dataset object and its location
    """
    dataset = version.download(
        model_format=format,
        location=DATASET_DIR,
        overwrite=True
    )

    print("Dataset location:", dataset.location)
    return dataset

def main():
    project, version = setup_roboflow()
    download_dataset(version)
    
if __name__ == "__main__":
    main()