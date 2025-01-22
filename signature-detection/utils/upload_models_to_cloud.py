import argparse
import os


def upload_files_to_gcs(
    local_folder: str, bucket_name: str, remote_folder: str
) -> None:
    """Uploads files from a local folder to a Google Cloud Storage bucket.

    Args:
        local_folder (str): Path to the local folder containing files to upload.
        bucket_name (str): Name of the GCS bucket.
        remote_folder (str): Path in the bucket where files will be uploaded.
    """
    from google.cloud import storage

    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)

                relative_path = os.path.relpath(local_file_path, local_folder)
                remote_file_path = os.path.join(remote_folder, relative_path)

                blob = bucket.blob(remote_file_path)

                # Upload the file to GCS (overwriting if it already exists)
                blob.upload_from_filename(local_file_path)

                print(f"File {file} uploaded to {remote_file_path}.")
    except Exception as e:
        print(f"Error uploading files to GCS: {e}")


def upload_files_to_azure(
    local_folder: str, container_name: str, remote_folder: str, connection_string: str
) -> None:
    """Uploads files from a local folder to an Azure Blob Storage container.

    Args:
        local_folder (str): Path to the local folder containing files to upload.
        container_name (str): Name of the Azure container.
        remote_folder (str): Path in the container where files will be uploaded.
        connection_string (str): Connection string for Azure Blob Storage.
    """
    from azure.storage.blob import BlobServiceClient

    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        container_client = blob_service_client.get_container_client(container_name)

        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder)
                remote_file_path = os.path.join(remote_folder, relative_path).replace(
                    "\\", "/"
                )

                with open(local_file_path, "rb") as data:
                    container_client.upload_blob(
                        name=remote_file_path, data=data, overwrite=True
                    )

                print(f"File {file} uploaded to {remote_file_path} in Azure.")
    except Exception as e:
        print(f"Error uploading files to Azure: {e}")


def main() -> None:
    """Main function to upload files to cloud providers."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_local_folder = os.path.abspath(os.path.join(base_dir, "..", "models"))
    default_remote_folder = "triton-server/image/signature-detection/models"

    parser = argparse.ArgumentParser(description="Upload files to cloud providers.")
    parser.add_argument("--provider", choices=["gcp", "az"], required=True, help="Cloud provider (gcp or azure)")
    parser.add_argument("--local-folder", type=str, default=default_local_folder, help="Local folder containing files to upload")
    parser.add_argument("--remote-folder", type=str, default=default_remote_folder, help="Remote folder path in the cloud storage")
    parser.add_argument("--bucket-name", type=str, help="Bucket name for GCP (required for GCP)")
    parser.add_argument("--container-name", type=str, help="Container name for Azure (required for Azure)")
    parser.add_argument("--connection-string", type=str, help="Connection string for Azure (required for Azure)")

    args = parser.parse_args()

    if args.provider == "gcp":
        if not args.bucket_name:
            raise ValueError("--bucket-name is required for GCP")
        upload_files_to_gcs(args.local_folder, args.bucket_name, args.remote_folder)

    elif args.provider == "az":
        if not args.container_name or not args.connection_string:
            raise ValueError("--container-name and --connection-string are required for Azure")
        upload_files_to_azure(args.local_folder, args.container_name, args.remote_folder, args.connection_string)

if __name__ == "__main__":
    main()
