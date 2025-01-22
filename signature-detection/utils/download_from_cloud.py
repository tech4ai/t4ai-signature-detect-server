import argparse
import os


def download_files_from_gcs(bucket_name: str, remote_folder: str, local_folder: str) -> None:
    """Downloads files from a Google Cloud Storage bucket.

    Args:
        bucket_name (str): Name of the GCS bucket.
        remote_folder (str): Path of the remote folder in the bucket.
        local_folder (str): Local path to save the files.
    """
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=remote_folder)

    for blob in blobs:
        relative_path = os.path.relpath(blob.name, remote_folder)
        local_file_path = os.path.join(local_folder, relative_path)

        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        if blob.size == 0:
            continue

        # Fazer o download do arquivo
        blob.download_to_filename(local_file_path)
        print(f"File {blob.name} downloaded to {local_file_path}.")


def download_files_from_azure(
    container_name: str, remote_folder: str, local_folder: str, connection_string: str
) -> None:
    """Downloads files from an Azure Blob Storage container.

    Args:
        container_name (str): Name of the Azure container.
        remote_folder (str): Path of the remote folder in the container.
        local_folder (str): Local path to save the files.
        connection_string (str): Connection string for Azure Blob Storage.
    """
    from azure.storage.blob import BlobServiceClient

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs(name_starts_with=remote_folder)

    for blob in blobs:

        relative_path = os.path.relpath(blob.name, remote_folder)
        local_file_path = os.path.join(local_folder, relative_path).replace("\\", "/")

        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        if blob.size == 0:
            continue

        with open(local_file_path, "wb") as file:
            blob_data = container_client.download_blob(blob.name)
            file.write(blob_data.readall())

        print(f"File {blob.name} downloaded to {local_file_path}.")


def main() -> None:
    """Main function to download files from cloud providers."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_local_folder = os.path.abspath(os.path.join(base_dir, "..", "models"))

    default_remote_folder = remote_folder = "triton-server/image/signature-detection/models"

    parser = argparse.ArgumentParser(description="Download files from cloud providers.")
    parser.add_argument("--provider", choices=["gcp", "az"], required=True, help="Cloud provider (gcp or azure)")
    parser.add_argument("--local-folder", type=str, default=default_local_folder, help="Local folder to save downloaded files")
    parser.add_argument("--remote-folder", type=str, default=default_remote_folder, help="Remote folder path in the cloud storage")
    parser.add_argument("--bucket-name", type=str, help="Bucket name for GCP (required for GCP)")
    parser.add_argument("--container-name", type=str, help="Container name for Azure (required for Azure)")
    parser.add_argument("--connection-string", type=str, help="Connection string for Azure (required for Azure)")

    args = parser.parse_args()

    if args.provider == "gcp":
        if not args.bucket_name:
            raise ValueError("--bucket-name is required for GCP")
        download_files_from_gcs(args.bucket_name, args.remote_folder, args.local_folder)

    elif args.provider == "az":
        if not args.container_name or not args.connection_string:
            raise ValueError("--container-name and --connection-string are required for Azure")
        download_files_from_azure(args.container_name, args.remote_folder, args.local_folder, args.connection_string)

if __name__ == "__main__":
    main()

