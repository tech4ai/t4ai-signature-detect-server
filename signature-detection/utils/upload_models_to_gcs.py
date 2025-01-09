import os

from google.cloud import storage


# Inicialize o cliente do Google Cloud Storage
def upload_files_to_gcs(local_folder, bucket_name, remote_folder):
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)

    # Walk through the local folder
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            
            relative_path = os.path.relpath(local_file_path, local_folder)
            remote_file_path = os.path.join(remote_folder, relative_path)

            blob = bucket.blob(remote_file_path)
            
            # Upload the file to GCS (overwriting if it already exists)
            blob.upload_from_filename(local_file_path)

            print(f"Arquivo {file} carregado para {remote_file_path}.")

if __name__ == '__main__':
    # Base directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the models folder and normalize it
    local_folder = os.path.abspath(os.path.join(base_dir, '..', 'models'))

    # Check if the folder exists
    if not os.path.exists(local_folder):
        print(f"A pasta {local_folder} não existe.")
        exit()

    bucket_name = 'iag-training'  # Bucket name
    remote_folder = 'triton-server/image/signature-detection/models'

    upload_files_to_gcs(local_folder, bucket_name, remote_folder)
