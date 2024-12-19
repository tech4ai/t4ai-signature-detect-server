from google.cloud import storage
import os

# Inicialize o cliente do Google Cloud Storage
def upload_files_to_gcs(local_folder, bucket_name, remote_folder):
    # Crie um cliente de storage
    storage_client = storage.Client()

    # Acesse o bucket no Google Cloud
    bucket = storage_client.get_bucket(bucket_name)

    # Percorrer os arquivos da pasta local
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            # Definir o caminho completo do arquivo local
            local_file_path = os.path.join(root, file)
            
            # Defina o caminho remoto (no bucket)
            # Vamos remover a parte inicial do caminho da pasta local para manter a estrutura no GCS
            relative_path = os.path.relpath(local_file_path, local_folder)
            remote_file_path = os.path.join(remote_folder, relative_path)

            # Crie o blob no GCS e faça o upload do arquivo local
            blob = bucket.blob(remote_file_path)

            # Faça o upload do arquivo para o GCS (substituindo se já existir)
            blob.upload_from_filename(local_file_path)

            print(f"Arquivo {file} carregado para {remote_file_path}.")

# Definir os parâmetros
base_dir = os.path.dirname(os.path.abspath(__file__))  # Obtém o diretório onde o script está
local_folder = os.path.join(base_dir, '..', 'signature-detection', 'models')  # Caminho correto para a pasta 'models'

# Check if the folder exists
if not os.path.exists(local_folder):
    print(f"A pasta {local_folder} não existe.")
    exit()

bucket_name = 'REDACTED_BUCKET_NAME'  # Nome do bucket
remote_folder = 'triton-server/image/signature-detection/models'  # Caminho remoto no bucket

# Chamar a função para fazer o upload dos arquivos
upload_files_to_gcs(local_folder, bucket_name, remote_folder)
