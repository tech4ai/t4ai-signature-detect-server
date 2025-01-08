import os

from google.cloud import storage


# Inicialize o cliente do Google Cloud Storage
def download_files_from_gcs(bucket_name, remote_folder, local_folder):
    # Crie um cliente de storage
    storage_client = storage.Client()

    # Acesse o bucket no Google Cloud
    bucket = storage_client.get_bucket(bucket_name)

    # Listar todos os blobs no bucket que estão no caminho remoto especificado
    blobs = bucket.list_blobs(prefix=remote_folder)

    for blob in blobs:
        # Definir o caminho local para onde o arquivo será baixado
        relative_path = os.path.relpath(blob.name, remote_folder)
        local_file_path = os.path.join(local_folder, relative_path)

        # Criar os diretórios necessários localmente, se não existirem
        local_file_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)

        # Fazer o download do arquivo
        blob.download_to_filename(local_file_path)
        print(f"Arquivo {blob.name} baixado para {local_file_path}.")

if __name__ == '__main__':
    # Definir os parâmetros
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Obtém o diretório onde o script está
    local_folder = os.path.join(base_dir, '..', 'signature-detection', 'models')  # Caminho local para salvar os modelos

    bucket_name = 'REDACTED_BUCKET_NAME'  # Nome do bucket
    remote_folder = 'triton-server/image/signature-detection/models'  # Caminho remoto no bucket

    # Chamar a função para baixar os arquivos
    download_files_from_gcs(bucket_name, remote_folder, local_folder)
