gcloud init
gcloud services enable artifactregistry.googleapis.com

LOCATION_ID="<YOUR_LOCATION_ID>" # Ex: us-central1

# Create a new repository
gcloud artifacts repositories create t4ai-nvidia-triton \
    --repository-format=docker \
    --location=$LOCATION_ID \
    --description="NVIDIA Triton Docker repository"

# Create the image docker container
NGC_TRITON_IMAGE_URI="nvcr.io/nvidia/tritonserver:24.11-py3"
MODEL_ARTIFACTS_REPOSITORY="<YOUR_MODEL_ARTIFACTS_REPOSITORY>" # Ex: gs://<YOUR_BUCKET_NAME>/<YOUR_MODEL_ARTIFACTS_FOLDER>
NGC_TRITON_TAG="$LOCATION_ID-docker.pkg.dev/<PROJECT_ID>/nvidia-triton/vertex-triton-inference:24.11"

docker pull $NGC_TRITON_IMAGE_URI
docker tag $NGC_TRITON_IMAGE_URI $NGC_TRITON_TAG

# Push the image to the repository
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push $NGC_TRITON_TAG

# Deploy the model

## Create the model
gcloud ai models upload \
    --region=us-central1 \
    --display-name=signature-detector \
    --container-image-uri=$NGC_TRITON_TAG \
    --artifact-uri=$MODEL_ARTIFACTS_REPOSITORY \
    --container-args='-model-control-mode=poll,--repository-poll-secs=60,--allow-http=true,--allow-vertex-ai=True'

## Create the endpoint
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name=signature-detection

## Deploy the model to the endpoint
ENDPOINT_ID=$(gcloud ai endpoints list \
    --region=us-central1 \
    --filter=display_name=signature-detection \
    --format='value(name)')

MODEL_ID=$(gcloud ai models list \
    --region=us-central1 \
    --filter=display_name=signature-detector \
    --format="value(name)")

gcloud ai endpoints deploy-model $ENDPOINT_ID \
    --region=us-central1 \
    --model=$MODEL_ID \
    --display-name=signature-detector \
    --machine-type=e2-standard-4 \
    --min-replica-count=1 \
    --max-replica-count=2 \
    --traffic-split=0=100