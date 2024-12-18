# Delete the deployed model and the endpoint
ENDPOINT_ID=$(gcloud ai endpoints list \
    --region=us-central1 \
    --filter=display_name=signature-detection \
    --format="value(name)")

DEPLOYED_MODEL_ID=$(gcloud ai endpoints describe $ENDPOINT_ID \
    --region=us-central1 \
    --format="value(deployedModels.id)")

gcloud ai endpoints undeploy-model $ENDPOINT_ID \
    --region=us-central1 \
    --deployed-model-id=$DEPLOYED_MODEL_ID

gcloud ai endpoints delete $ENDPOINT_ID \
    --region=us-central1 \
    --quiet

# Delete the model
MODEL_ID=$(gcloud ai models list \
    --region=us-central1 \
    --filter=display_name=signature-detector \
    --format="value(name)")

gcloud ai models delete $MODEL_ID \
    --region=us-central1 \
    --quiet

# Delete the repository
gcloud artifacts repositories delete t4ai-nvidia-triton \
  --location=us-central1 \
  --quiet