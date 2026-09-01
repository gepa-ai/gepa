#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${ARTIFACT_REPOSITORY:?Set ARTIFACT_REPOSITORY}"
: "${RUN_BUCKET:?Set RUN_BUCKET}"
: "${SERVICE_ACCOUNT:?Set SERVICE_ACCOUNT}"

JOB_NAME="${JOB_NAME:-glean-gepa-optimize}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/glean-gepa:${IMAGE_TAG}"

gcloud builds submit \
  --project "${PROJECT_ID}" \
  --config deploy/glean_gepa/cloudbuild.yaml \
  --substitutions "_IMAGE=${IMAGE_URI}" \
  .

gcloud run jobs deploy "${JOB_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE_URI}" \
  --service-account "${SERVICE_ACCOUNT}" \
  --tasks 1 \
  --parallelism 1 \
  --max-retries 1 \
  --task-timeout 24h \
  --cpu 2 \
  --memory 4Gi \
  --set-env-vars "GEPA_RUN_ROOT=/mnt/gepa/runs" \
  --add-volume "name=gepa-runs,type=cloud-storage,bucket=${RUN_BUCKET},readonly=false" \
  --add-volume-mount "volume=gepa-runs,mount-path=/mnt/gepa"

printf 'Deployed %s\n' "${JOB_NAME}"
printf 'Image: %s\n' "${IMAGE_URI}"
