#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"

JOB_NAME="${JOB_NAME:-glean-gepa-optimize}"
RUN_ID="${RUN_ID:-gepa-$(date -u +%Y%m%d-%H%M%S)}"
RUNNER_ARGS_JSON="${RUNNER_ARGS_JSON:-[\"--fake_flow\",\"--max_metric_calls\",\"10\"]}"

gcloud run jobs execute "${JOB_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --update-env-vars "^@^GEPA_RUN_ID=${RUN_ID}@GEPA_RUNNER_ARGS_JSON=${RUNNER_ARGS_JSON}" \
  --wait
