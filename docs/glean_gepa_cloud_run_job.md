# Running Glean GEPA as a Cloud Run Job

This is the smallest remote deployment for the current optimizer. One Cloud
Run task runs the existing synchronous GEPA loop, while a mounted Cloud Storage
bucket holds checkpoints, eval ID caches, child caches, logs, and final results.

Cloud Run is the process supervisor, not the evaluation system. Candidate
evaluation still happens in Cortex through `evalcli`.

## What is included

- `glean_gepa.remote_job`: converts Cloud Run environment variables into the
  ordinary `glean_gepa.runner` arguments and writes `remote_job_status.json`.
- `deploy/glean_gepa/Dockerfile`: reproducible Python/uv image.
- `deploy/glean_gepa/cloudbuild.yaml`: builds the image without local Docker.
- `deploy/glean_gepa/deploy_cloud_run.sh`: deploys a one-task Cloud Run Job with
  a writable Cloud Storage volume.
- `deploy/glean_gepa/run_cloud_job.sh`: starts an execution with JSON runner
  arguments.

Every execution writes under:

```text
gs://RUN_BUCKET/runs/RUN_ID/
  remote_job_status.json
  gepa_state.bin
  run_log.json
  candidates.json
  cache/
    glean_adapter_cache.json
    glean_eval_run_cache.json
    glean_children_cache.json
```

Reusing `RUN_ID` lets a retry load the same checkpoint and cached Cortex eval
IDs instead of launching everything again.

## Important Cortex authentication boundary

The current Scio `evalcli` is a Bazel-built Python binary that authenticates
with a user's IAP browser cookie. A macOS Bazel output cannot be copied into a
Linux Cloud Run image, and a personal browser cookie is not a durable
production identity.

Therefore:

1. The image can run `--fake_flow` immediately as an infrastructure smoke test.
2. A real Cortex run needs a canonical **Linux evalcli bundle or internal base
   image** at `/opt/evalcli/eval_cli`.
3. Production should give the job service identity access to Cortex. A manually
   supplied `CORTEX_IAP_COOKIE` is supported only as a short-lived development
   bridge; store it in Secret Manager and never in source or ordinary
   environment configuration.

The source-of-truth evalcli build is:

```bash
cd ~/workspace/scio
bazel build //python_scio/eval_cli:eval_cli
```

Build the production Linux bundle in Scio/CI, not from a developer's macOS
output. Place the resulting runtime under
`deploy/glean_gepa/evalcli_bundle/` before the image build, or use an internal
base image that already contains it.

## 1. Choose deployment values

```bash
export PROJECT_ID=dev-sandbox-334901
export REGION=us-central1
export ARTIFACT_REPOSITORY=gepa-jobs
export RUN_BUCKET="${PROJECT_ID}-gepa-runs"
export SERVICE_ACCOUNT="gepa-job@${PROJECT_ID}.iam.gserviceaccount.com"
export JOB_NAME=glean-gepa-optimize
```

Use a dedicated project/service account for a long-lived deployment. The values
above are suitable only if `dev-sandbox-334901` is the intended development
project.

## 2. Create the one-time cloud resources

These commands create billable/shared cloud resources. Run them once after the
project, region, and owner have been confirmed.

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  --project "${PROJECT_ID}"

gcloud artifacts repositories create "${ARTIFACT_REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --repository-format docker

gcloud storage buckets create "gs://${RUN_BUCKET}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --uniform-bucket-level-access

gcloud iam service-accounts create gepa-job \
  --project "${PROJECT_ID}" \
  --display-name "Glean GEPA optimization job"

gcloud storage buckets add-iam-policy-binding "gs://${RUN_BUCKET}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/storage.objectUser
```

The service account also needs BigQuery read/job permissions for the shell-error
analysis datasets, QE/reflection-model access, and the future Cortex service
permission. Grant those through the normal internal IAM path rather than
embedding credentials in the image.

## 3. Build and deploy

From the GEPA repository root:

```bash
deploy/glean_gepa/deploy_cloud_run.sh
```

The script uses Cloud Build, then deploys a single-task job with 2 CPUs, 4 GiB
memory, one retry, a 24-hour timeout, and the GCS bucket mounted at `/mnt/gepa`.
Cloud Run Jobs support task timeouts up to seven days; start at 24 hours and
increase only when actual run data justifies it.

## 4. Smoke-test the deployment

The default run script uses the offline deterministic fake flow:

```bash
export RUN_ID=smoke-$(date -u +%Y%m%d-%H%M%S)
deploy/glean_gepa/run_cloud_job.sh

gcloud storage ls "gs://${RUN_BUCKET}/runs/${RUN_ID}/"
gcloud storage cat "gs://${RUN_BUCKET}/runs/${RUN_ID}/remote_job_status.json"
```

The status should be `succeeded`, and the directory should contain a GEPA
checkpoint and candidate artifacts.

## 5. Start a real single-model optimization

First upload the seed candidate to the mounted bucket:

```bash
gcloud storage cp data/seed_candidate.json "gs://${RUN_BUCKET}/configs/writing-code-seed.json"
```

Then pass the normal runner arguments as a JSON array. The train/validation
versions are pinned so retries and comparisons remain reproducible.

```bash
export RUN_ID=writing-code-20260831-01
export RUNNER_ARGS_JSON='[
  "--seed_candidate", "/mnt/gepa/configs/writing-code-seed.json",
  "--run_dir", "/mnt/gepa/runs/writing-code-20260831-01",
  "--student_model", "gpt",
  "--judging_mode", "single_model",
  "--train_eval_versions", "20260813,20260820",
  "--val_eval_versions", "20260827",
  "--max_metric_calls", "10",
  "--eval_run_timeout_sec", "21600",
  "--reflection_samples", "all"
]'
deploy/glean_gepa/run_cloud_job.sh
```

For an asynchronous execution, remove `--wait` from
`deploy/glean_gepa/run_cloud_job.sh`. Inspect it with:

```bash
gcloud run jobs executions list \
  --job "${JOB_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"
```

## Failure and retry behavior

- Cortex eval IDs are persisted immediately after creation, before polling.
- A single Cortex eval wait defaults to six hours and fails clearly on timeout.
- GEPA state and caches live on the mounted bucket.
- Cloud Run retries the process once. It should use the same `GEPA_RUN_ID` so
  the existing state is reused.
- `remote_job_status.json` records the execution, attempt, arguments, final
  status, and traceback.
- Do not automatically create replacement Cortex runs after ambiguous create
  timeouts. Verify the original run by ID first.

## Promotion

The job only produces a candidate artifact. Before changing a production SC:

1. Run the winner on a separate held-out Cortex eval set.
2. Compare it against the seed/baseline on correctness and operational metrics.
3. Review the prompt diff and Cortex run links.
4. Promote through the normal SC review/deployment path.
