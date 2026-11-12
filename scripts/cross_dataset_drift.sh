#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RELEASE_ID="${RELEASE_ID:-market1501-vit-bnneck:v1-dvc}"
RELEASE_SLUG="${RELEASE_SLUG:-market1501-vit-bnneck-v1-dvc}"
TRITON_URL="${TRITON_URL:-http://localhost:8000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
WORKERS="${WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/cross-domain/market1501-v1-on-duke/drift}"
REFERENCE="${REFERENCE_PATH:-artifacts/pipeline/market1501-vit-bnneck-v1-dvc/reference/embeddings/reference_embeddings.npy}"
CURRENT="${OUTPUT_ROOT}/duke_query_embeddings.npy"
METADATA="${OUTPUT_ROOT}/duke_query_metadata.json"
MODEL_MANIFEST="${MODEL_MANIFEST_PATH:-artifacts/pipeline/${RELEASE_SLUG}/release/model_manifest.json}"
CAPTURE_CONFIG="configs/duke_v1_evaluation.yaml"
CAPTURE_SPLIT="query"
DATASET_POINTER="datasets/dukemtmc-reid.dvc"

if [[ -f "${PROJECT_ROOT}/.env.mlops" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env.mlops"
  set +a
fi
MONITORING_ROOT="${MONITORING_STORAGE_PATH:-../person-reid-monitoring}"
PROMETHEUS_OUTPUT="${MONITORING_ROOT}/textfile/reid_cross_dataset_drift.prom"
MONITORING_REPORT="${MONITORING_ROOT}/reports/duke_drift_latest.json"

cd "${PROJECT_ROOT}"
curl --fail --silent --max-time 5 "${TRITON_URL}/v2/health/ready" >/dev/null

# Avoid an expensive 126k-file remote comparison when both versioned inputs are
# already materialized. Pull missing targets separately because mixing a .dvc
# directory target and a dvc.yaml stage output triggers a DVC path error.
if [[ ! -d datasets/dukemtmc-reid ]]; then
  ./scripts/dvc_local.sh pull datasets/dukemtmc-reid.dvc
fi
if [[ ! -f "${REFERENCE}" ]]; then
  ./scripts/dvc_local.sh pull extract_reference
fi
if [[ ! -f "${MODEL_MANIFEST}" && "${MODEL_MANIFEST}" == artifacts/* ]]; then
  ./scripts/dvc_local.sh pull validate_release
fi
if [[ -n "${INBOX_DATASET_PATH:-}" ]]; then
  : "${INBOX_DATASET_HOST_PATH:?INBOX_DATASET_HOST_PATH is required for Data Inbox workflows}"
  : "${INBOX_DATASET_REVISION:?INBOX_DATASET_REVISION is required for Data Inbox workflows}"
  INBOX_RUNTIME="${RUNNER_TEMP:-/tmp}/reid-inbox-${CONTROL_CYCLE_ID:-manual}"
  CAPTURE_CONFIG="configs/runtime/duke_v1_inbox_${CONTROL_CYCLE_ID:-manual}.yaml"
  DATASET_POINTER="${INBOX_RUNTIME}/inbox_dataset.dvc"
  CAPTURE_SPLIT="train"
  conda run --no-capture-output -n reid python -m src.prepare_inbox_dataset \
    --source "${INBOX_DATASET_HOST_PATH}" \
    --base-dataset datasets/dukemtmc-reid \
    --base-config configs/duke_v1_evaluation.yaml \
    --output-root "${INBOX_RUNTIME}/dataset" \
    --output-config "${CAPTURE_CONFIG}" \
    --output-pointer "${DATASET_POINTER}" \
    --revision "${INBOX_DATASET_REVISION}" \
    --dataset-name "${INBOX_DATASET_NAME:-dukemtmc-reid}"
fi
mkdir -p "${MONITORING_ROOT}/textfile" "${MONITORING_ROOT}/reports"

conda run --no-capture-output -n reid python src/capture_external_embeddings.py \
  --config "${CAPTURE_CONFIG}" \
  --dataset-pointer "${DATASET_POINTER}" \
  --split "${CAPTURE_SPLIT}" \
  --max-samples "${SAMPLE_SIZE}" \
  --seed 42 \
  --workers "${WORKERS}" \
  --server-url "${TRITON_URL}" \
  --release-id "${RELEASE_ID}" \
  --output "${CURRENT}" \
  --metadata-output "${METADATA}" \
  --jsonl-output "${OUTPUT_ROOT}/duke_query_embeddings.jsonl"

conda run --no-capture-output -n reid python src/generate_drift_report.py \
  --reference "${REFERENCE}" \
  --current "${CURRENT}" \
  --release-id "${RELEASE_ID}" \
  --config configs/drift_detection.yaml \
  --metadata "${METADATA}" \
  --model-manifest "${MODEL_MANIFEST}" \
  --output "${OUTPUT_ROOT}/drift_report.json" \
  --prometheus-output "${PROMETHEUS_OUTPUT}"

cp "${PROMETHEUS_OUTPUT}" "${OUTPUT_ROOT}/reid_cross_dataset_drift.prom"
cp "${OUTPUT_ROOT}/drift_report.json" "${MONITORING_REPORT}"

if [[ "${LOG_DRIFT_MLFLOW:-0}" == "1" ]]; then
  conda run --no-capture-output -n reid python src/log_drift_mlflow.py \
    --report "${OUTPUT_ROOT}/drift_report.json" \
    --prometheus-metrics "${PROMETHEUS_OUTPUT}" \
    --policy configs/drift_detection.yaml \
    --experiment person-reid-drift \
    --output "${OUTPUT_ROOT}/drift_mlflow_run.json"
fi
