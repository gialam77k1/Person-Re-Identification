#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RELEASE_ID="${RELEASE_ID:-market1501-vit-bnneck:v1-dvc}"
RELEASE_SLUG="${RELEASE_SLUG:-market1501-vit-bnneck-v1-dvc}"
TRITON_URL="${TRITON_URL:-http://localhost:8000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
WORKERS="${WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/cross-domain/market1501-v1-on-msmt17/drift}"
REFERENCE="artifacts/pipeline/${RELEASE_SLUG}/reference/embeddings/reference_embeddings.npy"
CURRENT="${OUTPUT_ROOT}/msmt17_query_embeddings.npy"
METADATA="${OUTPUT_ROOT}/msmt17_query_metadata.json"
MODEL_MANIFEST="artifacts/pipeline/${RELEASE_SLUG}/release/model_manifest.json"

if [[ -f "${PROJECT_ROOT}/.env.mlops" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env.mlops"
  set +a
fi
MONITORING_ROOT="${MONITORING_STORAGE_PATH:-../person-reid-monitoring}"
PROMETHEUS_OUTPUT="${MONITORING_ROOT}/textfile/reid_cross_dataset_drift.prom"
MONITORING_REPORT="${MONITORING_ROOT}/reports/msmt17_drift_latest.json"

cd "${PROJECT_ROOT}"
curl --fail --silent --max-time 5 "${TRITON_URL}/v2/health/ready" >/dev/null

# Avoid an expensive 126k-file remote comparison when both versioned inputs are
# already materialized. Pull missing targets separately because mixing a .dvc
# directory target and a dvc.yaml stage output triggers a DVC path error.
if [[ ! -d datasets/MSMT17_V1 ]]; then
  ./scripts/dvc_local.sh pull datasets/MSMT17_V1.dvc
fi
if [[ ! -f "${REFERENCE}" ]]; then
  ./scripts/dvc_local.sh pull extract_reference
fi
if [[ ! -f "${MODEL_MANIFEST}" ]]; then
  ./scripts/dvc_local.sh pull validate_release
fi
mkdir -p "${MONITORING_ROOT}/textfile" "${MONITORING_ROOT}/reports"

conda run --no-capture-output -n reid python src/capture_external_embeddings.py \
  --config configs/msmt17_v1_evaluation.yaml \
  --dataset-pointer datasets/MSMT17_V1.dvc \
  --split query \
  --max-samples "${SAMPLE_SIZE}" \
  --seed 42 \
  --workers "${WORKERS}" \
  --server-url "${TRITON_URL}" \
  --release-id "${RELEASE_ID}" \
  --output "${CURRENT}" \
  --metadata-output "${METADATA}" \
  --jsonl-output "${OUTPUT_ROOT}/msmt17_query_embeddings.jsonl"

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
