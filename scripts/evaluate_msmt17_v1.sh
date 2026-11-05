#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RELEASE_ID="${RELEASE_ID:-market1501-vit-bnneck:v1-dvc}"
TRITON_URL="${TRITON_URL:-http://localhost:8000}"
WORKERS="${WORKERS:-4}"
MAX_QUERIES="${MAX_QUERIES:-0}"
MAX_GALLERY="${MAX_GALLERY:-0}"
OUTPUT="${OUTPUT:-artifacts/cross-domain/market1501-v1-on-msmt17/metrics/v1_on_msmt17.json}"

cd "${PROJECT_ROOT}"
curl --fail --silent --max-time 5 "${TRITON_URL}/v2/health/ready" >/dev/null
./scripts/dvc_local.sh pull datasets/MSMT17_V1.dvc

exec conda run --no-capture-output -n reid python src/evaluate_deployment_retrieval.py \
  --config configs/msmt17_v1_evaluation.yaml \
  --server-url "${TRITON_URL}" \
  --release-id "${RELEASE_ID}" \
  --dataset-pointer datasets/MSMT17_V1.dvc \
  --max-queries "${MAX_QUERIES}" \
  --max-gallery "${MAX_GALLERY}" \
  --workers "${WORKERS}" \
  --query-batch-size 128 \
  --save-embeddings \
  --log-mlflow \
  --output "${OUTPUT}"
