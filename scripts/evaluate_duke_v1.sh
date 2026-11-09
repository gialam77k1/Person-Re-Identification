#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RELEASE_ID="${RELEASE_ID:-market1501-vit-bnneck:v1-dvc}"
TRITON_URL="${TRITON_URL:-http://localhost:8000}"
WORKERS="${WORKERS:-4}"
MAX_QUERIES="${MAX_QUERIES:-0}"
MAX_GALLERY="${MAX_GALLERY:-0}"
OUTPUT="${OUTPUT:-artifacts/cross-domain/market1501-v1-on-duke/metrics/v1_on_duke.json}"

cd "${PROJECT_ROOT}"
curl --fail --silent --max-time 5 "${TRITON_URL}/v2/health/ready" >/dev/null
./scripts/dvc_local.sh pull datasets/dukemtmc-reid.dvc

exec conda run --no-capture-output -n reid python src/evaluate_deployment_retrieval.py \
  --config configs/duke_v1_evaluation.yaml \
  --server-url "${TRITON_URL}" \
  --release-id "${RELEASE_ID}" \
  --dataset-pointer datasets/dukemtmc-reid.dvc \
  --max-queries "${MAX_QUERIES}" \
  --max-gallery "${MAX_GALLERY}" \
  --workers "${WORKERS}" \
  --query-batch-size 128 \
  --save-embeddings \
  --log-mlflow \
  --output "${OUTPUT}"
