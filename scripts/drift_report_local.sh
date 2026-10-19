#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.mlops"
RELEASE_ID="${1:-market1501-vit-bnneck:v1-dvc}"
RELEASE_SLUG="${2:-market1501-vit-bnneck-v1-dvc}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Create it from .env.mlops.example first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

MONITORING_ROOT="${MONITORING_STORAGE_PATH:-../person-reid-monitoring}"
mkdir -p "${MONITORING_ROOT}/reports" "${MONITORING_ROOT}/textfile"

cd "${ROOT_DIR}"
./scripts/dvc_local.sh pull "artifacts/pipeline/${RELEASE_SLUG}/reference/embeddings/reference_embeddings.npy"
exec conda run --no-capture-output -n reid python src/generate_drift_report.py \
  --reference "artifacts/pipeline/${RELEASE_SLUG}/reference/embeddings/reference_embeddings.npy" \
  --current "${MONITORING_ROOT}/samples/current_embeddings.jsonl" \
  --release-id "${RELEASE_ID}" \
  --output "${MONITORING_ROOT}/reports/drift_latest.json" \
  --prometheus-output "${MONITORING_ROOT}/textfile/reid_drift.prom"
