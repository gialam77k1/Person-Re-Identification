#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.mlops"
RELEASE_ID="${1:-}"
RELEASE_SLUG="${2:-}"
REFERENCE_PATH=""

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Create it from .env.mlops.example first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

MONITORING_ROOT="$(
  "${ROOT_DIR}/scripts/resolve_monitoring_storage.sh" \
    "${MONITORING_STORAGE_PATH:-../person-reid-monitoring}"
)"
"${ROOT_DIR}/scripts/prepare_monitoring_storage.sh" "${MONITORING_ROOT}"
mkdir -p "${MONITORING_ROOT}/reports" "${MONITORING_ROOT}/textfile"
RUNTIME_SAMPLE_DIR="$(
  docker inspect --format '{{range .Mounts}}{{if eq .Destination "/app/monitoring-samples"}}{{.Source}}{{end}}{{end}}' \
    "${INFERENCE_CONTAINER_NAME:-reid-inference-api}" 2>/dev/null || true
)"
if [[ -z "${RUNTIME_SAMPLE_DIR}" ]]; then
  RUNTIME_SAMPLE_DIR="${MONITORING_ROOT}/samples"
fi
RUNTIME_MONITORING_ROOT="$(dirname "${RUNTIME_SAMPLE_DIR}")"
if [[ "${RUNTIME_MONITORING_ROOT}" != "${MONITORING_ROOT}" ]]; then
  # Compatibility for a service deployed before monitoring storage became
  # canonical across local and Actions checkouts.
  "${ROOT_DIR}/scripts/prepare_monitoring_storage.sh" "${RUNTIME_MONITORING_ROOT}"
fi

cd "${ROOT_DIR}"
if [[ -n "${RELEASE_ID}" && -z "${RELEASE_SLUG}" ]] || \
   [[ -z "${RELEASE_ID}" && -n "${RELEASE_SLUG}" ]]; then
  echo "release_id and release_slug overrides must be provided together." >&2
  exit 1
fi
if [[ -z "${RELEASE_ID}" ]]; then
  DEPLOYMENT_STATE="${DEPLOYMENT_STATE_ROOT:-../person-reid-deployments}/production/current.json"
  SERVING_ARTIFACTS_ROOT="$(
    docker inspect --format '{{range .Mounts}}{{if eq .Destination "/app/artifacts"}}{{.Source}}{{end}}{{end}}' \
      "${INFERENCE_CONTAINER_NAME:-reid-inference-api}" 2>/dev/null || true
  )"
  RESOLVE_ARGS=(
    --api-url "${INFERENCE_API_URL:-http://localhost:8080}"
    --deployment-state "${DEPLOYMENT_STATE}"
  )
  if [[ -n "${SERVING_ARTIFACTS_ROOT}" ]]; then
    RESOLVE_ARGS+=(--serving-artifacts-root "${SERVING_ARTIFACTS_ROOT}")
  fi
  ACTIVE_RELEASE="$(
    conda run --no-capture-output -n reid python src/resolve_serving_release.py \
      "${RESOLVE_ARGS[@]}"
  )"
  IFS=$'\t' read -r RELEASE_ID RELEASE_SLUG REFERENCE_PATH <<< "${ACTIVE_RELEASE}"
fi
if [[ -z "${RELEASE_ID}" || -z "${RELEASE_SLUG}" ]]; then
  echo "Could not resolve an active release_id and release_slug." >&2
  exit 1
fi
echo "Drift monitoring active release: ${RELEASE_ID} (${RELEASE_SLUG})"
if [[ -z "${REFERENCE_PATH}" ]]; then
  REFERENCE_PATH="artifacts/pipeline/${RELEASE_SLUG}/reference/embeddings/reference_embeddings.npy"
  ./scripts/dvc_local.sh pull "${REFERENCE_PATH}"
else
  echo "Using immutable serving reference: ${REFERENCE_PATH}"
fi
exec conda run --no-capture-output -n reid python src/generate_drift_report.py \
  --reference "${REFERENCE_PATH}" \
  --current "${RUNTIME_SAMPLE_DIR}/current_embeddings.jsonl" \
  --release-id "${RELEASE_ID}" \
  --output "${MONITORING_ROOT}/reports/drift_latest.json" \
  --prometheus-output "${MONITORING_ROOT}/textfile/reid_drift.prom"
