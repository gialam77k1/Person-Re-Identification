#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ID="${RELEASE_ID:?RELEASE_ID is required}"
RELEASE_SLUG="${RELEASE_SLUG:?RELEASE_SLUG is required}"
CANDIDATE_REPORTS_ROOT="${CANDIDATE_REPORTS_ROOT:-}"

if [[ ! "${RELEASE_SLUG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid release slug: ${RELEASE_SLUG}" >&2
  exit 2
fi

cd "${ROOT_DIR}"

if [[ -n "${CANDIDATE_REPORTS_ROOT}" ]]; then
  candidate_lock="${CANDIDATE_REPORTS_ROOT}/dvc.lock"
  publication="${CANDIDATE_REPORTS_ROOT}/artifacts/pipeline/${RELEASE_SLUG}/release/dvc_publication.json"
  test -s "${candidate_lock}"
  test -s "${publication}"
  stage="$(conda run --no-capture-output -n reid python -m src.resolve_release_publication \
    --publication "${publication}" \
    --release-id "${RELEASE_ID}" \
    --release-slug "${RELEASE_SLUG}")"
  cp "${candidate_lock}" dvc.lock
else
  stage="package_triton"
fi

./scripts/dvc_local.sh pull --with-deps "${stage}"

if [[ -n "${CANDIDATE_REPORTS_ROOT}" ]]; then
  mkdir -p "artifacts/pipeline/${RELEASE_SLUG}/release"
  cp "${publication}" \
    "artifacts/pipeline/${RELEASE_SLUG}/release/dvc_publication.json"
fi

dependency_args=(
  --manifest "artifacts/pipeline/${RELEASE_SLUG}/release/model_manifest.json"
  --repository-root "${ROOT_DIR}"
)
if [[ -n "${CANDIDATE_REPORTS_ROOT}" ]]; then
  dependency_args+=(--candidate-root "${CANDIDATE_REPORTS_ROOT}")
fi
conda run --no-capture-output -n reid python -m src.materialize_release_dependencies \
  "${dependency_args[@]}"

test -s "artifacts/pipeline/${RELEASE_SLUG}/release/model_manifest.json"
test -s "artifacts/pipeline/${RELEASE_SLUG}/release/validation_report.json"
test -d "artifacts/pipeline/${RELEASE_SLUG}/triton/model_repository"

echo "Materialized ${RELEASE_ID} from immutable DVC stage ${stage}."
