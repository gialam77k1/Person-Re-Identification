#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${ROOT_DIR}/artifacts/duke/market1501-duke-vit-bnneck-v2"
PIPELINE_ROOT="${ROOT_DIR}/artifacts/pipeline/market1501-duke-vit-bnneck-v2"
BEST_CHECKPOINT="${SOURCE_ROOT}/checkpoints/best_model.pth"
LAST_CHECKPOINT="${SOURCE_ROOT}/checkpoints/last_model.pth"
PUBLICATION_REPORT="${PIPELINE_ROOT}/release/dvc_publication.json"
STATUS_FILE="$(mktemp)"
trap 'rm -f "${STATUS_FILE}"' EXIT

for required in \
  "${BEST_CHECKPOINT}" \
  "${LAST_CHECKPOINT}" \
  "${PIPELINE_ROOT}/release/model_manifest.json" \
  "${PIPELINE_ROOT}/release/validation_report.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required v2 release artifact: ${required}" >&2
    exit 2
  fi
done

cd "${ROOT_DIR}"

# Keep both train checkpoints recoverable. The complete immutable release is
# already represented by DVC pipeline outputs and their dependencies.
./scripts/dvc_local.sh add "${BEST_CHECKPOINT}" "${LAST_CHECKPOINT}"
./scripts/dvc_local.sh push "${BEST_CHECKPOINT}.dvc" "${LAST_CHECKPOINT}.dvc"
./scripts/dvc_local.sh push --with-deps v2_package_triton
./scripts/dvc_local.sh status --cloud --with-deps --json v2_package_triton > "${STATUS_FILE}"

conda run --no-capture-output -n reid python -m src.verify_v2_artifact_publication \
  --manifest "${PIPELINE_ROOT}/release/model_manifest.json" \
  --validation-report "${PIPELINE_ROOT}/release/validation_report.json" \
  --dvc-status "${STATUS_FILE}" \
  --checkpoint-pointer "${BEST_CHECKPOINT}.dvc" \
  --checkpoint-pointer "${LAST_CHECKPOINT}.dvc" \
  --remote minio \
  --stage v2_package_triton \
  --output "${PUBLICATION_REPORT}"

# The small publication evidence is itself versioned, while model binaries
# remain exclusively in the DVC/MinIO object store.
./scripts/dvc_local.sh add "${PUBLICATION_REPORT}"
./scripts/dvc_local.sh push "${PUBLICATION_REPORT}.dvc"
./scripts/dvc_local.sh status --cloud --quiet "${PUBLICATION_REPORT}.dvc"

echo "V2 release artifacts are synchronized with MinIO: ${PUBLICATION_REPORT}"
