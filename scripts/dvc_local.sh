#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.mlops"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.mlops.example and set local credentials first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

export DVC_SITE_CACHE_DIR="${DVC_SITE_CACHE_DIR:-${ROOT_DIR}/.dvc/site-cache}"

cd "${ROOT_DIR}"
exec conda run --no-capture-output -n reid dvc "$@"
