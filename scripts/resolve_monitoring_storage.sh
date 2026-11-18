#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGURED_ROOT="${1:-${MONITORING_STORAGE_PATH:-../person-reid-monitoring}}"
NODE_EXPORTER="${NODE_EXPORTER_CONTAINER_NAME:-reid-node-exporter}"

TEXTFILE_SOURCE="$(
  docker inspect --format '{{range .Mounts}}{{if eq .Destination "/textfile"}}{{.Source}}{{end}}{{end}}' \
    "${NODE_EXPORTER}" 2>/dev/null || true
)"
if [[ -n "${TEXTFILE_SOURCE}" ]]; then
  dirname "${TEXTFILE_SOURCE}"
elif [[ "${CONFIGURED_ROOT}" == /* ]]; then
  realpath -m "${CONFIGURED_ROOT}"
else
  realpath -m "${ROOT_DIR}/${CONFIGURED_ROOT}"
fi
