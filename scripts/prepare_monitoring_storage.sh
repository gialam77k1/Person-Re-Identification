#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUESTED_ROOT="${1:-${MONITORING_STORAGE_PATH:-../person-reid-monitoring}}"
if [[ "${REQUESTED_ROOT}" == /* ]]; then
  MONITORING_ROOT="$(realpath -m "${REQUESTED_ROOT}")"
else
  MONITORING_ROOT="$(realpath -m "${ROOT_DIR}/${REQUESTED_ROOT}")"
fi
CONTAINER_NAME="${INFERENCE_CONTAINER_NAME:-reid-inference-api}"

if [[ "${MONITORING_ROOT}" == "/" || "${MONITORING_ROOT}" == "${ROOT_DIR}" ]]; then
  echo "Refusing unsafe monitoring storage path: ${MONITORING_ROOT}" >&2
  exit 1
fi

RUNTIME_IMAGE="$(docker inspect --format '{{.Config.Image}}' "${CONTAINER_NAME}")"
if [[ -z "${RUNTIME_IMAGE}" ]]; then
  echo "Could not resolve the running inference image from ${CONTAINER_NAME}." >&2
  exit 1
fi

docker run --rm \
  --user 0:0 \
  --entrypoint /bin/sh \
  --volume "${MONITORING_ROOT}:/monitoring" \
  "${RUNTIME_IMAGE}" \
  -c 'mkdir -p /monitoring/samples /monitoring/reports /monitoring/textfile && chown -R "$1:$2" /monitoring && chmod 0700 /monitoring/samples /monitoring/reports && chmod 0755 /monitoring/textfile' \
  -- "$(id -u)" "$(id -g)"

echo "Monitoring storage ready: ${MONITORING_ROOT}"
