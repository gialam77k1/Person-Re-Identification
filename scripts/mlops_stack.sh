#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.mlops.yml"
ENV_FILE="${PROJECT_ROOT}/.env.mlops"

usage() {
  echo "Usage: $0 {validate|up|status|verify|logs|down}"
}

require_env() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Thiếu ${ENV_FILE}. Hãy copy .env.mlops.example thành .env.mlops và đổi secret." >&2
    exit 1
  fi
  if grep -q "change-me" "${ENV_FILE}"; then
    echo ".env.mlops vẫn chứa giá trị change-me. Hãy đổi credential trước khi chạy." >&2
    exit 1
  fi
}

load_env() {
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
}

compose() {
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
}

validate() {
  require_env
  load_env
  command -v docker >/dev/null 2>&1 || {
    echo "Không tìm thấy Docker CLI." >&2
    exit 1
  }
  docker info >/dev/null
  compose config --quiet

  local model_repository="${TRITON_MODEL_REPOSITORY:-${PROJECT_ROOT}/artifacts/triton/market1501-vit-bnneck-v1/model_repository}"
  if [[ "${model_repository}" != /* ]]; then
    model_repository="${PROJECT_ROOT}/${model_repository#./}"
  fi
  local model_dir="${model_repository}/reid_embedding/1"
  [[ -f "${model_dir}/model.onnx" ]] || {
    echo "Thiếu ${model_dir}/model.onnx" >&2
    exit 1
  }
  [[ -f "${model_dir}/model_embedding.onnx.data" ]] || {
    echo "Thiếu ${model_dir}/model_embedding.onnx.data" >&2
    exit 1
  }

  echo "MLOps stack validation: PASS"
}

verify_url() {
  local name="$1"
  local url="$2"
  local attempt
  for attempt in $(seq 1 30); do
    if curl --fail --silent --max-time 5 "${url}" >/dev/null 2>&1; then
      echo "${name}: healthy"
      return 0
    fi
    sleep 2
  done
  echo "${name}: health check failed after 60 seconds (${url})" >&2
  return 1
}

verify() {
  require_env
  load_env
  compose exec -T postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null
  echo "PostgreSQL: healthy"
  verify_url "MinIO" "http://localhost:${MINIO_API_PORT:-9000}/minio/health/live"
  verify_url "MLflow" "http://localhost:${MLFLOW_PORT:-5000}/health"
  verify_url "Qdrant" "http://localhost:${QDRANT_HTTP_PORT:-6333}/healthz"
  verify_url "Triton" "http://localhost:${TRITON_HTTP_PORT:-8000}/v2/health/ready"
  compose run --rm minio-init >/dev/null
  echo "MinIO buckets: ready"
  echo "MLOps stack verification: PASS"
}

cd "${PROJECT_ROOT}"

case "${1:-}" in
  validate)
    validate
    ;;
  up)
    validate
    compose up -d --build
    compose ps
    ;;
  status)
    require_env
    compose ps
    ;;
  verify)
    verify
    ;;
  logs)
    require_env
    compose logs --tail=200 "${@:2}"
    ;;
  down)
    require_env
    compose down
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
