#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-all}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-reid}"

pass() {
  printf 'PASS: %s\n' "$1"
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || {
    printf 'FAIL: không tìm thấy lệnh %s\n' "${command_name}" >&2
    return 1
  }
  pass "command ${command_name}"
}

check_prerequisites() {
  require_command git
  require_command docker
  require_command curl
  require_command conda
  require_command nvidia-smi

  docker info >/dev/null
  docker compose version >/dev/null
  pass "Docker daemon và Docker Compose"

  conda run -n "${CONDA_ENV_NAME}" python --version >/dev/null
  conda run -n "${CONDA_ENV_NAME}" dvc --version >/dev/null
  pass "Conda environment ${CONDA_ENV_NAME} và DVC"

  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >/dev/null
  pass "NVIDIA GPU và driver"

  [[ -f "${PROJECT_ROOT}/.env.mlops" ]] || {
    printf 'FAIL: thiếu %s/.env.mlops\n' "${PROJECT_ROOT}" >&2
    return 1
  }
  if grep -q 'change-me' "${PROJECT_ROOT}/.env.mlops"; then
    printf 'FAIL: .env.mlops vẫn chứa giá trị change-me\n' >&2
    return 1
  fi
  pass ".env.mlops đã được cấu hình"

  "${PROJECT_ROOT}/scripts/mlops_stack.sh" validate
}

check_services() {
  "${PROJECT_ROOT}/scripts/mlops_stack.sh" verify
}

cd "${PROJECT_ROOT}"
case "${MODE}" in
  prerequisites)
    check_prerequisites
    ;;
  services)
    check_services
    ;;
  all)
    check_prerequisites
    check_services
    ;;
  *)
    echo "Usage: $0 {prerequisites|services|all}" >&2
    exit 2
    ;;
esac

echo "Self-hosted runner smoke check: PASS (${MODE})"
