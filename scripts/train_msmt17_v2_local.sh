#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="configs/msmt17_v2_train.yaml"
DECISION_PATH="artifacts/cross-domain/market1501-v1-on-msmt17/retraining/decision.json"
CHECKPOINT_PATH="artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth"
DATASET_PATH="datasets/MSMT17_V1"
DATASET_DVC_PATH="datasets/MSMT17_V1.dvc"
CHECKPOINT_DVC_PATH="artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth.dvc"
RUN_ROOT="artifacts/msmt17/market1501-msmt17-vit-bnneck-v2"
CONDA_ENV="${REID_CONDA_ENV:-reid}"
CHECK_ONLY=false

if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
elif [[ -n "${1:-}" ]]; then
  echo "Cách dùng: $0 [--check]" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Không tìm thấy conda trong PATH." >&2
  exit 1
fi

conda run -n "${CONDA_ENV}" python src/validate_msmt17_v2_training.py \
  --decision "${DECISION_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --dataset "${DATASET_PATH}" \
  --config "${CONFIG_PATH}" \
  --workspace .

if [[ ! -f .env.mlops ]]; then
  echo "Thiếu .env.mlops để kiểm tra MLflow và DVC remote." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env.mlops
set +a
MLFLOW_BASE_URL="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
MLFLOW_HEALTH_URL="${MLFLOW_BASE_URL%/}/health"
curl --fail --silent --show-error --max-time 5 "${MLFLOW_HEALTH_URL}" >/dev/null
echo "MLflow tracking server: OK (${MLFLOW_HEALTH_URL})"

"${PROJECT_ROOT}/scripts/dvc_local.sh" status -c \
  "${DATASET_DVC_PATH}" \
  "${CHECKPOINT_DVC_PATH}"
echo "DVC dataset/checkpoint remote: OK"

if [[ "${CHECK_ONLY}" == true ]]; then
  echo "MSMT17 v2 training preflight: OK"
  exit 0
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run root đã tồn tại: ${RUN_ROOT}" >&2
  echo "Không ghi đè artifact v2; hãy lưu hoặc đổi tên run hiện tại trước." >&2
  exit 1
fi

exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/train.py --config "${CONFIG_PATH}"
