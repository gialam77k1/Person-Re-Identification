#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-configs/duke_v2_train.yaml}"
DECISION_PATH="${DECISION_PATH:-artifacts/cross-domain/market1501-v1-on-duke/retraining/decision.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth}"
DATASET_PATH="${DATASET_PATH:-datasets/dukemtmc-reid}"
DATASET_DVC_PATH="${DATASET_DVC_PATH:-datasets/dukemtmc-reid.dvc}"
CHECKPOINT_DVC_PATH="${CHECKPOINT_DVC_PATH-artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth.dvc}"
RUN_ROOT="${RUN_ROOT:-artifacts/duke/market1501-duke-vit-bnneck-v2}"
CONDA_ENV="${REID_CONDA_ENV:-reid}"
CHECK_ONLY=false

# Keep allocator behavior identical between a local run and GitHub Actions.
# The value can still be overridden explicitly by the operator.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

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

validation_args=(
  --decision "${DECISION_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --dataset "${DATASET_PATH}"
  --config "${CONFIG_PATH}"
  --workspace .
)
if [[ -n "${DATASET_REVISION:-}" ]]; then
  validation_args+=(--dataset-revision "${DATASET_REVISION}")
fi
conda run -n "${CONDA_ENV}" python src/validate_duke_v2_training.py "${validation_args[@]}"

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

if [[ -n "${CHECKPOINT_DVC_PATH}" && "${DATASET_DVC_PATH}" == "datasets/dukemtmc-reid.dvc" ]]; then
  "${PROJECT_ROOT}/scripts/dvc_local.sh" status -c \
    "${DATASET_DVC_PATH}" \
    "${CHECKPOINT_DVC_PATH}"
elif [[ -n "${CHECKPOINT_DVC_PATH}" ]]; then
  test -s "${DATASET_DVC_PATH}"
  "${PROJECT_ROOT}/scripts/dvc_local.sh" status -c "${CHECKPOINT_DVC_PATH}"
fi
test -s "${CHECKPOINT_PATH}"
echo "DVC dataset/checkpoint remote: OK"

if [[ "${CHECK_ONLY}" == true ]]; then
  echo "DukeMTMC-reID v2 training preflight: OK"
  exit 0
fi

if [[ -e "${RUN_ROOT}" ]]; then
  if conda run -n "${CONDA_ENV}" python -m src.validate_v2_training_completion \
    --run-root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/logs/training_completion_report.json" >/dev/null 2>&1; then
    echo "Run v2 hoàn chỉnh đã tồn tại; tái sử dụng artifact thay vì ghi đè."
    exit 0
  fi
  echo "Run root đã tồn tại: ${RUN_ROOT}" >&2
  echo "Không ghi đè artifact v2; hãy lưu hoặc đổi tên run hiện tại trước." >&2
  exit 1
fi

exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/train.py --config "${CONFIG_PATH}"
