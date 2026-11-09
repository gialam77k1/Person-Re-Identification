#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-configs/msmt17_v2_eval.yaml}"
RELEASE_ID="${RELEASE_ID:-market1501-msmt17-vit-bnneck:v2}"
CHECKPOINT="${CHECKPOINT:-artifacts/msmt17/market1501-msmt17-vit-bnneck-v2/checkpoints/best_model.pth}"
REPORT="${REPORT:-artifacts/msmt17/market1501-msmt17-vit-bnneck-v2/evaluations/msmt17/metrics/evaluation_msmt17.json}"
CONDA_ENV="${REID_CONDA_ENV:-reid}"

cd "${PROJECT_ROOT}"
test -f "${CHECKPOINT}"
./scripts/dvc_local.sh pull datasets/MSMT17_V1.dvc
conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/evaluate.py --config "${CONFIG_PATH}" --checkpoint "${CHECKPOINT}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/validate_v2_evaluation.py \
    --report "${REPORT}" \
    --release-id "${RELEASE_ID}" \
    --dataset msmt17
