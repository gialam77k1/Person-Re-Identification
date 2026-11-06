#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-configs/msmt17_v2_train.yaml}"
RELEASE_ID="${RELEASE_ID:-market1501-msmt17-vit-bnneck:v2}"
CHECKPOINT="${CHECKPOINT:-artifacts/msmt17/market1501-msmt17-vit-bnneck-v2/checkpoints/best_model.pth}"
REPORT="${REPORT:-artifacts/msmt17/market1501-msmt17-vit-bnneck-v2/metrics/evaluation_latest.json}"
CONDA_ENV="${REID_CONDA_ENV:-reid}"

cd "${PROJECT_ROOT}"
test -f "${CHECKPOINT}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/evaluate.py --config "${CONFIG_PATH}" --checkpoint "${CHECKPOINT}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/validate_v2_evaluation.py \
    --report "${REPORT}" \
    --release-id "${RELEASE_ID}" \
    --dataset msmt17
