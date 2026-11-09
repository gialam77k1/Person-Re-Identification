#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-configs/duke_v2_eval.yaml}"
RELEASE_ID="${RELEASE_ID:-market1501-duke-vit-bnneck:v2}"
CHECKPOINT="${CHECKPOINT:-artifacts/duke/market1501-duke-vit-bnneck-v2/checkpoints/best_model.pth}"
REPORT="${REPORT:-artifacts/duke/market1501-duke-vit-bnneck-v2/evaluations/duke/metrics/evaluation_duke.json}"
CONDA_ENV="${REID_CONDA_ENV:-reid}"

cd "${PROJECT_ROOT}"
test -f "${CHECKPOINT}"
./scripts/dvc_local.sh pull datasets/dukemtmc-reid.dvc
conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/evaluate.py --config "${CONFIG_PATH}" --checkpoint "${CHECKPOINT}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/validate_v2_evaluation.py \
    --report "${REPORT}" \
    --release-id "${RELEASE_ID}" \
    --dataset duke
