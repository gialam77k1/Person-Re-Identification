#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MARKET1501_EVAL_CONFIG="${MARKET1501_EVAL_CONFIG:-configs/market1501_v2_eval.yaml}"
RELEASE_ID="${RELEASE_ID:-market1501-duke-vit-bnneck:v2}"
CHECKPOINT="${CHECKPOINT:-artifacts/duke/market1501-duke-vit-bnneck-v2/checkpoints/best_model.pth}"
REPORT="${MARKET1501_V2_REPORT:-artifacts/duke/market1501-duke-vit-bnneck-v2/evaluations/market1501/metrics/evaluation_market1501.json}"
CONDA_ENV="${REID_CONDA_ENV:-reid}"

cd "${PROJECT_ROOT}"
test -f "${CHECKPOINT}"
test -f "${MARKET1501_EVAL_CONFIG}"
./scripts/dvc_local.sh pull datasets/Market-1501-v15.09.15.dvc
conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/evaluate.py --config "${MARKET1501_EVAL_CONFIG}" --checkpoint "${CHECKPOINT}"
test -s "${REPORT}" || {
  echo "ERROR: Market-1501 evaluation did not create the required report: ${REPORT}" >&2
  exit 1
}
conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/validate_v2_evaluation.py \
    --report "${REPORT}" \
    --release-id "${RELEASE_ID}" \
    --dataset market1501
