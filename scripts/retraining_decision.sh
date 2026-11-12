#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVALUATION_REPORT="${EVALUATION_REPORT:-artifacts/cross-domain/market1501-v1-on-duke/metrics/v1_on_duke.json}"
DRIFT_REPORT="${DRIFT_REPORT:-artifacts/cross-domain/market1501-v1-on-duke/drift/drift_report.json}"
OUTPUT="${OUTPUT:-artifacts/cross-domain/market1501-v1-on-duke/retraining/decision.json}"
APPROVED_BY="${1:-${APPROVED_BY:-}}"

arguments=(
  --drift-report "${DRIFT_REPORT}"
  --evaluation-report "${EVALUATION_REPORT}"
  --config configs/retraining_decision.yaml
  --approved-by "${APPROVED_BY}"
  --output "${OUTPUT}"
  --base-release "${RELEASE_ID:-market1501-vit-bnneck:v1-dvc}"
  --target-dataset "${TARGET_DATASET:-dukemtmc-reid}"
)
if [[ "${REQUIRE_APPROVED:-0}" == "1" ]]; then
  arguments+=(--require-approved)
fi

cd "${PROJECT_ROOT}"
exec conda run --no-capture-output -n reid python src/retraining_decision_gate.py "${arguments[@]}"
