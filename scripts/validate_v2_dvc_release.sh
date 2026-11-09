#!/usr/bin/env bash
set -euo pipefail

PIPELINE_ROOT="${V2_PIPELINE_ROOT:-artifacts/pipeline/market1501-duke-vit-bnneck-v2}"
CHECKPOINT_ROOT="${PIPELINE_ROOT}/checkpoint"
DukeMTMC-reID_REPORT="${PIPELINE_ROOT}/evaluate_duke/metrics/evaluation_duke.json"
MARKET1501_REPORT="${PIPELINE_ROOT}/evaluate_market1501/metrics/evaluation_market1501.json"
FORGETTING_REPORT="${PIPELINE_ROOT}/release/catastrophic_forgetting_report.json"
REFERENCE_ROOT="${PIPELINE_ROOT}/reference/embeddings"

python -m src.catastrophic_forgetting_gate \
  --duke-v1 artifacts/cross-domain/market1501-v1-on-duke/metrics/v1_on_duke.json \
  --duke-v2 "${DukeMTMC-reID_REPORT}" \
  --market1501-v1-manifest model_releases/market1501-vit-bnneck/v1/model_manifest.json \
  --market1501-v2 "${MARKET1501_REPORT}" \
  --policy configs/catastrophic_forgetting.yaml \
  --mlflow-run "${CHECKPOINT_ROOT}/mlflow_run.json" \
  --output "${FORGETTING_REPORT}"

python -m src.validate_v2_dvc_release \
  --checkpoint-manifest "${CHECKPOINT_ROOT}/checkpoint_manifest.json" \
  --data-manifest "${PIPELINE_ROOT}/validate/data_manifest.json" \
  --duke-evaluation "${DukeMTMC-reID_REPORT}" \
  --market1501-evaluation "${MARKET1501_REPORT}" \
  --forgetting-report "${FORGETTING_REPORT}" \
  --onnx-manifest "${PIPELINE_ROOT}/onnx/exports/onnx_export_manifest.json" \
  --parity-report "${PIPELINE_ROOT}/parity/metrics/onnx_parity.json" \
  --reference-manifest "${REFERENCE_ROOT}/reference_manifest.json" \
  --checkpoint "${CHECKPOINT_ROOT}/best_model.pth" \
  --onnx "${PIPELINE_ROOT}/onnx/exports/model_embedding.onnx" \
  --reference-root "${REFERENCE_ROOT}" \
  --preprocessing-source model_releases/market1501-vit-bnneck/v1/preprocessing.json \
  --preprocessing-output "${PIPELINE_ROOT}/release/preprocessing.json" \
  --policy configs/model_validation_v2.yaml \
  --mlflow-run "${CHECKPOINT_ROOT}/mlflow_run.json" \
  --output "${PIPELINE_ROOT}/release/validation_report.json" \
  --manifest-output "${PIPELINE_ROOT}/release/model_manifest.json"
