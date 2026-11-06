#!/usr/bin/env bash
set -euo pipefail

PIPELINE_ROOT="${V2_PIPELINE_ROOT:-artifacts/pipeline/market1501-msmt17-vit-bnneck-v2}"
CHECKPOINT_ROOT="${PIPELINE_ROOT}/checkpoint"
MSMT17_REPORT="${PIPELINE_ROOT}/evaluate_msmt17/metrics/evaluation_msmt17.json"
MARKET1501_REPORT="${PIPELINE_ROOT}/evaluate_market1501/metrics/evaluation_market1501.json"
FORGETTING_REPORT="${PIPELINE_ROOT}/release/catastrophic_forgetting_report.json"
REFERENCE_ROOT="${PIPELINE_ROOT}/reference/embeddings"

python src/catastrophic_forgetting_gate.py \
  --msmt17-v1 model_releases/market1501-vit-bnneck/v1/msmt17_cross_domain_metrics.json \
  --msmt17-v2 "${MSMT17_REPORT}" \
  --market1501-v1-manifest model_releases/market1501-vit-bnneck/v1/model_manifest.json \
  --market1501-v2 "${MARKET1501_REPORT}" \
  --policy configs/catastrophic_forgetting.yaml \
  --mlflow-run "${CHECKPOINT_ROOT}/mlflow_run.json" \
  --output "${FORGETTING_REPORT}"

python src/validate_v2_dvc_release.py \
  --checkpoint-manifest "${CHECKPOINT_ROOT}/checkpoint_manifest.json" \
  --data-manifest "${PIPELINE_ROOT}/validate/data_manifest.json" \
  --msmt17-evaluation "${MSMT17_REPORT}" \
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
  --output "${PIPELINE_ROOT}/release/validation_report.json" \
  --manifest-output "${PIPELINE_ROOT}/release/model_manifest.json"
